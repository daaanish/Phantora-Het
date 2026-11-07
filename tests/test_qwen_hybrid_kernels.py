import builtins
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch

import qwen_hybrid_kernels as kernels


def recorded_examples():
    events = []
    adapters = kernels.KernelAdapters(lambda key, device: events.append(key))
    x = torch.empty(1, 2, 4, requires_grad=True)
    adapters.l2norm(x).sum().backward()
    x = torch.empty(1, 8, 4).transpose(1, 2).requires_grad_()
    weight = torch.empty(4, 3, requires_grad=True)
    bias = torch.empty(4, requires_grad=True)
    adapters.causal_conv1d_fn(x, weight, bias, activation="silu").sum().backward()
    inputs = [torch.empty(shape, requires_grad=True) for shape in (
        (1, 8, 2, 4), (1, 8, 2, 4), (1, 8, 4, 6), (1, 8, 4), (1, 8, 4),
    )]
    output, state = adapters.chunk_gated_delta_rule(*inputs)
    assert state is None
    output.sum().backward()
    return events


class KernelAdapterTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.adapters = kernels.KernelAdapters(lambda key, device: self.events.append(key))

    def test_cross_language_key_fixture(self):
        fixture = json.loads((Path(__file__).parent / "assets/qwen_hybrid_keys.json").read_text())
        self.assertEqual(recorded_examples(), fixture)

    def test_native_sink_sends_key_and_raw_current_stream_without_latency_lookup(self):
        send = Mock(return_value=0)
        library = ModuleType("test_native_library")
        library.phantora_gdn_call = send
        stream = ModuleType("test_stream")
        stream.cuda_stream = 1234
        with patch.object(kernels.ctypes, "CDLL", return_value=library), \
                patch.object(torch.cuda, "device", return_value=nullcontext()) as device_context, \
                patch.object(torch.cuda, "current_stream", return_value=stream):
            sink = kernels.NativeEventSink()
            key = recorded_examples()[0]
            sink(key, torch.device("cuda", 1))
        device_context.assert_called_once_with(torch.device("cuda", 1))
        raw_stream, payload, size = send.call_args.args
        self.assertEqual(raw_stream.value, 1234)
        self.assertEqual(size, len(payload))
        self.assertEqual(json.loads(payload), key)

    def test_l2norm_dtype_graph_and_incoming_stride(self):
        x = torch.empty(2, 3, 8, requires_grad=True)
        y = self.adapters.l2norm(x, output_dtype=torch.bfloat16)
        self.assertEqual(y.dtype, torch.bfloat16)
        y.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual([e["direction"] for e in self.events], ["forward", "backward"])
        self.assertEqual(self.events[1]["grad_output"]["strides"], [0, 0, 0])

    def test_convolution_keeps_weight_bias_and_input_gradients(self):
        x = torch.empty(2, 9, 4).transpose(1, 2).requires_grad_()
        weight = torch.empty(4, 3, requires_grad=True)
        bias = torch.empty(4, requires_grad=True)
        output = self.adapters.causal_conv1d_fn(x, weight, bias, activation="swish")
        self.assertEqual(output.stride(), x.stride())
        output.sum().backward()
        for tensor in (x, weight, bias):
            self.assertEqual(tensor.grad.shape, tensor.shape)
        self.assertEqual(self.events[0]["options"]["activation"], "silu")

    def test_chunk_keeps_all_five_gradients_and_fused_norm_option(self):
        inputs = [torch.empty(shape, dtype=torch.float32 if i == 3 else torch.bfloat16, requires_grad=True) for i, shape in enumerate((
            (2, 8, 2, 4), (2, 8, 2, 4), (2, 8, 4, 6), (2, 8, 4), (2, 8, 4),
        ))]
        output, state = self.adapters.chunk_gated_delta_rule(*inputs, use_qk_l2norm_in_kernel=True)
        self.assertIsNone(state)
        self.assertEqual(output.shape, inputs[2].shape)
        output.sum().backward()
        for tensor in inputs:
            self.assertEqual(tensor.grad.shape, tensor.shape)
            self.assertEqual(tensor.grad.dtype, tensor.dtype)
        self.assertEqual(len(self.events), 2)
        self.assertEqual(self.events[0]["options"]["use_qk_l2norm_in_kernel"], "true")

    def test_no_grad_and_partial_grad_mask(self):
        x, weight = torch.empty(1, 2, 4), torch.empty(2, 3, requires_grad=True)
        self.adapters.causal_conv1d_fn(x, weight).sum().backward()
        self.assertEqual([None if t is None else t["requires_grad"] for t in self.events[0]["inputs"]], [False, True, None])
        self.assertIsNotNone(weight.grad)
        with torch.no_grad():
            output = self.adapters.l2norm(weight)
        self.assertFalse(output.requires_grad)
        self.assertFalse(self.events[-1]["grad_enabled"])

    def test_unsupported_state_and_malformed_inputs_rejected(self):
        x, weight = torch.empty(1, 2, 4), torch.empty(2, 3)
        for kwargs in (dict(return_final_states=True), dict(seq_idx=x), dict(activation="relu")):
            with self.assertRaises(ValueError):
                self.adapters.causal_conv1d_fn(x, weight, **kwargs)
        with self.assertRaises(ValueError):
            self.adapters.l2norm(x, eps=float("nan"))
        self.assertEqual(self.events, [])

    def test_higher_order_gradients_are_explicitly_rejected(self):
        x = torch.empty(2, 4, requires_grad=True)
        output = self.adapters.l2norm(x)
        with self.assertRaisesRegex(RuntimeError, "higher-order"):
            torch.autograd.grad(output.sum(), x, create_graph=True)

    def test_installer_preserves_namespaces_repairs_aliases_and_imports_no_kernels(self):
        existing = ModuleType("fla")
        existing.unrelated = object()
        captured = ModuleType("megatron.core.ssm.gated_delta_net")
        sink = lambda key, device: None
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("triton", "fla", "causal_conv1d")):
                raise AssertionError(f"real kernel import attempted: {name}")
            return real_import(name, *args, **kwargs)

        with patch.object(kernels, "_ADAPTERS", None), patch.dict(sys.modules, {
            "fla": existing, "megatron.core.ssm.gated_delta_net": captured,
        }), patch.dict(os.environ, PHANTORA="1"), patch("builtins.__import__", guarded_import):
            adapters = kernels.install_simulation_kernels(sink)
            self.assertIs(kernels.install_simulation_kernels(sink), adapters)
            self.assertIs(sys.modules["fla"], existing)
            self.assertIs(captured.l2norm.__self__, adapters)
            self.assertIs(captured.chunk_gated_delta_rule.__self__, adapters)
            self.assertIs(captured.causal_conv1d_fn.__self__, adapters)
            self.assertTrue(captured.HAVE_FLA)
            self.assertIs(sys.modules["fla.modules.l2norm"].l2norm.__self__, adapters)
            self.assertTrue(hasattr(existing, "unrelated"))
        with patch.dict(os.environ):
            os.environ.pop("PHANTORA", None)
            with self.assertRaises(RuntimeError):
                kernels.install_simulation_kernels(sink)


if __name__ == "__main__":
    unittest.main()
