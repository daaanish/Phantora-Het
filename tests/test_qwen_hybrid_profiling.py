import contextlib
from dataclasses import asdict
import copy
import json
import runpy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent / "perfdb"))
from gdn_common import TORCH_REVISION, canonical_key, merge_profile, read_measured, validate_provenance
from gdn_profile import allocate_strided, assert_real_execution, build_thunk, checked_thunk
from gdn_references import causal_conv_reference, l2norm_reference, recurrent_gdn_reference
from qwen_hybrid_report import aggregate, compare, report
from qwen_hybrid_config import HybridConfig
from qwen_hybrid_control import expected_events
from gdn_validate import compare_tensor
import torch

FIXTURES = json.loads((Path(__file__).parent / "assets/qwen_hybrid_keys.json").read_text())
SOURCE = dict(gpu="test GPU", torch="2.9.1+cu128", torch_revision=TORCH_REVISION, cuda="12.8", triton="3.7.0", fla="0.4.0", causal_conv1d="1.5.3.post1", measurement="cupti_kernel_sum")


class ProfilingTests(unittest.TestCase):
    def test_keys_roundtrip_and_transport_guards(self):
        for key in FIXTURES:
            self.assertEqual(canonical_key(key), canonical_key(canonical_key(key)))
        key = copy.deepcopy(FIXTURES[0]); key["options"]["eps"] = "0.000001"
        self.assertEqual(canonical_key(key), canonical_key(FIXTURES[0]))
        for raw in ['{"version":1,"version":1}', ' ' * 32769]:
            with self.assertRaises(ValueError): canonical_key(raw)

    def test_atomic_positive_only_and_failure_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gdn.csv"
            self.assertTrue(merge_profile(path, FIXTURES[:1], SOURCE, lambda _: 100)["committed"])
            before = path.read_bytes()
            self.assertFalse(merge_profile(path, FIXTURES, SOURCE, lambda _: 0)["committed"])
            self.assertEqual(before, path.read_bytes())
            calls = iter([123, RuntimeError("kernel failure")])
            def partial(_):
                value = next(calls)
                if isinstance(value, Exception): raise value
                return value
            result = merge_profile(path, FIXTURES, SOURCE, partial)
            self.assertFalse(result["committed"])
            self.assertTrue(result["measured"])
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(len(read_measured(path, SOURCE)), 1)
            changed = dict(SOURCE, measurement="cuda_event_wall")
            with self.assertRaises(ValueError): read_measured(path, changed)
        for name, value in (("cuda", "12.7"), ("torch_revision", None), ("gpu", "")):
            with self.assertRaises(ValueError): validate_provenance(dict(SOURCE, **{name: value}))

    def test_distinct_strided_storage_and_incoming_gradient(self):
        info = dict(shape=[2, 3], strides=[1, 2], dtype="float32", requires_grad=True)
        left = allocate_strided(info, "q", torch, "cpu")
        right = allocate_strided(info, "k", torch, "cpu")
        self.assertNotEqual(left.data_ptr(), right.data_ptr())
        self.assertEqual(left.stride(), (1, 2)); self.assertTrue(left.is_leaf)
        zero = allocate_strided(dict(info, strides=[0, 0], requires_grad=False), "grad_output", torch, "cpu")
        self.assertEqual(zero.stride(), (0, 0))

    def test_backward_builds_graph_once_outside_timed_thunk(self):
        key = copy.deepcopy(FIXTURES[1])
        kernel = Mock(return_value=object())
        grad = Mock(return_value=(object(),))
        api = types.SimpleNamespace(float32=object(), set_grad_enabled=lambda _: contextlib.nullcontext(), autograd=types.SimpleNamespace(grad=grad))
        thunk = build_thunk(key, {"l2norm": kernel}, api, allocator=lambda *_: object())
        self.assertEqual(kernel.call_count, 1); self.assertEqual(grad.call_count, 0)
        thunk(); thunk()
        self.assertEqual(kernel.call_count, 1); self.assertEqual(grad.call_count, 2)
        self.assertTrue(grad.call_args.kwargs["retain_graph"])
        self.assertFalse(grad.call_args.kwargs["allow_unused"])

    def test_finiteness_rejected_before_timing(self):
        key = copy.deepcopy(FIXTURES[0])
        with patch("gdn_profile.allocate_strided", side_effect=lambda info, name, api: allocate_strided(info, name, api, "cpu")):
            with self.assertRaises(ValueError):
                checked_thunk(key, {"l2norm": lambda x, **_: x * float("nan")}, torch)

    def test_unit_normalized_chunk_inputs_keep_layout(self):
        key = next(copy.deepcopy(key) for key in FIXTURES if key["op"] == "chunk_gated_delta_rule" and key["direction"] == "forward")
        key["options"]["use_qk_l2norm_in_kernel"] = "false"
        captured = []
        def kernel(*inputs, **kwargs):
            captured.extend(inputs)
            return inputs[2], None
        thunk = build_thunk(key, {"chunk_gated_delta_rule": kernel}, torch,
            allocator=lambda info, name: allocate_strided(info, name, torch, "cpu"))
        thunk()
        for tensor, info in zip(captured[:2], key["inputs"][:2]):
            self.assertEqual(tensor.stride(), tuple(info["strides"]))
            self.assertEqual(tensor.requires_grad, info["requires_grad"])
            torch.testing.assert_close(tensor.float().square().sum(-1), torch.ones_like(tensor[..., 0]).float(), rtol=.02, atol=.02)

    def test_conv_bfloat16_accumulates_and_gates_in_fp32(self):
        torch.manual_seed(123)
        x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        weight = torch.randn(4, 3, dtype=torch.bfloat16)
        bias = torch.randn(4, dtype=torch.bfloat16)
        expected = torch.nn.functional.conv1d(x.float(), weight.float().unsqueeze(1), bias.float(), padding=2, groups=4)[..., :8]
        expected = torch.nn.functional.silu(expected).to(x.dtype)
        torch.testing.assert_close(causal_conv_reference(x, weight, bias, "silu"), expected, rtol=0, atol=0)

    def test_u64_timing_overflow_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            result = merge_profile(Path(folder) / "gdn.csv", FIXTURES[:1], SOURCE, lambda _: 2**64)
            self.assertFalse(result["committed"])

    def test_relative_l2_rejects_zero_or_large_error(self):
        expected = torch.full((4,), .001)
        with self.assertRaises(RuntimeError): compare_tensor(torch, torch.zeros(4), expected, .03, .03, .03, "zero")
        with self.assertRaises(RuntimeError): compare_tensor(torch, expected * 2, expected, .03, .03, .03, "large")
        self.assertEqual(compare_tensor(torch, expected, expected, .03, .03, .03, "exact")["relative_l2_error"], 0)

    def test_real_guard(self):
        for variable in ("PHANTORA", "LD_PRELOAD"):
            with patch.dict("os.environ", {variable: "1"}):
                with self.assertRaises(RuntimeError): assert_real_execution()

    def test_tiny_equations_and_all_gradients(self):
        x = torch.tensor([[[1., 2., 3.]]], requires_grad=True)
        weight = torch.tensor([[1., 10.]], requires_grad=True)
        self.assertTrue(torch.equal(causal_conv_reference(x, weight), torch.tensor([[[10., 21., 32.]]])))
        tiny = torch.tensor([[1e-4]])
        self.assertLess(l2norm_reference(tiny).item(), 0.11)
        q, k = [torch.ones(1, 2, 1, 1, requires_grad=True) for _ in range(2)]
        v = torch.tensor([2., 4.]).reshape(1, 2, 1, 1).requires_grad_()
        g = torch.full((1, 2, 1), -0.6931471805599453, requires_grad=True)
        beta = torch.full((1, 2, 1), .5, requires_grad=True)
        output = recurrent_gdn_reference(q, k, v, g, beta)
        torch.testing.assert_close(output.flatten(), torch.tensor([1., 2.25]))
        grads = torch.autograd.grad(output.sum(), (q, k, v, g, beta))
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))


class MatrixTests(unittest.TestCase):
    def test_explicit_topology_and_debug_first(self):
        commands = runpy.run_path(str(Path(__file__).parent / "docker/megatron/hybrid_matrix.py"))["commands"]
        args = types.SimpleNamespace(gpus_per_host="1,2,1,1,4", preset="both", image="hybrid", perf_db="db", mode="strict", iterations=4, out=Path("runs"))
        cases = commands(args)
        self.assertEqual(len(cases), 10)
        self.assertTrue(cases[4]["case"].startswith("debug"))
        self.assertTrue(cases[5]["case"].startswith("9b"))
        self.assertEqual(cases[4]["hosts"], 2)
        args.gpus_per_host = "2,2,1,1,4"
        with self.assertRaises(ValueError): commands(args)


class ReportTests(unittest.TestCase):
    def records(self):
        return [dict(version=1, run_id="run", rank=rank, world_size=2, iterations=3, durations=values, tokens_per_iteration=1024) for rank, values in enumerate(([99., 1., 3.], [99., 2., 2.]))]

    def bundle(self, kind="replay"):
        workload = HybridConfig(data_parallel_size=2, world_size=2, gradient_accumulation=4)
        counts = expected_events(workload, 0, 3)
        ranks = {str(rank): dict(started=True, finished=True, exited=True, observer_started=True, observer_finished=True, host=dict(hostname=f"host-{rank}", pid=rank + 1), expected=counts, actual=counts) for rank in range(2)}
        return dict(run_id="run", world_size=2, kind=kind,
            status=dict(version=1, run_id="run", expected_world=2, phase="complete", valid=True, mode="strict", ranks=ranks, errors=[], categories={}, malformed_messages=0, timing_misses={name: 0 for name in ("compute", "memcpy", "flash_attn", "gdn", "malformed_gdn")}),
            records=self.records(), runner=dict(run_id="run", valid=True, exit_code=0, worker_exit_codes=[0, 0]),
            match=dict(workload=asdict(workload), software={"image": "pinned"}, gpu="test GPU", gpus_per_host=[1, 1], network={"latency": 1}, cpu_time_policy=dict(replay="ignored", hardware="wall_clock_elapsed")))

    def test_max_rank_steady_series(self):
        result = aggregate(self.records(), "run", 2)
        self.assertEqual(result["steady_max_rank_seconds"], [2., 3.])
        self.assertEqual(result["tokens_per_second"], 409.6)
        self.assertTrue(compare(report(self.bundle()), report(self.bundle("hardware")))["passed"])
        hardware = report(self.bundle("hardware"))
        hardware["mean_seconds"] = 10.
        self.assertFalse(compare(report(self.bundle()), hardware)["passed"])

    def test_report_rejects_inconsistent_native_or_runner_evidence(self):
        for mutate in (
            lambda b: b["status"].pop("ranks"),
            lambda b: b["status"].update(version=2),
            lambda b: b["status"]["ranks"]["1"].update(observer_started=False),
            lambda b: b["status"]["ranks"]["1"].update(host=b["status"]["ranks"]["0"]["host"]),
            lambda b: b["status"]["ranks"]["1"]["actual"].update({"l2norm/backward": 0}),
            lambda b: b["status"]["timing_misses"].update(gdn=1),
            lambda b: b["status"].update(errors=["bad"]),
            lambda b: b["runner"].update(worker_exit_codes=[0, 7]),
            lambda b: b["runner"].update(exit_code=1),
            lambda b: b["records"][0].update(tokens_per_iteration=17),
            lambda b: b["match"]["workload"].update(world_size=7),
        ):
            bundle = copy.deepcopy(self.bundle())
            mutate(bundle)
            with self.assertRaises(ValueError): report(bundle)

    def test_missing_duplicate_stale_nonfinite_and_policy(self):
        for records in (self.records()[:1], self.records() * 2):
            with self.assertRaises(ValueError): aggregate(records, "run", 2)
        for field, value in (("run_id", "stale"), ("durations", [1., float("nan"), 2.]), ("durations", ["1.00", "2.00", "3.00"])):
            records = self.records(); records[0][field] = value
            with self.assertRaises(ValueError): aggregate(records, "run", 2)
        bundle = self.bundle(); bundle["status"]["valid"] = False
        with self.assertRaises(ValueError): report(bundle)
        left, right = report(self.bundle()), report(self.bundle("hardware"))
        right["match"]["cpu_time_policy"] = dict(replay="include", hardware="wall_clock_elapsed")
        with self.assertRaises(ValueError): compare(left, right)


if __name__ == "__main__": unittest.main()
