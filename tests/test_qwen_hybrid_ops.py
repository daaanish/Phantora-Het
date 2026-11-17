from pathlib import Path
import runpy
import struct
import unittest

import torch

build = runpy.run_path(str(Path(__file__).resolve().parents[1] / "phantora/phantora/src/qwen_ops.py"))["build"]


class QwenOpTests(unittest.TestCase):
    def payload(self, op, scalars=(), dims=(), keepdim=False):
        return dict(op=op, inputs=[dict(shape=[2, 3], dtype="Float")],
            scalars=[struct.unpack(">Q", struct.pack(">d", float(value)))[0] for value in scalars],
            dims=list(dims), keepdim=keepdim, dtype=None)

    def test_scalar_power_keys_run_requested_exponent(self):
        value = torch.arange(1, 7, dtype=torch.float32).view(2, 3)
        for exponent in (2, 3, -0.5):
            actual = build(self.payload("pow", [exponent]), allocator=lambda info: value)()
            torch.testing.assert_close(actual, value ** exponent)

    def test_reduction_dims_and_keepdim(self):
        value = torch.arange(1, 7, dtype=torch.float32).view(2, 3)
        for name in ("mean", "sum"):
            actual = build(self.payload(name, dims=[1], keepdim=True), allocator=lambda info: value)()
            torch.testing.assert_close(actual, getattr(value, name)(1, keepdim=True))

    def test_inplace_backward_division_runs_the_recorded_scalar(self):
        value = torch.ones(2, 3)
        actual = build(self.payload("div_", [4]), allocator=lambda info: value)()
        self.assertIs(actual, value)
        torch.testing.assert_close(actual, torch.full((2, 3), 0.25))

    def test_square_and_exp_out_replay_the_distinct_aten_calls(self):
        value = torch.linspace(-1, 1, 6).view(2, 3)
        original = value.clone()
        allocate = lambda info: value
        torch.testing.assert_close(build(self.payload("square"), allocator=allocate)(), original.square())
        torch.testing.assert_close(build(self.payload("exp"), allocator=allocate)(), original.exp())
        torch.testing.assert_close(value, original)
        payload = self.payload("exp_out")
        payload["inputs"] *= 2
        result = build(payload, allocator=allocate)()
        self.assertIs(result, value)
        torch.testing.assert_close(value, original.exp())

    def test_cat_reads_distinct_members_and_contiguous_copies_a_strided_source(self):
        value = torch.arange(6, dtype=torch.float32).view(2, 3)
        payload = self.payload("cat", dims=[1])
        payload["inputs"] *= 2
        actual = build(payload, allocator=lambda info: value)()
        torch.testing.assert_close(actual, torch.cat([value, value], 1))
        payload = self.payload("contiguous")
        calls = []
        original = torch.Tensor.contiguous

        def spy(tensor, *args, **kwargs):
            calls.append(tensor.is_contiguous())
            return original(tensor, *args, **kwargs)

        thunk = build(payload, allocator=lambda info: value)
        torch.Tensor.contiguous = spy
        try:
            result = thunk()
        finally:
            torch.Tensor.contiguous = original
        self.assertEqual(calls, [False])
        self.assertEqual(tuple(result.shape), (2, 3))
        self.assertTrue(result.is_contiguous())

    def test_traced_layouts_are_rebuilt_exactly(self):
        # contiguous of a transposed [1,8,4,16] source, as the layout-aware tracer sends it.
        payload = self.payload("contiguous")
        payload["inputs"] = [dict(shape=[1, 8, 4, 16], dtype="Float")]
        payload["layouts"] = [dict(stride=[0, 16, 128, 1], offset=0)]
        seen = []
        original = torch.Tensor.contiguous

        def spy(tensor, *args, **kwargs):
            seen.append((tensor.stride(), tensor.storage_offset()))
            return original(tensor, *args, **kwargs)

        thunk = build(payload, allocator=lambda info: self.fail("allocator used"), device="cpu")
        torch.Tensor.contiguous = spy
        try:
            result = thunk()
        finally:
            torch.Tensor.contiguous = original
        self.assertEqual(seen, [((0, 16, 128, 1), 0)])
        self.assertTrue(result.is_contiguous())
        # cat members keep their own strides and storage offsets, on distinct storage.
        payload = self.payload("cat", dims=[1])
        payload["inputs"] = [dict(shape=[2, 3], dtype="Float")] * 2
        payload["layouts"] = [dict(stride=[1, 2], offset=0), dict(stride=[5, 1], offset=7)]
        captured = []
        real_cat = torch.ops.aten.cat
        thunk = build(payload, device="cpu")
        try:
            torch.ops.aten.cat = lambda members, dim: captured.extend(members) or real_cat(members, dim)
            result = thunk()
        finally:
            torch.ops.aten.cat = real_cat
        self.assertEqual([(m.stride(), m.storage_offset()) for m in captured], [((1, 2), 0), ((5, 1), 7)])
        self.assertNotEqual(captured[0].data_ptr(), captured[1].data_ptr())
        self.assertEqual(tuple(result.shape), (2, 6))
        payload["layouts"] = payload["layouts"][:1]
        with self.assertRaises(ValueError):
            build(payload, device="cpu")

    def test_gate_forward_and_backward_dispatch(self):
        value = torch.ones(2, 3)
        for name in ("exp", "neg", "sigmoid", "silu", "rsqrt", "softplus"):
            payload = self.payload(name, scalars=[1, 20] if name == "softplus" else [])
            self.assertEqual(build(payload, allocator=lambda info: value)().shape, value.shape)
        torch.testing.assert_close(build(self.payload("neg"), allocator=lambda info: value)(), -value)
        for name in ("sigmoid_backward", "silu_backward", "softplus_backward"):
            payload = self.payload(name, scalars=[1, 20] if name == "softplus_backward" else [])
            payload["inputs"] *= 2
            self.assertEqual(build(payload, allocator=lambda info: value)().shape, value.shape)


if __name__ == "__main__":
    unittest.main()
