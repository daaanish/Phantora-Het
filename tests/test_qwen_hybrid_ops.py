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
