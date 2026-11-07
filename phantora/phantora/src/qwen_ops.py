"""Shared exact ATen thunks for Qwen gates/norms, embedded by Rust too."""

import json
import struct

import torch


def build(payload, allocator=None):
    if isinstance(payload, str):
        payload = json.loads(payload)["QwenOp"]
    cache = {}

    def allocate(info):
        dtype = {"Float": torch.float32, "Half": torch.float16, "BFloat16": torch.bfloat16,
                 "Double": torch.float64}[info["dtype"]]
        size = 1
        for dim in info["shape"]:
            size *= dim
        key = (size, dtype)
        if key not in cache:
            cache[key] = torch.randn(size, dtype=dtype, device="cuda")
        return cache[key].view(info["shape"])

    inputs = [(allocator or allocate)(info) for info in payload["inputs"]]
    scalars = [struct.unpack(">d", bits.to_bytes(8, "big"))[0] for bits in payload["scalars"]]
    op = payload["op"]
    if op in ("mean", "sum"):
        dtype = payload["dtype"]
        dtype = {"Float": torch.float32, "Half": torch.float16, "BFloat16": torch.bfloat16,
                 "Double": torch.float64, "Int64": torch.int64}.get(dtype) if dtype else None
        return lambda: getattr(torch, op)(inputs[0], dim=payload["dims"], keepdim=payload["keepdim"], dtype=dtype)
    if op == "pow":
        return lambda: torch.pow(inputs[0], scalars[0])
    if op == "div_":
        return lambda: inputs[0].div_(inputs[1] if len(inputs) == 2 else scalars[0])
    if op == "softplus":
        return lambda: torch.ops.aten.softplus(inputs[0], *scalars)
    if op == "softplus_backward":
        return lambda: torch.ops.aten.softplus_backward(*inputs, *scalars)
    if op in ("exp", "neg", "sigmoid", "silu", "rsqrt", "sigmoid_backward", "silu_backward"):
        return lambda: getattr(torch.ops.aten, op)(*inputs)
    raise ValueError(f"unsupported Qwen op: {op}")
