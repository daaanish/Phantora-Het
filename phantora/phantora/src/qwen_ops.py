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
    if op == "exp_out":
        # Megatron cross-entropy aliases input and output. The trace lacks strides
        # and storage identity, so both allocators replay using their existing aliasing rule.
        return lambda: torch.ops.aten.exp.out(inputs[0], out=inputs[1])
    if op == "cat":
        # Members of equal size come back aliased from both allocators; a real cat reads
        # distinct tensors, so duplicates are cloned once here, outside the timed call.
        seen, members = set(), []
        for value in inputs:
            members.append(value.clone() if id(value) in seen or value.data_ptr() in seen else value)
            seen.update((id(value), value.data_ptr()))
        return lambda: torch.ops.aten.cat(members, payload["dims"][0])
    if op == "contiguous":
        # The trace has no strides. Replay a real strided copy: a source of the recorded
        # shape whose storage has two dimensions swapped.
        shape = list(inputs[0].shape)
        source = None
        for first in range(len(shape) - 1, 0, -1):
            for second in range(first - 1, -1, -1):
                swapped = list(shape)
                swapped[first], swapped[second] = swapped[second], swapped[first]
                candidate = inputs[0].reshape(swapped).transpose(first, second)
                if not candidate.is_contiguous():
                    source = candidate
                    break
            if source is not None:
                break
        if source is None:
            raise ValueError("no strided layout exists for this shape")
        return lambda: source.contiguous()
    if op in ("exp", "neg", "sigmoid", "silu", "rsqrt", "square", "sigmoid_backward", "silu_backward"):
        return lambda: getattr(torch.ops.aten, op)(*inputs)
    raise ValueError(f"unsupported Qwen op: {op}")
