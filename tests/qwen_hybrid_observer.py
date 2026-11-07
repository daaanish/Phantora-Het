"""Opt-in supplement for CUDA device-only ops discarded by the pinned tracer.

TorchDispatchMode is thread-local. Actual backward/worker-thread visibility still
requires the Linux trace gate; activation is necessary, not proof of completeness.
"""
import json
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


def argument_metadata(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))
    if isinstance(value, (list, tuple)):
        return [argument_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(name): argument_metadata(item) for name, item in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value) if isinstance(value, (torch.device, torch.dtype)) else type(value).__name__


class DeviceOnlyObserver(TorchDispatchMode):
    def __init__(self, sink, target_device="cuda"):
        super().__init__()
        self.sink = sink
        self.target_device = target_device  # CPU override for actual-dispatch tests only

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        tensors = [value for value in tree_flatten((args, kwargs))[0] if isinstance(value, torch.Tensor)]
        device = kwargs.get("device")
        if device is not None and torch.device(device).type == self.target_device and not any(tensor.device.type == "cuda" for tensor in tensors):
            name = function._schema.name
            if name in ("aten::empty", "aten::empty_strided", "aten::empty_like"):
                kind = "allocation_only"
            elif name == "aten::_to_copy" and tensors and all(tensor.device.type == "cpu" for tensor in tensors):
                kind = "native_transfer"
            else:
                kind = "unsupported"
            signature = json.dumps(dict(schema=str(function._schema), args=argument_metadata(args),
                                        kwargs=argument_metadata(kwargs)), sort_keys=True)
            self.sink(dict(op=name, kind=kind, signature=signature[:4096]))
        return function(*args, **kwargs)
