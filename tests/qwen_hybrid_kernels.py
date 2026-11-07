"""Shape-only fused boundaries. No FLA, Triton or CUDA extensions are imported.

Placeholder gradients preserve training orchestration, not numerical training or
activation-memory predictions. Timings always come from the native keyed DB.
"""

import ctypes
import json
import math
import os
import sys
from types import ModuleType

import torch


def tensor_metadata(tensor):
    if tensor is None:
        return None
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("fused simulation requires floating point inputs")
    if tensor.ndim == 0 or any(size <= 0 for size in tensor.shape):
        raise ValueError("empty fused inputs are unsupported")
    return dict(shape=list(tensor.shape), strides=list(tensor.stride()),
                dtype=str(tensor.dtype).removeprefix("torch."), requires_grad=tensor.requires_grad)


class NativeEventSink:
    def __init__(self):
        self.library = ctypes.CDLL("libcuda.so.1")
        self.send = self.library.phantora_gdn_call
        self.send.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t)
        self.send.restype = ctypes.c_int

    def __call__(self, key, device):
        if device.type != "cuda":
            raise RuntimeError("native fused simulation events require a CUDA tensor")
        payload = json.dumps(key, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device).cuda_stream
            if self.send(ctypes.c_void_p(stream), payload, len(payload)) != 0:
                raise RuntimeError("native fused event rejected; check the key/schema and rebuilt stubs")


def _empty(tensor, dtype=None, preserve=False):
    if preserve:
        return torch.empty_strided(tensor.shape, tensor.stride(), dtype=dtype or tensor.dtype, device=tensor.device)
    return torch.empty(tensor.shape, dtype=dtype or tensor.dtype, device=tensor.device)


class _Boundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, options, sink, grad_enabled, *inputs):
        device = inputs[0].device
        if any(t is not None and t.device != device for t in inputs):
            raise ValueError("all fused inputs must be on the same device")
        ctx.inputs = tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) if t is not None else None for t in inputs)
        ctx.key = dict(version=1, op=op, direction="forward", inputs=[tensor_metadata(t) for t in inputs],
                       grad_enabled=grad_enabled, grad_output=None, options=options)
        ctx.sink, ctx.device = sink, device
        sink(ctx.key, device)
        if op == "chunk_gated_delta_rule":
            return _empty(inputs[2], dtype=inputs[0].dtype)
        dtype = getattr(torch, options["output_dtype"]) if op == "l2norm" else None
        return _empty(inputs[0], dtype=dtype, preserve=op == "causal_conv1d")

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError("hybrid placeholders do not support higher-order gradients")
        key = dict(ctx.key, direction="backward", grad_output=tensor_metadata(grad))
        ctx.sink(key, ctx.device)
        gradients = []
        for info, needed in zip(ctx.inputs, ctx.needs_input_grad[4:]):
            if info is None or not needed:
                gradients.append(None)
            else:
                shape, strides, dtype, device = info
                gradients.append(torch.empty_strided(shape, strides, dtype=dtype, device=device))
        return (None, None, None, None, *gradients)


class KernelAdapters:
    """Inject an event sink for CPU tests; production uses only NativeEventSink."""

    def __init__(self, sink):
        self.sink = sink

    def l2norm(self, x, eps=1e-6, output_dtype=None):
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("l2norm eps must be positive and finite")
        output_dtype = output_dtype or x.dtype
        if output_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("unsupported l2norm output dtype")
        return _Boundary.apply("l2norm", dict(eps=str(float(eps)), output_dtype=str(output_dtype).removeprefix("torch.")),
                               self.sink, torch.is_grad_enabled(), x)

    def causal_conv1d_fn(self, x, weight, bias=None, seq_idx=None, initial_states=None,
                         return_final_states=False, final_states_out=None, activation=None):
        if seq_idx is not None or initial_states is not None or return_final_states or final_states_out is not None:
            raise ValueError("hybrid excludes packed convolution and recurrent state")
        if x.ndim != 3 or weight.ndim != 2 or weight.shape[0] != x.shape[1] or weight.shape[1] not in (2, 3, 4):
            raise ValueError("expected convolution x[B,C,T], weight[C,width=2/3/4]")
        if bias is not None and bias.shape != (x.shape[1],):
            raise ValueError("convolution bias must have shape [C]")
        if activation not in (None, "silu", "swish"):
            raise ValueError("unsupported convolution activation")
        return _Boundary.apply("causal_conv1d", dict(activation="silu" if activation == "swish" else activation or "none"),
                               self.sink, torch.is_grad_enabled(), x, weight, bias)

    def chunk_gated_delta_rule(self, q, k, v, g, beta, scale=None, initial_state=None,
                               output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
        if initial_state is not None or output_final_state or cu_seqlens is not None or kwargs:
            raise ValueError("hybrid excludes recurrent state, packed sequences and extra chunk options")
        if q.ndim != 4 or q.shape != k.shape or v.ndim != 4 or q.shape[:2] != v.shape[:2]:
            raise ValueError("expected q/k[B,T,Hk,K], v[B,T,Hv,V]")
        if any(dim <= 0 for tensor in (q, k, v, g, beta) for dim in tensor.shape):
            raise ValueError("empty chunk inputs are unsupported")
        if v.shape[2] % q.shape[2] or g.shape != v.shape[:3] or beta.shape != g.shape:
            raise ValueError("chunk value heads must be a multiple of key heads; g/beta must match value heads")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("chunk q/k/v dtypes must match")
        if type(use_qk_l2norm_in_kernel) is not bool:
            raise ValueError("use_qk_l2norm_in_kernel must be a bool")
        scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("chunk scale must be positive and finite")
        output = _Boundary.apply("chunk_gated_delta_rule", dict(scale=str(scale),
                                 use_qk_l2norm_in_kernel=str(use_qk_l2norm_in_kernel).lower()),
                                 self.sink, torch.is_grad_enabled(), q, k, v, g, beta)
        return output, None


_ADAPTERS = None


def install_simulation_kernels(sink=None):
    """Install before importing upstream GDN, and repair captured aliases too."""
    global _ADAPTERS
    if os.environ.get("PHANTORA") is None:
        raise RuntimeError("simulation kernel aliases require PHANTORA")
    if _ADAPTERS is None:
        _ADAPTERS = KernelAdapters(sink or NativeEventSink())
    elif sink is not None and _ADAPTERS.sink is not sink:
        raise RuntimeError("simulation kernel sink is already installed")
    exports = {
        "fla.modules.l2norm": dict(l2norm=_ADAPTERS.l2norm),
        "fla.ops.gated_delta_rule": dict(chunk_gated_delta_rule=_ADAPTERS.chunk_gated_delta_rule),
        "causal_conv1d": dict(causal_conv1d_fn=_ADAPTERS.causal_conv1d_fn),
    }
    for path, names in exports.items():
        parts = path.split(".")
        for end in range(1, len(parts) + 1):
            name = ".".join(parts[:end])
            module = sys.modules.get(name)
            if module is None:
                module = ModuleType(name)
                module.__path__ = []
                sys.modules[name] = module
            if end > 1:
                setattr(sys.modules[".".join(parts[:end - 1])], parts[end - 1], module)
        sys.modules[path].__dict__.update(names)
    captured = sys.modules.get("megatron.core.ssm.gated_delta_net")
    if captured is not None:
        captured.l2norm = _ADAPTERS.l2norm
        captured.chunk_gated_delta_rule = _ADAPTERS.chunk_gated_delta_rule
        captured.causal_conv1d_fn = _ADAPTERS.causal_conv1d_fn
        captured.HAVE_FLA = True
    return _ADAPTERS
