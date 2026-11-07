"""Independent tiny equations, not a Megatron model or simulation replacement.

Qwen reference pinned at:
https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen3_5/modeling_qwen3_5.py
"""

import torch
import torch.nn.functional as F


def l2norm_reference(x, eps=1e-6, output_dtype=None):
    value = x.float()
    return (value * torch.rsqrt(value.square().sum(-1, keepdim=True) + eps)).to(output_dtype or x.dtype)


def causal_conv_reference(x, weight, bias=None, activation=None):
    dtype = x.dtype
    x, weight = x.float(), weight.float()
    bias = bias.float() if bias is not None else None
    width = weight.shape[-1]
    outputs = []
    for time in range(x.shape[-1]):
        terms = [x[..., time - (width - 1 - tap)] * weight[:, tap]
                 for tap in range(width) if time >= width - 1 - tap]
        value = sum(terms)
        outputs.append(value + bias if bias is not None else value)
    output = torch.stack(outputs, dim=-1)
    return (F.silu(output) if activation == "silu" else output).to(dtype)


def recurrent_gdn_reference(q, k, v, g, beta, scale=None, use_qk_l2norm_in_kernel=False):
    if use_qk_l2norm_in_kernel:
        q, k = l2norm_reference(q), l2norm_reference(k)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    dtype = q.dtype
    ratio = v.shape[2] // q.shape[2]
    q, k = q.float().repeat_interleave(ratio, dim=2), k.float().repeat_interleave(ratio, dim=2)
    state = torch.zeros(q.shape[0], v.shape[2], q.shape[-1], v.shape[-1], dtype=torch.float32, device=q.device)
    outputs = []
    for time in range(q.shape[1]):
        state = state * g[:, time].float().exp()[..., None, None]
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, time], state)
        delta = beta[:, time].float()[..., None] * (v[:, time].float() - prediction)
        state = state + k[:, time, :, :, None] * delta[:, :, None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, time] * scale, state))
    return torch.stack(outputs, dim=1).to(dtype)
