"""Opt-in local Megatron specs; imports Megatron/FLA only when building specs."""

from copy import deepcopy
import os

import torch
from torch import nn


class QwenRMSNorm(nn.Module):
    """Qwen pre/post/QK norm: FP32 normalization and zero-centered scale."""

    def __init__(self, config, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        cpu_init = getattr(config, "use_cpu_initialization", False) or not torch.cuda.is_available()
        device = "cpu" if cpu_init else torch.cuda.current_device()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=config.params_dtype, device=device))
        self.weight.sequence_parallel = config.sequence_parallel

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * (1 + self.weight.float())).to(x.dtype)


class GDNOutputRMSNorm(QwenRMSNorm):
    """Standard scale-one RMSNorm; Megatron applies the SiLU gate after this."""

    def __init__(self, config, hidden_size, eps=1e-6):
        super().__init__(config, hidden_size, eps)
        nn.init.ones_(self.weight)

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        # The pinned Qwen reference casts normalized values before scale/gating.
        return self.weight * value.to(x.dtype)


def require_hybrid_kernels():
    if os.environ.get("PHANTORA") is not None:
        from qwen_hybrid_kernels import install_simulation_kernels
        install_simulation_kernels()


def transformer_overrides(workload):
    model = workload.model
    return dict(
        transformer_impl="local",
        experimental_attention_variant="gated_delta_net", linear_attention_freq=4,
        linear_conv_kernel_dim=model.conv_kernel_dim,
        linear_key_head_dim=model.linear_key_head_dim,
        linear_value_head_dim=model.linear_value_head_dim,
        linear_num_key_heads=model.linear_num_key_heads,
        linear_num_value_heads=model.linear_num_value_heads,
        attention_output_gate=True, attention_dropout=0.0, hidden_dropout=0.0,
        layernorm_epsilon=model.rms_norm_eps, layernorm_zero_centered_gamma=True,
        bias_activation_fusion=False, bias_dropout_fusion=False,
        masked_softmax_fusion=False, persist_layer_norm=False,
    )


def build_hybrid_spec(workload, pp_rank):
    require_hybrid_kernels()
    from causal_conv1d import causal_conv1d_fn
    if not callable(causal_conv1d_fn):
        raise RuntimeError("hybrid requires the fused causal_conv1d implementation")
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

    base = get_gpt_layer_local_spec(normalization="RMSNorm", qk_layernorm=True)
    layers = []
    for _, layer_type in workload.pipeline_layers(pp_rank):
        layer = deepcopy(base)
        layer.submodules.input_layernorm = QwenRMSNorm
        layer.submodules.pre_mlp_layernorm = QwenRMSNorm
        if layer_type == "gdn":
            layer.submodules.self_attention = ModuleSpec(
                module=GatedDeltaNet,
                params=dict(bias=False, conv_bias=False, use_qk_l2norm=True),
                submodules=GatedDeltaNetSubmodules(
                    in_proj=ColumnParallelLinear, out_norm=GDNOutputRMSNorm,
                    out_proj=RowParallelLinear,
                ),
            )
        else:
            attention = layer.submodules.self_attention.submodules
            attention.q_layernorm = QwenRMSNorm
            attention.k_layernorm = QwenRMSNorm
        layers.append(layer)
    return TransformerBlockSubmodules(layer_specs=layers, layer_norm=QwenRMSNorm)


def finalize_hybrid_grads(model, *args, **kwargs):
    """Retain Megatron DP/QK reductions and sum shared GDN scales across TP."""
    from megatron.core import parallel_state
    from megatron.core.distributed import finalize_model_grads

    finalize_model_grads(model, *args, **kwargs)
    group = parallel_state.get_tensor_model_parallel_group()
    if torch.distributed.get_world_size(group) == 1:
        return
    for chunk in model if isinstance(model, list) else [model]:
        for name, param in chunk.named_parameters():
            if name.endswith("self_attention.out_norm.weight") and param.requires_grad:
                grad = getattr(param, "main_grad", None)
                if grad is None:
                    grad = param.grad
                if grad is None:
                    raise RuntimeError(f"missing hybrid output norm gradient: {name}")
                torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM, group=group)


def validate_hybrid_structure(model, workload, pp_rank):
    """Actual-model smoke assertions; equations/kernel and HF fidelity are separate gates."""
    model = getattr(model, "module", model)
    expected = workload.pipeline_layers(pp_rank)
    layers = model.decoder.layers
    if len(layers) != len(expected):
        raise RuntimeError("hybrid local layer count mismatch")
    checked = []
    dims, tp = workload.model, workload.tensor_parallel_size
    def shape(parameter, expected, name):
        if tuple(parameter.shape) != expected:
            raise RuntimeError(f"hybrid {name} shape mismatch: {tuple(parameter.shape)} != {expected}")
    for layer, (global_index, kind) in zip(layers, expected):
        if layer.layer_number != global_index + 1:
            raise RuntimeError("hybrid global pipeline layer index mismatch")
        attention = layer.self_attention
        actual_gdn = attention.__class__.__name__ == "GatedDeltaNet"
        if actual_gdn != (kind == "gdn"):
            raise RuntimeError("hybrid attention pattern mismatch")
        shape(layer.mlp.linear_fc1.weight, (2 * dims.ffn_hidden_size // tp, dims.hidden_size), "SwiGLU fc1")
        shape(layer.mlp.linear_fc2.weight, (dims.hidden_size, dims.ffn_hidden_size // tp), "SwiGLU fc2")
        if actual_gdn:
            key_dim = dims.linear_key_head_dim * dims.linear_num_key_heads
            value_dim = dims.linear_value_head_dim * dims.linear_num_value_heads
            shape(attention.in_proj.weight, ((2 * key_dim + 2 * value_dim + 2 * dims.linear_num_value_heads) // tp, dims.hidden_size), "GDN input projection")
            shape(attention.conv1d.weight, ((2 * key_dim + value_dim) // tp, 1, dims.conv_kernel_dim), "GDN convolution")
            shape(attention.A_log, (dims.linear_num_value_heads // tp,), "GDN A_log")
            shape(attention.dt_bias, (dims.linear_num_value_heads // tp,), "GDN dt_bias")
            expected_norm = (workload.model.linear_value_head_dim,)
            if tuple(attention.out_norm.weight.shape) != expected_norm:
                raise RuntimeError("shared GDN output norm shape mismatch")
            if tuple(attention.out_proj.weight.shape) != (workload.model.hidden_size,
                    workload.model.linear_num_value_heads * workload.model.linear_value_head_dim // workload.tensor_parallel_size):
                raise RuntimeError("GDN output projection shape mismatch")
        else:
            shape(attention.linear_qkv.weight, ((2 * dims.num_attention_heads + 2 * dims.num_query_groups) * dims.head_dim // tp, dims.hidden_size), "gated QKV")
            shape(attention.linear_proj.weight, (dims.hidden_size, dims.num_attention_heads * dims.head_dim // tp), "attention projection")
            shape(attention.q_layernorm.weight, (dims.head_dim,), "Q norm")
            shape(attention.k_layernorm.weight, (dims.head_dim,), "K norm")
        checked.append(dict(global_layer=global_index, kind=kind))
    return checked
