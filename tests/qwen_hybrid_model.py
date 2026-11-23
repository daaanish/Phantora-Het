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


class FlashCoreAttention(nn.Module):
    """Causal flash attention for the full-attention layers (and the MTP layer).

    Replaces Megatron's local unfused DotProductAttention (baddbmm -> softmax -> bmm with an
    s x s score matrix) with PyTorch SDPA on the flash backend, which is what real Qwen3.5
    training uses. The backend is forced, so a host or simulator that cannot run flash fails
    loudly instead of silently switching kernels. Grouped KV heads go to the kernel natively
    (enable_gqa) without the repeat_interleave copies of the unfused path. The causal mask is
    implicit (is_causal); no mask tensor is read. Same signature as DotProductAttention.
    """

    def __init__(self, config, layer_number, attn_mask_type, attention_type,
                 attention_dropout=None, softmax_scale=None, cp_comm_type=None,
                 pg_collection=None, **_):
        super().__init__()
        from megatron.core.transformer.enums import AttnMaskType
        if attn_mask_type != AttnMaskType.causal:
            raise ValueError(f"FlashCoreAttention supports causal masks only, got {attn_mask_type}")
        dropout = config.attention_dropout if attention_dropout is None else attention_dropout
        if dropout:
            raise ValueError("FlashCoreAttention assumes attention_dropout == 0")
        if getattr(config, "context_parallel_size", 1) != 1:
            raise ValueError("FlashCoreAttention does not support context parallelism")
        self.config = config
        self.softmax_scale = softmax_scale  # None = 1/sqrt(head_dim), Megatron's default too

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        if attention_bias is not None or packed_seq_params is not None:
            raise ValueError("FlashCoreAttention supports neither attention_bias nor packed sequences")
        sq, b, heads, head_dim = query.shape
        # [s, b, h, d] -> [b, h, s, d]
        q, k, v = (t.permute(1, 2, 0, 3) for t in (query, key, value))
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.softmax_scale,
                enable_gqa=k.shape[1] != heads)
        # [b, h, s, d] -> [s, b, h * d]
        return out.permute(2, 0, 1, 3).reshape(sq, b, heads * head_dim)


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
        # None (not 0) keeps Megatron's MTP code paths fully off, as in every run before QH-07w.
        mtp_num_layers=workload.mtp_num_layers or None,
    )


def build_hybrid_spec(workload, pp_rank):
    require_hybrid_kernels()
    from causal_conv1d import causal_conv1d_fn
    if not callable(causal_conv1d_fn):
        raise RuntimeError("hybrid requires the fused causal_conv1d implementation")
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

    base = get_gpt_layer_local_spec(normalization="RMSNorm", qk_layernorm=True)
    layers = [_layer_spec(base, layer_type) for _, layer_type in workload.pipeline_layers(pp_rank)]
    return TransformerBlockSubmodules(layer_specs=layers, layer_norm=QwenRMSNorm)


def _layer_spec(base, layer_type):
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.spec_utils import ModuleSpec

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
        # Flash attention always (since 2026-11-22); earlier runs used Megatron's local unfused path.
        attention.core_attention = FlashCoreAttention
    return layer


def build_hybrid_mtp_spec(workload, pp_rank):
    """Qwen3.5 MTP block for the stage that owns it, or None.

    Checkpoint layout (mtp.*): pre_fc_norm_embedding / pre_fc_norm_hidden -> fc (2h -> h) ->
    one gated full-attention layer with Q/K norm and dense SwiGLU MLP -> norm -> the shared,
    tied LM head. Megatron's MTP layer has the same pieces (enorm, hnorm, eh_proj,
    transformer_layer, final_layernorm); the norms here are Qwen's zero-centered RMSNorm.
    """
    if not workload.mtp_num_layers or pp_rank != workload.pipeline_parallel_size - 1:
        return None
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.multi_token_prediction import (
        MultiTokenPredictionBlockSubmodules, MultiTokenPredictionLayer,
        MultiTokenPredictionLayerSubmodules,
    )
    from megatron.core.transformer.spec_utils import ModuleSpec

    base = get_gpt_layer_local_spec(normalization="RMSNorm", qk_layernorm=True)
    layer = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=QwenRMSNorm, hnorm=QwenRMSNorm, eh_proj=ColumnParallelLinear,
            transformer_layer=_layer_spec(base, "full_attention"), layer_norm=QwenRMSNorm,
        ),
    )
    return MultiTokenPredictionBlockSubmodules(layer_specs=[layer] * workload.mtp_num_layers)


def _mtp_output_weight_unused(model):
    """True on a stage that owns MTP but not the input embedding, with tied embeddings."""
    return (getattr(model, "mtp_process", False) and not model.pre_process
            and model.post_process and model.share_embeddings_and_output_weights)


def drop_unused_mtp_output_weight(model):
    """Remove the output-layer weight Megatron Core 0.16 allocates but never uses with MTP + PP.

    On a last pipeline stage that owns MTP, Megatron keeps a copy of the word embedding and, with
    tied embeddings, computes the logits from it (shared_embedding_or_output_weight). The output
    layer still allocates its own weight there (skip_weight_param_allocation only checks
    pre_process), which never receives a gradient; DDP with overlap_grad_reduce then fails the
    per-parameter ready-count assertion in zero_grad_buffer. PP 1 is unaffected (pre_process is
    true on the MTP stage). Call before wrapping in DDP. Returns True if a weight was removed.
    """
    model = getattr(model, "module", model)
    if not _mtp_output_weight_unused(model) or model.output_layer.weight is None:
        return False
    model.output_layer.weight = None
    return True


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
            if not isinstance(attention.core_attention, FlashCoreAttention):
                raise RuntimeError("hybrid full attention must use FlashCoreAttention")
        checked.append(dict(global_layer=global_index, kind=kind))
    mtp = getattr(model, "mtp", None)
    owns_mtp = workload.mtp_num_layers and pp_rank == workload.pipeline_parallel_size - 1
    if not owns_mtp:
        if mtp is not None:
            raise RuntimeError("hybrid MTP block built on a stage that should not own it")
        return checked
    if mtp is None or len(mtp.layers) != workload.mtp_num_layers:
        raise RuntimeError("hybrid MTP layer count mismatch")
    if _mtp_output_weight_unused(model) and model.output_layer.weight is not None:
        raise RuntimeError("unused output-layer weight on the MTP stage; call drop_unused_mtp_output_weight")
    for depth, mtp_layer in enumerate(mtp.layers):
        shape(mtp_layer.eh_proj.weight, (dims.hidden_size // tp, 2 * dims.hidden_size), "MTP eh_proj")
        for name in ("enorm", "hnorm", "final_layernorm"):
            if not isinstance(getattr(mtp_layer, name), QwenRMSNorm):
                raise RuntimeError(f"hybrid MTP {name} is not Qwen RMSNorm")
        layer = mtp_layer.transformer_layer
        attention = layer.self_attention
        if attention.__class__.__name__ == "GatedDeltaNet":
            raise RuntimeError("hybrid MTP layer must use full attention")
        shape(attention.linear_qkv.weight, ((2 * dims.num_attention_heads + 2 * dims.num_query_groups) * dims.head_dim // tp, dims.hidden_size), "MTP gated QKV")
        shape(attention.q_layernorm.weight, (dims.head_dim,), "MTP Q norm")
        if not isinstance(attention.core_attention, FlashCoreAttention):
            raise RuntimeError("hybrid MTP attention must use FlashCoreAttention")
        shape(layer.mlp.linear_fc1.weight, (2 * dims.ffn_hidden_size // tp, dims.hidden_size), "MTP SwiGLU fc1")
        checked.append(dict(mtp_depth=depth, kind="mtp_full_attention"))
    return checked
