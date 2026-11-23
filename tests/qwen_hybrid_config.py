"""CPU-only contracts for the opt-in dense Qwen hybrid workload.

These configurations describe workloads, not validated runtime support or timings.
Layer indices are zero-based global indices, including after pipeline partitioning.
"""

from dataclasses import dataclass, replace
import math

from pipeline_split import stage_ranges, validate_stage_layers

QWEN35_CONFIG_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
QWEN35_CONFIG_SOURCE = (
    "https://huggingface.co/Qwen/Qwen3.5-9B/resolve/"
    f"{QWEN35_CONFIG_REVISION}/config.json"
)

QWEN35_08B_CONFIG_SOURCE = (
    "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/"
    "2fc06364715b967f1860aea9cf38778875588b17/config.json"
)


def _positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_query_groups: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    conv_kernel_dim: int = 4
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    rotary_base: float = 10000000.0
    tie_word_embeddings: bool = False
    # MTP depth of the released checkpoint (HF mtp_num_hidden_layers); 0 for fixtures without one.
    # The workload decides whether to train it (HybridConfig.mtp_num_layers).
    mtp_num_hidden_layers: int = 0

    def __post_init__(self):
        for name in (
            "num_layers", "hidden_size", "ffn_hidden_size", "vocab_size",
            "num_attention_heads", "num_query_groups", "head_dim",
            "linear_num_key_heads", "linear_num_value_heads",
            "linear_key_head_dim", "linear_value_head_dim", "conv_kernel_dim",
        ):
            _positive_int(name, getattr(self, name))
        if type(self.mtp_num_hidden_layers) is not int or self.mtp_num_hidden_layers < 0:
            raise ValueError("mtp_num_hidden_layers must be a non-negative integer")
        if self.num_attention_heads % self.num_query_groups:
            raise ValueError("num_attention_heads must be divisible by num_query_groups")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
        if self.conv_kernel_dim not in (2, 3, 4):
            raise ValueError("fused causal_conv1d supports widths 2, 3 and 4")
        rotary_dim = self.head_dim * self.partial_rotary_factor
        if not 0 < self.partial_rotary_factor <= 1 or rotary_dim % 1 or rotary_dim % 2:
            raise ValueError("partial_rotary_factor must select a positive even rotary dimension")
        if not 0 < self.rms_norm_eps < 1:
            raise ValueError("rms_norm_eps must lie between 0 and 1")
        if not math.isfinite(self.rotary_base) or self.rotary_base <= 0:
            raise ValueError("rotary_base must be positive and finite")

    @property
    def layer_types(self):
        return tuple("full_attention" if i % 4 == 3 else "gdn" for i in range(self.num_layers))


# Deliberately small architecture fixture; it is not a released Qwen checkpoint.
DEBUG_MODEL = ModelSpec(
    name="qwen3.5-debug", num_layers=4, hidden_size=256, ffn_hidden_size=768,
    vocab_size=1024, num_attention_heads=4, num_query_groups=2, head_dim=64,
    linear_num_key_heads=4, linear_num_value_heads=8,
    linear_key_head_dim=128, linear_value_head_dim=128,
)

# Text config snapshot inspected 2026-10-29; no network access is needed at runtime.
QWEN35_9B = ModelSpec(
    name="qwen3.5-9b-text", num_layers=32, hidden_size=4096,
    ffn_hidden_size=12288, vocab_size=248320, num_attention_heads=16,
    num_query_groups=4, head_dim=256, linear_num_key_heads=16,
    linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128,
    mtp_num_hidden_layers=1,
)


QWEN35_08B = ModelSpec(
    name="qwen3.5-0.8b-text", num_layers=24, hidden_size=1024,
    ffn_hidden_size=3584, vocab_size=248320, num_attention_heads=8,
    num_query_groups=2, head_dim=256, linear_num_key_heads=16,
    linear_num_value_heads=16, linear_key_head_dim=128, linear_value_head_dim=128,
    tie_word_embeddings=True, mtp_num_hidden_layers=1,
)

# Reduced-depth timing fixture: 0.8B layer shapes, one 3 GDN + 1 full-attention group.
# It is not a released Qwen checkpoint; use it to fit longer sequences in memory.
QWEN35_08B_L4 = replace(QWEN35_08B, name="qwen3.5-0.8b-text-4layer", num_layers=4)
# Intermediate depths between the 4-layer fixture and the released 24 layers (2026-11-22), same
# shapes and 3 GDN : 1 full-attention pattern; also not released checkpoints.
QWEN35_08B_L8 = replace(QWEN35_08B, name="qwen3.5-0.8b-text-8layer", num_layers=8)
QWEN35_08B_L12 = replace(QWEN35_08B, name="qwen3.5-0.8b-text-12layer", num_layers=12)
QWEN35_08B_L16 = replace(QWEN35_08B, name="qwen3.5-0.8b-text-16layer", num_layers=16)

HYBRID_PRESETS = {"debug": DEBUG_MODEL, "0.8b": QWEN35_08B, "0.8b-l4": QWEN35_08B_L4,
                  "0.8b-l8": QWEN35_08B_L8, "0.8b-l12": QWEN35_08B_L12, "0.8b-l16": QWEN35_08B_L16,
                  "9b": QWEN35_9B}
HYBRID_CONFIG_SOURCES = {
    QWEN35_08B: QWEN35_08B_CONFIG_SOURCE,
    QWEN35_08B_L4: f"{QWEN35_08B_CONFIG_SOURCE}; depth reduced from 24 to 4 layers",
    QWEN35_08B_L8: f"{QWEN35_08B_CONFIG_SOURCE}; depth reduced from 24 to 8 layers",
    QWEN35_08B_L12: f"{QWEN35_08B_CONFIG_SOURCE}; depth reduced from 24 to 12 layers",
    QWEN35_08B_L16: f"{QWEN35_08B_CONFIG_SOURCE}; depth reduced from 24 to 16 layers",
    QWEN35_9B: QWEN35_CONFIG_SOURCE,
}


@dataclass(frozen=True)
class HybridConfig:
    model: ModelSpec = DEBUG_MODEL
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    world_size: int = 1
    sequence_length: int = 128
    micro_batch_size: int = 1
    gradient_accumulation: int = 8
    global_batch_size: int = 8
    context_parallel_size: int = 1
    expert_parallel_size: int = 1
    sequence_parallel: bool = False
    virtual_pipeline_size: int | None = None
    recompute_activations: bool = False
    packed_sequences: bool = False
    use_cache: bool = False
    vision: bool = False
    # Trained MTP depth (Megatron mtp_num_layers); 0 = off, the behaviour of every run before QH-07w.
    mtp_num_layers: int = 0
    # Decoder layers per PP stage, e.g. (7, 9, 9, 7); None = Megatron's even split.
    pipeline_stage_layers: tuple[int, ...] | None = None
    dtype: str = "bfloat16"

    def __post_init__(self):
        for name in (
            "tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size",
            "world_size", "sequence_length", "micro_batch_size",
            "gradient_accumulation", "global_batch_size", "context_parallel_size",
            "expert_parallel_size",
        ):
            _positive_int(name, getattr(self, name))
        if self.world_size != self.tensor_parallel_size * self.data_parallel_size * self.pipeline_parallel_size:
            raise ValueError("world_size must equal TP * DP * PP")
        if self.context_parallel_size != 1 or self.expert_parallel_size != 1:
            raise ValueError("hybrid initially requires CP=1 and EP=1")
        if self.virtual_pipeline_size is not None:
            raise ValueError("virtual pipeline parallelism is unsupported")
        if type(self.mtp_num_layers) is not int or self.mtp_num_layers < 0:
            raise ValueError("mtp_num_layers must be a non-negative integer")
        if self.mtp_num_layers > self.model.mtp_num_hidden_layers:
            raise ValueError(f"mtp_num_layers exceeds the {self.model.mtp_num_hidden_layers} MTP layer(s) of {self.model.name}")
        if self.mtp_num_layers and self.model.layer_types[-1] != "full_attention":
            raise ValueError("MTP reuses the last decoder layer's spec, which must be full attention")
        for name in ("sequence_parallel", "recompute_activations", "packed_sequences", "use_cache", "vision"):
            if getattr(self, name):
                raise ValueError(f"{name} is unsupported for the initial hybrid workload")
        if self.dtype != "bfloat16":
            raise ValueError("hybrid initially requires bfloat16")
        if self.pipeline_stage_layers is not None:
            validate_stage_layers(self.pipeline_stage_layers, self.model.num_layers, self.pipeline_parallel_size)
        elif self.model.num_layers % self.pipeline_parallel_size:
            raise ValueError("num_layers must be divisible by PP")
        # Restrict to whole head shards; replicated KV heads need a separate gate.
        for name in (
            "hidden_size", "ffn_hidden_size", "vocab_size", "num_attention_heads",
            "num_query_groups", "linear_num_key_heads", "linear_num_value_heads",
        ):
            if getattr(self.model, name) % self.tensor_parallel_size:
                raise ValueError(f"{name} must be divisible by TP")
        if self.global_batch_size != self.effective_batch_size:
            raise ValueError("global_batch_size must equal micro_batch_size * gradient_accumulation * DP")

    @property
    def effective_batch_size(self):
        return self.micro_batch_size * self.gradient_accumulation * self.data_parallel_size

    @property
    def tokens_per_iteration(self):
        return self.effective_batch_size * self.sequence_length

    def pipeline_layers(self, stage):
        """Return (global index, layer type) pairs for a contiguous PP stage (even or uneven split)."""
        if type(stage) is not int or not 0 <= stage < self.pipeline_parallel_size:
            raise ValueError("pipeline stage must be in [0, PP)")
        if self.pipeline_stage_layers is not None:
            start, width = stage_ranges(self.pipeline_stage_layers)[stage]
        else:
            width = self.model.num_layers // self.pipeline_parallel_size
            start = stage * width
        return tuple(enumerate(self.model.layer_types[start:start + width], start=start))


def configure_hybrid_args(args, argv, world_size):
    """Apply an explicit preset while rejecting conflicting architectural flags."""
    model = HYBRID_PRESETS[args.hybrid_preset]
    supplied = {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}
    fixed = dict(
        num_layers=model.num_layers, hidden_size=model.hidden_size,
        ffn_hidden_size=model.ffn_hidden_size, vocab_size=model.vocab_size,
        num_attention_heads=model.num_attention_heads, num_query_groups=model.num_query_groups,
        kv_channels=model.head_dim, qk_layernorm=True, swiglu=True,
        position_embedding_type="rope", rotary_base=model.rotary_base,
        normalization="RMSNorm", disable_bias_linear=True,
    )
    for name, value in fixed.items():
        if f"--{name}" in supplied and getattr(args, name) != value:
            raise ValueError(f"--{name} conflicts with hybrid preset {args.hybrid_preset}")
        setattr(args, name, value)
    if "--tensor_parallel_size" not in supplied:
        args.tensor_parallel_size = 1
    if "--sequence_length" not in supplied:
        args.sequence_length = 128
    tp_pp = args.tensor_parallel_size * args.pipeline_model_parallel_size
    _positive_int("TP * PP", tp_pp)
    if world_size % tp_pp:
        raise ValueError("world_size must be divisible by TP * PP")
    dp = world_size // tp_pp
    if args.data_parallel_size is not None and args.data_parallel_size != dp:
        raise ValueError("requested DP does not match world_size / (TP * PP)")
    gbs = args.global_batch_size if args.global_batch_size is not None else 8
    denominator = args.micro_batch_size * dp
    _positive_int("micro_batch_size * DP", denominator)
    if gbs % denominator:
        raise ValueError("global_batch_size must be divisible by micro_batch_size * DP")
    if "--gradient_accumulation" not in supplied:
        args.gradient_accumulation = gbs // denominator
    if args.num_moe_experts is not None:
        raise ValueError("hybrid initially excludes MoE")
    return HybridConfig(
        model=model, tensor_parallel_size=args.tensor_parallel_size, data_parallel_size=dp,
        pipeline_parallel_size=args.pipeline_model_parallel_size, world_size=world_size,
        sequence_length=args.sequence_length, micro_batch_size=args.micro_batch_size,
        gradient_accumulation=args.gradient_accumulation, global_batch_size=gbs,
        expert_parallel_size=args.expert_model_parallel_size,
        virtual_pipeline_size=args.virtual_pipeline_model_parallel_size,
        recompute_activations=args.recompute_activations,
        mtp_num_layers=getattr(args, "mtp_layers", 0),
        pipeline_stage_layers=getattr(args, "pipeline_stage_layers", None),
    )
