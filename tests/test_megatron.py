from phantora_utils import (
    time_pair,
    enable_function_tracer,
    disable_function_tracer,
    install_phantora_megatron_moe_patches,
)
import os
import sys
if os.environ.get("PHANTORA") is not None and any(arg.split("=", 1)[0] == "--hybrid-preset" for arg in sys.argv):
    from qwen_hybrid_kernels import install_simulation_kernels
    install_simulation_kernels()
import functools
import torch
from torch.utils.data import DataLoader
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.core.distributed import (
    DistributedDataParallelConfig as DDPConfig,
    DistributedDataParallel as DDP,
    finalize_model_grads,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_no_pipelining,
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig


class NullTokenizer(MegatronTokenizer):
    def __init__(self, vocab_size):
        super().__init__(None, vocab_size=vocab_size)
        self._vocab_size_without_eod = int(vocab_size)
        self._eod_id = self._vocab_size_without_eod

    def tokenize(self, text):
        return [int(x) for x in text.split(" ")]

    def detokenize(self, ids):
        text = [str(x) for x in ids]
        return " ".join(text)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        offsets, start_idx = [], 0
        for id_ in ids:
            offsets.append(start_idx)
            start_idx += 1 + len(str(id_))
        return offsets

    @property
    def vocab_size(self):
        return self._vocab_size_without_eod + 1

    @property
    def vocab(self):
        raise NotImplementedError

    @property
    def inv_vocab(self):
        raise NotImplementedError

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eod_id

    @property
    def additional_special_tokens_ids(self):
        return None


def get_model(
    tensor_parallel_size,
    pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size,
    vp_stage,
    num_layers,
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
    num_query_groups,
    vocab_size,
    sequence_length,
    recompute_activations,
    swiglu,
    position_embedding_type,
    rotary_base,
    normalization,
    disable_bias_linear,
    num_moe_experts=None,
    moe_router_topk=2,
    moe_token_dispatcher_type="alltoall",
    expert_model_parallel_size=1,
    kv_channels=None,
    qk_layernorm=False,
    hybrid_config=None,
):
    # MoE config is only added when num_moe_experts is set, so dense models are
    # unchanged. moe_token_dispatcher_type="alltoall" is what makes Megatron emit
    # the expert all-to-all (grouped ncclSend/ncclRecv across EP ranks).
    moe_config_kwargs = {}
    if hybrid_config is not None:
        from qwen_hybrid_model import require_hybrid_kernels, transformer_overrides
        require_hybrid_kernels()
        moe_config_kwargs.update(transformer_overrides(hybrid_config))
    if num_moe_experts is not None:
        moe_config_kwargs = dict(
            num_moe_experts=num_moe_experts,
            moe_router_topk=moe_router_topk,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            expert_model_parallel_size=expert_model_parallel_size,
            moe_grouped_gemm=False,
        )

    transformer_config = TransformerConfig(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        # Megatron requires sequence parallelism when expert + tensor parallelism
        # are combined; enable it only in that case so dense / EP=1 runs are unchanged.
        sequence_parallel=(tensor_parallel_size > 1 and expert_model_parallel_size > 1),
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        # head_dim; None -> Megatron's default hidden_size // num_attention_heads.
        # Qwen3 and gpt-oss use head_dim != hidden_size // num_attention_heads.
        kv_channels=kv_channels,
        # QK-norm (RMSNorm on Q and K); used by Qwen3. Megatron default is False.
        qk_layernorm=qk_layernorm,
        gated_linear_unit=swiglu,
        activation_func=torch.nn.functional.silu if swiglu else torch.nn.functional.gelu,
        normalization=normalization,
        add_bias_linear=not disable_bias_linear,
        perform_initialization=hybrid_config is not None,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        attention_backend=AttnBackend.local if hybrid_config is not None else AttnBackend.flash,
        recompute_granularity="selective" if recompute_activations else None,
        **moe_config_kwargs,
    )

    if hybrid_config is not None:
        from qwen_hybrid_model import build_hybrid_spec
        layer_spec = build_hybrid_spec(hybrid_config, parallel_state.get_pipeline_model_parallel_rank())
    elif num_moe_experts is not None:
        layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts,
            moe_grouped_gemm=False,
            normalization=normalization,
        )
    else:
        layer_spec = get_gpt_layer_local_spec()

    gpt_model = Float16Module(
        transformer_config,
        GPTModel(
            config=transformer_config,
            transformer_layer_spec=layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=sequence_length,
            position_embedding_type=position_embedding_type,
            rotary_base=rotary_base,
            **(dict(rotary_percent=hybrid_config.model.partial_rotary_factor,
                    share_embeddings_and_output_weights=False) if hybrid_config is not None else {}),
            pre_process=parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=vp_stage
            ),
            post_process=parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=vp_stage
            ),
            vp_stage=vp_stage,
        ),
    )

    if hybrid_config is not None:
        from qwen_hybrid_model import validate_hybrid_structure
        import json
        checked = validate_hybrid_structure(gpt_model, hybrid_config,
            parallel_state.get_pipeline_model_parallel_rank())
        print("HYBRID_STRUCTURE " + json.dumps(dict(layers=checked,
            memory_validated=False, hf_whole_model_equivalence=False)), flush=True)

    dp_config = DDPConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        bucket_size=40000000,
    )

    model = DDP(transformer_config, dp_config, gpt_model)
    transformer_config.no_sync_func = model.no_sync
    if hybrid_config is not None:
        from qwen_hybrid_model import finalize_hybrid_grads
        transformer_config.finalize_model_grads_func = finalize_hybrid_grads
    else:
        transformer_config.finalize_model_grads_func = finalize_model_grads

    return model


def get_optimizer(model):
    optim_config = OptimizerConfig(
        optimizer="adam",
        lr=5e-5,
        bf16=True,
        use_distributed_optimizer=True,
        clip_grad=0.0,  # no grad clipping because norm depends on communication and can cause error in simulation
    )

    model_chunks = model if isinstance(model, list) else [model]
    optim = get_megatron_optimizer(optim_config, model_chunks, use_gloo_process_groups=False)

    return optim


def iter_model_chunks(model):
    return model if isinstance(model, list) else [model]


def get_train_data_iterator(vocab_size, micro_batch_size, sequence_length):
    config = GPTDatasetConfig(
        random_seed=42,
        sequence_length=sequence_length,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=NullTokenizer(vocab_size-1),
    )
    config.mock = True
    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [None, None, None], lambda: True, config
    ).build()
    train_dataloader = DataLoader(datasets[0], batch_size=micro_batch_size)
    return iter(train_dataloader)


def forward_step_func(device, data_iterator, model):
    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        loss_mask = loss_mask.view(-1).to(output_tensor.dtype)
        loss = torch.sum(output_tensor.view(-1) * loss_mask) / loss_mask.sum()
        return loss, {"lm-loss": loss}

    data = next(data_iterator)
    tokens = data["tokens"].to(device)
    attention_mask = data["attention_mask"].to(device)
    position_ids = data["position_ids"].to(device)
    labels = data["labels"].to(device)
    loss_mask = data["loss_mask"].to(device)

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, functools.partial(loss_func, loss_mask)


def main(
    tensor_parallel_size,
    pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size,
    num_layers,
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
    num_query_groups,
    vocab_size,
    sequence_length,
    micro_batch_size,
    gradient_accumulation,
    recompute_activations,
    iterations,
    swiglu,
    position_embedding_type,
    rotary_base,
    normalization,
    disable_bias_linear,
    num_moe_experts=None,
    moe_router_topk=2,
    moe_token_dispatcher_type="alltoall",
    expert_model_parallel_size=1,
    kv_channels=None,
    qk_layernorm=False,
    hybrid_config=None,
):
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if hybrid_config is not None:
        from qwen_hybrid_model import require_hybrid_kernels
        require_hybrid_kernels()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if hybrid_config is not None:
        from qwen_hybrid_runtime_check import check_runtime
        check_runtime()
    torch.cuda.memory.reset_peak_memory_stats(device)

    torch.distributed.init_process_group(
        world_size=world_size, rank=rank, device_id=device
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        create_gloo_process_groups=False,
    )

    if num_moe_experts is not None:
        # Make MoE survive payload-free simulation: TE-absence shim + load-balanced
        # routing so the dispatch/all-to-all sizes are well-defined and uniform.
        install_phantora_megatron_moe_patches()

    model_parallel_cuda_manual_seed(42)

    if virtual_pipeline_model_parallel_size is None:
        model = get_model(
            tensor_parallel_size,
            pipeline_model_parallel_size,
            None,
            None,
            num_layers,
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            num_query_groups,
            vocab_size,
            sequence_length,
            recompute_activations,
            swiglu,
            position_embedding_type,
            rotary_base,
            normalization,
            disable_bias_linear,
            num_moe_experts=num_moe_experts,
            moe_router_topk=moe_router_topk,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            expert_model_parallel_size=expert_model_parallel_size,
            kv_channels=kv_channels,
            qk_layernorm=qk_layernorm,
            hybrid_config=hybrid_config,
        )
        model = model.to(device)
        model.train()
    else:
        model = []
        for vp_stage in range(virtual_pipeline_model_parallel_size):
            parallel_state.set_virtual_pipeline_model_parallel_rank(vp_stage)
            model_chunk = get_model(
                tensor_parallel_size,
                pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size,
                vp_stage,
                num_layers,
                hidden_size,
                ffn_hidden_size,
                num_attention_heads,
                num_query_groups,
                vocab_size,
                sequence_length,
                recompute_activations,
                swiglu,
                position_embedding_type,
                rotary_base,
                normalization,
                disable_bias_linear,
                num_moe_experts=num_moe_experts,
                moe_router_topk=moe_router_topk,
                moe_token_dispatcher_type=moe_token_dispatcher_type,
                expert_model_parallel_size=expert_model_parallel_size,
                kv_channels=kv_channels,
                qk_layernorm=qk_layernorm,
            )
            model_chunk = model_chunk.to(device)
            model_chunk.train()
            model.append(model_chunk)
        parallel_state.set_virtual_pipeline_model_parallel_rank(0)

        no_sync_funcs = [model_chunk.no_sync for model_chunk in model]
        for model_chunk in model:
            model_chunk.config.no_sync_func = no_sync_funcs
            model_chunk.config.finalize_model_grads_func = finalize_model_grads

    if rank == 0:
        model_size = sum(
            p.numel()
            for model_chunk in iter_model_chunks(model)
            for p in model_chunk.parameters()
        )
        print(f"Model size: {model_size}")
    optim = get_optimizer(model)

    pp_size = pipeline_model_parallel_size
    num_microbatches = gradient_accumulation if hybrid_config is not None else (
        max(gradient_accumulation, pp_size * 4) if pp_size > 1 else gradient_accumulation
    )
    if hybrid_config is not None:
        import json
        from dataclasses import asdict
        from qwen_hybrid_config import QWEN35_CONFIG_SOURCE, QWEN35_9B
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        print("HYBRID_CONFIG " + json.dumps(dict(
            rank=rank, pp_rank=pp_rank, workload=asdict(hybrid_config),
            layers=hybrid_config.pipeline_layers(pp_rank),
            tokens_per_iteration=hybrid_config.tokens_per_iteration,
            source=QWEN35_CONFIG_SOURCE if hybrid_config.model == QWEN35_9B else "synthetic debug dimensions",
            memory_validated=False,
            full_attention_backend="local_unfused",
        ), sort_keys=True), flush=True)

    duras = []
    duras_wall = []
    region = None
    if hybrid_config is not None and os.environ.get("PHANTORA_COVERAGE") == "1":
        from qwen_hybrid_control import TrainingRegion, expected_events
        region = TrainingRegion(rank, world_size, expected_events(hybrid_config, pp_rank, iterations))
        region.start_observer()
        region.emit("start")
    for i in range(iterations):
        if virtual_pipeline_model_parallel_size is None:
            train_iterator = get_train_data_iterator(vocab_size, micro_batch_size, sequence_length)
        else:
            train_iterator = [
                get_train_data_iterator(vocab_size, micro_batch_size, sequence_length)
                for _ in range(virtual_pipeline_model_parallel_size)
            ]
        start, start_wall = time_pair()
        for model_chunk in iter_model_chunks(model):
            model_chunk.zero_grad_buffer()
        optim.zero_grad()
        if pp_size > 1 and virtual_pipeline_model_parallel_size is not None:
            forward_backward_pipelining_with_interleaving(
                forward_step_func=functools.partial(forward_step_func, device),
                data_iterator=train_iterator,
                model=model,
                num_microbatches=num_microbatches,
                seq_length=sequence_length,
                micro_batch_size=micro_batch_size,
                forward_only=False,
            )
        elif pp_size > 1:
            forward_backward_pipelining_without_interleaving(
                forward_step_func=functools.partial(forward_step_func, device),
                data_iterator=train_iterator,
                model=model,
                num_microbatches=num_microbatches,
                seq_length=sequence_length,
                micro_batch_size=micro_batch_size,
                forward_only=False,
            )
        else:
            forward_backward_no_pipelining(
                forward_step_func=functools.partial(forward_step_func, device),
                data_iterator=train_iterator,
                model=model,
                num_microbatches=num_microbatches,
                seq_length=sequence_length,
                micro_batch_size=micro_batch_size,
                forward_only=False,
            )
        optim.step()
        torch.cuda.synchronize()
        end, end_wall = time_pair()
        print(f"rank {rank} iter {i} time: {end - start:.2f} wall: {end_wall - start_wall:.2f}\n", end="")
        duras.append(end - start)
        duras_wall.append(end_wall - start_wall)

    if region is not None:
        region.stop_observer()
        region.emit("finish")

    if hybrid_config is not None:
        print("HYBRID_METRICS " + json.dumps(dict(
            version=1, run_id=os.environ.get("PHANTORA_RUN_ID"), rank=rank,
            world_size=world_size, durations=duras, wall_durations=duras_wall,
            tokens_per_iteration=hybrid_config.tokens_per_iteration,
            iterations=iterations, memory_validated=False,
        ), sort_keys=True), flush=True)
    peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"Rank {rank} Time: {duras} Avg Time: {sum(duras[1:]) / (iterations - 1):.2f}\n", end="")
    memory_label = " Peak (unvalidated hybrid memory)" if hybrid_config is not None else " Peak"
    print(f"Rank {rank}{memory_label}: {peak_vram_mib:<.2f}MiB\n", end="")
    print(f"Rank {rank} Wall: {duras_wall} Avg Wall: {sum(duras_wall[1:]) / (iterations - 1):.2f}\n", end="")
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-preset", choices=["debug", "9b"], default=None)
    parser.add_argument("--global_batch_size", type=int, default=None)
    parser.add_argument("--data_parallel_size", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--pipeline_model_parallel_size", type=int, default=1)
    parser.add_argument("--virtual_pipeline_model_parallel_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--ffn_hidden_size", type=int, default=11008)
    parser.add_argument("--num_attention_heads", type=int, default=32)
    parser.add_argument("--num_query_groups", type=int, default=None,
        help="Number of KV heads for GQA (None = MHA, i.e., equal to num_attention_heads).")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--sequence_length", type=int, default=4096)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--recompute_activations", action="store_true")
    parser.add_argument("--iterations", type=int, default=4)
    # Architectural knobs (names mirror Megatron's TransformerConfig / GPTModel fields).
    parser.add_argument("--swiglu", action="store_true",
        help="Use SwiGLU gated MLP (Megatron's `gated_linear_unit=True`).")
    parser.add_argument("--position_embedding_type", type=str, default="learned_absolute",
        choices=["learned_absolute", "rope"])
    parser.add_argument("--rotary_base", type=float, default=10000.0)
    parser.add_argument("--normalization", type=str, default="LayerNorm",
        choices=["LayerNorm", "RMSNorm"])
    parser.add_argument("--disable_bias_linear", action="store_true",
        help="Set `add_bias_linear=False` (Megatron default is True).")
    # MoE knobs (names mirror Megatron's TransformerConfig). num_moe_experts=None => dense.
    parser.add_argument("--num_moe_experts", type=int, default=None,
        help="Number of routed experts (None = dense, no MoE).")
    parser.add_argument("--moe_router_topk", type=int, default=2)
    parser.add_argument("--moe_token_dispatcher_type", type=str, default="alltoall",
        choices=["allgather", "alltoall"])
    parser.add_argument("--expert_model_parallel_size", type=int, default=1)
    parser.add_argument("--kv_channels", type=int, default=None,
        help="Attention head_dim; None = hidden_size // num_attention_heads. "
             "Qwen3 and gpt-oss set this explicitly.")
    parser.add_argument("--qk_layernorm", action="store_true",
        help="Apply RMSNorm to Q and K (QK-norm); used by Qwen3.")
    args = parser.parse_args()
    hybrid_config = None
    if args.hybrid_preset is not None:
        import sys
        from qwen_hybrid_config import configure_hybrid_args
        try:
            hybrid_config = configure_hybrid_args(args, sys.argv[1:], int(os.environ.get("WORLD_SIZE", "1")))
        except ValueError as exc:
            parser.error(str(exc))
    elif args.global_batch_size is not None or args.data_parallel_size is not None:
        parser.error("--global_batch_size/--data_parallel_size require --hybrid-preset")
    if hybrid_config is not None and args.iterations < 2:
        parser.error("--iterations must be at least 2 for a steady-state average")

    enable_function_tracer()
    main(
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        vocab_size=args.vocab_size,
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        recompute_activations=args.recompute_activations,
        iterations=args.iterations,
        num_query_groups=args.num_query_groups,
        swiglu=args.swiglu,
        position_embedding_type=args.position_embedding_type,
        rotary_base=args.rotary_base,
        normalization=args.normalization,
        disable_bias_linear=args.disable_bias_linear,
        num_moe_experts=args.num_moe_experts,
        moe_router_topk=args.moe_router_topk,
        moe_token_dispatcher_type=args.moe_token_dispatcher_type,
        expert_model_parallel_size=args.expert_model_parallel_size,
        kv_channels=args.kv_channels,
        qk_layernorm=args.qk_layernorm,
        hybrid_config=hybrid_config,
    )
    disable_function_tracer()
