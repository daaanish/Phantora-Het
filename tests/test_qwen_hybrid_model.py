import argparse
from dataclasses import replace
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from qwen_hybrid_config import DEBUG_MODEL, HybridConfig, configure_hybrid_args
from qwen_hybrid_model import (
    GDNOutputRMSNorm, QwenRMSNorm, build_hybrid_spec,
    finalize_hybrid_grads, require_hybrid_kernels, transformer_overrides, validate_hybrid_structure,
)


def fake_modules(exports):
    modules = {}
    for path, attributes in exports.items():
        parts = path.split(".")
        for end in range(1, len(parts) + 1):
            name = ".".join(parts[:end])
            if name not in modules:
                modules[name] = ModuleType(name)
        modules[path].__dict__.update(attributes)
    return modules


class HybridModelTests(unittest.TestCase):
    def test_actual_structure_guard_with_explicit_doubles(self):
        workload = HybridConfig()
        class GatedDeltaNet: pass
        layers = []
        for index, kind in workload.pipeline_layers(0):
            attention = GatedDeltaNet() if kind == "gdn" else SimpleNamespace()
            if kind == "gdn":
                dims = workload.model
                key_dim = dims.linear_key_head_dim * dims.linear_num_key_heads
                value_dim = dims.linear_value_head_dim * dims.linear_num_value_heads
                attention.in_proj = SimpleNamespace(weight=torch.empty(2 * key_dim + 2 * value_dim + 2 * dims.linear_num_value_heads, dims.hidden_size))
                attention.conv1d = SimpleNamespace(weight=torch.empty(2 * key_dim + value_dim, 1, dims.conv_kernel_dim))
                attention.A_log = torch.empty(dims.linear_num_value_heads)
                attention.dt_bias = torch.empty(dims.linear_num_value_heads)
                attention.out_norm = SimpleNamespace(weight=torch.empty(workload.model.linear_value_head_dim))
                attention.out_proj = SimpleNamespace(weight=torch.empty(workload.model.hidden_size, workload.model.linear_num_value_heads * workload.model.linear_value_head_dim))
            if kind != "gdn":
                dims = workload.model
                attention.linear_qkv = SimpleNamespace(weight=torch.empty((2 * dims.num_attention_heads + 2 * dims.num_query_groups) * dims.head_dim, dims.hidden_size))
                attention.linear_proj = SimpleNamespace(weight=torch.empty(dims.hidden_size, dims.num_attention_heads * dims.head_dim))
                attention.q_layernorm = SimpleNamespace(weight=torch.empty(dims.head_dim))
                attention.k_layernorm = SimpleNamespace(weight=torch.empty(dims.head_dim))
            mlp = SimpleNamespace(linear_fc1=SimpleNamespace(weight=torch.empty(2 * workload.model.ffn_hidden_size, workload.model.hidden_size)),
                linear_fc2=SimpleNamespace(weight=torch.empty(workload.model.hidden_size, workload.model.ffn_hidden_size)))
            layers.append(SimpleNamespace(layer_number=index + 1, self_attention=attention, mlp=mlp))
        model = SimpleNamespace(decoder=SimpleNamespace(layers=layers))
        self.assertEqual(len(validate_hybrid_structure(model, workload, 0)), 4)
        layers[0].layer_number = 99
        with self.assertRaises(RuntimeError): validate_hybrid_structure(model, workload, 0)

    def test_norm_equations_and_gradients(self):
        config = SimpleNamespace(params_dtype=torch.bfloat16, sequence_parallel=False, use_cpu_initialization=True)
        for cls, centered in ((QwenRMSNorm, True), (GDNOutputRMSNorm, False)):
            norm = cls(config, 8)
            x = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
            with torch.no_grad():
                norm.weight.copy_(torch.linspace(-0.5, 0.5, 8))
            value = x.float()
            reference = value / torch.sqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            reference = (reference * (norm.weight.float() + 1)).to(x.dtype) if centered else norm.weight * reference.to(x.dtype)
            actual = norm(x)
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(actual, reference))
            actual.square().sum().backward()
            self.assertIsNotNone(x.grad)
            self.assertIsNotNone(norm.weight.grad)

    def test_gdn_norm_bfloat16_rounding_matches_cast_before_scale(self):
        config = SimpleNamespace(params_dtype=torch.bfloat16, sequence_parallel=False, use_cpu_initialization=True)
        norm = GDNOutputRMSNorm(config, 32)
        x = torch.linspace(0.1, 2.7, 32, dtype=torch.bfloat16).view(1, 32)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.3, 1.7, 32))
        value = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        expected = norm.weight * value.to(x.dtype)
        late_cast = (norm.weight.float() * value).to(x.dtype)
        self.assertFalse(torch.equal(expected, late_cast))
        self.assertTrue(torch.equal(norm(x), expected))

    def test_shared_output_norm_gradient_is_sum_of_head_shards(self):
        config = SimpleNamespace(params_dtype=torch.float32, sequence_parallel=False, use_cpu_initialization=True)
        x = torch.randn(2, 8, 16)
        gate = torch.randn_like(x)
        full = GDNOutputRMSNorm(config, 16)
        (full(x) * torch.nn.functional.silu(gate)).square().sum().backward()
        grads = []
        for xs, gs in zip(x.chunk(2, dim=1), gate.chunk(2, dim=1)):
            shard = GDNOutputRMSNorm(config, 16)
            (shard(xs) * torch.nn.functional.silu(gs)).square().sum().backward()
            grads.append(shard.weight.grad)
        torch.testing.assert_close(full.weight.grad, grads[0] + grads[1])

    def test_simulation_installs_kernel_adapters(self):
        with patch.dict(os.environ, {"PHANTORA": "1"}), \
                patch("qwen_hybrid_kernels.install_simulation_kernels") as install:
            require_hybrid_kernels()
        install.assert_called_once_with()

    def test_spec_wiring_and_global_pipeline_pattern_with_test_doubles(self):
        class GDN:
            pass

        class Column:
            pass

        class Row:
            pass

        def local_spec(**kwargs):
            self.assertEqual(kwargs, dict(normalization="RMSNorm", qk_layernorm=True))
            return SimpleNamespace(submodules=SimpleNamespace(
                self_attention=SimpleNamespace(module="full", submodules=SimpleNamespace()),
                mlp=SimpleNamespace(module="local_mlp"),
            ))

        modules = fake_modules({
            "causal_conv1d": {"causal_conv1d_fn": lambda: None},
            "megatron.core.models.gpt.gpt_layer_specs": {"get_gpt_layer_local_spec": local_spec},
            "megatron.core.ssm.gated_delta_net": {"GatedDeltaNet": GDN, "GatedDeltaNetSubmodules": SimpleNamespace},
            "megatron.core.tensor_parallel.layers": {"ColumnParallelLinear": Column, "RowParallelLinear": Row},
            "megatron.core.transformer.spec_utils": {"ModuleSpec": SimpleNamespace},
            "megatron.core.transformer.transformer_block": {"TransformerBlockSubmodules": SimpleNamespace},
        })
        config = HybridConfig(model=replace(DEBUG_MODEL, num_layers=12), pipeline_parallel_size=2, world_size=2)
        with patch.dict(sys.modules, modules), patch.dict(os.environ):
            os.environ.pop("PHANTORA", None)
            spec = build_hybrid_spec(config, 1)
        self.assertIs(spec.layer_norm, QwenRMSNorm)
        for layer, (_, kind) in zip(spec.layer_specs, config.pipeline_layers(1)):
            self.assertIs(layer.submodules.input_layernorm, QwenRMSNorm)
            self.assertIs(layer.submodules.pre_mlp_layernorm, QwenRMSNorm)
            self.assertEqual(layer.submodules.mlp.module, "local_mlp")
            attention = layer.submodules.self_attention
            if kind == "gdn":
                self.assertIs(attention.module, GDN)
                self.assertIs(attention.submodules.in_proj, Column)
                self.assertIs(attention.submodules.out_proj, Row)
                self.assertIs(attention.submodules.out_norm, GDNOutputRMSNorm)
                self.assertFalse(attention.params["conv_bias"])
            else:
                self.assertIs(attention.submodules.q_layernorm, QwenRMSNorm)
                self.assertIs(attention.submodules.k_layernorm, QwenRMSNorm)

    def test_finalize_keeps_megatron_and_sums_only_gdn_scale(self):
        grad = torch.ones(8)
        param = SimpleNamespace(main_grad=grad, requires_grad=True)
        chunk = SimpleNamespace(named_parameters=lambda: [
            ("decoder.layers.0.self_attention.out_norm.weight", param),
            ("decoder.layers.0.input_layernorm.weight", param),
        ])
        original = Mock()
        modules = fake_modules({
            "megatron.core": {"parallel_state": SimpleNamespace(get_tensor_model_parallel_group=lambda: "tp")},
            "megatron.core.distributed": {"finalize_model_grads": original},
        })
        with patch.dict(sys.modules, modules), patch.object(torch.distributed, "get_world_size", return_value=2), \
                patch.object(torch.distributed, "all_reduce") as reduce:
            finalize_hybrid_grads([chunk], None)
        original.assert_called_once_with([chunk], None)
        reduce.assert_called_once_with(grad, op=torch.distributed.ReduceOp.SUM, group="tp")

    def test_transformer_flags(self):
        overrides = transformer_overrides(HybridConfig())
        self.assertTrue(overrides["attention_output_gate"])
        self.assertEqual(overrides["attention_dropout"], 0)
        self.assertEqual(overrides["hidden_dropout"], 0)
        self.assertEqual(overrides["layernorm_epsilon"], 1e-6)


class HybridCliTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(
            hybrid_preset="debug", tensor_parallel_size=4, pipeline_model_parallel_size=1,
            data_parallel_size=None, global_batch_size=None, micro_batch_size=1,
            sequence_length=4096, gradient_accumulation=1, num_moe_experts=None,
            expert_model_parallel_size=1, virtual_pipeline_model_parallel_size=None,
            recompute_activations=False,
        )

    def test_hybrid_defaults_and_dp_accumulation(self):
        args = self.args()
        config = configure_hybrid_args(args, [], 2)
        self.assertEqual(config.data_parallel_size, 2)
        self.assertEqual(config.gradient_accumulation, 4)
        self.assertEqual(args.tensor_parallel_size, 1)
        self.assertEqual(args.sequence_length, 128)
        self.assertEqual(args.normalization, "RMSNorm")
        self.assertTrue(args.qk_layernorm)

    def test_explicit_accumulation_is_not_rewritten(self):
        args = self.args()
        args.gradient_accumulation = 2
        args.global_batch_size = 4
        config = configure_hybrid_args(args, ["--gradient_accumulation=2"], 2)
        self.assertEqual(config.gradient_accumulation, 2)

    def test_conflicting_architecture_or_batch_rejected(self):
        args = self.args()
        args.num_layers = 8
        with self.assertRaisesRegex(ValueError, "conflicts"):
            configure_hybrid_args(args, ["--num_layers", "8"], 1)
        args = self.args()
        args.gradient_accumulation = 2
        with self.assertRaisesRegex(ValueError, "global_batch_size"):
            configure_hybrid_args(args, ["--gradient_accumulation", "2"], 1)


if __name__ == "__main__":
    unittest.main()
