"""Run with python3 -m unittest discover -s tests -p test_qwen_hybrid_config.py."""

import unittest
from dataclasses import replace

from qwen_hybrid_config import DEBUG_MODEL, QWEN35_9B, HybridConfig


class HybridConfigTests(unittest.TestCase):
    def test_text_preset_keeps_explicit_attention_dimensions(self):
        config = HybridConfig(model=QWEN35_9B)
        self.assertEqual(config.model.num_layers, 32)
        self.assertEqual(config.model.layer_types.count("full_attention"), 8)
        self.assertEqual(config.model.head_dim, 256)
        self.assertEqual(config.model.linear_key_head_dim, 128)
        self.assertEqual(config.model.linear_value_head_dim, 128)

    def test_global_pattern_and_crossing_pipeline_boundary(self):
        config = HybridConfig(model=replace(DEBUG_MODEL, num_layers=12), pipeline_parallel_size=2, world_size=2)
        self.assertEqual(config.pipeline_layers(1), (
            (6, "gdn"), (7, "full_attention"), (8, "gdn"),
            (9, "gdn"), (10, "gdn"), (11, "full_attention"),
        ))
        joined = config.pipeline_layers(0) + config.pipeline_layers(1)
        self.assertEqual(joined, tuple(enumerate(config.model.layer_types)))

    def test_validation_matrix_has_exact_batch(self):
        for tp, dp, pp, ga in ((1, 1, 1, 8), (2, 1, 1, 8), (1, 2, 1, 4), (1, 1, 2, 8), (2, 2, 2, 4)):
            with self.subTest(tp=tp, dp=dp, pp=pp):
                config = HybridConfig(tensor_parallel_size=tp, data_parallel_size=dp,
                                      pipeline_parallel_size=pp, world_size=tp * dp * pp,
                                      gradient_accumulation=ga)
                self.assertEqual(config.effective_batch_size, 8)
                self.assertEqual(config.tokens_per_iteration, 1024)
                self.assertEqual(config.gradient_accumulation, ga)

    def test_pp_does_not_inflate_accumulation(self):
        config = HybridConfig(pipeline_parallel_size=2, world_size=2,
                              gradient_accumulation=1, global_batch_size=1)
        self.assertEqual(config.gradient_accumulation, 1)
        self.assertEqual(config.effective_batch_size, 1)

    def test_world_and_batch_mismatch_rejected(self):
        for kwargs in ({"world_size": 2}, {"global_batch_size": 16}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HybridConfig(**kwargs)

    def test_positive_integer_fields(self):
        for name in ("sequence_length", "micro_batch_size", "gradient_accumulation", "tensor_parallel_size"):
            for value in (0, -1, True, 1.5):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    HybridConfig(**{name: value})

    def test_unsupported_modes_rejected(self):
        cases = {"context_parallel_size": 2, "expert_parallel_size": 2,
                 "sequence_parallel": True, "virtual_pipeline_size": 1,
                 "recompute_activations": True, "packed_sequences": True,
                 "use_cache": True, "vision": True, "mtp": True, "dtype": "float16"}
        for name, value in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                HybridConfig(**{name: value})

    def test_partition_and_head_divisibility(self):
        for model, kwargs in (
            (replace(DEBUG_MODEL, num_layers=5), {"pipeline_parallel_size": 2}),
            (replace(DEBUG_MODEL, num_query_groups=1), {"tensor_parallel_size": 2}),
            (replace(DEBUG_MODEL, linear_num_key_heads=1), {"tensor_parallel_size": 2}),
            (replace(DEBUG_MODEL, ffn_hidden_size=769), {"tensor_parallel_size": 2}),
        ):
            with self.subTest(model=model), self.assertRaises(ValueError):
                HybridConfig(model=model, world_size=2, **kwargs)

    def test_model_head_and_rotary_constraints(self):
        for kwargs in ({"num_query_groups": 3}, {"linear_num_value_heads": 7},
                       {"partial_rotary_factor": 0}, {"partial_rotary_factor": 0.1},
                       {"head_dim": 0}, {"rms_norm_eps": 0}, {"conv_kernel_dim": 5},
                       {"rotary_base": float("inf")}, {"rotary_base": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(DEBUG_MODEL, **kwargs)

    def test_invalid_stage_rejected(self):
        for stage in (-1, 1, True, 0.5):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                HybridConfig().pipeline_layers(stage)


if __name__ == "__main__":
    unittest.main()
