"""Fail-fast dependency/API check inside the opt-in Linux hybrid image."""

import inspect
import json
import os
from importlib import metadata

from packaging.requirements import Requirement
from packaging.version import Version


EXPECTED_STACK = {
    "megatron-core": "0.16.0", "flash-linear-attention": "0.4.0",
    "causal-conv1d": "1.5.3.post1", "triton": "3.7.0",
    "transformers": "4.56.2", "deepspeed": "0.17.5",
    "torchtitan": "0.2.0", "numpy": "1.26.4",
}


def check_metadata():
    expected = EXPECTED_STACK
    for name, version in expected.items():
        actual = metadata.version(name)
        if actual != version:
            raise RuntimeError(f"{name}: expected {version}, found {actual}")
    for name in ("megatron-core", "flash-linear-attention", "causal-conv1d"):
        for dependency in metadata.requires(name) or []:
            requirement = Requirement(dependency)
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
            actual = metadata.version(requirement.name)
            if actual not in requirement.specifier:
                raise RuntimeError(f"{name} requires {requirement}, found {actual}")

    if Version(metadata.version("torch")).base_version != "2.9.1":
        raise RuntimeError("hybrid requires patched torch 2.9.1 distribution metadata")
    return {name: metadata.version(name) for name in (*expected, "torch")}


def check_runtime(replay=None):
    replay = os.environ.get("PHANTORA") is not None if replay is None else replay
    versions = check_metadata()
    import torch
    if Version(torch.__version__).base_version != "2.9.1":
        raise RuntimeError(f"hybrid image must preserve patched torch 2.9.1, found {torch.__version__}")
    revision = getattr(torch.version, "git_version", None)
    if revision != "bbb19f8c856fcf8d7c60267fecb976b4749e2cff":
        raise RuntimeError(f"hybrid requires the pinned patched torch revision, found {revision}")
    if getattr(torch.version, "cuda", None) != "12.8":
        raise RuntimeError("hybrid requires torch built with CUDA 12.8")
    versions.update(torch_revision=revision, cuda=torch.version.cuda)
    if not callable(getattr(torch.profiler, "enable_function_tracer", None)):
        raise RuntimeError("patched function tracer is missing")
    if replay:
        from qwen_hybrid_kernels import install_simulation_kernels
        install_simulation_kernels()
    from causal_conv1d import causal_conv1d_fn
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    checks = (
        (causal_conv1d_fn, ("x", "weight", "bias", "activation")),
        (l2norm, ("x",)),
        (chunk_gated_delta_rule, ("q", "k", "v", "g", "beta")),
        (GatedDeltaNet, ("config", "submodules", "pg_collection")),
        (GatedDeltaNetSubmodules, ("in_proj", "out_norm", "out_proj")),
        (GPTModel, ("rotary_percent", "share_embeddings_and_output_weights")),
        (TransformerConfig, ("attention_output_gate", "linear_attention_freq")),
    )
    for function, names in checks:
        params = inspect.signature(function).parameters
        for name in names:
            if name not in params:
                raise RuntimeError(f"unsupported signature: {function.__name__} lacks {name}")
    from qwen_hybrid_config import HybridConfig
    from qwen_hybrid_model import build_hybrid_spec
    block = build_hybrid_spec(HybridConfig(), 0)
    assert len(block.layer_specs) == 4
    print("HYBRID_RUNTIME " + json.dumps(versions, sort_keys=True))
    print(f"Hybrid dependency/import/signature/spec checks passed (replay={replay}); CUDA execution remains unvalidated.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", help="Check distributions without importing torch/FLA/CUDA extensions")
    args = parser.parse_args()
    if args.metadata_only:
        print("HYBRID_METADATA " + json.dumps(check_metadata(), sort_keys=True))
    else:
        check_runtime()
