#!/usr/bin/env python3
"""Profile actual pinned fused boundaries; never imports simulation adapters."""

import argparse
import csv
import ctypes
from importlib import metadata
import json
import os
from pathlib import Path
import sys

from gdn_common import PINS, TORCH_REVISION, atomic_text, canonical_key, merge_profile, option_float


def assert_real_execution():
    if "PHANTORA" in os.environ or os.environ.get("LD_PRELOAD", "").strip():
        raise RuntimeError("real fused profiling forbids PHANTORA and LD_PRELOAD")
    aliases = sys.modules.get("qwen_hybrid_kernels")
    if aliases is not None and getattr(aliases, "_ADAPTERS", None) is not None:
        raise RuntimeError("simulation aliases are installed in this process")
    driver = ctypes.CDLL("libcuda.so.1")
    if hasattr(driver, "phantora_gdn_call"):
        raise RuntimeError("the CUDA library is a Phantora stub")


def load_kernels(gpu, method):
    assert_real_execution()
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != gpu:
        raise RuntimeError("actual GPU does not match --gpu-name")
    source = dict(gpu=gpu, torch=torch.__version__, torch_revision=torch.version.git_version, cuda=torch.version.cuda,
        triton=metadata.version("triton"), fla=metadata.version("flash-linear-attention"),
        causal_conv1d=metadata.version("causal-conv1d"), measurement=method)
    from gdn_common import validate_provenance
    validate_provenance(source, gpu)
    from causal_conv1d import causal_conv1d_fn
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    kernels = dict(causal_conv1d=causal_conv1d_fn, l2norm=l2norm, chunk_gated_delta_rule=chunk_gated_delta_rule)
    for name, function in kernels.items():
        prefix = "causal_conv1d" if name == "causal_conv1d" else "fla."
        if not getattr(function, "__module__", "").startswith(prefix):
            raise RuntimeError(f"unexpected/placeholder kernel implementation: {name}")
    return torch, kernels, source


def allocate_strided(info, name, torch_api=None, device="cuda"):
    if torch_api is None:
        import torch as torch_api
    span = 1 + sum((size - 1) * stride for size, stride in zip(info["shape"], info["strides"]))
    base = torch_api.empty(span, dtype=getattr(torch_api, info["dtype"]), device=device)
    if name == "g":
        base.uniform_(-0.1, -0.01)
    elif name == "beta":
        base.uniform_(0.1, 0.9)
    else:
        base.uniform_(-0.2, 0.2)
    return base.as_strided(info["shape"], info["strides"]).detach().requires_grad_(info["requires_grad"])


def build_thunk(key, kernels, torch_api=None, allocator=None):
    if torch_api is None:
        import torch as torch_api
    key = json.loads(canonical_key(key))
    names = {"l2norm": ["x"], "causal_conv1d": ["x", "weight", "bias"],
             "chunk_gated_delta_rule": ["q", "k", "v", "g", "beta"]}[key["op"]]
    allocator = allocator or (lambda info, name: allocate_strided(info, name, torch_api))
    inputs = [allocator(info, name) if info is not None else None for name, info in zip(names, key["inputs"])]
    if key["op"] == "chunk_gated_delta_rule" and key["options"]["use_qk_l2norm_in_kernel"] == "false":
        # Stabilize long recurrences without changing recorded layout or leaf flags.
        with torch_api.no_grad():
            for value in inputs[:2]:
                unit = value.float() * torch_api.rsqrt(value.float().square().sum(-1, keepdim=True) + 1e-6)
                value.copy_(unit.to(value.dtype))
    function = kernels[key["op"]]
    options = key["options"]
    if key["op"] == "l2norm":
        kwargs = dict(eps=option_float(options["eps"]), output_dtype=getattr(torch_api, options["output_dtype"]))
    elif key["op"] == "causal_conv1d":
        kwargs = dict(activation=None if options["activation"] == "none" else options["activation"])
    else:
        kwargs = dict(scale=option_float(options["scale"]), use_qk_l2norm_in_kernel=options["use_qk_l2norm_in_kernel"] == "true",
                      initial_state=None, output_final_state=False)

    def forward():
        with torch_api.set_grad_enabled(key["grad_enabled"]):
            return function(*inputs, **kwargs)

    if key["direction"] == "forward":
        return forward
    # Setup/forward graph and incoming gradient allocation are outside timing.
    output = forward()
    output = output[0] if isinstance(output, tuple) else output
    gradient = allocator(key["grad_output"], "grad_output")
    leaves = [value for value, info in zip(inputs, key["inputs"]) if info is not None and info["requires_grad"]]
    return lambda: torch_api.autograd.grad(output, leaves, grad_outputs=gradient, retain_graph=True, allow_unused=False)


def checked_thunk(key, kernels, torch_api):
    thunk = build_thunk(key, kernels, torch_api)
    result = thunk()  # validation outside warmup and measurement
    if key["direction"] == "backward":
        expected = [info for info in key["inputs"] if info is not None and info["requires_grad"]]
        outputs = result
    else:
        outputs = [result[0] if isinstance(result, tuple) else result]
        info = key["inputs"][2] if key["op"] == "chunk_gated_delta_rule" else key["inputs"][0]
        expected = [dict(info, dtype=key["options"].get("output_dtype", info["dtype"]))]
    if len(outputs) != len(expected):
        raise ValueError("unexpected fused output arity")
    for output, info in zip(outputs, expected):
        if list(output.shape) != info["shape"] or output.dtype != getattr(torch_api, info["dtype"]) or not torch_api.isfinite(output).all().item():
            raise ValueError("invalid fused output/gradient shape, dtype or finiteness")
    return thunk


def read_keys(path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if "key" not in (reader.fieldnames or []):
            raise ValueError("expected a fused key or discovery CSV")
        keys = []
        for row in reader:
            if "status" in row and row["status"] != "missing":
                raise ValueError("unexpected discovery status")
            keys.append(canonical_key(row["key"]))
    if not keys:
        raise ValueError("no fused keys requested")
    return keys


def device_metadata(torch_api):
    metadata = {}
    try:
        props = torch_api.cuda.get_device_properties(0)
        metadata["properties"] = {name: getattr(props, name) for name in
            ("name", "major", "minor", "total_memory", "multi_processor_count")}
    except Exception as error:
        metadata["properties_error"] = str(error)
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,pci.bus_id", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=True)
        metadata["driver_and_pci"] = result.stdout.strip()
    except Exception as error:
        metadata["driver_error"] = str(error)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--method", choices=["cupti_kernel_sum", "cuda_event_wall"], default="cupti_kernel_sum")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        parser.error("warmup must be nonnegative and iters positive")
    torch, kernels, source = load_kernels(args.gpu_name, args.method)
    torch.manual_seed(42)
    manifest = args.out / "manifest.md"
    if manifest.exists():
        gpu_lines = [line.removeprefix("- **GPU:** ").strip() for line in manifest.read_text().splitlines() if line.startswith("- **GPU:** ")]
        if gpu_lines != [args.gpu_name]:
            raise ValueError("output DB manifest GPU mismatch")
    elif args.out.exists() and any(args.out.glob("*.csv")):
        raise ValueError("existing output tables require a validated GPU manifest")
    from bench import time_kernel, time_wall
    timer = time_kernel if args.method == "cupti_kernel_sum" else time_wall
    result = merge_profile(args.out / "gdn.csv", read_keys(args.keys), source,
        lambda key: timer(checked_thunk(key, kernels, torch), args.warmup, args.iters))
    result.update(device_metadata=device_metadata(torch), provenance=source, warmup=args.warmup, iters=args.iters, memory_validated=False)
    if result["committed"] and not manifest.exists():
        atomic_text(manifest, f"# Phantora performance database\n\n- **GPU:** {args.gpu_name}\n- **Schema version:** 1\n")
    atomic_text(args.out / "gdn.profile.json", json.dumps(result, indent=2))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["committed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
