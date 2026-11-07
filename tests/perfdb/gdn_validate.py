#!/usr/bin/env python3
"""Optional real GPU fused forward/all-input-gradient checks against tiny equations."""

import argparse
import json
import math
from pathlib import Path

from gdn_common import atomic_text
from gdn_profile import load_kernels
from gdn_references import causal_conv_reference, l2norm_reference, recurrent_gdn_reference


def compare_tensor(torch, actual, expected, rtol, atol, relative_l2, label):
    difference = torch.linalg.vector_norm((actual.float() - expected.float()).double()).item()
    reference = torch.linalg.vector_norm(expected.float().double()).item()
    error = difference / reference if reference else (0.0 if difference == 0 else float("inf"))
    if not math.isfinite(error) or error > relative_l2:
        raise RuntimeError(f"{label} relative L2 error {error} exceeds {relative_l2}")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return dict(label=label, relative_l2_error=error, reference_l2=reference, difference_l2=difference)


def validate(torch, kernels, rtol, atol, relative_l2):
    torch.manual_seed(42)
    cases = []
    x = torch.randn(1, 8, 16, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(8, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    bias = torch.randn(8, device="cuda", dtype=torch.bfloat16) * 0.1
    cases.append(("causal_conv1d", [x, weight, bias], dict(activation="silu"), causal_conv_reference))
    cases.append(("l2norm", [x.transpose(1, 2).contiguous()], {}, l2norm_reference))
    q, k, v = [torch.randn(1, 16, 2, 128, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(3)]
    g = torch.full((1, 16, 2), -0.05, device="cuda", dtype=torch.float32)
    beta = torch.full((1, 16, 2), 0.5, device="cuda", dtype=torch.bfloat16)
    for internal_norm in (False, True):
        cases.append(("chunk_gated_delta_rule", [q, k, v, g, beta],
            dict(use_qk_l2norm_in_kernel=internal_norm), recurrent_gdn_reference))
    results = []
    for name, tensors, options, reference in cases:
        actual_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
        reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
        actual = kernels[name](*actual_inputs, **options)
        actual = actual[0] if isinstance(actual, tuple) else actual
        expected = reference(*reference_inputs, **options)
        errors = [compare_tensor(torch, actual, expected, rtol, atol, relative_l2, name + "/output")]
        gradient = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, actual_inputs, gradient, allow_unused=False)
        expected_grads = torch.autograd.grad(expected, reference_inputs, gradient, allow_unused=False)
        for index, (actual_grad, expected_grad) in enumerate(zip(actual_grads, expected_grads)):
            if not torch.isfinite(actual_grad).all() or not torch.isfinite(expected_grad).all():
                raise RuntimeError(f"nonfinite {name} gradient {index}")
            errors.append(compare_tensor(torch, actual_grad, expected_grad, rtol, atol, relative_l2, f"{name}/gradient{index}"))
        results.append(dict(op=name, options=options, differentiable_inputs=len(tensors), errors=errors, passed=True))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--relative-l2", type=float, default=0.03)
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in (args.rtol, args.atol, args.relative_l2)):
        parser.error("tolerances must be finite and positive")
    torch, kernels, source = load_kernels(args.gpu_name, "cupti_kernel_sum")
    result = dict(runtime={key: value for key, value in source.items() if key != "measurement"},
        memory_validated=False, whole_model_validated=False, rtol=args.rtol, atol=args.atol, relative_l2_tolerance=args.relative_l2,
        whole_model_limits=["Core packed projections and external L2/head repetition differ from HF composition",
            "Core BF16 A_log.exp differs from HF A_log.float().exp; quantify before whole-model equivalence"])
    try:
        result.update(cases=validate(torch, kernels, args.rtol, args.atol, args.relative_l2), passed=True)
    except Exception as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
    atomic_text(args.out, json.dumps(result, indent=2))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
