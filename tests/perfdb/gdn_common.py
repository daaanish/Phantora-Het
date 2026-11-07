"""Stdlib-only schema-1 fused keys, provenance and atomic measured CSV writes."""

import csv
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import os

TORCH_REVISION = "bbb19f8c856fcf8d7c60267fecb976b4749e2cff"
PINS = dict(triton="3.7.0", fla="0.4.0", causal_conv1d="1.5.3.post1")
DTYPES = {"float32", "float16", "bfloat16"}


def option_float(text):
    value = struct.unpack(">d", bytes.fromhex(text[4:]))[0] if text.startswith("f64:") else float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("fused numeric option must be finite and positive")
    return value


def canonical_key(raw):
    def unique_fields(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate fused JSON field")
            result[name] = value
        return result
    text = raw if isinstance(raw, str) else json.dumps(raw)
    if len(text.encode("utf-8")) > 32768:
        raise ValueError("fused payload exceeds transport limit")
    key = json.loads(text, object_pairs_hook=unique_fields)
    if set(key) != {"version", "op", "direction", "inputs", "grad_enabled", "grad_output", "options"} or type(key["version"]) is not int or key["version"] != 1:
        raise ValueError("unsupported fused key schema")
    if key["direction"] not in ("forward", "backward") or type(key["grad_enabled"]) is not bool:
        raise ValueError("invalid fused direction/grad flag")
    options = {"l2norm": {"eps", "output_dtype"}, "causal_conv1d": {"activation"},
               "chunk_gated_delta_rule": {"scale", "use_qk_l2norm_in_kernel"}}
    if key["op"] not in options or set(key["options"]) != options[key["op"]] or any(type(v) is not str for v in key["options"].values()):
        raise ValueError("unsupported fused options")
    for name in ("eps", "scale"):
        if name in key["options"]:
            key["options"][name] = "f64:" + struct.pack(">d", option_float(key["options"][name])).hex()
    count = {"l2norm": 1, "causal_conv1d": 3, "chunk_gated_delta_rule": 5}[key["op"]]
    if not isinstance(key["inputs"], list) or len(key["inputs"]) != count:
        raise ValueError("invalid input arity")
    for index, tensor in enumerate(key["inputs"]):
        if tensor is None and key["op"] == "causal_conv1d" and index == 2:
            continue
        validate_tensor(tensor)
    x = key["inputs"][0]
    output_shape, output_dtype = x["shape"], x["dtype"]
    if key["op"] == "l2norm":
        output_dtype = key["options"]["output_dtype"]
        if output_dtype not in DTYPES:
            raise ValueError("invalid L2norm dtype")
    elif key["op"] == "causal_conv1d":
        weight, bias = key["inputs"][1:]
        if len(x["shape"]) != 3 or len(weight["shape"]) != 2 or weight["shape"][0] != x["shape"][1] or weight["shape"][1] not in (2, 3, 4):
            raise ValueError("invalid convolution shape")
        if bias is not None and bias["shape"] != [x["shape"][1]]:
            raise ValueError("invalid convolution bias")
        if key["options"]["activation"] not in ("none", "silu"):
            raise ValueError("invalid convolution activation")
    else:
        q, k, v, g, beta = key["inputs"]
        if len(q["shape"]) != 4 or len(v["shape"]) != 4 or q["shape"] != k["shape"] or q["shape"][:2] != v["shape"][:2] or v["shape"][2] % q["shape"][2] or g["shape"] != v["shape"][:3] or beta["shape"] != g["shape"]:
            raise ValueError("invalid chunk shapes")
        if q["dtype"] != k["dtype"] or q["dtype"] != v["dtype"] or key["options"]["use_qk_l2norm_in_kernel"] not in ("true", "false"):
            raise ValueError("invalid chunk dtype/options")
        output_shape = v["shape"]
    if key["direction"] == "backward":
        grad = key["grad_output"]
        validate_tensor(grad)
        if not key["grad_enabled"] or grad["requires_grad"] or not any(t and t["requires_grad"] for t in key["inputs"]) or grad["shape"] != output_shape or grad["dtype"] != output_dtype:
            raise ValueError("invalid incoming gradient")
    elif key["grad_output"] is not None:
        raise ValueError("forward key must not contain incoming gradients")
    return json.dumps(key, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_tensor(tensor):
    if not isinstance(tensor, dict) or set(tensor) != {"shape", "strides", "dtype", "requires_grad"}:
        raise ValueError("invalid tensor metadata")
    shape, strides = tensor["shape"], tensor["strides"]
    if not isinstance(shape, list) or not isinstance(strides, list) or not 1 <= len(shape) <= 8 or len(shape) != len(strides) or any(type(n) is not int or n <= 0 for n in shape) or any(type(n) is not int or n < 0 for n in strides):
        raise ValueError("invalid tensor shape/strides")
    if tensor["dtype"] not in DTYPES or type(tensor["requires_grad"]) is not bool:
        raise ValueError("invalid tensor dtype/gradient flag")
    if math.prod(shape) > 2**63 - 1 or 1 + sum((n - 1) * stride for n, stride in zip(shape, strides)) > 2**63 - 1:
        raise ValueError("tensor size/span overflow")


def validate_provenance(source, gpu=None):
    if set(source) != {"gpu", "torch", "torch_revision", "cuda", "triton", "fla", "causal_conv1d", "measurement"}:
        raise ValueError("invalid fused provenance fields")
    if not source["gpu"] or (gpu is not None and source["gpu"] != gpu) or source["torch"].split("+")[0] != "2.9.1" or source["cuda"] != "12.8" or source["torch_revision"] != TORCH_REVISION or any(source[name] != version for name, version in PINS.items()) or source["measurement"] not in ("cupti_kernel_sum", "cuda_event_wall"):
        raise ValueError("incompatible fused timing provenance")


def read_measured(path, source):
    rows = {}
    if not path.exists():
        return rows
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["key", "nanos", "provenance"]:
            raise ValueError("invalid measured CSV header")
        for row in reader:
            key = canonical_key(row["key"])
            provenance = json.loads(row["provenance"])
            validate_provenance(provenance, source["gpu"])
            if provenance != source or not 0 < int(row["nanos"]) <= 2**64 - 1 or key in rows:
                raise ValueError("zero/duplicate/mixed-provenance measured timing")
            rows[key] = int(row["nanos"])
    return rows


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def merge_profile(path, keys, source, measure):
    """One transaction: any failed key preserves the prior measured file bytes."""
    validate_provenance(source)
    rows = read_measured(path, source)
    failures = []
    measured = {}
    for key in sorted({canonical_key(key) for key in keys}):
        if key in rows:
            continue
        try:
            nanos = measure(json.loads(key))
            if type(nanos) is not int or not 0 < nanos <= 2**64 - 1:
                raise ValueError("profiling returned a nonpositive/noninteger duration")
            measured[key] = nanos
        except Exception as error:
            failures.append(dict(key=key, error=f"{type(error).__name__}: {error}"))
    if failures:
        return dict(committed=False, failures=failures, measured=measured, retained_entries=len(rows))
    rows.update(measured)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["key", "nanos", "provenance"])
    for key, nanos in sorted(rows.items()):
        writer.writerow([key, nanos, json.dumps(source, sort_keys=True)])
    atomic_text(path, output.getvalue())
    return dict(committed=True, failures=[], measured=measured, retained_entries=len(rows))
