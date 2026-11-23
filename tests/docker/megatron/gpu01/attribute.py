"""Per-op attribution of real GPU time vs Phantora's charged time (QH-07r method, reusable).

usage: attribute.py REPLAY_DIR TRACE_DIR OUT_CSV [--rank R]
  REPLAY_DIR  a run's replay/ (events.jsonl.gz from --sim-log debug, status.json, worker.log)
  TRACE_DIR   a run's trace-real/traces/ (full-rank-R.json from the trace-real stage)
Joins every real kernel/memcpy/memset to its outermost ATen op (Kineto External id + op nesting)
and groups both sides by op. Phantora's side is two steady replay steps on rank R, averaged.
Prints the real step split (compute busy, NCCL, GPU empty with no launch pending, launch latency)
and writes one CSV row per op group, largest |real - charged| first."""
import collections, csv, gzip, json, re, sys
from pathlib import Path

replay_dir, trace_dir, out_csv = map(Path, sys.argv[1:4])
RANK = int(sys.argv[sys.argv.index("--rank") + 1]) if "--rank" in sys.argv else 0
FUSED = ("ChunkGatedDeltaRule", "L2NormFunction", "CausalConv1dFn")
MAP = {  # Phantora event label -> group
    "MatMul": "matmul/linear", "BMM": "bmm", "BAddBMM": "baddbmm", "ConvDType": "to (dtype conversion)",
    "SoftmaxBackward": "_softmax_backward_data", "MaskedFill": "masked_fill", "MaskedFill_": "masked_fill_",
    "Softmax": "softmax", "Mul": "mul", "MulScalar": "mul", "Mul_": "mul_", "Add": "add", "Add_": "add_",
    "DivScalar": "div", "Div": "div", "Sum": "sum", "Pow": "pow", "Dropout": "dropout",
    "ForeachMulScalar_": "_foreach_mul_", "ForeachAddCDiv_": "_foreach_addcdiv_", "ForeachAddCMul_": "_foreach_addcmul_",
    "cuda:GdnCall": "fused GDN (conv, l2norm, chunk)", "cuda:CudaMemcpyAsync": "device memcpy",
    "QwenOp:exp_out": "exp", "QwenOp:exp": "exp",
    "SDPA": "flash attention forward", "SDPABackward": "flash attention backward",
}
ATEN = {"aten::matmul": "matmul/linear", "aten::mm": "matmul/linear", "aten::linear": "matmul/linear",
        "aten::to": "to (dtype conversion)",
        "aten::scaled_dot_product_attention": "flash attention forward",
        "aten::_scaled_dot_product_flash_attention": "flash attention forward",
        "aten::_scaled_dot_product_flash_attention_backward": "flash attention backward"}


def step_ms():
    vals = []
    for line in open(replay_dir / "worker.log", errors="replace"):
        if "HYBRID_METRICS " in line:
            vals.append(json.JSONDecoder().raw_decode(line.split("HYBRID_METRICS ", 1)[1])[0])
    durs = [max(x["durations"][i] for x in vals) for i in range(1, len(vals[0]["durations"]))]
    return 1000 * sum(durs) / len(durs)


def phantora():
    rows = []
    for line in gzip.open(replay_dir / "events.jsonl.gz", "rt"):
        o = json.loads(line)
        if not isinstance(o, dict):
            rows.append(o)
    step_us = step_ms() * 1000
    nodes = sorted({r[2] for r in rows if r[3] != "N"})
    last = max(r[1] for r in rows if r[3] == "N")
    lo, hi = last - 2.2 * step_us, last - 0.2 * step_us   # two steady steps before the end
    agg = collections.defaultdict(lambda: [0.0, 0.0])
    for t, e, n, k, label, dur, sid in rows:
        if k != "N" and n == nodes[RANK] and lo <= t < hi:
            g = MAP.get(label) or (label[7:] if label.startswith("QwenOp:") else "phantora:" + label)
            agg[g][0] += 0.5; agg[g][1] += (e - t) / 2e3
    return agg, step_us / 1e3


def real():
    ev = json.load(open(trace_dir / f"full-rank-{RANK}.json"))["traceEvents"]
    ops = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation", "python_function") and e.get("ph") == "X"]
    bytid = collections.defaultdict(list)
    for e in ops:
        bytid[e["tid"]].append(e)
    parent, byext = {}, {}
    for lst in bytid.values():
        lst.sort(key=lambda e: (e["ts"], -e["dur"]))
        stack = []
        for e in lst:
            while stack and stack[-1]["ts"] + stack[-1]["dur"] <= e["ts"]:
                stack.pop()
            parent[id(e)] = stack[-1] if stack else None
            stack.append(e)
            x = e["args"].get("External id")
            if x is not None:
                byext[x] = e
    api = {}
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = e["args"].get("correlation")
            if c is not None and (c not in api or "aunch" in e["name"]):
                api[c] = e
    span = [e for e in ev if e.get("cat") == "Trace"][0]
    res = []
    for e in ev:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        op, chain = byext.get(e["args"].get("External id")), []
        while op is not None:
            chain.append(op); op = parent[id(op)]
        aten = [o for o in chain if o["name"].startswith("aten::")]
        a = api.get(e["args"].get("correlation"))
        res.append(dict(cat=e["cat"], k=e["name"], ts=e["ts"], dur=e["dur"], chain=[o["name"] for o in chain],
                        outer=aten[-1]["name"] if aten else None, launch=a["ts"] if a else None))
    res.sort(key=lambda r: r["ts"])
    return res, (span["ts"], span["ts"] + span["dur"])


def split(res, span):
    busy_end = span[0]; host = queued = comp = nccl = 0.0
    for r in res:
        s, en = r["ts"], r["ts"] + r["dur"]
        if s > busy_end:
            if r["launch"] is not None and r["launch"] >= busy_end:
                host += s - busy_end
            else:
                queued += s - busy_end
        if en > busy_end:
            add = en - max(busy_end, s)
            if "nccl" in r["k"]:
                nccl += add
            else:
                comp += add
            busy_end = en
    return dict(traced_step_ms=(span[1] - span[0]) / 1e3, compute_busy_ms=comp / 1e3, nccl_residence_ms=nccl / 1e3,
                gpu_empty_no_launch_pending_ms=host / 1e3, gpu_empty_launch_latency_ms=queued / 1e3)


def rgroup(r):
    if "nccl" in r["k"]:
        return None
    if any(f in n for n in r["chain"] for f in FUSED):
        return "fused GDN (conv, l2norm, chunk)"
    if r["cat"] == "gpu_memcpy":
        return "device memcpy"
    if r["cat"] == "gpu_memset":
        return "device memset"
    if r["outer"]:
        return ATEN.get(r["outer"], r["outer"][6:])
    top = r["chain"][-1] if r["chain"] else "<no op>"
    return "torch.compile region" if "Compiled" in top else "no ATen op: " + top[:60]


P, replay_step = phantora()
res, span = real()
R = collections.defaultdict(lambda: [0, 0.0])
for r in res:
    g = rgroup(r)
    if g is not None:
        R[g][0] += 1; R[g][1] += r["dur"] / 1e3
cat_of = {}
status = json.load(open(replay_dir / "status.json"))
for cat, names in status.get("categories", {}).items():
    for n in names:
        cat_of[n.replace("aten::", "")] = cat
tr, tp = sum(v[1] for v in R.values()), sum(v[1] for v in P.values())
print(json.dumps(dict(rank=RANK, replay_step_ms=round(replay_step, 3), real_compute_kernels_ms=round(tr, 1),
                      phantora_charged_compute_ms=round(tp, 1), **{k: round(v, 1) for k, v in split(res, span).items()})))
groups = sorted(set(R) | set(P), key=lambda g: -abs(R.get(g, [0, 0])[1] - P.get(g, [0, 0])[1]))
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["op_group", "real_gpu_activities", "real_gpu_ms", "phantora_events", "phantora_charged_ms",
                "shortfall_ms", "real_over_charged", "phantora_status"])
    for g in groups:
        rn, rm = R.get(g, [0, 0.0]); pn, pm = P.get(g, [0, 0.0])
        st = cat_of.get(g, "") or ("modeled" if g in P else "")
        w.writerow([g, rn, f"{rm:.3f}", f"{pn:.0f}", f"{pm:.3f}", f"{rm - pm:.3f}", f"{rm / pm:.3f}" if pm else "", st])
        if abs(rm - pm) >= 2.0:
            print(f"{g:42s} real {rm:9.1f} ({rn:6d})  charged {pm:9.1f} ({pn:6.0f})  short {rm - pm:+8.1f}  {st}")
