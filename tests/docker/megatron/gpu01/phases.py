"""Collect Megatron phase timers (HYBRID_PHASES lines) from a phases pass.

usage: phases.py [--ranks N] LOG...   (N = expected ranks, default 2)
Prints JSON: {"ranks": {rank: {phase: ms_per_step}}, "max_rank": {phase: max over ranks},
"step_ms": mean over steps 1..N-1 of the slowest rank (with timers on)} or {} unless all N
reported. Timers can nest: forward-backward contains forward-compute, backward-compute and
all-grads-sync; optimizer-* run after it."""
import json, sys
args = sys.argv[1:]
want = 2
if args[:1] == ["--ranks"]:
    want, args = int(args[1]), args[2:]
rows = []
for path in args:
    try:
        for line in open(path, errors="replace"):
            if "HYBRID_PHASES " in line:
                rows.append(json.JSONDecoder().raw_decode(line.split("HYBRID_PHASES ", 1)[1])[0])
    except OSError:
        pass
ranks = {x["rank"]: x for x in rows}
if len(ranks) != want:
    print("{}")
    sys.exit(0)
names = sorted({name for x in ranks.values() for name in x["ms_per_step"]})
n = len(ranks[0]["durations"])
steps = [max(x["durations"][i] for x in ranks.values()) for i in range(1, n)]
print(json.dumps(dict(
    ranks={r: {k: round(v, 3) for k, v in x["ms_per_step"].items()} for r, x in sorted(ranks.items())},
    max_rank={k: round(max(x["ms_per_step"].get(k, 0.0) for x in ranks.values()), 3) for k in names},
    step_ms=round(1000 * sum(steps) / len(steps), 3),
)))
