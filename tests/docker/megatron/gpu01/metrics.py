"""Print mean over steps 1..N-1 of the slowest rank's duration (ms), or blank.

usage: metrics.py [--ranks N] LOG...   (N = expected ranks, default 2; blank unless all N reported)"""
import json, sys
args = sys.argv[1:]
want = 2
if args[:1] == ["--ranks"]:
    want, args = int(args[1]), args[2:]
m = []
for path in args:
    try:
        for line in open(path, errors="replace"):
            if "HYBRID_METRICS " in line:
                m.append(json.loads(line.split("HYBRID_METRICS ", 1)[1]))
    except OSError:
        pass
ranks = {x["rank"]: x for x in m}
if len(ranks) == want:
    n = len(ranks[0]["durations"])
    it = [max(x["durations"][i] for x in ranks.values()) for i in range(1, n)]
    print(f"{1000 * sum(it) / len(it):.3f}")
else:
    print("")
