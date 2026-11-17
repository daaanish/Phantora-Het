"""Print mean over steps 1..N-1 of the slower rank's duration (ms), or blank."""
import json, sys
m = []
for path in sys.argv[1:]:
    try:
        for line in open(path, errors="replace"):
            if "HYBRID_METRICS " in line:
                m.append(json.loads(line.split("HYBRID_METRICS ", 1)[1]))
    except OSError:
        pass
ranks = {x["rank"]: x for x in m}
if len(ranks) == 2:
    n = len(ranks[0]["durations"])
    it = [max(x["durations"][i] for x in ranks.values()) for i in range(1, n)]
    print(f"{1000 * sum(it) / len(it):.3f}")
else:
    print("")
