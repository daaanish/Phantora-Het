"""Append newly profiled timings to a run's perf DB. Usage: merge.py <run-dir>

Reads <run>/profile/measured*/{compute,memcpy}.csv and appends to <run>/db only rows with
a positive time whose key the DB does not already have; zero/negative rows are counted,
never added. Same rule as the QH-07q sweep merge. Writes <run>/profile/merge.json."""
import csv, json, sys
from pathlib import Path

run = Path(sys.argv[1]); db = run / "db"; prof = run / "profile"
out = {}
for name in ("compute.csv", "memcpy.csv"):
    with (db / name).open(newline="") as f:
        r = csv.reader(f); header = next(r); old = list(r)
    new = []
    for src in ("measured", "measured-exp"):
        path = prof / src / name
        if path.exists():
            with path.open(newline="") as f:
                r = csv.reader(f)
                if next(r, None) == header:
                    new += list(r)
    have = {tuple(x[:-1]) for x in old}
    pos = [x for x in new if int(x[-1]) > 0 and tuple(x[:-1]) not in have]
    nonpos = [x for x in new if int(x[-1]) <= 0]
    with (db / name).open("a", newline="") as f:
        csv.writer(f).writerows(pos)
    out[name] = dict(prior=len(old), profiled=len(new), added=len(pos), nonpositive=len(nonpos),
                     nonpositive_keys=[x[0] for x in nonpos])
g = db / "gdn.profile.json"
out["gdn_committed"] = json.loads(g.read_text()).get("committed") if g.exists() else None
(prof / "merge.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "nonpositive_keys"}) for k, v in out.items()}))
