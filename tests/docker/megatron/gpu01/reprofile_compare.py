"""Helper for reprofile_sessions.sh (A15).

collect OUT RUNDIR...   union of keys every run's profile session measured -> OUT/keys, OUT/keys-exp
compare OUT RUNDIR...   per session: median of (session timing / fixed re-timing) over its bench.py keys;
                        plus old-vs-new bench.py agreement in the reference session. Writes OUT/sessions.json.
A session at ~0.80 recorded 4 of 5 iterations (A15); ~1.00 is fine."""
import csv, glob, json, statistics, sys
from pathlib import Path

mode, out = sys.argv[1], Path(sys.argv[2])
runs = sorted({p for g in sys.argv[3:] for p in glob.glob(g)} if any("*" in g for g in sys.argv[3:]) else set(sys.argv[3:]))
runs = [Path(r) for r in runs if (Path(r) / "profile" / "measured" / "compute.csv").exists()]


def table(path):
    """{key tuple: nanos} and header of a DB-style CSV (nanos is the last column)."""
    if not path.exists():
        return [], {}
    rows = list(csv.reader(open(path)))
    return rows[0], {tuple(r[:-1]): int(r[-1]) for r in rows[1:] if r and r[-1].lstrip("-").isdigit()}


def session(run):
    m = {}
    for sub, kind in (("measured", "bench"), ("measured-exp", "exp")):
        for name in ("compute.csv", "memcpy.csv"):
            h, t = table(run / "profile" / sub / name)
            for k, v in t.items():
                m[(name, k)] = (v, kind, h)
    return m


if mode == "collect":
    keys = {}
    for run in runs:
        for (name, k), (v, kind, h) in session(run).items():
            keys.setdefault((kind, name), (h, {}))[1][k] = v
    for kind, sub in (("bench", "keys"), ("exp", "keys-exp")):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for name in ("compute.csv", "memcpy.csv"):
            h, t = keys.get((kind, name), (None, {}))
            if h is None:
                h = ["key", "nanos"] if name == "compute.csv" else ["kind", "size_bytes", "nanos"]
            with open(out / sub / name, "w", newline="") as f:
                w = csv.writer(f); w.writerow(h); w.writerows([*k, v] for k, v in sorted(t.items()))
    n = {f"{kind}/{name}": len(v[1]) for (kind, name), v in keys.items()}
    print(json.dumps(dict(sessions=len(runs), unique_keys=n)))
    sys.exit(0)

ref = {}
for v in ("new", "old"):
    for sub in (v, v + "-exp"):
        for name in ("compute.csv", "memcpy.csv"):
            for k, x in table(out / sub / name)[1].items():
                ref.setdefault(v, {})[(name, k)] = x
new, old = ref.get("new", {}), ref.get("old", {})
both = [k for k in new if k in old and new[k] > 0 and old[k] > 0]
oldnew = statistics.median(old[k] / new[k] for k in both) if both else None
res = dict(reference=str(out), old_over_new_median=round(oldnew, 4) if oldnew else None, keys_compared_old_new=len(both),
           new_short_captures=sum("warning: profiler captured" in l for f in out.glob("new*.log") for l in open(f)),
           sessions={})
for run in runs:
    ratios = [v / new[(name, k)] for (name, k), (v, kind, h) in session(run).items()
              if kind == "bench" and new.get((name, k), 0) > 0 and v > 0 and "SDPABackward" not in k[0]]
    if not ratios:
        continue
    med = statistics.median(ratios)
    res["sessions"][run.name] = dict(keys=len(ratios), median_ratio=round(med, 3),
                                     p10=round(sorted(ratios)[len(ratios) // 10], 3), p90=round(sorted(ratios)[9 * len(ratios) // 10], 3),
                                     verdict="BAD (short capture, A15)" if med < 0.9 else ("ok" if med < 1.1 else "CHECK (slow)"))
(out / "sessions.json").write_text(json.dumps(res, indent=1))
print(f"old/new bench.py on the same keys in this session: median {res['old_over_new_median']} over {len(both)} keys; "
      f"short captures in new: {res['new_short_captures']}")
for name, s in res["sessions"].items():
    print(f"{s['median_ratio']:6.3f} (p10 {s['p10']}, p90 {s['p90']}, {s['keys']:4d} keys)  {s['verdict']:26s} {name}")
