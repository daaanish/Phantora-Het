"""Summarize a clocks.csv written by ph_gpu_log (paths.sh) over busy samples (utilization > 0).

usage: clocks.py CLOCKS_CSV
Prints JSON per GPU index: busy sample count, median / min SM clock (MHz), median memory clock,
median and max power (W), max temperature (C), and the throttle-reason bitmasks seen with counts
(e.g. 0x4 SW power cap, 0x20 SW thermal, 0x40 HW thermal, 0x80 HW power brake, 0x400 idle-ish)."""
import collections, csv, json, statistics, sys

rows = collections.defaultdict(list)
with open(sys.argv[1]) as f:
    for r in csv.reader(f):
        if not r or r[0].strip() == "timestamp" or len(r) < 8:
            continue
        try:
            ts, idx, sm, mem, power, temp, reasons, util = (x.strip() for x in r[:8])
            if float(util) > 0:
                rows[idx].append((float(sm), float(mem), float(power), float(temp), reasons))
        except ValueError:
            continue
out = {}
for idx, v in sorted(rows.items()):
    sm, mem, pw, tp = ([x[i] for x in v] for i in range(4))
    out[idx] = dict(busy_samples=len(v), sm_mhz_median=statistics.median(sm), sm_mhz_min=min(sm),
                    mem_mhz_median=statistics.median(mem), power_w_median=round(statistics.median(pw), 1),
                    power_w_max=round(max(pw), 1), temp_c_max=max(tp),
                    throttle_reasons=dict(collections.Counter(x[4] for x in v).most_common()))
print(json.dumps(out, indent=1))
