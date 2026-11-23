# Qwen3.5 hybrid with data parallelism (QH-07z, 2026-11-22)

First DP > 1 runs of the hybrid workload, real vs Phantora, with `run.sh --dp` (added in
`0845d55`; runs on commit `feb7835`). Data: [`phantora_vs_real_dp.csv`](phantora_vs_real_dp.csv).
Earlier results are unchanged.

## Conditions

- gpu01, RTX PRO 6000 Blackwell Max-Q. `0.8b-l4` preset, seq 8192, MB 1, 8 microbatches per
  replica (global batch = 8 × DP), TP 1, PP 1, 4 iterations; step time = mean of iterations 2–4
  of the slowest rank. Real runs with expandable segments (preset default).
- One change between the two rows: DP 1 (1 GPU) → DP 2 (2 GPUs). Each GPU does the same
  8 microbatches; DP 2 adds the gradient reduce-scatter and parameter all-gather of Megatron's
  distributed optimizer, and halves optimizer state per GPU.
- Stages discovery, profile (GPU 0 only), replay, real in one run each. Base DB
  `runs/20261119-175512-mg-l4-tp1pp4-1f070b99/db` (QH-07y; copied, never modified).
- Real DP 2 confirmed in both ranks' `HYBRID_CONFIG` (DP 2, world 2, global batch 16,
  accumulation 8) and `HYBRID_METRICS` (131,072 tokens per iteration).
- Status `valid` false only for the usual unmapped-operation list and the known
  multi-argument `aten::to`; every rank's events match the coverage plan.
- Network: same single-host `netconfig.toml` (271.9 Gbit/s measured loopback) as all runs.

## Results

| DP | GPUs | real ms | Phantora replay ms | replay vs real |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1,726.312 | 1,502.404 | −12.97% |
| 2 | 2 | 1,739.056 | 1,517.678 | −12.73% |
| DP 2 − DP 1 | | +12.744 (+0.74%) | +15.274 (+1.02%) | Phantora predicts 119.9% of the change |

Real peak memory: 35,982 MiB at DP 1; 31,628 / 31,630 MiB at DP 2 (optimizer state sharded).

## What one GPU had to add for DP

DP 1 needed 7 new compute keys (Adam `_foreach_*` on the TP 1 full-model parameter lists and one
`Add`). DP 2 needed 20 new compute keys and 4 new memcpy keys, all timed on GPU 0:
`_foreach_*` on the halved optimizer shard, `zeros_like` (f32) and `mul_` (bf16, gradient
scaling), 4 extra bf16 dtype-conversion shapes, and 4 device-to-device copies (0.47 MB to
294 MB) of the distributed optimizer's buffers.

## Observations

- The error stays the same with and without DP (−13.0% vs −12.7%), as it did for PP 2 (−11.8%).
- Absolute DP cost is small at this size (13 ms real, 15 ms predicted, about 1% of the step), so
  the 120% ratio is a 2.5 ms difference; it is not a strong test of the communication model. A
  larger gradient volume (24 layers) or larger DP would test it better.

## Rerun with every key profiled fresh (QH-07z-fresh, 2026-11-23)

**Why:** A15 (some profile sessions recorded 4 of 5 iterations) was checked by re-timing the
2026-11-22/23 sessions. Both QH-07z sessions passed (median ratio 0.998 over 7 keys, 1.002 over
24 keys; `gpu01:~/Phantora/reprofile-20261123/sessions.txt`), but most keys in their replays came
from the base DB built from earlier sessions (QH-07m/u, before the commit-pinned workflow), which
the re-timing tool cannot read. To remove the doubt, both points were rerun from an **empty** DB.

**What changed (three things at once, so the shift is not attributable to any one):**
1. Base DB: empty (`runs/empty-db-20261123`, headers and manifest only). Every key, including the
   fused GDN keys, profiled on GPU 0 in one session per point with the A15-fixed profiler (c98260c).
2. Code: commit `d5376d2`, which uses flash attention (SDPA) in the full-attention layers (QH-07ab).
   `--mtp 0` as in QH-07z (MTP became the default on 2026-11-22).
3. Real runs repeated on that commit; gpu01 driver now reports 620.70.06 (570.195.03 at QH-02;
   the QH-07z runs' driver was not recorded).

Everything else as above (0.8b-l4, seq 8192, MB 1, GA 8 per replica, TP 1, 4 iterations).

| DP | GPUs | real ms | Phantora replay ms | replay vs real |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1,404.648 | 1,178.618 | −16.09% |
| 2 | 2 | 1,404.534 | 1,192.515 | −15.10% |
| DP 2 − DP 1 | | −0.114 | +13.897 | — |

- Profiling from nothing on one GPU: 226 compute + 20 memcpy keys (DP 1) and 240 + 24 (DP 2),
  about 2 minutes each; 13 non-positive keys remain missing (14 before; one fewer under flash).
  Each session logged 5 short captures (17–19 of 20 iterations), which the fixed profiler
  averaged over the captured iterations instead of under-reporting.
- Coverage and real configuration checked as above (real DP 2: both ranks DP 2, global batch 16).
- Real peak memory: 29,122 MiB (DP 1), 23,962 / 23,964 MiB (DP 2), lower than QH-07z (flash has
  no mask tensor).

**Observations.**
- The error at this size is now −15 to −16% (was −13%). Flash attention, the fresh DB and the
  driver changed together, so this run cannot say which moved it.
- Real DP cost is now ~0 ms (−0.1), against +12.7 ms in QH-07z; Phantora predicts +13.9 ms in
  both conditions. Within one real run the steady iterations already spread by ~13 ms
  (1,396–1,409 ms at DP 1), so DP cost at this size is inside run-to-run noise in both conditions.
  Whether the real gradient reduce-scatter is hidden behind backward compute (and Phantora charges
  it) needs a larger gradient volume, e.g. 24 layers at DP 2, or the phase timers (`--phases`).
