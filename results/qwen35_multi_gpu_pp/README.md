# Qwen3.5-0.8B on 4 and 8 simulated GPUs from one-GPU profiles (QH-07y, 2026-11-19)

Phantora's claim is that one GPU is enough to predict multi-GPU training. gpu01 has 2 GPUs, so
every earlier result was TP 2 or PP 2. These runs simulate 4 and 8 GPUs with pipeline parallelism,
starting from the timing database already measured on gpu01's Blackwell GPUs, and time any
missing kernel on **GPU 0 only**. Data: [`phantora_multi_gpu_pp.csv`](phantora_multi_gpu_pp.csv).
Earlier results are unchanged.

## Conditions

- Commit `1f070b9` (same code as `2b7f44f`, tracker-only change), gpu01 workflow `run.sh`,
  stages discovery, profile, replay; controls discovery, replay. No real stage (2 GPUs only).
- Base DB `runs/20261119-171647-pp2-even-l4-8192-2b7f44f1/db` (QH-07u `db-layout` + QH-07x TP 1
  keys; copied per run, never modified). Each run's `db/` adds only its own new rows.
- Seq 8192, MB 1, GA 8 (8 microbatches, Megatron 1F1B), 4 iterations, DP 1; step time = mean of
  iterations 2–4 of the slowest rank. Even PP split.
- Why PP only: Qwen3.5-0.8B has 2 KV groups, so TP ≤ 2. DP was hard-coded to 1 in the gpu01
  scripts at the time (`--dp` added afterwards in `0845d55`, not used here).
- **Simulated network assumption:** all 4 or 8 GPUs sit on one host at gpu01's measured
  loopback rate (271.9 Gbit/s, PCIe, no NVLink), the same `netconfig.toml` as all earlier runs.
  No such 8-GPU gpu01 exists; this is a hypothetical machine, not a measured one.
- Status `valid` is false in every run only for the usual unmapped-operation list and the known
  multi-argument `aten::to` form. Every rank completed and matched its stage's expected events.

## Results

| model | TP × PP | GPUs | layers / stage | Phantora replay ms | real ms | replay vs real |
|---|---|---:|---:|---:|---:|---:|
| 4 layers | 1 × 2 (control) | 2 | 2 | 1,380.875 | 1,564.740 | −11.75% |
| 4 layers | 1 × 4 | 4 | 1 | 1,316.566 | — | — |
| 4 layers | 2 × 2 | 4 | 2 | 720.140 | — | — |
| 4 layers | 2 × 4 | 8 | 1 | 678.790 | — | — |
| 24 layers | 2 × 1 (control) | 2 | 24 | 2,651.727 | 3,030.833 | −12.51% |
| 24 layers | 2 × 2 | 4 | 12 | 1,689.128 | — | — |
| 24 layers | 2 × 4 | 8 | 6 | 1,278.660 | — | — |
| 24 layers | 1 × 8 | 8 | 3 | 1,859.894 | — | — |

Both controls reproduce their earlier replays (+0.004 ms and +0.000 ms).

## What one GPU had to add

The only keys the existing DB lacked (beyond the 14 known untimed ConvDType/Dropout keys) were
the Adam optimizer's `_foreach_mul_`, `_foreach_addcmul_` and `_foreach_addcdiv_` on the new
per-stage parameter lists: 12 to 33 rows per run, each run's profiling took 11–12 s on GPU 0.
This is assumption A14 in the tracker (layout changes optimizer tensor-list keys). No layer,
attention, GDN, MLP or communication-shape key was new. Their simulated cost is small: replay
minus discovery (where they were charged 0) is 3.0–6.1 ms per step (0.2–0.5%).

## Observations (inferred, no real reference above 2 GPUs)

- 4-layer TP 1: PP 4 is only 4.7% faster than PP 2. Likely the last stage, which carries the
  248,320-word LM head and loss at seq 8192, sets the pace once a stage holds one layer (not
  verified with per-stage timings).
- 24 layers: TP 2 PP 1 → PP 2 → PP 4 gives 2,651.7 → 1,689.1 → 1,278.7 ms (1.57× and 2.07×
  speed-up on 2× and 4× the GPUs).
- At 8 GPUs, TP 2 × PP 4 (1,278.7 ms) beats TP 1 × PP 8 (1,859.9 ms).
- The 2-GPU controls run 11.8–12.5% below real. Whether that error holds at 4 and 8 GPUs cannot
  be checked on gpu01; it needs a machine with more GPUs of the same model.
