# Qwen3.5-0.8B full sweep with flash attention: real vs Phantora (QH-07ac, 2026-11-22/23)

The first sweep with everything on: flash attention (QH-07ab), the A14 flash-backward profiler fix,
MTP, cat/contiguous handlers with traced layouts, real-run prefetch, and depths between 4 and 24
layers (new `0.8b-l8/l12/l16` presets). Data: [`phantora_vs_real_flash.csv`](phantora_vs_real_flash.csv).
Earlier unfused sweeps are in `../qwen35_08b_mtp_sweep/` and `../qwen35_08b_sequence_sweep/`.

**A15 correction (2026-11-23).** Some of the profile sessions behind these points recorded only 4 of
5 iterations, so every timing they produced was 0.80x (A15 in
[`../phantora_assumptions_vs_reality.md`](../phantora_assumptions_vs_reality.md)). Every key those
sessions measured was re-timed with the fixed profiler, the affected points were replayed from DB
copies with only those timings replaced, and their Phantora numbers below were **overwritten** (a
one-time exception to the never-overwrite rule, decided by the user because it is one wrong
measurement in otherwise sound data). The CSV keeps each original value in `a15_original_replay_ms`
and the correcting run in `a15_corrected_run_dir`; the original runs stay on gpu01. Marked `*` below.
Real numbers are unaffected (real runs never use the profiler).

## Conditions

- Commit `06baf8d` via `gpu01/run.sh`, batch `gpu01:~/Phantora/runs/batch-flash-sweep-20261122.sh`,
  2026-11-22 23:19 to 2026-11-23 01:17. Stages discovery, profile, replay, real at every point.
  32k at 8/12/16 layers: commit `b18e228` (same model code), 2026-11-23 12:19–12:31, real stages;
  their Phantora numbers come from the A15-corrected replays (commit `17c6542`).
- MTP 1 layer, flash attention (`FlashCoreAttention`), BF16, synthetic data, MB 1, GA 8, 4 iterations;
  step = mean of iterations 2–4 of the slower rank. TP 2, DP 1 unless noted.
- `expandable_segments` on in every real run, 24 layers included (it was off for 24 layers before).
  At 24L seq 8192 this changed real by +0.2% (2,645.7 → 2,651.8 ms) and peak by −0.9 GiB.
- Base DB for every point: the QH-07ab fixed-profiler flash run's `db` (24L, seq 8192), copied per
  run; each run profiled its own missing keys (32k at 8/12/16 layers reused the 4-layer 32k run's).
  Replay `valid` is false only for the 14 known untimed keys (13 at DP 2).
- Layer counts 8/12/16 are reduced-depth fixtures: same embedding, final norm, tied LM head, loss and
  MTP layer as the released model, fewer 3 GDN + 1 attention groups. Not released checkpoints.

## Phantora vs real (TP 2)

| layers | 1,024 | 4,096 | 8,192 | 12,288 | 16,384 | 24,576 | 32,768 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | −42.0%* | −19.4%* | −17.4% | — | −16.5% | −11.5% | −10.2%* |
| 8 | — | — | −16.6% | — | −15.6% | −9.6% | −8.3%* |
| 12 | — | — | −16.1% | — | −15.1% | −8.6% | −7.4%* |
| 16 | — | — | −16.2% | — | −14.7%* | −8.4% | −7.6%*† |
| 24 | −65.3%* | −23.0%* | −16.3% | −16.5% | −15.5% | −7.5%* | OOM‡ |

TP 1 DP 2, 4 layers, seq 8192: Phantora 2,136.0 ms vs real 2,545.2 ms, −16.1%.

\* A15-corrected. † Phantora runs out of simulated memory at the real 97,887 MiB (real fits at
89.4 GiB, so A13); replayed with 131,072 MiB, and 6 point-specific keys were never profiled (charged
0), so slightly low. ‡ 24 layers at 32k not run; estimated ~108 GiB.

## Real peak memory (GiB, device level)

| layers | 1,024 | 4,096 | 8,192 | 12,288 | 16,384 | 24,576 | 32,768 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 10.1 | 15.8 | 22.2 | — | 36.3 | 50.3 | 64.4 |
| 8 | — | — | 24.7 | — | 40.5 | 56.4 | 72.4 |
| 12 | — | — | 27.2 | — | 45.0 | 62.7 | 80.5 |
| 16 | — | — | 30.0 | — | 49.7 | 69.6 | 89.4 |
| 24 | 14.3 | 23.3 | 35.4 | 47.3 | 59.3 | 83.3 | — |

With flash, 24 layers fit at 24k (83 GiB); unfused, 24 layers ran out of memory at 16k.

## All points

| layers | seq | Phantora ms | real ms | diff | |
|---:|---:|---:|---:|---:|---|
| 24 | 1,024 | 325.8 | 939.3 | −65.3% | A15-corrected (was 288.4) |
| 24 | 4,096 | 1,114.8 | 1,447.6 | −23.0% | A15-corrected (was 990.9) |
| 24 | 8,192 | 2,220.7 | 2,651.8 | −16.3% | |
| 24 | 12,288 | 3,467.0 | 4,151.9 | −16.5% | |
| 24 | 16,384 | 4,829.6 | 5,716.6 | −15.5% | |
| 24 | 24,576 | 8,703.2 | 9,405.5 | −7.5% | A15-corrected (was 7,925.1) |
| 16 | 8,192 | 1,787.1 | 2,132.3 | −16.2% | |
| 16 | 16,384 | 3,881.4 | 4,549.9 | −14.7% | A15-corrected (was 3,553.4) |
| 16 | 24,576 | 6,756.2 | 7,372.6 | −8.4% | |
| 16 | 32,768 | 9,632.6 | 10,423.3 | −7.6% | 128 GiB simulated memory, 6 keys uncharged |
| 12 | 8,192 | 1,570.3 | 1,872.5 | −16.1% | |
| 12 | 16,384 | 3,362.9 | 3,959.9 | −15.1% | |
| 12 | 24,576 | 5,826.6 | 6,371.2 | −8.6% | |
| 12 | 32,768 | 8,274.5 | 8,938.2 | −7.4% | A15-corrected DB (uncorrected 7,600.1) |
| 8 | 8,192 | 1,353.5 | 1,622.4 | −16.6% | |
| 8 | 16,384 | 2,872.3 | 3,401.9 | −15.6% | |
| 8 | 24,576 | 4,895.9 | 5,415.2 | −9.6% | |
| 8 | 32,768 | 6,912.7 | 7,535.5 | −8.3% | A15-corrected DB (uncorrected 6,359.2) |
| 4 | 1,024 | 162.4 | 279.9 | −42.0% | A15-corrected (was 144.6) |
| 4 | 4,096 | 576.1 | 714.3 | −19.4% | A15-corrected (was 508.1) |
| 4 | 8,192 | 1,136.7 | 1,375.5 | −17.4% | |
| 4 | 16,384 | 2,384.5 | 2,856.1 | −16.5% | |
| 4 | 24,576 | 3,960.8 | 4,477.0 | −11.5% | |
| 4 | 32,768 | 5,550.9 | 6,180.7 | −10.2% | A15-corrected (was 5,118.2) |

## Observations

Measured:
- Phantora is below real at every point, and the error hardly depends on depth: −16.1% to −17.4% at
  8k for all five depths, −14.7% to −16.5% at 16k, −7.5% to −11.5% at 24k, −7.4% to −10.2% at 32k.
- The error shrinks with sequence length, mostly between 16k and 24k.
- Short sequences (≤ 4k) are far off: with flash the GPU work per step is small enough that host time
  sets the pace, which Phantora ignores (A1).
- Phase timers (QH-07ad, `../qwen35_flash_phases/`) put nearly all of the gap in forward and backward,
  split about evenly between them, at every point measured.

Not yet explained:
- What the remaining ~16% at 8–16k is and why it drops to ~8% at 24–32k. Candidates from the unfused
  attribution (QH-07r): unmapped ops (A5, e.g. cross-entropy `sub_`/`max`), `aten::to` forms (A6),
  same-shape binary ops profiled fast (A4), host gaps (A1/A2). A traced real run (trace-real) would
  measure it per op.
