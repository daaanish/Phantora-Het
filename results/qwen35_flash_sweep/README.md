# Qwen3.5-0.8B full sweep with flash attention: real vs Phantora (QH-07ac, 2026-11-22/23)

The first sweep with everything on: flash attention (QH-07ab), the A14 flash-backward profiler fix,
MTP, cat/contiguous handlers with traced layouts, real-run prefetch, and depths between 4 and 24
layers (new `0.8b-l8/l12/l16` presets). Data: [`phantora_vs_real_flash.csv`](phantora_vs_real_flash.csv).
Earlier unfused sweeps are unchanged in `../qwen35_08b_mtp_sweep/` and `../qwen35_08b_sequence_sweep/`.

## Conditions

- Commit `06baf8d` via `gpu01/run.sh`, batch `gpu01:~/Phantora/runs/batch-flash-sweep-20261122.sh`,
  2026-11-22 23:19 to 2026-11-23 01:17. Stages discovery, profile, replay, real at every point.
- MTP 1 layer, flash attention (`FlashCoreAttention`), BF16, synthetic data, MB 1, GA 8, 4 iterations;
  step = mean of iterations 2–4 of the slower rank. TP 2, DP 1 unless noted.
- `expandable_segments` on in every real run, 24 layers included (it was off for 24 layers before).
  At 24L seq 8192 this changed real by +0.2% (2,645.7 → 2,651.8 ms) and peak by −0.9 GiB.
- One base DB for every point: the QH-07ab fixed-profiler flash run's `db` (24L, seq 8192), copied
  per run; each run profiled its own missing keys. Replay `valid` is false only for the 14 known
  untimed keys (13 at DP 2).
- Layer counts 8/12/16 are reduced-depth fixtures with the released shapes, not released checkpoints.

## Phantora vs real (TP 2)

| layers | 1,024 | 4,096 | 8,192 | 12,288 | 16,384 | 24,576 | 32,768 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | −48.4% | −28.9% | −17.4% | — | −16.5% | −11.5% | −17.2% |
| 8 | — | — | −16.6% | — | −15.6% | −9.6% | — |
| 12 | — | — | −16.1% | — | −15.1% | −8.6% | — |
| 16 | — | — | −16.2% | — | −21.9% | −8.4% | — |
| 24 | −69.3% | −31.6% | −16.3% | −16.5% | −15.5% | −15.7% | — |

TP 1 DP 2, 4 layers, seq 8192: Phantora 2,136.0 ms vs real 2,545.2 ms, −16.1%.

## Real peak memory (GiB, device level)

| layers | 1,024 | 4,096 | 8,192 | 12,288 | 16,384 | 24,576 | 32,768 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 10.1 | 15.8 | 22.2 | — | 36.3 | 50.3 | 64.4 |
| 8 | — | — | 24.7 | — | 40.5 | 56.4 | — |
| 12 | — | — | 27.2 | — | 45.0 | 62.7 | — |
| 16 | — | — | 30.0 | — | 49.7 | 69.6 | — |
| 24 | 14.3 | 23.3 | 35.4 | 47.3 | 59.3 | 83.3 | — |

With flash every point fits, including 24 layers at 24k (83 GiB); unfused, 24 layers ran out of
memory at 16k.

## All points

| layers | seq | Phantora ms | real ms | diff |
|---:|---:|---:|---:|---:|
| 24 | 1,024 | 288.4 | 939.3 | −69.3% |
| 24 | 4,096 | 990.9 | 1,447.6 | −31.6% |
| 24 | 8,192 | 2,220.7 | 2,651.8 | −16.3% |
| 24 | 12,288 | 3,467.0 | 4,151.9 | −16.5% |
| 24 | 16,384 | 4,829.6 | 5,716.6 | −15.5% |
| 24 | 24,576 | 7,925.1 | 9,405.5 | −15.7% |
| 16 | 8,192 | 1,787.1 | 2,132.3 | −16.2% |
| 16 | 16,384 | 3,553.4 | 4,549.9 | −21.9% |
| 16 | 24,576 | 6,756.2 | 7,372.6 | −8.4% |
| 12 | 8,192 | 1,570.3 | 1,872.5 | −16.1% |
| 12 | 16,384 | 3,362.9 | 3,959.9 | −15.1% |
| 12 | 24,576 | 5,826.6 | 6,371.2 | −8.6% |
| 8 | 8,192 | 1,353.5 | 1,622.4 | −16.6% |
| 8 | 16,384 | 2,872.3 | 3,401.9 | −15.6% |
| 8 | 24,576 | 4,895.9 | 5,415.2 | −9.6% |
| 4 | 1,024 | 144.6 | 279.9 | −48.4% |
| 4 | 4,096 | 508.1 | 714.3 | −28.9% |
| 4 | 8,192 | 1,136.7 | 1,375.5 | −17.4% |
| 4 | 16,384 | 2,384.5 | 2,856.1 | −16.5% |
| 4 | 24,576 | 3,960.8 | 4,477.0 | −11.5% |
| 4 | 32,768 | 5,118.2 | 6,180.7 | −17.2% |

## Observations

Measured:
- Phantora is below real at every point. From seq 8,192 up the error is mostly −15% to −17%,
  nearly independent of depth: 16.1–17.4% at 8,192 for all five depths.
- Short sequences are far off (−29% to −69% at ≤ 4,096): with flash the GPU work per step is small
  enough that host time sets the pace, which Phantora ignores (A1). With flash this region extends
  to longer sequences than unfused (24L 4,096: −31.6% flash vs −17.7% unfused).
- Seq 24,576 at 8/12/16 layers is closer (−8.4% to −9.6%), but 4 layers (−11.5%) and 24 layers
  (−15.7%) at the same length are not, and 4 layers at 32,768 is −17.2%. 16 layers at 16,384
  (−21.9%) stands out from its row. Each point is a single real run.

Not yet explained (to check, not assumed):
- Why 24,576 is closer at 8–16 layers only, and the 16L 16,384 outlier. Repeating those real runs
  would show whether it is run-to-run variation; phase passes would show which phase moves.
- What the remaining ~16% is. Earlier attribution (QH-07r, unfused, seq 8192) pointed at host gaps
  between kernels, unmapped ops (A5) and the multi-argument `aten::to` (A6).
