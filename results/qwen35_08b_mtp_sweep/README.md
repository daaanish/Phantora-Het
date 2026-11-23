# Qwen3.5-0.8B with MTP, every fix on: real vs Phantora (QH-07aa, 2026-11-22)

The first sweep of the full target model (24 decoder layers + its one MTP layer) with every
simulator and harness fix made so far. Data: [`phantora_vs_real_mtp.csv`](phantora_vs_real_mtp.csv).
Earlier sweeps without MTP are unchanged in `../qwen35_08b_sequence_sweep/`.

## Conditions

- Commit `feb78357` via `tests/docker/megatron/gpu01/run.sh`, batch script
  `gpu01:~/Phantora/runs/batch-mtp-sweep-20261122.sh`. MTP passed explicitly (`--mtp 1`);
  the later `run.sh` default change did not affect these runs.
- Attention: Megatron's local **unfused** full attention (s × s scores), as in every run before
  QH-07ab (flash attention from 2026-11-22 on). Label these numbers "unfused" when comparing.
- Fixes in force: `cat`/`contiguous` handlers with traced strides (QH-07u), real-run batch
  prefetch (A11), MTP layer (QH-07w). Replay ignores CPU time (A1), as in every earlier run.
- BF16, synthetic data, MB 1, GA 8, 4 iterations; step time = mean of iterations 2–4 of the
  slower rank. Real runs on gpu01's 2× RTX PRO 6000 Blackwell; profiling on GPU 0.
  `expandable_segments` on for `0.8b-l4` real runs, off for `0.8b` (as in QH-07).
- Base DBs (copied per run, never changed): 24L from QH-07w `mtp-8192`, 4L TP2 from QH-07w
  `mtp-l4-24576`, TP1 from today's `dp2-l4-tp1`. Each run profiles the keys it lacks.
- `valid` is false in every run only for the 14 known untimed ConvDType/Dropout keys plus
  (with TP2) the `eh_proj` no-op cast.

## Results

| layers | parallelism | seq | Phantora ms | real ms | Phantora vs real | real peak GiB | note |
|---:|---|---:|---:|---:|---:|---:|---|
| 24 | TP2 | 1,024 | 328.4 | 962.9 | −65.9% | 15.0 | host-bound |
| 24 | TP2 | 4,096 | 1,346.8 | 1,636.0 | −17.7% | 27.5 | |
| 24 | TP2 | 8,192 | 3,300.1 | 3,776.0 | −12.6% | 49.0 | control |
| 24 | TP2 | 12,288 | 5,353.2 | 6,729.0 | −20.4% | 78.9 | see below |
| 24 | TP2 | 16,384 | OOM | OOM | — | — | QH-07w |
| 4 | TP2 | 1,024 | 162.8 | 285.2 | −42.9% | 10.1 | host-bound |
| 4 | TP2 | 4,096 | 640.3 | 798.6 | −19.8% | 16.4 | |
| 4 | TP2 | 8,192 | 1,452.2 | 1,702.1 | −14.7% | 27.3 | |
| 4 | TP2 | 16,384 | 3,714.1 | 4,196.5 | −11.5% | 50.9 | |
| 4 | TP2 | 24,576 | 7,054.1 | 7,547.8 | −6.5% | 90.4 | control |
| 4 | TP1 PP2 even | 8,192 | failed | failed | — | — | Megatron assertion |
| 4 | TP1 PP2 split 1,3 | 8,192 | failed | failed | — | — | same |
| 4 | TP1 DP2 | 8,192 | 2,773.0 | 3,185.4 | −12.9% | 47.5 | |

**Controls reproduce.** Replay matches QH-07w exactly (3,300.120 and 7,054.083 ms). Real moved
−0.5% (3,795.7 → 3,776.0) and +0.1% (7,539.3 → 7,547.8), run-to-run variation.

## Observations

- Phantora is below real at every point. Error falls as the GPU dominates: 4L goes
  −19.8 → −14.7 → −11.5 → −6.5% from seq 4096 to 24576, the same trend as without MTP.
- Host-bound sizes (seq 1024) remain far off, as expected with CPU time ignored (A1).
- **24L seq 12288 (−20.4%) breaks the trend** (8192 gives −12.6%; no-MTP 24L 16384 gave
  −9.3%). Not investigated. One candidate (unverified): this 24L real run had
  `expandable_segments` off at 79 GiB peak, so allocator retries may inflate real time.
  A rerun with `--expandable` would test it.
- DP2 error (−12.9%) is in line with TP2 at the same size; the no-MTP DP2 run (QH-07z, `../qwen35_data_parallel/`) gave −12.7%.

## Failure: MTP with pipeline parallelism

Both PP2 runs fail on rank 1 (the last stage, which holds the MTP layer and the tied output
weight) at the reset after iteration 1:
`megatron/core/distributed/param_and_grad_buffer.py:206 assert len(self.per_param_grad_ready_counts) == len(self.params)`,
i.e. a parameter in that bucket group got no gradient in the first iteration. Real and
simulation fail identically, so it is a model/harness issue under PP + MTP, not a simulator
error. Cause not yet found; PP results with MTP are missing until it is fixed.
