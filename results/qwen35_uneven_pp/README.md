# Qwen3.5-0.8B with uneven pipeline splits, MTP on (QH-07af, 2026-11-23)

Does Phantora predict how the split of decoder layers across pipeline stages changes the training
step time? Real vs Phantora on gpu01's 2 GPUs (TP 1, PP 2), plus two 8-GPU predictions at 32k.
Data: [`phantora_vs_real_uneven_pp.csv`](phantora_vs_real_uneven_pp.csv). Earlier MTP-off result
(QH-07x, 4 layers, split 1,3): `../qwen35_pipeline_parallel/`, unchanged.

## What a split means

"Split a,b" = a decoder layers on stage 0 (GPU 0), b on the last stage (GPU 1). Fixed pieces:

- Stage 0 also holds the token embedding (248,320 × 1,024).
- The last stage also holds the final norm, the LM head (tied to the embedding; Megatron keeps a
  copy on the last stage and all-reduces its gradient with stage 0 every step), the loss, and the
  MTP layer (gated full attention + SwiGLU).

Qwen3.5's 3:1 pattern puts full attention (FA) at layers 3, 7, 11; the rest are Gated DeltaNet
(GDN). Per-stage layer lists in the CSV; the 12-layer ones were checked against each real run's
`HYBRID_CONFIG` (per-rank layer list).

## Conditions

- gpu01, 2× RTX PRO 6000 Blackwell Max-Q. Commit `d464f13` (same code as `b8bd065`, the MTP + PP
  fix; tracker-only changes), gpu01 workflow `run.sh`, stages discovery, profile, replay, real.
- Seq 8192, MB 1, GA 8 (8 microbatches, Megatron 1F1B), 4 iterations, TP 1 × DP 1 × PP 2, MTP 1,
  flash attention, simulated VRAM 97,887 MiB. Real runs with expandable segments.
- **Step time** = one training iteration (8 microbatches forward and backward through both
  stages, gradient sync, optimizer), mean of iterations 2–4 of the slowest rank. Not per stage.
- DB chain: starts from the QH-07ae fix test's `db/` (fresh, A15-fixed profiles), each run from
  the previous run's `db/` (CSV `base_db`). Each run profiled 19–25 keys on GPU 0 and added 6–12:
  Adam `_foreach_*` keys on the new per-stage parameter lists (A14). The same 13 known untimed
  keys stay missing in every replay.
- Coverage: every rank matched its stage's expected events; status `valid` false only for the
  usual unmapped-operation list.
- The 4-layer even point is the QH-07ae fix test (commit `b8bd065`), not rerun.

## Results: 4 layers (`0.8b-l4`, GDN 0–2 + FA 3), seq 8192, TP 1 × DP 1 × PP 2, MTP 1

| split | real ms | Phantora ms | replay vs real | real Δ vs even | Phantora Δ vs even | real peak MiB GPU0 / GPU1 |
|---|---:|---:|---:|---:|---:|---:|
| 3,1 | 2,268.362 | 1,928.379 | −14.99% | −74.1 (−3.2%) | −68.6 (−3.4%) | 16,444 / 35,942 |
| 2,2 (even) | 2,342.473 | 1,996.999 | −14.75% | — | — | 14,798 / 37,048 |
| 1,3 | 2,421.477 | 2,065.719 | −14.69% | +79.0 (+3.4%) | +68.7 (+3.4%) | 13,170 / 38,110 |

## Results: 12 layers (`0.8b-l12`, first 12 of the full model, 9 GDN + 3 FA), seq 8192, TP 1 × DP 1 × PP 2, MTP 1

| split | real ms | Phantora ms | replay vs real | real Δ vs 6,6 | Phantora Δ vs 6,6 | Phantora / real Δ | real peak MiB GPU0 / GPU1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5,7 | 2,834.741 | 2,431.127 | −14.24% | +77.6 (+2.8%) | +68.8 (+2.9%) | 88.6% | 19,860 / 42,486 |
| 6,6 (even) | 2,757.123 | 2,362.376 | −14.32% | — | — | — | 21,688 / 41,402 |
| 7,5 | 2,684.668 | 2,293.821 | −14.56% | −72.5 (−2.6%) | −68.6 (−2.9%) | 94.6% | 23,494 / 40,320 |
| 8,4 | 2,588.243 | 2,216.020 | −14.38% | −168.9 (−6.1%) | −146.4 (−6.2%) | 86.7% | 25,364 / 39,214 |
| 9,3 | 2,516.359 | 2,147.609 | −14.65% | −240.8 (−8.7%) | −214.8 (−9.1%) | 89.2% | 27,350 / 38,110 |
| 10,2 | 2,441.976 | 2,079.181 | −14.86% | −315.1 (−11.4%) | −283.2 (−12.0%) | 89.9% | 29,438 / 37,048 |
| 11,1 | 2,370.497 | 2,010.771 | −15.18% | −386.6 (−14.0%) | −351.6 (−14.9%) | 90.9% | 31,826 / 35,942 |

10,2 and 11,1 were added after the first five (`batch-unevenpp-ext-20261123.sh`) because 9,3 was
still the fastest. Before running them, the per-stage rates below predicted Phantora ≈ 2,070 and
≈ 1,990 ms; measured 2,079.2 and 2,010.8. 11,1 is the fastest split at 12 layers.

## Prediction: 24 layers (`0.8b`, 18 GDN + 6 FA), seq 32768, TP 2 × DP 1 × PP 4 (8 simulated GPUs), MTP 1, simulation only

| split | Phantora ms | vs even |
|---|---:|---:|
| 6,6,6,6 (even; QH-07ae, `b8bd065`) | 7,097.234 | — |
| 7,7,6,4 | 6,383.876 | −713.4 (−10.1%) |
| 8,7,6,3 | 6,146.413 | −950.8 (−13.4%) |

Same simulated network as QH-07ae (one host, gpu01 loopback rate); no real reference above 2 GPUs.
The even point used a different DB chain (`k32m` runs); the uneven runs started from the
`k32m-24-tp2dp2pp2` DB and added only optimizer keys (18 and 21).

## Observations

- Measured: with MTP on, moving decoder layers off the last stage makes the step faster at every
  split tested, in real runs and in Phantora. Phantora ranks the splits in the same
  order as real (all 10 seq-8192 splits) and predicts 87–95% of each change from the even split.
- Measured: Phantora's error is flat across splits (−14.2% to −15.2%), so the split does not
  change the bias.
- Measured: real peak memory moves from GPU 1 to GPU 0 as layers move to stage 0; GPU 1 stays
  higher at every split (LM head, fp32 logits and MTP).

## Per-stage compute (Megatron phase timers), seq 8192, TP 1 × DP 1 × PP 2, MTP 1

Data: [`per_stage_phase_timers.csv`](per_stage_phase_timers.csv). Separate passes
(`run.sh --stages phases-sim,phases-real`), base DB = `upp-l12-57`'s `db/` (the end of the chain
above). Stage compute = `forward-compute` + `backward-compute` of that rank, ms per step (8
microbatches), mean of iterations 2–4. The timers synchronize CUDA, so real step times run 10–16 ms
above the headline runs; simulated steps match the replays within 0.1 ms. Headline step times stay
the ones above.

Per-step compute, 12 layers (`0.8b-l12`), ms:

| split | stage 0 real | stage 0 Phantora | last stage real | last stage Phantora | last stage / step (real) |
|---|---:|---:|---:|---:|---:|
| 5,7 | 484.8 | 408.2 (−15.8%) | 2,739.7 | 2,352.4 (−14.1%) | 96% |
| 6,6 | 514.9 | 486.6 (−5.5%) | 2,655.5 | 2,274.0 (−14.4%) | 96% |
| 7,5 | 592.5 | 565.0 (−4.6%) | 2,570.5 | 2,195.6 (−14.6%) | 95% |
| 8,4 | 693.0 | 654.1 (−5.6%) | 2,461.1 | 2,106.5 (−14.4%) | 95% |
| 9,3 | 771.0 | 732.5 (−5.0%) | 2,376.4 | 2,028.1 (−14.7%) | 94% |
| 10,2 | 848.8 | 810.8 (−4.5%) | 2,288.8 | 1,949.8 (−14.8%) | 93% |
| 11,1 | 940.4 | 889.2 (−5.4%) | 2,204.3 | 1,871.4 (−15.1%) | 93% |

4 layers (`0.8b-l4`): stage 0 / last stage real 258.0 / 2,204.9 (3,1), 178.7 / 2,283.5 (2,2),
100.1 / 2,376.5 (1,3); Phantora 240.7 / 1,871.4, 162.3 / 1,949.8, 83.9 / 2,028.2.

Check: the last stage of 12-layer 9,3 (GDN 9, GDN 10, FA 11 + tail) and of 4-layer 1,3 (GDN 1,
GDN 2, FA 3 + tail) hold the same layer types; Phantora gives 2,028.1 and 2,028.2 ms, real 2,376.4
and 2,376.5 ms. Likewise 12-layer 10,2 vs 4-layer 2,2 (Phantora 1,949.8 both; real 2,288.8 vs
2,283.5) and 12-layer 11,1 vs 4-layer 3,1 (Phantora 1,871.4 both; real 2,204.3 vs 2,204.9).

## Observations

- Measured: the last stage is the bottleneck at every split. Its compute is 93–96% of the real
  step; stage 0 computes 17–39% of the step (12 layers) and waits the rest. Even at 11,1 the
  last stage (FA 11 + tail, 2,204.3 ms real) does over twice the work of stage 0 (11 layers +
  embedding, 940.4 ms).
- Inferred from differences between splits (not timed directly): one GDN layer costs about 79 ms
  per step (real 79.3, Phantora 78.4; 4-layer 2,2 → 3,1 stage 0), one FA layer about 100 ms real /
  89 ms Phantora (12-layer 7,5 → 8,4 stage 0). The fixed tail (final norm, LM head, loss, MTP) is
  then about 2,100 ms real / 1,780 ms Phantora per step: roughly 23–27 GDN layers' worth, and 88% of
  the last stage at split 9,3. With a 248,320-word vocabulary and MTP's second head and loss, the
  tail outweighs all 12 decoder layers (about 1,000 ms).
- So at 12 layers, every split up to 11,1 leaves the last stage the slower one; moving each layer
  off it should save its full cost. Prediction (Phantora, inferred from the rates above, not run
  yet): 10,2 ≈ 2,070 ms, 11,1 ≈ 1,990 ms. Measured afterwards: 2,079.2 and 2,010.8 ms
  (real 2,442.0 and 2,370.5); 11,1 is the fastest split.
- Measured: Phantora's per-stage error is −14.1% to −15.1% on the last stage at every split, and
  −4.5% to −5.6% on stage 0 (5,7 at −15.8% is the exception: its real stage 0 forward, 188.1 ms, is
  higher than at 6,6 with one more layer; unexplained, single run). The step-time error therefore
  comes mostly from the last stage, and there mostly from forward: last-stage forward real 1,214.7
  vs Phantora 939.8 ms (−22.6%), backward 1,161.7 vs 1,088.4 ms (−6.3%) at 9,3. Which ops in the
  tail's forward are under-charged is not attributed yet (candidate assumption; needs per-op
  attribution, `attribute.py`).
