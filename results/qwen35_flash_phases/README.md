# Megatron phase timers, real vs Phantora, flash sweep points (QH-07ad, 2026-11-23)

Where in the training step Phantora's gap to real training sits. Megatron's own phase timers
(`test_megatron.py --phase_timings`, the same timers as vanilla Phantora's Llama/GPT phase study) on
five QH-07ac points, in separate passes from the headline timing (`run.sh --phases`: `phases-sim`,
`phases-real`). Data: [`phases_phantora_vs_real.csv`](phases_phantora_vs_real.csv), slowest rank, ms
per step, iterations 1–3.

## Conditions

- Same workload as the QH-07ac sweep point: MTP, flash attention, TP 2, MB 1, GA 8, 4 iterations,
  expandable segments. Commit `b18e228`; each pass replays with its sweep point's DB.
- Points marked * use the simulated phases from the A15-corrected DB (`a15-ph-*` runs, commit
  `17c6542`); the first simulated pass used a DB with 0.8x timings (A15). Real phases are unaffected.
- The timers synchronize CUDA at every start and stop (and barrier at level 1). Their cost here:
  simulated step time identical to the replay's (the simulated clock is unaffected), real step +0.2% to
  +0.5% (24L 8k 2,657.7 vs 2,651.8 ms; 16L 16k 4,570.9 vs 4,549.9 ms).
- Timers nest: `forward-backward` contains forward and backward compute and the gradient sync; the
  optimizer and parameter all-gather follow it.

## Results

| layers | seq | Phantora step ms | real step ms | forward | backward | Adam ms (Phantora / real) | share of gap in fwd+bwd |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 8,192 | 2,220.7 | 2,657.7 | −14.3% | −15.9% | 4.1 / 21.2 | 92% |
| 24 | 24,576 | 8,703.2 | 9,422.1 | −6.9% | −7.4% | 4.1 / 21.3 | 95% * |
| 4 | 8,192 | 1,136.7 | 1,382.4 | −22.2% | −11.4% | 3.0 / 9.8 | 95% |
| 16 | 16,384 | 3,881.4 | 4,570.9 | −13.5% | −15.3% | 3.6 / 16.7 | 96% * |
| 16 | 24,576 | 6,756.2 | 7,390.6 | −8.8% | −7.8% | 3.7 / 16.8 | 96% |

## Observations

Measured:
- 92–96% of Phantora's gap is inside forward and backward compute. Gradient sync, parameter
  all-gather and the optimizer copies are a few ms on either side.
- Forward and backward are under-predicted by about the same fraction at 16 and 24 layers, and that
  fraction shrinks with sequence length (about 15% at 8–16k, about 7–9% at 24k).
- At 4 layers the forward is −22% but the backward −11%. The per-step pieces that do not scale with
  depth (embedding, LM head over 248,320 words, two cross-entropies for MTP) are a larger share of the
  forward there.
- Adam is charged 3–4 ms against 10–21 ms real (−70% to −81%), consistent with the optimizer
  `_foreach_*` shortfalls in the unfused ledger (QH-07r, ~16 ms): small in absolute terms.

Inferred (not measured here):
- The 4-layer forward skew points at the loss path; the unfused ledger had the cross-entropy `sub_` and
  `max` uncharged (A5). A traced real run would confirm per op.
- At 24 layers the gap grows less than in proportion to the work: 431 ms at 8k, 702 ms at 24k
  (headline numbers), for about 3.5x the step time. So part of the gap is fixed per step (kernel count,
  host gaps, unmapped small ops) and part scales with the work. (Before the A15 correction the 24k
  point looked proportional; that reading was based on the 0.8x DB.) The traced run splits the two.
