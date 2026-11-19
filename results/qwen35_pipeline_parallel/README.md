# Qwen3.5 hybrid with pipeline parallelism (QH-07x, 2026-11-19)

First pipeline-parallel runs of the hybrid workload, real vs Phantora, through the
commit-pinned gpu01 workflow (`run.sh --pp 2 [--stage-layers 1,3]`, commit `2b7f44f`).
Data: [`phantora_vs_real_pp2.csv`](phantora_vs_real_pp2.csv). Earlier results (all TP 2, PP 1)
are unchanged.

## Conditions

- gpu01, 2× RTX PRO 6000 Blackwell Max-Q; TP 1, PP 2, DP 1 (one stage per GPU).
- `0.8b-l4` preset (0.8B shapes, layers gdn, gdn, gdn, full_attention), seq 8192, MB 1, GA 8
  (8 microbatches per step, Megatron 1F1B), 4 iterations; step time = mean of iterations 2–4 of
  the slowest rank. Real runs with expandable segments (preset default).
- Stages discovery, profile, replay, real in one run each; base DB = QH-07u `db-layout`
  (copied, never modified). TP 1 shapes were new: 194 compute keys missing, 194 profiled,
  180 added, 14 non-positive (the same 14 known untimed keys as QH-07u); replay then misses 14.
- Coverage plan per PP rank (`pp_rank = rank // TP`): every rank's training-region GDN event
  counts equal its stage's expected counts in discovery and replay. Status `valid` is false
  only because of the usual unmapped-operation list (as in every earlier run).
- Real runs confirmed the split: `HYBRID_CONFIG` shows layers [0,1] | [2,3] (even) and
  [0] | [1,2,3] (split 1,3; Megatron layout `Et*1|t*3L`).

## Results

| split | real ms | Phantora replay ms | replay vs real |
|---|---:|---:|---:|
| 2,2 (even) | 1,564.740 | 1,380.871 | −11.75% |
| 1,3 | 1,641.614 | 1,449.087 | −11.73% |
| uneven − even | +76.874 (+4.9%) | +68.216 (+4.9%) | Phantora predicts 88.7% of the change |

Phantora keeps the same relative error at both splits and ranks them correctly (1,3 is
slower: the last stage carries 3 layers plus the LM head and loss). Peak memory (real, MiB):
even 14,654 / 31,878, split 1,3 13,126 / 32,542 (GPU 0 / GPU 1; GPU 1 holds the LM head and
loss logits).
