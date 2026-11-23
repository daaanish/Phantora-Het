# Flash attention for Qwen3.5-0.8B: real vs Phantora (QH-07ab, 2026-11-22)

From QH-07ab on, the full-attention layers and the MTP layer use PyTorch SDPA on the flash
backend (`FlashCoreAttention`, native grouped KV, implicit causal mask, no mask tensor)
instead of Megatron's local unfused attention. Data: [`unfused_vs_flash.csv`](unfused_vs_flash.csv).

## Conditions

24 layers + MTP, TP 2, seq 8192, MB 1, GA 8, 4 iterations, step = mean of iterations 2–4 of the
slower rank; gpu01, 2× RTX PRO 6000 Blackwell. Each flash run starts from the unfused point's
DB (QH-07aa `z-24-8192`), so only the flash keys are new.

## Results

| attention | profiler | Phantora ms | real ms | Phantora vs real | real peak GiB |
|---|---|---:|---:|---:|---:|
| unfused (QH-07aa) | upstream | 3,300.1 | 3,776.0 | −12.6% | 49.0 |
| flash | upstream (A14 bug) | 3,307.2 | 2,645.7 | +25.0% | 36.3 |
| flash | fixed | **2,220.7** | 2,645.7 | **−16.1%** | 36.3 |

- Real: flash is 30% faster (−1,130.3 ms) and peaks 12.7 GiB lower.
- Phantora with the fixed profiler predicts −1,079.4 ms for the switch, 95.5% of the real change.
- Per-op replay change (ms per step, rank 0): removed BMM −299.8, softmax backward −200.8,
  masked_fill −172.6 and −90.9, baddbmm −170.4, softmax −80.9, mask H2D copies −92.7, dtype
  casts −124.8; added flash forward +38.3 (0.68 ms × 56) and backward +117.6 (2.10 ms × 56).
- The error vs real (−16.1%) is a little larger than unfused (−12.6%): the remaining
  undercharge (A1 host gaps, A5/A6 unmapped and `aten::to` forms) is now a larger share of a
  shorter step. Not yet attributed.

## A14: the first flash replay was 25% too high

Upstream Phantora's profiler (`tests/perfdb/bench.py`) timed the flash backward key through
`scaled_dot_product_attention` without `enable_gqa`. With 4 query heads and 1 K/V head per rank
PyTorch then used the math backend: 21.5 ms per call vs 2.0 ms real. Fixed in `6f881ca`; details
and scope in [`../phantora_assumptions_vs_reality.md`](../phantora_assumptions_vs_reality.md) (A14).
