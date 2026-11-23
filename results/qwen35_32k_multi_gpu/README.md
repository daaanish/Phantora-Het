# Qwen3.5-0.8B at 32k context on 4 and 8 simulated GPUs (QH-07ae, 2026-11-23)

Phantora predictions for 24-layer Qwen3.5-0.8B training at seq 32768 on 4 and 8 GPUs, from timings
measured on one gpu01 Blackwell GPU. 24 layers at 32k do not fit on gpu01's 2 GPUs in real
training, so the real reference is a 12-layer, TP 2 point. Data:
[`phantora_32k_multi_gpu.csv`](phantora_32k_multi_gpu.csv) (every run, including failures).

## Conditions

- gpu01, RTX PRO 6000 Blackwell Max-Q, driver 620.70.06. seq 32768, MB 1, 8 microbatches per
  replica, 4 iterations; step time = mean of iterations 2–4 of the slowest rank. Flash attention.
  Simulated VRAM 97,887 MiB. Real runs (2 GPUs only) with expandable segments (default for
  reduced-depth presets).
- **Every timing profiled fresh on GPU 0 with the A15-fixed profiler.** The first run started from
  an empty DB (`runs/empty-db-20261123`, headers and manifest only); each later run started from
  the previous run's `db/` (exact chain in the CSV `base_db` column). The existing 32k DB was not
  used: it came from `f-l4-32768`, a session A15 flagged as short-captured, and older DBs have keys
  found too low (tracker, 2026-11-23).
- Simulated network: all GPUs on one host at gpu01's measured loopback rate (271.9 Gbit/s, PCIe),
  as in every earlier run. An 8-GPU gpu01 does not exist; this is a hypothetical machine.
- Coverage: every completed run's ranks match their expected events; status `valid` false only for
  the usual unmapped-operation list and the multi-argument `aten::to`.
- Three conditions, in order run (all kept):
  - **A**, MTP 1 on `398fe2b`: anchor plus five 24-layer configurations, which all failed.
  - **B**, MTP 0 on `398fe2b`: side condition run as a workaround while A failed. Not the target
    model (Qwen3.5 has one MTP layer); recorded, not extended.
  - **C**, MTP 1 on `b8bd065`, the fix below: fix test on 2 real GPUs, and the three TP 2
    configurations from A.

## Real reference (2 GPUs, 12 layers, TP 2)

| MTP | real ms | Phantora replay ms | replay vs real | real peak MiB per GPU |
|---:|---:|---:|---:|---:|
| 1 (A) | 8,938.117 | 8,235.293 | −7.86% | 82,402 |
| 0 (B) | 6,145.347 | 5,839.374 | −4.98% | 55,978 |
| MTP cost | +2,792.8 | +2,395.9 | Phantora predicts 85.8% | |

The MTP-on anchor profiled 243 compute and 20 memcpy keys from nothing in about 2 minutes on
GPU 0. It agrees with the A15-corrected replay of the earlier 12-layer 32k point (−7.4%).

## 24 layers, seq 32768, MTP 1 (condition C, simulation only)

| TP × DP × PP | GPUs | Phantora replay ms | new keys on GPU 0 |
|---|---:|---:|---|
| 2 × 1 × 2 | 4 | 8,753.656 | 3 compute |
| 2 × 1 × 4 | 8 | 7,097.234 | 21 compute |
| 2 × 2 × 2 | 8 | 8,769.010 | 47 compute, 9 memcpy |
| 1 × 1 × 4 | 4 | — | simulated OOM (condition A) |
| 1 × 1 × 8 | 8 | — | simulated OOM (condition A) |

Condition B, MTP 0, for reference: 6,357.743 / 4,701.189 / 6,368.952 ms for the same three TP 2
configurations.

## Failures and the fix

- **MTP with PP failed in Megatron (all TP 2 runs in A).** The assertion
  `len(self.per_param_grad_ready_counts) == len(self.params)` in `zero_grad_buffer` fired on every
  last-stage rank. Megatron Core 0.16.0 allocates `output_layer.weight` on a last stage that owns
  MTP but not the input embedding (`skip_weight_param_allocation` only checks `pre_process`). With
  tied embeddings it computes the logits from that stage's embedding copy, so the output weight
  never gets a gradient, and DDP with overlapped gradient reduction rejects it. PP 1 is unaffected,
  because the MTP stage is also the first stage. Fixed in the harness (`b8bd065`,
  `drop_unused_mtp_output_weight`): the weight is removed before DDP, and
  `validate_hybrid_structure` checks it is gone.
- **Fix test on real GPUs:** 4 layers, TP 1, PP 2, MTP 1, seq 8192 trained with MTP and PP on
  both GPUs (`HYBRID_MTP_PP` printed once, last stage). Real 2,342.473 ms vs replay 1,996.999 ms
  (−14.75%); real peak 14,798 / 37,048 MiB. Base DB: QH-07z-fresh DP 2 `db/`, 26 keys added.
  Also 35 unit tests in the gpu01 image, including a new one for the removal.
- **TP 1 at 32k does not fit.** TP1 × PP4 and TP1 × PP8 ran out of simulated memory on the last
  stage, requesting 30.31 GiB with 88.27 / 78.78 GiB already allocated. 30.31 GiB is the fp32
  logits, 32,768 × 248,320 × 4 bytes. Without TP the vocabulary is not split, and more pipeline
  stages cannot help because the output layer sits on one stage. These runs were before the fix,
  which frees roughly 1.5 GiB of unused weight and gradient on that stage at TP 1; not enough to
  close a ~21 GiB gap, but not rerun. Simulated memory is pessimistic (A13), so real behaviour is
  unverified.

## Observations (inferred; no real reference above 2 GPUs)

- MTP adds a constant ~2,396–2,400 ms in all three 24-layer configurations, the same as in the
  12-layer anchor (2,395.9). MTP sits on the last stage, so its cost adds straight to the step.
- PP 2 → PP 4 (4 → 8 GPUs) cuts the step by 19% (8,753.7 → 7,097.2 ms). The last stage, which
  carries the output layer, loss and MTP over 32k tokens, likely limits the gain (not checked
  per stage).
- Adding DP 2 to TP2 × PP2 costs +15.4 ms in simulation. The real figure would likely differ:
  at seq 8192 Phantora charged the optimizer only 22% of real, so it misses the optimizer savings
  of DP (A5, `../qwen35_data_parallel/`).
