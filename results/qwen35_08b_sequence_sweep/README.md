# Qwen3.5-0.8B sequence-length sweep: Phantora vs real

gpu01, 2026-11-16. Two NVIDIA RTX PRO 6000 Blackwell Max-Q (97,887 MiB each).

Data: [phantora_vs_real_by_sequence_length.csv](phantora_vs_real_by_sequence_length.csv)

| Sequence length | Real | Phantora | Difference | Shortfall |
|---:|---:|---:|---:|---:|
| 128 | 0.898 s | 0.097 s | −89.2% | 0.801 s |
| 512 | 0.894 s | not run | | |
| 2048 | 0.915 s | not run | | |
| 4096 | 1.357 s | 1.018 s | −25.0% | 0.339 s |
| 8192 | 3.192 s | 2.540 s | −20.4% | 0.651 s |
| 16384 | 9.075 s | 7.411 s | −18.3% | 1.664 s |
| 32768 | out of memory | not run | | |

## With batch prefetching (added 2026-11-16)

The test harness used to fetch each microbatch inside the timed step, so the
host built the sample (including a sequence-by-sequence attention mask) while
the GPU sat empty at the start of every step. `tests/test_megatron.py` now
fetches a step's batches before the timer starts (`--no_prefetch_batches`
restores the old behavior). Phantora's numbers are identical in both columns
because it never charged host time. The earlier real numbers are kept.

The last two columns are points rerun with the QH-07u build (`cat`/`contiguous`
handlers with traced layouts); blank means not rerun. See the QH-07u and QH-07v sections below.

24 layers:

| Sequence length | Real, no prefetch | Real, prefetch | Phantora | Diff, no prefetch | Diff, prefetch | Phantora, QH-07u build | Diff, prefetch, QH-07u build |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.898 s | 0.902 s | 0.097 s | −89.2% | −89.3% | | |
| 512 | 0.894 s | 0.902 s | not run | | | | |
| 2048 | 0.915 s | 0.874 s | not run | | | | |
| 4096 | 1.357 s | 1.317 s | 1.018 s | −25.0% | −22.7% | | |
| 8192 | 3.192 s | 3.031 s | 2.540 s | −20.4% | −16.2% | 2.652 s | −12.5% |
| 16384 | 9.075 s | 8.444 s | 7.411 s | −18.3% | −12.2% | 7.661 s | −9.3% |

4 layers:

| Sequence length | Real, no prefetch | Real, prefetch | Phantora | Diff, no prefetch | Diff, prefetch | Phantora, QH-07u build | Diff, prefetch, QH-07u build |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.184 s | 0.186 s | 0.030 s | −83.9% | −84.0% | | |
| 1024 | 0.187 s | 0.186 s | 0.093 s | −50.4% | −50.0% | | |
| 2048 | 0.273 s | 0.261 s | 0.169 s | −38.1% | −35.1% | | |
| 4096 | 0.495 s | 0.461 s | 0.351 s | −29.1% | −24.0% | | |
| 8192 | 1.103 s | 0.943 s | 0.786 s | −28.8% | −16.7% | | |
| 12288 | 1.922 s | 1.561 s | 1.328 s | −30.9% | −14.9% | | |
| 16384 | 2.932 s | 2.291 s | 1.987 s | −32.2% | −13.3% | | |
| 24576 | 5.521 s | 4.086 s | 3.778 s | −31.6% | −7.5% | 3.844 s | −5.9% |
| 32768 | 10.164 s | 7.609 s | 5.908 s | −41.9% | −22.3% | | |

Prefetch runs: `gpu01:/home/dathar/Phantora/qh07s-prefetch-3ztcW0` (one run per
size; a traced 24-layer 8192 run gave 3.039 s). Both CSVs carry the new values
in `real_prefetch_ms`, `difference_prefetch_pct`, `shortfall_prefetch_ms`.

## Ablation: charging GDN-specific operations (seq 8192, 24 layers)

Data: [ablation_seq8192.csv](ablation_seq8192.csv), [ablation_seq8192_per_shape.csv](ablation_seq8192_per_shape.csv)

Each row adds one handler to the simulator; everything else (timing database, workload, network) is held fixed.

| Configuration | Phantora | Change | Diff vs real, no prefetch (3.192 s) | Diff vs real, prefetch (3.031 s) | Compute still uncharged |
|---|---:|---:|---:|---:|---:|
| Baseline | 2.540 s | | −20.4% | −16.2% | 350 ms |
| + `cat` | 2.566 s | +25 ms | −19.6% | −15.4% | 325 ms |
| + `cat` + `contiguous` | 2.606 s | +40 ms | −18.4% | −14.0% | 285 ms |
| + `cat` + `contiguous`, traced layout (QH-07u) | 2.652 s | +46 ms | −16.9% | −12.5% | 239 ms |

- The handlers recover 65 ms of the 118 ms these two operations really take (`cat` 25 of 53 ms, `contiguous` 40 of 65 ms).
- The rest is a profiling problem, not a missing handler: the simulator's trace does not carry tensor strides, so the profiler has to guess the memory layout. Where the guess matches (`contiguous` on `[1,3072,8192]`) the timing is within 0.4% of the in-situ kernel; where it does not (`contiguous` on `[1,8192,8,128]`, the GDN split-backward `cat` whose inputs are strided gradients) the profile is 3–4x too fast. The per-shape file lists every case.
- The planned acceptance gate (profiled within 15% of in-situ) therefore fails for 11 of 12 shapes, so these handlers are not yet applied to the full sweeps.
- Build and runs: `gpu01:/home/dathar/Phantora/qh07t-gdnfix-eoqL0c`.

### Traced layout (QH-07u, 2026-11-17)

One change from the row above: the tracer now sends each tensor's strides and storage offset, and the `cat`/`contiguous` keys carry them, so the profiler rebuilds the real layout instead of guessing. Everything else (QH-07m base DB, workload, network) is unchanged; the other 14 missing keys are the same.

- Charged per rank-step: `cat` 50.4 ms (was 25.3; real 53.4), `contiguous` 60.9 ms (was 39.9; real 64.7). Together 111.3 of 118.1 ms.
- Per-shape gate: 8 of 12 pass (was 1). The GDN split-backward `cat` is 230 µs against 233 real (was 81); `contiguous` on `[1,8192,8,128]` averages 51.8 against 55.6 (was 13); every rotary `cat` is within 13%.
- Still failing: `[1,8192,1024]x3` GDN `cat` (0.71) and the gated-attention `cat` (0.73) take row-order inputs, so layout is not their cause (A9); `contiguous` `[8192,1,4,256]` (0.58) has the fused-region count mismatch noted above; `[1,8192,8]` (0.83) is 2–3 µs.
- The traced layouts show why the guess failed: the `[1,8192,8,128]` source is a three-way permutation (strides `[0,1,1048576,8192]`), not a two-dimension swap, and the rotary and split-backward `cat` members are slices of wider buffers.
- 20 layout keys replace 13 shape-only ones; three differ only by a storage offset that is a multiple of 16 MiB, profiled separately and within 0.3 µs of each other.
- Build and runs: `gpu01:/home/dathar/Phantora/qh07u-layout-IxfR7j`; patched torch prefix `gpu01:/tmp/phantora-qh07u-keep/torch-deps`.

## Sweep points rerun with the QH-07u build (QH-07v, 2026-11-17)

The two points closest to real, rerun with the `cat`/`contiguous` handlers and traced layouts. Each starts from its own sweep's final timing DB plus the 20 newly profiled strided keys; real times are the existing prefetch runs (the simulator change does not affect them). New columns in both sweep CSVs end in `_qh07v`.

| Point | Real (prefetch) | Phantora, sweep | Diff | Phantora, QH-07u build | Diff | `cat`+`contiguous` charged |
|---|---:|---:|---:|---:|---:|---:|
| 4 layers, seq 24576 | 4,086.2 ms | 3,778.2 ms | −7.5% | 3,843.5 ms | −5.9% | 65.3 ms |
| 24 layers, seq 16384 | 8,444.1 ms | 7,410.8 ms | −12.2% | 7,660.5 ms | −9.3% | 249.7 ms |

- Both replays exit 0 with the same 14 zero-time keys missing (`valid=false`), no other new keys and no large-message warnings.
- Per-op real times were only attributed at seq 8192, so how close the new charges are to real at these sizes is not measured.
- Runs: `gpu01:/home/dathar/Phantora/qh07v-points-aQv4GH`.

## Missing-compute ledger

[missing_compute_ledger.csv](missing_compute_ledger.csv) lists every operation
group at seq 8192 whose charged time differs from real by 1 ms or more, with
whether original Phantora has a handler for it, which part of the model it
belongs to (GDN-specific, shared Megatron attention, cross-entropy, optimizer,
…), where it originates, and its status. Rows are never removed; fixes update
`status`.

## Setup

- Qwen3.5-0.8B text preset: 24 layers (18 GDN, 6 unfused full attention), hidden 1024, BF16.
- TP 2, DP 1, PP 1, microbatch 1, gradient accumulation 8, 4 iterations.
- Times are the mean over iterations 1–3 of the slower rank's step time.
- Phantora: CPU time ignored, CUPTI kernel timings measured on one Blackwell, fixed 271.916 Gbit/s loopback rate, QH-07c native build.
- The 128 real time is the original QH-03a run; a rerun in the same sweep gave 902.0 ms.

## Caveats

- Every Phantora run is `valid=false` under strict coverage: 14 compute keys showed no GPU activity when profiled and are absent from the database, and some operators remain unmapped.
- Real peak memory is the device-level `nvidia-smi` peak sampled every 2 s and includes allocator cache. Phantora's peak is its own unvalidated estimate.
- At 128–2048 the real step is host-bound (flat near 0.9 s); a GPU-work replay cannot predict it.
- For 4096–16384 the shortfall is close to 0.13 s + 0.207 × Phantora time. This is a three-point observation, not an attribution.

## 4-layer variant (overnight sweep)

Data: [phantora_vs_real_4_layer_by_sequence_length.csv](phantora_vs_real_4_layer_by_sequence_length.csv)

Same 0.8B layer shapes at 4 layers (3 GDN + 1 unfused full attention), preset `0.8b-l4`. Same TP 2, microbatch 1, accumulation 8. Real runs use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which 32768 needs to fit.

| Sequence length | Real | Phantora | Difference | Real peak memory |
|---:|---:|---:|---:|---:|
| 128 | 0.184 s | 0.030 s | −83.9% | 8.5 GiB |
| 1024 | 0.187 s | 0.093 s | −50.4% | 8.8 GiB |
| 2048 | 0.273 s | 0.169 s | −38.1% | 9.8 GiB |
| 4096 | 0.495 s | 0.351 s | −29.1% | 12.7 GiB |
| 8192 | 1.103 s | 0.786 s | −28.8% | 19.2 GiB |
| 12288 | 1.922 s | 1.328 s | −30.9% | 27.5 GiB |
| 16384 | 2.932 s | 1.987 s | −32.2% | 33.8 GiB |
| 24576 | 5.521 s | 3.778 s | −31.6% | 59.8 GiB |
| 32768 | 10.164 s | 5.908 s | −41.9% | 96.4 GiB |

- The difference does not keep shrinking: it bottoms out near −29% at 4096–8192 and sits at −31% to −32% from 12288 to 24576.
- 32768 is an outlier. The real run is at the GPU memory limit, and Phantora with the real 97,887 MiB limit hit the same out-of-memory failure as the real default allocator, so its replay used a raised simulated limit. Whether the extra gap is real memory-pressure overhead or Phantora under-charging at this size is not established.
- Run directory: `gpu01:/home/dathar/Phantora/qh07q-l4-sweep-3CPzh4` (`results.csv`, `progress.log`, one folder per size; `seq-32768-attempt1` is the failed first Phantora attempt).

## Where the gap comes from at sequence length 8192 (24 layers)

Data: [attribution_seq8192.csv](attribution_seq8192.csv), one row per operation group, rank 0, per step.

One real step traced with the PyTorch profiler (3.21 s; untraced 3.19 s) against a Phantora replay with per-event logging (2.540 s, unchanged).

| Per step, rank 0 | Real | Phantora | Difference |
|---|---:|---:|---:|
| GPU compute (kernels and device copies) | 2,610 ms | 2,260 ms | 350 ms |
| TP all-reduce time not overlapped with compute | 331 ms | 280 ms | 51 ms |
| GPU empty between kernels | 269 ms | 0 ms | 269 ms |
| Step | 3,210 ms | 2,540 ms | 670 ms |

GPU compute difference by cause (sums to 350 ms):

| Cause | ms |
|---|---:|
| Operations Phantora never charges (unmapped: `contiguous` 65, `cat` 53, `sub_` 44, `max` 21, optimizer `_foreach_*`, others) | +203 |
| `to` dtype conversions: half of the large BF16↔FP32 conversions use a call form Phantora rejects | +121 |
| Other charged operations that come out low (`mul` 38, `add_` 37, `_softmax_backward_data` 31, fused GDN 15, device memcpy 13, matmul 13, …) | +176 |
| `torch.compile` regions and memset, not seen | +6 |
| Charged operations that come out high (`masked_fill` −70, `baddbmm` −52, `add` −14, `bmm` −11, …) | −156 |

- `masked_fill` is double-charged: its 148 ms timing already includes the 1 GiB clone, and the intercepted device memcpy is charged again (70 ms).
- Same-shape binary operations (`mul`, `add_`) are measured 2–3x too fast, consistent with the profiler aliasing both operands to one buffer.
- The 14 shapes that profile as zero matter very little: 8 conversion events and 440 dropout events per step, with no measurable real GPU time behind them.
- "GPU empty between kernels" is 190 ms with no launch pending (host) plus 79 ms of launch latency. Phantora with CPU time ignored charges none of it.

Traces: `gpu01:/home/dathar/Phantora/qh07r-attrib-x6LKFo` (`real/*.json`, `replay/events.jsonl.gz`). Scripts: `Phantora-Het/.cache/qwen-hybrid/qh07r/`.

Evidence log and run commands: `Phantora-Het/QWEN_HYBRID_TRACKER.md` (QH-07j through QH-07o).
