# Phantora assumptions that differ from reality

Started 2026-11-16. A running record of modelling assumptions in Phantora that
we have found to differ from what real training does, with the evidence and
the measured size of the effect. Add to it whenever a new one is found; never
remove an entry, update its status instead.

"Original" means the assumption is in upstream Phantora
(`Phantora-Vanilla/Phantora-main`, commit 301d9d9). "Fork" means it comes from
the Qwen hybrid work in `Phantora-Het`. Unless noted, numbers are per training
step on rank 0 for Qwen3.5-0.8B, 24 layers, TP 2, sequence length 8192,
microbatch 1, accumulation 8, on gpu01 (two RTX PRO 6000 Blackwell Max-Q).
Evidence rows are in `Phantora-Het/QWEN_HYBRID_TRACKER.md` under the IDs given.

## Summary

| # | Assumption | Where | Measured effect | Status |
|---|---|---|---|---|
| A1 | A training step costs only GPU work plus communication; host time is zero | Original (CPU time ignored by default in the Megatron launcher) | −89% at seq 128, where the host sets the pace | Open; avoid host-bound sizes |
| A2 | Kernels run back to back with no gap | Original | 79 ms (about 2 µs × 36,000 kernels) | Open |
| A3 | A tensor is fully described by shape, dtype and device | Original (trace format) | Layout copies profile 3–4x too fast when the guessed layout is wrong | Fixed for `cat`/`contiguous` in the fork (QH-07u): traced strides close 46 of the 53 ms gap; other ops still keyed by shape |
| A4 | Profiling tensors of equal size and dtype can share one buffer | Original (profiler allocation cache) | Same-shape `mul`/`add_` profile 2–3x too fast (cause suspected, not isolated) | Open |
| A5 | An operation with no handler costs nothing | Original; the fork's coverage mode reports them | 203 ms of unmapped operations | Partly fixed (`cat`, `contiguous`) |
| A6 | `aten::to` only matters in its dtype-only form | Original | 121 ms; half the large BF16↔FP32 conversions use another form | Open |
| A7 | An operation's profiled time and its intercepted device memcpy are separate work | Original | `masked_fill` charged twice for its 1 GiB clone, +70 ms too high | Open |
| A8 | Whole-microsecond resolution per event is enough | Original | At most 30 ms over ~30,000 events; not isolated | Open |
| A9 | A kernel timed in isolation takes as long as in the training run | Original | Mostly true (matmul within 2%); aligned `cat` 23–29% low for unknown reasons | Open |
| A10 | The simulated process runs the same code path as the real one | Original | Megatron's compiled regions run as separate ops in simulation (6 ms of fused kernels in real); workers also advertise an sm_86 GPU | Open; effect small here |
| A11 | Preparing input data inside the timed step is free | Test harness, original and fork | 163 ms stall at the start of every real step | Fixed in the harness (prefetch) |
| A12 | One measured bandwidth describes all collectives | Fork's setup | All-reduce within about 50 ms of real | Acceptable at this size |
| A13 | Simulated GPU memory behaves like real GPU memory | Original | Peak not validated (78.7 GiB simulated vs 95.6 GiB device-level at seq 16384); it did reproduce the real out-of-memory failure at 32768 | Unvalidated |
| A14 | The flash-attention backward can be profiled through SDPA without the grouped-KV flag | Original (`tests/perfdb/bench.py`, `SDPABackward`) | Flash backward with 4 Q / 1 KV head timed via the math fallback: 21.5 ms vs 2.0 ms real per call, +1,204 ms per step at seq 8192 (replay 25% above real) | Fixed in the fork (QH-07ab): flash forced, `enable_gqa` inferred from head counts; replay rerun pending |

## Details

### A1. Host time is zero

Phantora advances its clock by profiled GPU kernel time and simulated network
time. With `PHANTORA_IGNORE_CPU_TIME=1` (the default in the Megatron launcher)
the time the host spends in Python, the PyTorch dispatcher and the CUDA driver
between kernel launches is not counted. The alternative clock counts the
replay process's CPU time, which includes simulator overhead and overshoots
(3.33 s against 0.90 s real at seq 128).

Reality: at seq 128 a step issues about 36,000 kernels averaging 4 µs each
(141 ms of GPU work) but takes 0.9 s, because the host needs about 25 µs per
launch. Real step time is flat from seq 128 to 2048. Evidence: QH-07j.

### A2. No gap between consecutive kernels

Phantora chains events on a stream with zero delay. In the real trace each
kernel starts about 2.3 µs (median) after the previous one ends even when its
launch was issued long before; over 36,096 kernels that is 79 ms. Evidence:
QH-07r follow-up analysis.

### A3. Shape, dtype and device describe a tensor

The patched PyTorch tracer sends each tensor as shape, dtype and device. The
profiler later rebuilds a tensor from that description on a real GPU and times
the operation. Strides, contiguity and storage identity are not sent, so the
profiler cannot reproduce the real memory layout.

Reality: for operations that move data, layout is most of the cost.

- `contiguous` on `[1,8192,8,128]` BF16: 13 µs profiled, 56 µs in the real run.
- `cat` in the GDN split backward: 81 µs profiled, 233 µs real, because the
  real inputs are strided gradients and PyTorch uses its generic kernel.
- `contiguous` on `[1,3072,8192]`: 212.0 µs profiled, 212.8 µs real, because
  the guessed layout happened to match.

The fork's fused GDN events do carry strides, and five of six fused timings
are within 1 ms per step of real. Evidence: QH-07t, `ablation_seq8192_per_shape.csv`.

Fix (fork, QH-07u, 2026-11-17): the tracer now sends `stride` and `offset`
for every strided tensor, and the `cat` and `contiguous` keys carry them; the
profiler rebuilds that exact layout. Other keys are unchanged, so existing
database rows stay valid. Measured effect at seq 8192, per rank-step:
`cat` 25.3 → 50.4 ms (real 53.4), `contiguous` 39.9 → 60.9 ms (real 64.7);
GDN split-backward `cat` 81 → 230 µs (real 233); `contiguous` on
`[1,8192,8,128]` 13 → 52 µs mean (real 56). Step error vs real (prefetch)
−14.0% → −12.5%. The real `[1,8192,8,128]` source turned out to be a
three-way permutation, not the two-dimension swap the profiler guessed.
Not covered: ordinary ops (e.g. same-shape `mul`, A4) still have no layout in
their keys; two row-order `cat`s remain about 27% low for a reason that is not
layout (A9).

### A4. Equal-size profiling tensors may alias

`torch_estimate.rs` and `tests/perfdb/bench.py` cache one buffer per
(element count, dtype) and hand out views of it, so `mul(x, y)` with
equal-shaped operands is profiled as `mul(x, x)`. Real operands are distinct.
In the trace, same-shape `mul` on `[8192,1,1024]` FP32 takes 29.7 µs against
11 µs profiled. Aliasing is the suspected cause; it has not been tested in
isolation, and layout (A3) may contribute. First noted in
`Phantora-Vanilla/gpu01_timing_investigation_2026-11-08.md`.

### A5. Unhandled operations cost nothing

An ATen operation with no case in `torch_call.rs` is dropped. Original Phantora
does this silently; the fork's coverage mode lists them as `unmapped_compute`.
At seq 8192 they total 203 ms: `contiguous` 65, `cat` 53, `sub_` 44, `max` 21,
optimizer `_foreach_lerp_/sqrt/div_/add_` about 9, and smaller ones. Which are
GDN-specific and which are shared with dense Megatron models is recorded in
`qwen35_08b_sequence_sweep/missing_compute_ledger.csv`. Evidence: QH-07r, QH-07t.

### A6. Only the dtype-only form of `aten::to` matters

The handler accepts `to(tensor, dtype, bool, bool)` and ignores other forms.
At the attention-score shape there are 96 BF16↔FP32 conversions per step in
each direction; 48 are charged. The rest come from the backward pass, which
uses a longer argument list. Evidence: QH-07r.

### A7. Profiled time and intercepted memcpy are independent

Phantora charges device-to-device `cudaMemcpyAsync` calls it intercepts, and
separately charges each operation's profiled time. `masked_fill` (out of
place) clones its input first; its profiled 148 ms already includes that clone,
and the 70 ms memcpy is then charged again. Evidence: QH-07r.

### A8. Microsecond resolution

`simulator.rs` converts each profiled duration to whole microseconds
(`as_micros()`), discarding up to 1 µs per event. Bounded at 30 ms per step
here; it mostly matters when kernels are a few microseconds long.

### A9. Isolated timing equals in-situ timing

True within 2% for matrix multiplies and several large attention kernels.
Not true everywhere even when the kernel variant matches: aligned `cat` cases
profile 23–29% below the real run with distinct, contiguous inputs. Cause
unknown.

### A10. Same code path in simulation and reality

- Megatron's `jit_fuser` functions are compiled into fused Triton kernels in
  real runs. In simulation the same code runs as individual operations, which
  are charged one by one (for example 48 extra `contiguous` copies and 64
  extra `silu` events per step that have no separate kernel in reality).
- Simulated workers report GPU compute capability 8.6 regardless of the
  target GPU, which could change dispatch decisions. Effect not measured.

### A11. Input preparation inside the step is free

The test harness fetched each microbatch inside the timed step. Megatron's mock
dataset builds every sample, including a sequence-by-sequence attention mask,
on the host. The GPU sat empty for 163 ms at the start of every real step;
later microbatches hid the cost behind queued GPU work. Simulation never saw
it. Fixed by prefetching the step's batches before the timer
(`tests/test_megatron.py`, `--no_prefetch_batches` restores the old path).
Evidence: QH-07r, QH-07s.

### A12. One bandwidth for all collectives

The replay uses a single loopback rate measured for an 8 MiB all-reduce. At
seq 8192 simulated all-reduce time is within about 50 ms of the real NCCL
kernel residence. Not checked at other payload sizes.

### A13. Simulated memory

Simulated peak memory is labelled unvalidated and is lower than the
device-level real peak. One point in its favor: with the real 97,887 MiB
limit the simulation failed at seq 32768 with the same allocation request and
fragmentation as the real run (16 GiB requested, 64.7 GiB allocated, 23.9 GiB
reserved). Evidence: QH-07q.

### A14. Flash backward profiled without grouped KV

Phantora charges the flash-attention backward at
`aten::_scaled_dot_product_flash_attention_backward`, and its Rust estimator
calls that op directly. The timing-database profiler (`bench.py`) instead
rebuilds the forward with `scaled_dot_product_attention(q, k, v, is_causal)`
and times `out.backward`. It does not pass `enable_gqa`, and the key does not
record it. With grouped KV (Qwen3.5-0.8B per TP 2 rank: 4 query heads, 1 KV
head) PyTorch cannot use the flash or memory-efficient kernels without the flag
and falls back to the math backend, which materializes the s x s scores. The
profiled key was 21.5 ms per call; the real flash backward at the same shapes
is 2.0 ms (real autograd 2.705 ms forward+backward minus 0.688 ms forward;
direct op call 2.08 ms). With 56 calls per step this added 1,204 ms and put the
first flash replay at 3,307.2 ms vs 2,645.7 ms real (+25.0%).

Fix in the fork: build the graph under `sdpa_kernel(FLASH_ATTENTION)` with
`enable_gqa = k heads != q heads`. Checked on gpu01: the key times 2.09 ms
(was 22.2 ms); an equal-heads key is unchanged (2.12 vs 2.13 ms), so earlier
non-GQA timings stay valid. Affects any GQA model with flash attention, not only
Qwen. Evidence: QH-07ab.

Scope (checked 2026-11-22). The code is unchanged from upstream: `bench.py`
`SDPABackward` in `Phantora-Vanilla/Phantora-main` (301d9d9) is identical. It
is latent there: upstream's shipped `tests/perfdb/l40s/compute.csv` has four
SDPA keys (two forward, two backward), all with equal Q and K/V head counts
(`[2,16,1024,128]`, `[2,16,8192,16]`), so its published timings are not
affected. It triggers for any workload that passes grouped K/V heads to SDPA
natively (`enable_gqa=True`, fewer K/V heads than Q heads on a rank): GQA or
MQA models such as Llama 3, Qwen and Mistral when they do not repeat K/V
before the call. Workloads that repeat K/V to the query head count first never
reach it. Upstream's SDPA forward key already records `gqa`; only the backward
key and its profiler dropped it. Not specific to Qwen or to the hybrid fork;
our flash module is simply the first workload here to use native GQA.
