# Fused hybrid timing contract (key schema 1)

`gdn.csv` is optional for legacy databases. Measured rows have columns
`key,nanos,provenance`. Only positive nanoseconds are accepted. Provenance is JSON:
`gpu`, `torch`, `torch_revision`, `triton`, `fla`, `causal_conv1d`, `measurement`.
GPU must match the database manifest; all rows must share provenance. Supported
measurement methods are `cupti_kernel_sum` and `cuda_event_wall`. Schema 1 is bound
to the pinned hybrid runtime versions checked in `gdn.rs`; changing implementations
requires deliberate compatibility review rather than silently reusing timings.

The JSON key contains `version`, `op`, `direction`, `inputs`, `grad_enabled`,
`grad_output`, `options`. Each input is tensor metadata or null for convolution
bias. Tensor metadata records shape, strides, dtype and requires-grad. The incoming
backward gradient is keyed separately, including expanded/zero strides. Operations
are `causal_conv1d`, `l2norm`, `chunk_gated_delta_rule`; directions are `forward` and
`backward`. Options record activation, epsilon/output dtype, or scale/internal
QK-normalization, respectively. State, packing, extra options and higher derivatives
are unsupported. Core GDN's separate L2-normalization and head repetition remain
outside chunk timing when internal QK-normalization is false.

Rust uses the same typed parser for transported and CSV keys. Epsilon and scale
are canonicalized as `f64:` followed by sixteen hexadecimal IEEE-754 bits. A
profiler can decode these with `struct.unpack('>d', bytes.fromhex(text[4:]))[0]`.
JSON field order and alternate equivalent decimal spellings cannot create misses.
`tests/assets/qwen_hybrid_keys.json` is the shared Python/Rust protocol fixture.

Missing timings have no synthetic estimate. Discovery charges zero, marks results
INVALID, and writes `<db>.missing/gdn.missing.csv` with columns `key,status` and
status `missing`, plus `gdn.status.json`. With no DB configured the directory is
`gdn.missing`. Discovery artifacts cannot be loaded as measured timing rows.
Native on-demand real fused profiling is not implemented. Use the offline fused
profiler below rather than passing these keys to `bench.py`.

Native C translates the raw current CUDA stream handle before Rust sends the
semantic event. Server scheduling flushes buffered preceding Torch calls and uses
the existing stream dependency chain, including subsequent NCCL. Placeholder
allocations/gradients model orchestration, not numerical GDN or activation memory.
Linux compilation, native scheduling tests and real-kernel calibration remain
separate validation gates.

The surrounding GDN output norm follows the pinned Qwen reference's BF16 cast
before scale/gate, rather than delaying that cast until after FP32 scaling:
[Qwen3.5 reference](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
CPU tests distinguish this rounding behavior. They do not validate the complete
initialized Megatron model or real fused CUDA execution.

The offline fused profiler is `tests/perfdb/gdn_profile.py`. It requires the exact
patched torch revision, torch 2.9.1 built with CUDA **12.8**, Triton 3.7.0, FLA
0.4.0 and causal-conv1d 1.5.3.post1, plus the actual GPU name. Provenance includes
`cuda`. The candidate combination still needs its Linux import and kernel gates.
Run without PHANTORA, LD_PRELOAD or simulation aliases; stub libraries are rejected.

```bash
python3 tests/perfdb/gdn_profile.py --keys RUN/perfdb.missing/gdn.missing.csv \
  --out tests/perfdb/MATCHED_GPU --gpu-name 'ACTUAL GPU NAME'
python3 tests/perfdb/gdn_validate.py --gpu-name 'ACTUAL GPU NAME' \
  --out .cache/qwen-hybrid/real-fused-validation.json
```

Each argument has distinct storage and recorded shape/dtype/stride/leaf flags;
expanded incoming gradients preserve zero strides. Inputs are stable and q/k are
unit-normalized when internal normalization is disabled. One finite result with
correct output/all-gradient shapes and dtypes is checked outside measurement.
Backward builds its forward graph outside timing and repeatedly calls
`autograd.grad(..., retain_graph=True)`, including the fused backward's internal
recomputation. Default timing is CUPTI kernel sum to match ordinary device timing.
`--method cuda_event_wall` is an explicit alternative, with distinct provenance;
methods cannot be mixed. CPU-time policy is separate from either device timer.

Only positive u64 nanosecond measurements are committed. A single failed key preserves the old
CSV bytes, and `gdn.profile.json` records failures and uncommitted candidates.
Successful merges are atomic and retain prior measured entries. No equation or
placeholder fallback produces timing rows. The numerical validator compares tiny
independent causal convolution, FP32 sum-of-squares L2 and recurrent delta-rule
equations, including every differentiable argument. Its tolerances are explicit
CLI values. Every output and gradient must additionally pass relative L2 error
(default 0.03, `--relative-l2`); reports record those errors and tolerances, so a
zero vector cannot pass solely through absolute tolerance. Convolution reference
products/accumulation/bias/SiLU use FP32 before one cast to input dtype. Passing
these checks does not establish whole-model HF equivalence. Best-effort device
properties and driver/PCI identity are archived separately in profile reports;
the strict provenance still requires the exact software/CUDA/GPU/method pins.

Core's packed GDN projections and external L2-before-head-repetition differ from
HF's separate projections and internal L2 composition. Core evaluates BF16
`A_log.exp()` where the pinned HF reference uses `A_log.float().exp()`. These are
pending whole-model fidelity checks; the validator records them explicitly.
The real initialized trainer emits `HYBRID_STRUCTURE` global ownership and shared
norm/output-projection shape assertions before DDP. Actual forward/backward and
optimizer/TP/PP execution must still pass on GPUs; source/mock CPU checks cannot
prove that gate. Memory estimates remain **unvalidated**.
