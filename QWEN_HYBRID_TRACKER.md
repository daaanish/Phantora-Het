# Qwen hybrid training simulation: plan and evidence

Last updated: 2026-11-16 (America/Indiana/Indianapolis).

## Objective and current position

Simulate dense Qwen3.5-9B text-decoder training with TP, DP and PP using local
Megatron projections/MLPs and actual fused GDN kernel measurements. Add a clearly
labeled dense Next-style adaptation after the Qwen3.5 milestone is validated.

**Current phase:** QH-02 passed on `gpu01` via Apptainer. Real QH-03/03a
training smokes passed with the CUDA 12.8 assembler override. QH-07a Phantora
discovery completed both 0.8B TP2 ranks with exact fused counts; strict timing
coverage is incomplete. Docker is unverified.
**GitHub handoff:** QH-P01 passed. Implementation commit
`9adeef71f58c20b8210d387bd20d84405062d177` is published to `origin/main` and the
remote SHA was verified. Vanilla remains in its separate checkout; the user's
selected workflow uses main in Phantora-Het.
**Accuracy finding (QH-07j–m, 2026-11-16):** the 0.8B seq-128 gap is host
time, not missing GPU work. Real steps issue ~36,000 GPU activities of ~4 µs
each (~141 ms of compute per rank); the GPU sits empty waiting for the next
launch for most of the rest, and real step time stays ~0.9 s from seq 128 to
2048. At seq 8192 the GPU is the bottleneck and the partial replay is 2.540 s
vs 3.192 s real (−20.4%); at seq 4096 it is 1.018 s vs 1.357 s (−25.0%) and
at seq 16384 it is 7.411 s vs 9.075 s (−18.3%), in
line with the Llama/GPT depth sweep. Seq 32768 has no real result (OOM).
Consolidated table and CSV: `results/qwen35_08b_sequence_sweep/`.
**Attribution (QH-07r):** at seq 8192 the 670 ms traced gap is 350 ms GPU
compute not charged, 269 ms GPU empty between kernels (host), 51 ms all-reduce.
The 14 zero-activity keys are negligible.
**Next executable task for accuracy:** in order of size: map `contiguous`,
`cat`, `sub_`, `max` and the missing `_foreach_*` ops; accept the multi-argument
`aten::to` conversion form; stop double-charging `masked_fill`'s clone; profile
binary ops with distinct operand buffers; then decide how to model host gaps.
Do not tune bandwidth or treat CPU-on replay as a calibration fix. Seq 128 is
not a useful timing-validation point unless host launch time is modeled.
The original Core 0.13.1 SIF remains untouched.
Use `tests/docker/megatron/HYBRID.md` and record actual commands, versions,
results and logs here. Gate outcomes below distinguish Linux replay from real CUDA.
**Completion:** Megatron upgrade, single-GPU Qwen debug and real two-GPU 0.8B
training smoke passed on gpu01. The full Qwen simulation milestone is open;
fidelity, coverage and calibration are
separate gates. The gpu01 compatibility patch is applied in its isolated run and
recorded here; the current local changes have not been published to GitHub.

Requested roles: Sol Max parent owns planning, assumptions, review and this file;
Sol Medium subagent writes code and orchestrates implementation/tests. Keep work
in small reviewed batches. Preserve user changes and existing runtime defaults.

## Baseline and environment

- Initial repository commit: `301d9d976a13067cc0f5438912395a11efba5f8e` (`main`).
- Published implementation revision:
  [`9adeef71f58c20b8210d387bd20d84405062d177`](https://github.com/daaanish/Phantora-Het/commit/9adeef71f58c20b8210d387bd20d84405062d177),
  43 task files, 4,816 insertions and 73 deletions. A tracker-only follow-up records
  this actual publication evidence; it does not change the implementation revision.
- Pre-existing untracked file: `AGENTS.md`; leave it intact.
- Patched PyTorch gitlink: `bbb19f8c856fcf8d7c60267fecb976b4749e2cff`, branch
  `2.9.1-phantora`; submodule is not initialized locally.
- Official Qwen3.5-9B config revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`,
  resolved through the official Hugging Face model API; no weights downloaded.
- Reviewed upstream source revisions: Megatron Core tag `core_v0.16.0` resolves to
  `3bec9aa97dda898d16ff5a89bac0ed2b6682b172`; FLA tag `v0.4.0` links to
  `2e0f6c31f9db50f010cb157c712c048b8a3a89d2`; causal-conv1d tag `v1.5.3.post1`
  resolves to `52949d0891770c65243ed24db8abb66ef37741b7`. These identify reviewed
  sources; installed wheel/source equivalence still needs the Linux runtime gate.
- Qwen equation reference pinned to Transformers
  `bd15bc95a89e728bbc1224084eb3b5829428c353`. This reference source is separate
  from the retained runtime's transformers 4.56.2; we do not instantiate the HF model.
- Existing image: CUDA 12.8, Python 3.11.9, patched torch 2.9.1,
  megatron-core 0.13.1, flash-attn 2.7.3, transformers 4.56.2,
  numpy 1.26.4, triton 3.7.0. These describe source pins, not a verified local build.
- Local machine: macOS arm64, Python 3.14.2, stock CPU torch 2.10.0;
  no CUDA or patched tracer, no discoverable cargo/rustc or Megatron package.
- Docker client exists; a subsequent elevated read-only check confirmed the
  daemon is unavailable. This was not a Docker build test.
- On 2026-11-14 the user authorized `gpu01` for the Megatron upgrade. Verified
  SSH access to `gpu01.pronet.rcac.purdue.edu` and Rocky Linux 9.6 x86_64; 64 CPUs,
  354 GiB RAM available, 528 GiB free in local `/tmp` at preflight. Docker absent;
  Apptainer 1.4.2-1.el9 and Podman available. Two NVIDIA RTX PRO 6000 Blackwell
  Max-Q (sm_120, 97,887 MiB each), driver 570.195.03; both idle at preflight, not
  reserved. Existing `/home/dathar/Phantora/Phantora/phantora-gpu01.sif` (8.37 GB)
  contains Core 0.13.1, Torch 2.9.1 with CUDA 12.8 / sm_120 and a callable tracer,
  Transformers 4.56.2, Triton 3.7.0, NVCC 12.8.61, cbindgen 0.29.4 and stable
  Rust/Cargo 1.99.1 when their image paths are supplied. FLA and causal-conv1d
  are absent. Torch reports `git_version='Unknown'`; the image's PyTorch `.git`
  pointer targets an absent submodule git directory. The adjacent host checkout
  is at the pinned SHA; that observation alone does not establish what was built.
- Publication destination: `https://github.com/daaanish/Phantora-Het.git`, branch
  `main`. On 2026-11-07, a fresh fetch confirmed local HEAD, `origin/main` and
  `FETCH_HEAD` all equal the baseline above, with no divergence.
- Publication scope: Qwen source, tests, Docker changes, runbooks and this tracker.
  The pre-existing untracked `AGENTS.md` stays local. Raw evidence under ignored
  `.cache/qwen-hybrid/` also stays local; its results are summarized below, and a
  Git clone will not contain those log files. No measured GPU timing DB exists yet.

Prepared Linux checkout command (not executed on a Linux host in this session):

```bash
git clone --branch main --recurse-submodules https://github.com/daaanish/Phantora-Het.git
cd Phantora-Het
```

For an existing checkout, use `git pull --ff-only origin main` followed by
`git submodule update --init --recursive`. Continue with the ordered validation
gates in `tests/docker/megatron/HYBRID.md` and record new results here.

## Tasks and gates

States: planned / in progress / implemented / passed / failed / pending environment.
The four evidence columns are independent; implementation does not imply validation.

| ID | Deliverable and dependencies | Owner | Implementation | Local tests | Linux tests | GPU validation | Acceptance gate |
|---|---|---|---|---|---|---|---|
| QH-00 | Central tracker and baseline | Parent | implemented | n/a | n/a | n/a | Plan, decisions, assumptions and evidence recorded |
| QH-01 | Config, global pattern, TP/DP/PP and batch contracts; QH-00 | Medium | implemented | passed (10 tests, hardened guards) | pending environment | n/a | Positive/negative CPU tests; exact effective batch |
| QH-02 | Isolated pinned hybrid runtime; QH-01 | Medium | Core 0.16.0 in durable isolated gpu01 runtime; Docker separate | 62 CPU tests, AST and whitespace pass | passed on Apptainer: native build/7 tests, exact probes and no-miss Llama A/B; Docker build pending | Blackwell Llama record and GPU-backed import gate pass; model/kernel checks in QH-03/06 | Exact stack imports/signatures/tracer pass; same recorded Llama workload completes without timing misses on default and upgraded runtimes |
| QH-03 | Local hybrid layer spec, debug and 9B presets; QH-01/02 | Medium | implemented | CPU equation/wiring/CLI tests passed; BF16 correction checked | exact runtime and debug structure pass on Apptainer | single-GPU debug 4 iterations passed with CUDA 12.8 ptxas; numerical fidelity and distributed GPU tests pending | Correct global PP ownership and reference forward/backward |
| QH-03a | Official Qwen3.5-0.8B text preset and real gpu01 TP=2 smoke; QH-02/03 | Medium + parent review | implemented | 11 config / 63 total hybrid tests passed | passed on exact QH-02 Apptainer runtime | passed on both Blackwell GPUs | Both ranks construct all 24 layers (18 GDN, 6 full), use tied embeddings, and complete four real BF16 steps, TP=2/DP=1/PP=1; fresh weights, synthetic seq 128/MB 1/GBS 8 |
| QH-04 | Autograd-safe fused boundaries, native events and timing DB; QH-01/03 | Medium | implemented | passed (28 cumulative CPU tests) | 3 GDN Rust unit tests passed; live event transport/stream ordering pending | pending environment | Graph, serialization, key, stream ordering and no-double-count tests |
| QH-05 | Coverage/discovery/strict status and all-host runner; QH-04 | Medium | implemented | passed (41 cumulative CPU tests) | 4 coverage Rust unit tests passed; integration failures/Compose pending | n/a | Missing events/timings and crashed ranks fail without hanging |
| QH-05a | Observe device-only CUDA factories omitted by pinned tracer; QH-05/source review | Medium | implemented | passed (61 cumulative CPU tests with review fixes) | CUDA/TLS/native tests pending environment | n/a | Per-rank observer active; allocation observations classified and initialized/unknown compute invalidates strict status |
| QH-06 | Actual fused profiler and provenance checks; QH-04/05/05a | Medium | implemented | passed (61 cumulative CPU tests) | pending environment | pending environment | Separate fwd/bwd measured; invalid or mismatched timings rejected |
| QH-07 | Debug and full-depth replay matrix; QH-02 through 06/05a | Medium + parent review | executable commands prepared; execution pending | 10-case command preparation passed | pending environment | pending environment | Complete per-config coverage and expected computation/communication |
| QH-07a | One matched Qwen3.5-0.8B real vs Phantora case on gpu01; QH-02/03a/04/05/05a/06 | Medium + parent | Apptainer discovery launcher exercised | existing checks | both ranks completed discovery with exact fused counts; strict invalid (timing keys and ATen coverage) | no 0.8B profiling started | Identical TP2/DP1/PP1 BF16 seq128 MB1 GA8, exact rank lifecycle and GDN counts, no misses, archived placement/network/CPU assumptions, compare max-rank steady-state step time |
| QH-07b | User-requested partial 0.8B real vs replay comparison; QH-07a | Medium + parent | completed, no strict changes | checked unrounded max-rank aggregation | provisional replay complete: 0.095299 s/step vs 0.897536 s real, −89.38%; native strict invalid | measured 275 positive compute, 32 copy and all 6 fused keys | Exact matched workload and complete rank/GDN counts; show remaining misses and unsupported ops explicitly, no strict validation or calibration claim |
| QH-07c | Layer-boundary coverage audit and first opt-in op fixes; QH-07b | Medium + parent | ledger, `square`/`exp_out` mappings and profiler thunks implemented | 64 Python tests pass | 14 Rust tests pass (one pre-existing ignored), `make dist` and matching discovery pass | five newly modeled keys await measurement | Preserve default mappings, discover supported keys without double counting, then measure those exact keys |
| QH-07d | Measure the five QH-07c keys and repeat the single 0.8B comparison; QH-07b/c | Medium + parent | completed; five rows merged without changes to prior keys | checked rank-max aggregation | matching replay complete: 0.096515 s vs 0.095020 before on same native build; strict invalid | all five positive CUPTI timings measured on idle gpu01 Blackwell | Preserve every prior DB entry; all five hit; compare pre/post and real on identical max-rank post-warmup math, no strict claim |
| QH-07e | Diagnose the 801 ms/step 0.8B discrepancy; QH-07d | Medium + parent | diagnostic passes complete | checked saved unrounded clocks | CPU-on same DB 3.326479 s/step; sensitivity only | repeated real 0.870984 s/step, one profiled step and 828 paired TP events analyzed | Gap reproduced; peer-arrival waiting established under profiler, source of real rank lag awaits low-overhead evidence |
| QH-07f | GPT/Llama-style matched phase diagnostic; QH-07d/e | Medium + parent | matching scratch-only Megatron timers complete | compared both-rank phase records | CPU-off phase pass completed 96.213 ms/step, strict invalid | real phase pass completed 907.784 ms/step | Forward/backward accounts for 767.20 of 811.57 ms instrumented gap; nested timers diagnostic only |
| QH-07g | Check whether collective peer skew survives lower-overhead tracing; QH-07e/f | Medium + parent | completed, scratch-only | common time base and matching collective sequence checked | reused saved replay result | CUDA-only capture: 828 pairs, 455 µs median starts versus 0.60 µs finishes; +19.4% overhead | Skew persists with lighter trace; earlier rank flips, source of unprofiled lag not proven |
| QH-07h | Lightweight host-side collective issue clocks; QH-07g | Medium + parent | completed, scratch-only timestamps | 828 signatures match, common monotonic clocks | reused CPU-off baseline | 882.105 ms warm step (+1.31% vs prior same-index control); 139 µs median API entry skew | Host issue skew exists without a GPU profiler; high tail concentrated early, source and impact on step gap not proven |
| QH-07i | Matched per-layer timing to localize forward/backward gap; QH-07f/h | Medium + parent | scratch-only capture complete | marker completeness/order and 2,500 µs stub unit probe pass | same-build/DB CPU-off replay complete, 14 existing compute misses | exact matched real run complete | Both layer types undershot; 87–95% of sampled gap inside block envelopes, includes exposed waits |
| QH-07j | Classify GPU idle time in saved QH-07g traces and test host-bound hypothesis with a real sequence-length sweep; QH-07g/i | Parent | completed, scratch-only analysis and unmodified trainer | launch/activity pairing by correlation ID checked on both ranks | n/a | real seq 128/512/2048: 902.0/894.3/915.4 ms | Step time flat across 16x tokens; pace-setting rank's GPU empty with no launch pending for 73% of the profiled step |
| QH-07k/l | Find GPU-bound 0.8B sizes and the gpu01 memory limit; QH-07j | Parent | completed, unmodified trainer with size flags only | rank-max aggregation over steps 1–3 | n/a | seq 2048 MB 2/4/8 GA 8: 1.226/2.197/4.476 s; seq 4096/8192/16384 MB 1: 1.357/3.192/9.075 s; seq 32768 OOM | Step time doubles with tokens from 8k tokens/microbatch; seq 16384 peaks at 95.6 of 97.9 GiB |
| QH-07m | Matched 0.8B seq-8192 real vs replay; QH-07d/l | Parent | completed; new rows only in a separate DB copy | checked rank-max aggregation | discovery and replay complete on QH-07c native build; strict invalid, 14 compute keys missing | 161 compute (147 positive), 7 copy and 6 fused keys measured on one idle Blackwell | Replay 2.540 s vs real 3.192 s (−20.4%); no strict or calibration claim |
| QH-07w | Opt-in Qwen3.5 MTP layer (`--mtp-layers`, `run.sh --mtp`); QH-03a | Parent | implemented: Megatron `MultiTokenPredictionBlock` with Qwen RMSNorm enorm/hnorm/final norm, `eh_proj` 2h→h, one gated full-attention + SwiGLU layer, shared tied LM head; default 0 leaves every earlier configuration unchanged | 72 hybrid CPU tests pass in the gpu01 image (68 at HEAD 9383f7c + 4 MTP tests) | pending: real Megatron construction and discovery with MTP | pending: real and replay with `--mtp 1` | Both ranks build the MTP block and pass the structure check; real and replay complete; MTP cost is recorded as new columns next to the no-MTP numbers |
| QH-08 | Hardware-matched calibration; QH-07 | Medium + parent review | report/compare tools prepared; execution pending | report math/mismatch cases passed | pending environment | pending environment | At most 10% steady-state iteration error in each accepted case |
| QH-09 | Dense Next-style adaptation; QH-08 | Medium | planned | planned | pending environment | pending environment | Explicit dense adaptation; separate validated configurations |
| QH-P01 | Publish the reviewed local batch for Linux handoff; QH-00 and local batch | Medium Git execution + parent review | passed: implementation pushed to origin/main | preflight/staged checks and remote SHA verification passed; existing CPU evidence retained | n/a | n/a | Reviewed task files committed and pushed to origin/main; remote commit verified; user-local instructions/logs preserved |

Before each task: record scope, dependencies, assumptions and acceptance checks.
After each test/change: append command, result and artifact location; update the
task's relevant evidence column and next action. Record failed attempts as well
as successes. Do not change a pending environment gate to passed without evidence.

### Results documentation principles

1. **Never overwrite a measurement.** When the method changes, keep the earlier
   numbers under a named condition and add the new ones as a new column or row.
2. **Every number carries its conditions.** Model, configuration, timing window,
   aggregation rule, date and the run directory it came from.
3. **One change at a time, and name it.** State what changed, what is expected
   to move and what should stay the same, then check both.
4. **Separate measured from inferred.** Label estimates, fits and guesses as
   such, next to the number.
5. **Record failures and invalid runs too.** Out-of-memory runs, failed attempts
   and runs flagged invalid stay in the record with the reason.
6. **Document as you go, not at the end.** Update the results files and this
   tracker when each result lands.
7. **Record every simulator assumption found to differ from reality.** Add it to
   `results/phantora_assumptions_vs_reality.md` with where it lives
   (original Phantora or this fork), the evidence and the measured effect.

## Implementation decisions

The [migration record](tests/docker/megatron/HYBRID.md#migration-record) records
what existed, broke and changed in original Phantora, the Megatron update and
the Qwen3.5 GDN run.

1. Use the released Qwen3.5-9B **text decoder** as the first fidelity target.
2. Retain local Megatron projection/MLP implementations and add fused FLA GDN;
   Transformer Engine is outside the first milestone.
3. Preserve the existing distributed optimizer, overlapping gradient reduction
   and parameter gather; DP traffic includes reduce-scatter/all-gather.
4. Build the 3:1 layer pattern globally before PP slicing. Report actual local
   ownership; never restart the pattern on each stage.
5. Hybrid runs honor exact gradient accumulation. Legacy PP's
   `max(accumulation, PP * 4)` behavior must not alter hybrid workloads.
6. Begin with independent causal-conv1d, l2norm and chunk_gated_delta_rule
   forward/backward boundaries. Ordinary surrounding work remains observable.
7. Native semantic events carry operation metadata and use translated stub stream
   IDs. Existing bare cuda_add_latency is insufficient as a timing-key protocol.
8. Keep discovery available but label incomplete estimates invalid. Strict mode
   aggregates all ranks and returns nonzero without abandoning synchronous workers.
9. Validate time and throughput first. Peak activation memory remains unvalidated.
10. Use an opt-in hybrid runtime and preserve default image/launcher/DB behavior.
11. Initial implemented CLI is `--hybrid-preset debug|9b`, with optional explicit
    `--global_batch_size` and `--data_parallel_size`. Presets reject conflicting
    architecture overrides; TP/PP, sequence and batch settings remain explicit.
12. Build hybrid image with `docker build --target phantora-hybrid -t phantora:hybrid .`.
    The final `phantora-default` alias preserves an untargeted build's old stack.
    This build command has not been run in the current environment.
13. Native contract selected for QH-04: append a semantic `GdnCall` carrying
    translated stream identity and a versioned JSON key. Canonical typed keys
    distinguish measured `gdn.csv` entries from discovery-only missing artifacts.
14. QH-06 profiling will default to `cupti_kernel_sum`, matching the inherited
    ordinary device-timing policy. An explicit CUDA-event wall-time alternative
    must retain distinct provenance and cannot be mixed into that table.
15. Performance reporting will discard warmup iteration 0, require every rank's
    complete unrounded duration series, take the maximum rank time per iteration,
    then average those steady-state maxima. Throughput uses actual global tokens
    over that duration. Hardware comparisons must match workload, software and
    GPU/host placement/topology, with the network/CPU-time assumptions visible.
    The matched-case CPU policy is an explicit pair: replay ignores CPU time by
    default, while hardware measurements are real wall-clock elapsed time. Report
    both and the effective policy for each kind; do not label hardware as CPU-free.

## Assumptions and simplifications ledger

| ID | Classification | Assumption or limit | Consequence / revisit trigger |
|---|---|---|---|
| A01 | verified user choice and available hardware | Dense Qwen3.5 first; use the two Blackwell devices on gpu01 for initial validation | One- and two-device cases are possible on gpu01; the eight-GPU real calibration case needs a matching larger allocation |
| A02 | deliberate simplification | BF16, fixed-length text, synthetic data | No numerical training-quality or dataset-throughput claim |
| A03 | deliberate simplification | CP=1, EP=1, no MoE | World size is TP*DP*PP; reject incompatible hybrid flags |
| A04 | deliberate simplification | No SP, virtual PP, recomputation, packing, caches, vision or MTP initially | Reject unsupported combinations rather than silently approximate. 2026-11-17 (QH-07w): MTP is now supported as an opt-in (`--mtp-layers 1`, the released depth); default stays off so earlier numbers keep their meaning |
| A05 | verified code behavior | Distributed optimizer and overlap are enabled | Preserve both DDP and optimizer settings; validate actual collectives |
| A06 | provisional dependency choice | Core 0.16.0, FLA 0.4.0 and its separate `fla-core==0.4.0` kernels, causal-conv1d 1.5.3.post1 | Verify imports/signatures/runtime before claiming support; record revisions |
| A07 | verified upstream/code fact | Stock hybrid builder requires TE, current repo uses local specs | Construct an equivalent local spec; do not pretend to model TE performance |
| A08 | verified upstream fact | Full-attention output gating and Qwen-specific norms/partial RoPE are required | Pattern alone is not an architecture match; verify reference equations |
| A09 | deliberate simplification | Simulation values/placeholder gradients are not numerical GDN | Test graph connectivity separately from real numerical correctness |
| A10 | deliberate simplification | Fused saved activations and workspace are not initially reproduced | Peak VRAM and capacity predictions are unvalidated |
| A11 | verified code limitation | Unmapped torch ops can be dropped; driver Triton launches have no timing | Explicit semantic events and strict coverage are required |
| A12 | verified code limitation | cuda_add_latency and simulator clock use microseconds; DB stores ns | Convert once consistently; existing clock quantization remains a limit |
| A13 | verified code limitation | Current runner mainly returns host-1 status; server has no all-rank finalization | Add bounded all-host waits and machine-readable final run status |
| A14 | verified code implication | Depth/TP/DP/PP can change optimizer tensor-list keys and bucket layout | Discover full configurations; never assume four-layer DB completeness |
| A15 | provisional accuracy target | <=10% error per hardware-matched steady-state timing case | Diagnose mismatches; no arbitrary timing multipliers or averaging away failures |
| A16 | inherited simulator assumption | Existing network model and CPU-time policy remain in force | Include topology and CPU-time settings in each report; no universal accuracy claim |
| A17 | verified scope distinction | Released Qwen3-Next reference is MoE | Any later dense variant must be labeled an adaptation, not the released model |
| A18 | measurement requirement | Timing keys are specific to rank-local work, implementation and GPU | Reject failed/zero/stale/mixed-provenance timings; profile backward separately |
| A19 | validation default | Report tokens from actual batch and max-rank steady-state duration | Do not use rank-0 alone or silently changed accumulation for scaling comparisons |
| A20 | verified source implication; distributed test pending | GDN output-norm weight is shared across head shards; Core 0.16's default TP gradient finalizer does not include it when SP is off | Add/test the necessary TP SUM for this shared gradient; include its real communication in timing |
| A21 | deliberate implementation choice | Local surrounding norm/MLP/attention work uses unfused PyTorch paths; fused bias/softmax paths disabled for hybrid observability | Timings describe this backend, including local full attention; do not claim TE or fully fused attention performance |
| A22 | verified reference correction; fixed and CPU checked | Pinned Qwen GDN output RMSNorm normalizes in FP32, casts to input dtype before multiplying its scale, then applies the FP32 gate/final cast | Earlier FP32-through-gate assumption was incorrect; corrected local norm and BF16 roundoff regression pass. Actual GPU/model comparison remains pending |
| A23 | verified import requirement | FLA can probe the driver at import time | Replay must install required simulation aliases before upstream GDN imports, without initializing real FLA/Triton |
| A24 | deliberate provenance requirement | GDN measured entries require the pinned torch revision and candidate package versions | Run the fused profiler in the matching hybrid environment; stock local torch is for CPU checks, not accepted calibration data |
| A25 | source issue under investigation in QH-05 | Legacy scalar-pow timing ignores exponent; new norm backward can require exponent 3 | Do not conflate materially different hybrid kernels; preserve exponent-2 keys where applicable and record compatibility changes |
| A26 | inherited profiling simplification | Existing ordinary-op profiler aliases operands with the same element count/dtype within an operation | Can alter memory traffic, especially optimizer lists; retain visibly and require hardware calibration rather than assuming accuracy. New fused profiling should allocate distinct inputs |
| A27 | inherited timing-key limitation | Ordinary TorchCall tensor keys contain shape/dtype, not full strides or alias relationships | New fused keys include strides; ordinary-layout fidelity remains part of hardware calibration and must not be described as validated from CPU tests |
| A28 | inherited provenance limitation | Legacy ordinary timing tables record GPU identity but do not enforce the complete software provenance used by the new GDN table | Accepted hybrid evidence must also archive the exact ordinary-profiling runtime; existing DB contents alone do not establish this |
| A29 | verified source issue; opt-in correction implemented | Legacy replay accepts zero ordinary timing entries as cache hits | Hybrid startup removes loaded zero compute/transfer/flash rows except narrow proven no-ops, causing used placeholders to be rediscovered and invalidated. Native regressions prepared, not executed; old replay defaults preserved |
| A30 | source verified; execution pending | Pinned custom tracer emits only outermost ATen operations with a CUDA tensor input (convolution_backward is a special exception) | Do not charge both parent and child. Contiguous/clone/square need their outer boundary modeled accurately; they remain unsupported pending layout-aware trace evidence |
| A31 | deliberate implementation difference; numerical audit pending | Upstream Core uses one packed GDN input projection and external L2 normalization before repeating heads; the pinned HF reference uses separate projections and internal normalization after repetition | Architecture dimensions match, but kernel composition/timing describes the selected local Core backend. Whole-model reference tests must account for the mapped projection layout and precision rather than assuming identity |
| A32 | verified precision difference; revisit before numerical fidelity gate | Core 0.16 computes `A_log.exp()` in parameter dtype, while the pinned HF reference explicitly casts `A_log` to FP32 first | Current BF16 Core gate math has different rounding. Do not claim whole-model numerical equivalence to HF; quantify/correct this in the real-model reference gate before declaring that gate passed |
| A33 | provenance requirement implemented; actual hardware evidence pending | The selected CUDA runtime is 12.8 in addition to pinned torch/FLA/Triton/causal-conv versions | Python/native fused provenance now require torch's CUDA build version; archive driver/device details during actual profiling. GPU name alone does not establish identical operating conditions |
| A34 | source verified coverage gap; QH-05a correction implemented/CPU checked | Pinned tracer discards device-only factories before server parsing, so native `cuda_argument_index` alone cannot observe them | Added opt-in dispatch observer and required per-rank activation/deactivation. Unknown/initialized CUDA factories invalidate status; allocations/native transfers are classified separately. Actual CUDA/backward-thread TLS visibility remains pending Linux |
| A35 | provisional numerical test policy | BF16 reference comparisons need both rounding-aware tolerances and a relative norm check | A loose absolute tolerance must not allow zero output/gradients to pass. Correct convolution reference accumulation to FP32 and record actual GPU errors before accepting numerical validation |
| A36 | verified measurement distinction | Real `time_pair()` uses wall-clock elapsed time, while default replay ignores CPU time | Calibration compares this known approximation to real iteration time. Record both policies as an explicit pair; requiring one identical single-policy label would misrepresent the hardware measurement |
| A37 | deliberate initial runner scope | One homogeneous GPU timing DB and uniform virtual GPUs per host are selected for each run | Mixed GPU types/unequal per-host GPU counts are not implemented by this runner; actual calibration must match the chosen placement |
| A38 | inherited network model, uncalibrated here | Runner reuses two-spine/two-host-per-rack topology, 800 Gbps host/rack uplinks and 2880 Gbps loopback speed from the existing template | These are modeled rates, not measurements of allocated hardware; archive the full generated netconfig and investigate network mismatch during calibration |
| A39 | verified host/runtime difference | gpu01 uses Apptainer and sm_120; the default Docker build targets sm_80/sm_90 | Use the host's Blackwell runtime for the first Linux gate; log SIF vs Docker differences and leave the Docker build status independent |
| A40 | verified provenance blocker; exact wheel built on gpu01 | Existing gpu01 SIF has the tracer and sm_120 but Torch `git_version='Unknown'` because its copied `.git` pointer is dangling; the Docker build also copies just the submodule subtree without its parent Git metadata | Use the separately verified wheel for exact runtime/Qwen evidence; the original SIF and Docker build remain separate. Do not accept fused timing provenance from the original SIF Torch |
| A41 | verified release/build context issue | FLA 0.4.0 publishes the kernels in the required `fla-core==0.4.0` distribution; the original hybrid `--no-deps` install omitted it. `.dockerignore` also omits tests/, including files required by the hybrid target | Explicitly pin both FLA distributions and include only required helper files in the image context. Check actual installed metadata on gpu01 separately from Docker build validation |
| A42 | verified 0.16 runtime and coverage change | Core 0.16 uses object gather for optimizer setup and adds a scalar offset in unfused SwiGLU even when configured as zero | Preserve real optimizer metadata through CPU Gloo in simulation; keep timing of the add. Old L40S DB is incomplete for the upgraded path. Record/compare a separate Blackwell DB; never mix rows between GPUs |
| A43 | verified gpu01 toolchain requirement | Triton 3.7.0 bundled Blackwell ptxas 13.1 failed binary load on the observed 570.195.03 driver; the same workload with image ptxas 12.8 passed | Set `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` only in this runtime. Archive driver and effective assembler with fused measurements; schema 1 does not enforce this distinction, so do not mix assembler environments. Revisit on another driver/GPU |
| A44 | verified 0.8B configuration; initial workload scope | Official 0.8B has 24 layers (18 GDN/6 full), 2 KV heads and tied token embeddings | TP=2 can take one KV head per rank; preserve tying only where configured. Initial run is fresh weights, synthetic text, short context, no vision/MTP or checkpoint loading |

## Workload and validation matrix

Start with a four-layer debug model and fixed sequence/local microbatch. Use global
batch 8 with microbatch 1. Repeat full-depth discovery before accepting the 9B preset.
The debug fixture deliberately uses hidden size 256, FFN 768, vocabulary 1024,
four full-attention heads / two KV heads of dimension 64, four GDN key heads /
eight value heads of dimension 128, and sequence length 128. These are fixture
reductions, not the released model's dimensions.

| TP | DP | PP | Accumulation | Virtual ranks |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 8 | 1 |
| 2 | 1 | 1 | 8 | 2 |
| 1 | 2 | 1 | 4 | 2 |
| 1 | 1 | 2 | 8 | 2 |
| 2 | 2 | 2 | 4 | 8 |

Include a PP partition crossing a four-layer pattern boundary. Check negative
configuration cases, every differentiable fused input's gradient shape/presence,
PHANTORA guards, fwd/bwd key distinctions, DB round trips, current-stream ordering,
missing-key discovery, unknown-compute failure, dropped-event detection, rank
failure timeout, legacy replay regressions and actual fused numerical/reference tests.
Real distributed calibration needs the corresponding real GPU count and topology;
one GPU can provide local kernel measurements but not multi-GPU ground truth.

## Evidence log

| Date | Command / inspection | Result / artifact |
|---|---|---|
| 2026-10-29 | Planning: ast.parse over existing tests/**/*.py | 8 Python files parsed without syntax errors; capability baseline only |
| 2026-10-29 | Planning: stock torch CPU square/sum backward smoke check | Passed on torch 2.10.0; not a hybrid or GDN test |
| 2026-10-29 | git status / git log / git submodule status | Baseline commit above; only pre-existing AGENTS.md untracked; PyTorch uninitialized |
| 2026-10-29 | Source review by requested Sol Medium subagent and parent | Confirmed FFI units, missing-op behavior, Triton stub limits, PP batch change, runner limitations |
| 2026-10-29 | `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_qwen_hybrid_config.py -v` (Medium) | 10 tests passed: global pattern, crossing PP ownership, fixed-batch matrix, invalid modes/shards and architecture fields; parent reviewed sources |
| 2026-10-29 | `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_qwen_hybrid*.py' -v` (Medium) | 19 tests passed; [log](.cache/qwen-hybrid/qh02-qh03-cpu.log). Includes norm equations, actual shared-gradient shard sum, test-double spec/finalizer wiring, CLI invariants |
| 2026-10-29 | AST parsing / stock-torch helper import / `git diff --check` (Medium) | Passed; logs [parse](.cache/qwen-hybrid/qh02-qh03-parse.log), [helper import](.cache/qwen-hybrid/qh02-qh03-stock-torch.log), [diff](.cache/qwen-hybrid/qh02-qh03-diff-check.log). Test doubles do not validate installed Megatron APIs |
| 2026-10-29 | Tagged source download attempt using curl (Medium) | DNS unavailable; used web source inspection instead. No Docker build/native/GPU execution was attempted |
| 2026-10-29 | `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_qwen_hybrid*.py' -v` (Medium, QH-04) | 28 tests passed; [CPU log](.cache/qwen-hybrid/qh04-cpu.log), [AST parse](.cache/qwen-hybrid/qh04-parse.log), [diff check](.cache/qwen-hybrid/qh04-diff-check.log). Tests cover placeholder graph/gradients, no-grad/partial-grad cases, aliases, transport guards and Python key fixture |
| 2026-10-29 | `cargo test -p phantora gdn` (Medium) | NOT RUN: exit 127, cargo absent; [log](.cache/qwen-hybrid/qh04-native-tests.log). Rust key/DB/transport tests prepared but not passed |
| 2026-10-29 | Elevated read-only Docker version and Rust-path checks (Medium) | Docker daemon unavailable and usual Rust installations absent; [Docker log](.cache/qwen-hybrid/qh04-docker-version.log), [Rust availability](.cache/qwen-hybrid/qh04-rust-availability.log). No approval rejection; no build attempted |
| 2026-10-29 | `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_qwen_hybrid*.py' -v` (Medium, QH-05 attempt 1) | 36 tests passed; [log](.cache/qwen-hybrid/qh05-cpu-attempt-01.log). Includes real tiny worker failure/timeout cleanup, status identity/rank checks and CPU ATen thunks. Native simulator lifecycle remains unexecuted |
| 2026-10-29 | QH-05 CPU attempt 2 and AST check (Medium) | 36 tests and 15-file AST check passed before subsequent review fixes; [tests](.cache/qwen-hybrid/qh05-cpu-attempt-02.log), [AST](.cache/qwen-hybrid/qh05-parse-attempt-01.log). Updated code needs a fresh gate |
| 2026-10-29 | QH-05 CPU attempt 3 (same unittest command, Medium) | 40 tests passed; [log](.cache/qwen-hybrid/qh05-cpu-attempt-03.log). Includes BF16 cast-before-scale, metadata-only dependency checks without GPU imports, wrong torch revision and DB GPU identity checks |
| 2026-10-29 | Stock CPU norm/gate operator diagnostic (Medium) | [Audit](.cache/qwen-hybrid/qh05-stock-cpu-op-audit-attempt-01.log). Confirmed `neg`/`div_` need mappings; unresolved copies/composites remain strict failures, not hidden exemptions. This does not establish patched-tracer coverage |
| 2026-10-29 | QH-05 final CPU gate and supporting checks (Medium) | 41 tests passed; [tests](.cache/qwen-hybrid/qh05-cpu-attempt-04.log). Sixteen Python files parsed, runner help and tracked diff whitespace checks passed: [AST](.cache/qwen-hybrid/qh05-parse-attempt-02.log), [CLI](.cache/qwen-hybrid/qh05-runner-help-attempt-01.log), [diff](.cache/qwen-hybrid/qh05-diff-check-attempt-02.log). Native zero-row/status/parser regressions remain pending Linux |
| 2026-10-29 | QH-06 CPU attempt 1 (Medium, same discovery unittest command) | 50 tests passed; [log](.cache/qwen-hybrid/qh06-cpu-attempt-01.log). Covers key/provenance/atomic failure retention, strided distinct inputs, backward graph setup, finite-result rejection, independent recurrent gradients and max-rank reporting. Test-double timing is not a measurement |
| 2026-10-29 | Scoped elevated read-only `curl --connect-timeout 10 --max-time 20 -fsSL https://raw.githubusercontent.com/QDelta/pytorch/bbb19f8c856fcf8d7c60267fecb976b4749e2cff/torch/csrc/profiler/standalone/function_tracer.cpp` (parent) | Passed; reviewed full pinned source after web/raw cache failures. Confirmed outermost-ATen suppression and device-only factory omission; no repository mutation or build. Source URL below |
| 2026-10-29 | QH-06 CPU attempts 2 through 5 (Medium) | 52/52/53/53 tests passed; [attempt 2](.cache/qwen-hybrid/qh06-cpu-attempt-02.log), [3](.cache/qwen-hybrid/qh06-cpu-attempt-03.log), [4](.cache/qwen-hybrid/qh06-cpu-attempt-04.log), [5](.cache/qwen-hybrid/qh06-cpu-attempt-05.log). No failed attempts reported. Native/GPU execution still absent |
| 2026-10-29 | QH-06 AST/help/whitespace and matrix preparation (Medium) | 19 files parsed, profiler/report help and diff checks passed; logs under [.cache/qwen-hybrid](.cache/qwen-hybrid/). Ten executable debug/9B commands prepared without Docker: [matrix log](.cache/qwen-hybrid/qh06-matrix-attempt-01.log), [manifest](.cache/qwen-hybrid/qh06-matrix-prepared/matrix.json) |
| 2026-10-29 | QH-05a/review CPU attempt 1 (Medium) | 58 tests passed; [log](.cache/qwen-hybrid/qh05a-cpu-attempt-01.log). Actual stock-CPU dispatch tests cover allocation vs initialized factories, transfer classification and no duplicate tensor-op observation. Reporter/native lifecycle, numeric rejection and shape guards strengthened; actual CUDA/TLS behavior remains pending |
| 2026-10-29 | Scoped elevated read-only `curl` of pinned Core `attention.py`, filtered to projection/norm definitions (parent) | Verified gated QKV rows `(2*Qheads+2*KVheads)*head_dim/TP`, row projection `[hidden,Qheads*head_dim/TP]` and Q/K norm dimensions. Reference: https://github.com/NVIDIA/Megatron-LM/blob/3bec9aa97dda898d16ff5a89bac0ed2b6682b172/megatron/core/transformer/attention.py |
| 2026-10-29 | QH-05a/review CPU attempts 2 through 5 (Medium, same unittest discovery command) | 61 tests passed in each preserved attempt; [2](.cache/qwen-hybrid/qh05a-cpu-attempt-02.log), [3](.cache/qwen-hybrid/qh05a-cpu-attempt-03.log), [4](.cache/qwen-hybrid/qh05a-cpu-attempt-04.log), [final 5](.cache/qwen-hybrid/qh05a-cpu-attempt-05.log). Final includes representative factory argument metadata; no failing test attempts reported |
| 2026-10-29 | Final AST/whitespace and report CLI checks (Medium) | 22 Python files parsed and tracked diff check passed; [AST](.cache/qwen-hybrid/qh05a-parse-attempt-02.log), [diff](.cache/qwen-hybrid/qh05a-diff-check-attempt-03.log), [report help](.cache/qwen-hybrid/qh05a-report-help-attempt-01.log). Rust formatting/compilation remain unrun because the toolchain is unavailable |
| 2026-11-07 | Publication preflight: git status, remote and baseline inspection (parent) | Still on main at 301d9d976a13067cc0f5438912395a11efba5f8e; 15 modified tracked files plus new Qwen files, tracker and pre-existing AGENTS.md. No commit or push had occurred. Medium begins remote/evidence/diff checks; this is not a new runtime validation |
| 2026-11-07 | Publication preflight: fresh origin/main fetch, ancestry comparison, `git diff --check`, new-file whitespace/content inspection and saved-log review (Medium) | Passed: 43 task files (15 modified tracked, 28 new); local/fetched heads all equal the baseline, ahead/behind 0/0. No generated/binary/symlink/conflict-marker/common secret-pattern blocker found. Saved final logs confirm 61 passing CPU tests and 22 parsed files; task sources unchanged since that gate, so tests were not repeated. Restricted-DNS fetch failed; scoped elevated fetch succeeded. AGENTS.md and ignored logs excluded |
| 2026-11-07 | Explicit-path `git add`, `git diff --cached --check`, scoped commit, `git push origin main`, `git ls-remote origin refs/heads/main`, `git status --short --branch` (Medium; parent reviewed commit/status/log) | Passed: implementation commit 9adeef71f58c20b8210d387bd20d84405062d177, 43 files. Normal push failed restricted DNS; scoped elevated retry succeeded without force. Remote main, local main and origin/main matched that exact SHA. Only pre-existing AGENTS.md remains untracked; PyTorch gitlink unchanged. [Publication log](.cache/qwen-hybrid/qhp01-publish.log). No runtime gate changed |
| 2026-11-14 | Read-only SSH and host preflight (Medium): `ssh -o BatchMode=yes -o ConnectTimeout=8 -o ConnectionAttempts=1 -o StrictHostKeyChecking=yes`, `uname`, distro, runtime/tool, `df`, `free` and `nvidia-smi` checks | Short alias `gpu01` did not resolve; configured full hostname also failed inside restricted network, then scoped elevated SSH succeeded. Rocky 9.6 x86_64, Apptainer 1.4.2, no Docker, two sm_120 RTX PRO 6000 Blackwell Max-Q devices idle at snapshot. Existing 8.37 GB baseline SIF located. This establishes access and capacity only |
| 2026-11-14 | `apptainer exec --cleanenv` baseline Python/toolchain probes, source checkout gitlink/HEAD and embedded Git metadata inspection (Medium) | Installed Core 0.13.1 and Torch 2.9.1/CUDA 12.8/sm_120/tracer confirmed. Host submodule HEAD matches bbb19f8c856fcf8d7c60267fecb976b4749e2cff; embedded Git pointer broken and runtime reports Unknown. First Cargo probe lacked RUSTUP_HOME; setting image paths resolved toolchain (1.99.1). Strict provenance does not pass. [Preflight log](.cache/qwen-hybrid/qh02-gpu01-preflight-20261114.log) |
| 2026-11-14 | `git archive` from cac681b0b30ca0e679d10714375d601c7bfa4047; start `apptainer exec --cleanenv` native `make dist` with four jobs and seeded offline Cargo caches (Medium) | Source staged successfully; build started in `/tmp/phantora-qh02-V8gf5y/source`. Durable host records at `gpu01:/home/dathar/Phantora/qh02-gpu01-20261114-NRVgKQ/`, including source SHA/archive digest, native build script, log and exit status. Build result pending |
| 2026-11-14 | `make dist` and `cargo test --release -p phantora --lib gdn` / `cargo test --release -p phantora --lib coverage` in SIF (Medium) | Build exit 0 with compiler warnings, release compile 1m17s plus C link; 3 GDN and 4 coverage tests passed. See `native-build.*` and `native-tests.*` in the durable gpu01 run directory above. Live event transport/order remains pending |
| 2026-11-14 | Original Llama3-8B GPU-free replay, Core 0.13.1, L40S records, 4 layers, seq 1024, microbatch/GA 1, TP=2, DP=PP=1, 3 iterations, 81920 MiB simulated VRAM (Medium) | Attempt 1 with `CUDA_VISIBLE_DEVICES=''` failed device count on both ranks; contained `--containall` run without `--nv` and with mask unset passed, reporting 100.240 / 90.406 / 90.406 ms simulated on both ranks. No missing-key directory. See `llama-replay-attempt2.sh`, `llama-default-attempt2/`, driver log and exit file in durable run directory. No physical GPU kernel timing measured |
| 2026-11-14 | Parent checked published PyPI metadata for [FLA 0.4.0](https://pypi.org/pypi/flash-linear-attention/0.4.0/json), [FLA core 0.4.0](https://pypi.org/pypi/fla-core/0.4.0/json) and [Megatron Core 0.16.0](https://pypi.org/pypi/megatron-core/0.16.0/json); reviewed `.dockerignore` and Dockerfile | `flash-linear-attention==0.4.0` requires `fla-core==0.4.0`; FLA core requires Torch >=2.5 and einops. Core requires Torch >=2.6, numpy and packaging >=24.2. Existing `--no-deps` plan needs the explicit FLA core pin; hybrid helper COPY needs a context exception. Static findings, not installed-runtime results |
| 2026-11-14 | Packaging changes and `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_qwen_hybrid*.py' -v`, AST parse, `git diff --check` (Medium; diff reviewed by parent) | 62 CPU tests passed, including missing/mismatched `fla-core` with correct wrapper. Scoped `.dockerignore` includes exactly four copied helpers; the image explicitly installs `fla-core` and forces causal-conv1d sdist; profiler records kernel distribution version and checks wrapper. [CPU](.cache/qwen-hybrid/qh02-package-cpu-20261114.log), [parse](.cache/qwen-hybrid/qh02-package-parse-20261114.log), [diff](.cache/qwen-hybrid/qh02-package-diff-20261114.log). Docker build not run |
| 2026-11-14 | Apptainer `pip install --no-deps --target /qhbuild/upgraded-deps` for candidate wheel pins (Medium) | Exit 0, Core 0.16.0, FLA/core 0.4.0, packaging 24.2, einops 0.8.1 and ninja 1.11.1.4 installed in candidate prefix. Torch 2.9.1 still resolves from the original read-only SIF. See `candidate-wheels.log`/`.exit` in the durable gpu01 run directory. Causal extension compiles separately against that Torch |
| 2026-11-14 | First upgraded Llama replay, same arguments and binary/DB as original (Medium; `llama-upgraded-attempt2/workload.log`) | Both ranks exited 1 at import from `megatron.core.datasets.megatron_tokenizer`: Core 0.16 renamed the compatible base to `MegatronLegacyTokenizer`. Adjust fallback narrowly and rerun both Core versions |
| 2026-11-14 | Upgraded Llama retry with tokenizer compatibility import (Medium; `llama-upgraded-attempt3/workload.log`) | Reached optimizer construction. Both ranks failed Core 0.16 `_get_param_groups` with `EOFError` while unpickling a world-group `all_gather_object` through simulated NCCL, which moves no payload. Route that setup metadata via Gloo only during simulated optimizer construction and retest |
| 2026-11-14 | Identical GPU-free Llama replay on 0.13.1 vs 0.16.0 after tokenizer and Gloo fixes (Medium, attempt 4) | Same native binary and workload, both workers exit 0 and all three iterations per rank complete. 0.13.1 driver 0, fully covered. 0.16.0 driver 1, missing `Add(bf16 [1024,1,7168], scalar f64 [])`; identical printed times here are incomplete and not accepted for 0.16. `llama-{default,upgraded}-attempt4/` contains workload and simulator logs, exit files and the candidate missing-key CSV. Pinned Core 0.16 `mlp.py` shows the SwiGLU `glu_linear_offset` add; keep it modeled |
| 2026-11-14 | Pinned causal-conv1d `pip wheel --no-deps --no-build-isolation --no-binary causal-conv1d` in contained SIF (Medium) | First build filled container-private tmpfs during nvcc. Retry with `TMPDIR=/qhbuild/compiler-tmp` on disk-backed scratch, four jobs and sm_120; `candidate-causal-build-attempt2.*` pending |
| 2026-11-14 | Parent read-only SSH/Apptainer source inspection (`git submodule status --recursive`, embedded Torch version generator and CMake cache) | All 78 source submodule status rows matched the recorded commits; embedded generator returns `Unknown` if `git rev-parse HEAD` fails. CMake cached `/phantora/pytorch`, Release, sm_120, tests on and cuDNN off. Plan a revision-bearing rebuild from a verified source tree; the old SIF is not relabeled |
| 2026-11-14 | GPU-free upgraded Llama debug trace (Medium, attempt 5) | Confirmed 24 exact `aten::add` events for the missing [1024,1,7168] scalar-add shape (4 layers × 3 iterations × 2 ranks), plus 24 related in-place autograd adds. The offset-zero operation executes and must remain timed. Both physical GPUs idle at next probe; authorize only one for scoped Llama timing record |
| 2026-11-14 | Read-only Apptainer Git metadata bind and blob comparison (Medium; `/tmp/phantora-qh02-provenance-seKzsh/{metadata-mount,tracked-compare}.log`) | Embedded source resolves exact bbb19f8c... with the original Git directory mounted read-only. Checked 79 repo roots / 78 pinned gitlinks, 115,161 tracked blobs / 1,454,833,579 bytes: 87 byte differences, no missing files or mode differences. Verify exact Git-declared CRLF normalization before considering reuse; this probe does not prove the existing runtime revision |
| 2026-11-14 | Git attribute/CRLF audit, retained build size and `ninja -C build -n install` on read-only SIF (Medium; `tracked-eol.log`, `build-reuse-assessment.log`, `ninja-dry-run.log`) | All 87 raw blob differences exactly matched Git-declared `eol=crlf`; no unmatched tracked sources or gitlinks. Retained source/cache 5.2 GiB. Dry run requests upstream version generation, 4 Python C++ objects, 3 links and install, with no full CUDA compile. Authorize official rebuild in a private copy; dry-run evidence is not a wheel or runtime test |
| 2026-11-14 | Revision-bearing Torch build started on gpu01 (Medium; private copied source, retained cache, RO Git metadata, four jobs, 45-min bound) | Durable artifacts `gpu01:/home/dathar/Phantora/qh02-torch-revision-20261114-9K5hTQ/`: `rebuild.sh`, preflight audits, live `rebuild.log`, eventual `exit-status`. Original SIF remains read-only. Build and wheel verification pending |
| 2026-11-14 | Record ordinary Llama timings on one idle sm_120 Blackwell GPU with CUPTI, both Core versions, same Llama arguments, no L40S rows (Medium) | 0.13.1 recorded 90 compute / 14 memcpy rows; 0.16.0 added one compute for a union of 91 compute / 14 memcpy. DB under durable run `blackwell-db/`. GPU used only for measured kernel timings; independently replay without GPUs |
| 2026-11-14 | Causal-conv1d 1.5.3.post1 CUDA extension sdist retry (Medium) | Built sm_120 wheel successfully on disk-backed TMPDIR against SIF Torch. 135,044,977 bytes, SHA256 `0a71e97165142f7b8d8e66c81746cab3e0972ab659e6fa3745a42e293436c361`. `candidate-causal-build-attempt2.*`; installed imports/signatures still to check |
| 2026-11-14 | Candidate `qwen_hybrid_runtime_check.py --metadata-only`, installed causal/FLA signature probe, replay alias plus GDN/GPT/Transformer signatures and 4-layer spec (Medium) | All three diagnostic commands exit 0. The exact unmodified runtime probe exits 1 at Torch git_version Unknown. GPU-free FLA warns no Triton backend and no real kernel runs. `candidate-{metadata,installed-imports,replay-alias-imports,exact-runtime}.{log,exit}` in durable gpu01 run directory |
| 2026-11-14 | Copied-cache Ninja scope gate and upstream `pip wheel` progress (Medium) | Scope gate passed. Real CMake command `cmake --build . --target install --config Release -j 4` began with upstream version regeneration and expected C++ objects, no full CUDA compile. Wheel/install checks pending |
| 2026-11-14 | Official offline `pip wheel` and temporary `pip install --no-deps --target` from verified pinned tree/cache (Medium) | Build/verify exit 0 in ~1m43s; no full CUDA compile. `torch-2.9.1-cp311-cp311-linux_x86_64.whl` (341,622,030 bytes) SHA256 `e241702e47191ae81f66023d5c612c90cc7683efc9631dfd29c5250edf6f06b8` verified. Runtime reports exact bbb19f8c856fcf8d7c60267fecb976b4749e2cff, CUDA 12.8, sm_120, callable tracer; Torch extension and seven loaded Torch libraries all came from the isolated wheel. Wheel/checksums in `qh02-torch-revision-20261114-9K5hTQ/`; `wheel-verification.log` is CPU-only |
| 2026-11-14 | Scoped Blackwell CUPTI record and GPU-free replay A/B, `blackwell` run artifacts (Medium) | All four phases exit 0. Two ranks × 3 iterations per version, zero missing timing dirs, identical DB/source/native hashes. Both ranks report ms: 0.13.1 `[53.571939468, 49.218177795, 49.217939377]`; 0.16.0 `[53.672075272, 49.318075180, 49.317836761]`. The new SwiGLU add is measured at 25,184 ns. These are simulations using SIF Torch `Unknown`, not ground-truth/calibration measurements. Both GPUs idle after cleanup |
| 2026-11-14 | Durable candidate assembly; `cuobjdump` on causal wheel; final focused Python checks after trainer changes (Medium) | Verified fwd/bwd/update sm_120 and staged package prefixes and wheels at `gpu01:/home/dathar/Phantora/qh02-gpu01-20261114-NRVgKQ/candidate-artifacts/`. Includes `upgraded-deps`, `causal-deps`, `candidate-wheelhouse`, `SHA256SUMS`, `RECORD-SHA256SUMS`, `prefix-manifest.log`. 62 CPU tests plus AST/diff checks pass. Apptainer bind `candidate-artifacts:/qhcandidate:ro` and `PYTHONPATH=/qhcandidate/causal-deps:/qhcandidate/upgraded-deps` select candidate without changing default Python |
| 2026-11-14 | Assemble exact candidate and replay Core 0.13.1 on exact Torch (Medium; `exact-gate/`) | Wheel install and default Llama replay exit 0, no misses, expected times retained; simulator and worker lib mappings verified. Saved `compat.patch`, source and native archives/hashes, config and launch/build/replay/verification scripts in durable run's `exact-gate/`. Core 0.16.0 replay, exact causal compile and production runtime probes in progress |
| 2026-11-14 | Exact-Torch Llama replay Core 0.16.0 (Medium; `exact-gate/llama-upgraded/`) | Both exact-wheel 0.13.1 and 0.16.0 runs exit 0 with 3 iterations on 2 ranks, zero misses, identical source/native/DB hashes. Simulator and each rank mapped the isolated wheel's libtorch/libc10 only. Blackwell DB reused as a compatibility fixture; no new timing-accuracy claim |
| 2026-11-14 | Build pinned causal-conv1d against exact Torch; unmodified production `qwen_hybrid_runtime_check.py` in contained replay alias mode and with one visible Blackwell (Medium; `exact-gate/`) | All exit 0; exact causal wheel SHA256 `687916686ea20967418bb6f6b2776083191564cefe81c34f5c91881ee6350fe9`, sm_120 forward/backward/update cubins confirmed. Both probes identify patched bbb19f8c856fcf8d7c60267fecb976b4749e2cff, CUDA 12.8, expected Core/FLA/core/causal/Triton pins and correct loaded library paths. See `exact-causal-wheelhouse/`, `runtime-replay.log`, `runtime-real.log`. Import success does not establish real model/fused numerical behavior |
| 2026-11-14 | First exact runtime one-GPU real debug torchrun, four layers/seq 128, TP=DP=PP=1, no stubs (Medium; `exact-gate/debug.log`, `debug.exit`, GPU snapshots) | Exit 1 before iteration 0. Reached real GDN forward; FLA L2 norm Triton autotune failed in CUDA driver `load_binary` with `device kernel image is invalid`. Core 0.16 and Torch imports passed; no iteration/performance/model-reference claim. Preserve log and try the version-specific Blackwell `ptxas` override with a new cache |
| 2026-11-14 | Same real debug command and exact runtime, private fresh Triton cache and `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` (Medium; `exact-gate/debug-ptxas{,-driver}.log`, `debug-ptxas.exit`) | Workload and validation both exit 0. Observed bundled ptxas 13.1 vs selected image ptxas 12.8. One RTX PRO 6000 Blackwell Max-Q, 3 GDN + 1 local full-attention with shape assertions, TP=DP=PP=1, BF16, seq 128, microbatch 1, accumulation/global batch 8; four optimizer iterations and 1,024 tokens/iteration. Initial including JIT/autotuning ~75.162 s, later ~0.152 s each; smoke only. Both GPUs idle afterward; no running jobs and no shared image, driver or environment change |
| 2026-11-14 | Preserve exact source and executable test artifacts (Medium) | Published base cac681b0b30ca0e679d10714375d601c7bfa4047 plus scoped `exact-gate/compat.patch`, SHA256 `78525d1eaaee69ea33a0edbf3114ac9e6b3ed17f2d6265b3e89ce0872907fd09`. `source.sha256`, source/native tar archives, exact wheel checksums, `netconfig.toml`, environment and launch scripts/logs under `gpu01:/home/dathar/Phantora/qh02-gpu01-20261114-NRVgKQ/exact-gate/`. No new Git publication occurred |
| 2026-11-14 | QH-03a `[0.8b]` preset config checks and gpu01 preflight (Medium; pinned config and diff independently reviewed by parent) | 11 focused CPU config tests pass. [Official revision `2fc0636`](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json): 24 layers, hidden 1024, FFN 3584, vocab 248320, 8 Q / 2 KV heads of 256, 16 GDN key/value heads of 128, tied embeddings. Both 97,887 MiB Blackwell GPUs idle at preflight; verified existing Torch wheel SHA256. Real run not yet passed |
| 2026-11-14 | `python3 -m unittest discover -s tests -p 'test_qwen_hybrid_*.py'` and `git diff --check` after adding 0.8B (Medium) | 63 CPU tests passed; whitespace check passed |
| 2026-11-14 | Real GPU `torchrun --standalone --nproc-per-node=2 tests/test_megatron.py --hybrid-preset 0.8b --tensor_parallel_size 2 --data_parallel_size 1 --pipeline_model_parallel_size 1 --sequence_length 128 --micro_batch_size 1 --global_batch_size 8 --iterations 4` (Medium) | Exit 0, both ranks emitted structure and 4 iterations; 24 layers (18 GDN/6 local unfused full attention), tied embeddings configured, BF16, accumulation 8, 1,024 tokens/step. Rank 0 wall seconds `[81.672134,0.899464,0.896455,0.896690]`, rank 1 `[81.662239,0.888507,0.885574,0.886381]`; discard cold step, mean max-rank 0.897536 s/step (~1,141 tokens/s, smoke only). Exact patched Torch 2.9.1 bbb19f8c…, CUDA 12.8 sm_120, Core 0.16.0, FLA 0.4.0, causal 1.5.3.post1, Triton 3.7.0 with image ptxas 12.8.61. `gpu01:/home/dathar/Phantora/qwen35-08b-tp2-20261114-EMj2SG/verified.json` and per-rank logs. No checkpoint, simulated run or HF numerical comparison |
| 2026-11-14 | QH-03a artifact verification (Medium) | Both ranks emitted global indices 0–23; each reported 8,653.65 MiB PyTorch peak allocated (~8.45 GiB), which excludes other GPU allocations and does not validate simulated memory. Driver 570.195.03; both devices idle before/after. Pinned config SHA256 `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| 2026-11-15 | QH-07a scoped elevated read-only SSH and resolver check (Medium) | `gpu01.pronet.rcac.purdue.edu` fails to resolve from the Mac; no SSH session, workload, timing, or remote mutation. Prepare discovery locally so the first GPU-free run can start when access is restored |
| 2026-11-15 | QH-07a strict SSH reconnect after user restored access (Medium) | Exit 0; exact runtime scripts/archive and prior real baseline artifacts located. Proceed to bounded GPU-free discovery on the matching case |
| 2026-11-15 | QH-07a bounded Apptainer discovery setup and first attempt (Medium) | Verified exact Torch, causal and native artifact hashes; exact runtime probe passed. Netconfig uses actual gpu01 hostname and measured 271.9159771887 Gbit/s 8 MiB NCCL rate. Simulator ready; both worker imports failed because DeepSpeed tried to write to read-only `HOME`. Retry with isolated writable home/cache under `gpu01:/home/dathar/Phantora/qh07a-gpu01-20261115-2QD4Rq/discover-gnb3yu6o` |
| 2026-11-15 | User requested stop; scoped launcher/worker/simulator cleanup (Medium; parent checked exact run path) | Worker exit -15, `runner.json` records `valid=false` and `timing_result_accepted=false`; no leftover scoped run processes. Retry ran with no exposed NVIDIA devices. No physical GPU availability query, profiling, or comparable result |
| 2026-11-15 | Resume at user request: review stopped artifacts, verify archived native/config, wheel checksums and trainer/config/runtime source, launch fresh bounded discovery (Medium) | Stopped run had reached iterations 0–2 on both ranks, then stopped by request. Partial status already showed missing timing keys and unclassified ATen/device-only ops, so no valid timing. Fresh identical GPU-free run `gpu01:/home/dathar/Phantora/qh07a-gpu01-20261115-2QD4Rq/discover-qsp4agg7` in progress |
| 2026-11-15 | Completed Qwen3.5-0.8B TP2 discovery on exact Megatron 0.16 runtime, same settings as real QH-03a (Medium; parent checked local summary) | Worker/launcher exit 0, native `phase=complete` and all rank/observer lifecycle flags true. Four iterations per rank, 1,024 tokens/step. Per rank: 576 causal conv, 576 chunk GDN and 1,152 L2 events in **each** direction, exactly as planned. Strict invalid: 204 compute / 18 memcpy / 6 GDN timing signatures missing, zero flash or malformed GDN; 43 distinct diagnostics (39 unmapped compute, two unmapped device-only, two malformed signatures) and 19,072 malformed compute events (19,008 `aten::to`, 64 `aten::exp`). [Local evidence](.cache/qwen-hybrid/qh07a/evidence-qsp4agg7/summary.json) contains names, rank/server logs and missing manifests. No GPU kernels or accepted simulated time |
| 2026-11-15 | QH-07b profiler preflight `nvidia-smi` (Medium) | Both gpu01 GPUs idle at snapshot (2 MiB/0%, no compute applications). Profile scoped to GPU 0 with a second availability check immediately before launch; replay will be GPU-free |
| 2026-11-15 | QH-07b actual CUPTI profiling on gpu01 GPU 0, exact Torch/FLA/causal stack, scoped CUDA 12.8 ptxas (Medium) | Exit 0. Fresh measurements attempted for the union of 295 compute and 32 memcpy keys; no exceptions or fallback to old timings. All 6 fused timings positive (forward/backward ns): conv 2,988/11,048; L2 1,087/1,198; chunk 29,715/42,881. Raw zero-activity rows excluded from the replay DB; count to be recorded at replay. Durable `gpu01:/home/dathar/Phantora/qh07b-gpu01-20261115-FGikEJ`. No strict changes; one GPU-free replay in progress |
| 2026-11-15 | Parent compared original `torch_call.rs` (byte-identical to `301d9d9`) against QH-07a's unmapped-compute names | None of the 39 names had a direct original ATen dispatch case. Original `aten::to` handled only dtype-only calls, and no explicit `aten::exp` case existed. This confirms source-level coverage gaps; it does not establish the original Llama trace encountered them or that native layers account for none of their work |
| 2026-11-15 | QH-07b replay with the positive-only CUPTI tables; compare complete real/replay rank series (Medium, parent checked [local summary](.cache/qwen-hybrid/qh07a/evidence-qh07b-FGikEJ/summary.json)) | Profiling and replay exit 0. Both ranks complete all exact fused counts; six GDN rows hit, no memcpy or GDN misses. Of 20 raw zero-activity compute probes excluded, 14 used keys still missing (12 ConvDType, 2 Dropout); 39 unmapped compute names, two device-only names and two malformed signatures remain. At 1,024 tokens/step, mean max-rank seconds over steps 1–3: real 0.897536287 (~1,141 tokens/s), partial replay 0.095298608 (~10,745 tokens/s modeled), difference −89.3822%. The CPU-time-ignored replay and one-size fixed 271.91597719 Gbit/s network assumption remain. Native `valid=false`; no calibration claim. Remote `qh07b-gpu01-20261115-FGikEJ/discover-6l3hktw6` |
| 2026-11-15 | QH-07c parent layer-boundary audit of QH-07b `status.json` prior to code | 5,056 unmapped `square` calls across ranks/steps, consistent with 79 norm calls/microbatch (48 pre/post block norms, 18 GDN output, 12 Q/K full-attention, one final norm) × 64 rank-microbatches. Also 768 `sin` and 768 `cos` rotary calls, 8,832 `contiguous` calls whose actual copy/no-op split is unknown, and 64 malformed `exp` out signatures. Counts are not timings. Add `square`/`exp.out` first; do not treat all layout operations as allocating or free |
| 2026-11-15 | QH-07c opt-in `square`/`exp_out` mapping and local `test_qwen_hybrid_*.py` suite (Medium; parent reviewed diff) | 64 tests passed and `git diff --check` clean. `exp_out` is a separate key with shape/dtype/device checks and the true ATen out call; old default parser and unary exp rule unchanged. Linux private native build/test and live discovery pending |
| 2026-11-15 | Original CPU timer source audit and private Linux verification launch (Medium) | Original uses `CLOCK_PROCESS_CPUTIME_ID` and subtracts explicit simulator send/receive work. CPU-on still measures replay-side Python/stub work, not the real CUDA host timeline; treat a future toggle as sensitivity evidence. Exact-wheel CPU-only native build/test and bounded discovery launched at `gpu01:/home/dathar/Phantora/qh07c-gpu01-20261115-KiNNh8` |
| 2026-11-15 | QH-07c private Linux Rust tests, `make dist` and GPU-free matched 0.8B TP2 discovery (Medium) | 14 Rust tests passed (one existing ignored), build and 4 steps on both ranks passed; exact fused counts and no GDN/copy/flash misses. All 5,056 `square` occurrences now recognized and the 64 rejected `exp` out calls no longer malformed. Five unique unmeasured keys (four `square`, one `exp_out`) surfaced, so 19 compute timing misses total; 19,008 malformed `to`, 38 other unmapped compute names and two device-only names remain. No new timings measured; `valid=false`. [Local evidence](.cache/qwen-hybrid/qh07c/evidence-KiNNh8/summary.json), remote `qh07c-gpu01-20261115-KiNNh8/discover-eh8hbitd` |
| 2026-11-15 | QH-07d read-only GPU availability preflight (Medium) | CUDA process present on both gpu01 devices (548 MiB allocated on each; instantaneous 0% utilization does not show reservation/ownership). No profiling or new replay launched. Do not add rows or present a post-change timing without new measurements |
| 2026-11-15 | QH-07d local staging while GPU unavailable (Medium) | [Five-key reference](.cache/qwen-hybrid/qh07d/keys/compute.csv) and [bounded launch script](.cache/qwen-hybrid/qh07d/launch-from-local.sh) prepared; exact key subset and Python/shell parse checks passed. Launcher requires idle GPU, checks exact runtime/inputs, prepares finite aliased `exp_out` inputs outside CUPTI, preserves the old DB and runs matched replay. No measurements produced |
| 2026-11-15 | User reported GPU free; resume QH-07d | Execute staged launcher with fresh idle check, five real profiles, then matched GPU-free replay. No comparison until exit, coverage counts and DB merge verified |
| 2026-11-15 | QH-07d idle check and exact-stack one-Blackwell CUPTI profile (Medium) | Positive ns: `square` `[1024,128]` 1,338; `[128,1,1,256]` 1,151; `[128,1,1024]` 1,170; `[128,1,4,256]` 1,164. `exp_out` `[128,1,124160]` 81,592, aliased with fresh finite max-shifted logits outside measurement. Only these five rows added; previous tables and rows verified unchanged. Bounded GPU-free replay started under `gpu01:/home/dathar/Phantora/qh07d-gpu01-rNvEfL` |
| 2026-11-15 | QH-07d matched replay and comparison (Medium; [local evidence](.cache/qwen-hybrid/qh07d/evidence-rNvEfL/summary.json)) | Profile and replay exit 0, both ranks finish with exact GDN counts. Five keys hit; missing compute 19→14, zero GDN/copy/flash misses; 38 unmapped compute, two device-only and one malformed `to` signature remain. Mean max-rank seconds over steps 1–3: same-native before measurement 0.095019976, after 0.096515338 (+1.495 ms), real 0.897536287 (replay 89.25% below real). Older QH-07b partial was 0.095298608 with pre-patch native; not the isolated before/after control. Network and CPU assumptions unchanged, `valid=false`. Remote `qh07d-gpu01-rNvEfL/discover-frtuj7n2` |
| 2026-11-15 | QH-07e idle preflight and bounded diagnostic launch (Medium) | Both GPUs idle at preflight. Serial unprofiled reproduction, real single warmed-step CUDA timeline, and same-DB CPU-on replay at `gpu01:/home/dathar/Phantora/qh07e-gpu01-2SvSaE`. CUDA profiling uses scratch-only ranges; no default changes |
| 2026-11-15 | User asked to match prior GPT/Llama phase check; parent reviewed prior phase CSV and vanilla `--phase_timings` implementation | QH-07f records the same outer forward/backward and inner forward, backward, optimizer and sync timers where available, in separate real and simulated passes. Prior timers use synchronized `time_pair()[0]`, may change overlap and can nest. Keep native/profile/headline clocks distinct |
| 2026-11-15 | QH-07e diagnostic runs complete; QH-07f matched timer passes launched (Medium) | Unprofiled real head time reproduces at approximately 0.86 s/step. CPU-on same-DB replay 3.326479 s/step vs 0.096515 CPU-off; exact rank/GDN lifecycle and 14 compute misses, zero GDN misses. The CPU-on clock adds replay process CPU, not real host runtime; it overpredicts and is not a correction. Real warmed CUDA trace and phase analyses pending; phase artifacts `gpu01:/home/dathar/Phantora/qh07e-phase-gpu01-bxEXCt` |
| 2026-11-15 | QH-07e exact unprofiled maxima and first CUDA trace review (Medium) | Real max-rank warmed steps 0.875124/0.869056/0.868772 s (mean 0.870984 s), ~3% below original 0.897536. One profiled step ~1.268 s (46% overhead), non-NCCL kernel sums ~140 ms/rank. Captured NCCL residence asymmetric (~930 ms rank 0, ~14 ms rank 1); rank 1 has ~1.109 s with no captured GPU activity. Trace is a synchronization/progress clue, not a clean estimate of NCCL cost. Matched phase pass still running |
| 2026-11-15 | QH-07e paired real TP collective events by order in trace with shared clock origin (Medium) | 828 matching named all-reduces (803 BF16, 25 FP32). Rank 0 entered all 828 before rank 1; median start gap 695.5 µs (p95 2.243 ms), median absolute finish gap 0.498 µs (p95 1.328 µs). Median NCCL residence 712.2 vs 17.12 µs. This points to waiting for peer arrival, not 930 ms of link transfers; percentages from the more expensive profiled step cannot be transferred to the unprofiled run |
| 2026-11-15 | QH-07e late rank GPU/CPU intervals (Medium) | Rank 1 forward/backward window 1,228.44 ms in the perturbed step; GPU activity union 128.77 ms, gaps 1,099.67 ms. Of gaps 375.80 ms overlaps captured main-thread ATen or CUDA API spans; 723.88 ms overlaps neither. Cannot call the latter Python or real CPU time. [Trace summary](.cache/qwen-hybrid/qh07e/evidence-2SvSaE/trace-summary.json) and [peer events](.cache/qwen-hybrid/qh07e/evidence-2SvSaE/peer-summary.json) |
| 2026-11-15 | QH-07f identical synchronized Core timers, real and CPU-off replay, both ranks (Medium; [phase JSON](.cache/qwen-hybrid/qh07e/evidence-phase-bxEXCt/phase-summary.json)) | Both exit 0; exact GDN counts and same 14 ordinary misses. Mean max-rank step real/replay 907.784/96.213 ms. Max per-rank phase means: fwd+bwd 857.912/90.711 ms (767.200 ms delta, 94.53% of 811.571 ms step gap), forward 408.638/33.935, backward 446.313/56.768, inner optimizer 21.041/4.211, main-grad copy 2.824/1.290, main→model copy 1.875/0.000. Timers synchronize and nest; zero is a reported modeled timer, not proven absence of real work. Remote `qh07e-phase-gpu01-bxEXCt` |
| 2026-11-15 | User resumed QH-07g, fresh idle preflight and serial diagnostic launch (Medium) | Both gpu01 Blackwells idle at preflight. Rerun the exact 0.8B real workload unprofiled then capture one warmed step with CUDA-only Kineto, two epoch markers and no additional GPU fences inside the timed step; reuse private warmed cache. `gpu01:/home/dathar/Phantora/qh07g-gpu01-LDOx0Q`. Check perturbation and clock pairing before interpreting results |
| 2026-11-15 | QH-07g CUDA-only matched real trace and same-session control (Medium; [summary](.cache/qwen-hybrid/qh07g/evidence-LDOx0Q/summary.json)) | Captured step 1039.879 ms vs 870.660 ms unprofiled (+19.44%, earlier full Kineto +45.9%). Equal time base with markers and no CPU op events. 828 matching all-reduces, all overlap; median/p95 absolute start skew 455.106/1644.322 µs, finish 0.597/1.379 µs. Rank 1 earlier in 709 pairs, opposite the earlier trace. GPU union rank0/1 261.061/686.766 ms. Both exit 0; no remaining compute jobs. This weakens a fixed-rank hardware explanation but does not prove unprofiled skew or cause |
| 2026-11-15 | QH-07h fresh idle check and lightweight real TP2 host clocks (Medium) | Both GPUs idle at preflight. Capture one warmed step with monotonic entry/exit timestamps around the Python all-reduce calls, shape/dtype/group/async signature and coarse phase stamps. No profiler or added CUDA fences, output after the step, private launch `gpu01:/home/dathar/Phantora/qh07h-gpu01-m4x3d2` |
| 2026-11-15 | QH-07h matching real lightweight host timestamps (Medium; [summary](.cache/qwen-hybrid/qh07h/evidence-m4x3d2/summary.json)) | Stamped step 882.105 ms vs adjacent 874.648/871.608 and prior same-index 870.660 (+1.31%). 828 calls/rank match (392 async); monotonic offset spread 0.271 µs. Median absolute entry/exit gaps 138.893/139.409 µs. All 101 entry gaps >5 ms fall in first 101 calls; rank 1 begins step 10.955 ms later. Remaining 727 entry pairs p50/p95 120.631/909.891 µs. API interval union ~42 ms/rank; host entry/exit are not device completion. Do not sum inherited lead as independent waiting; origin of the larger step gap remains unassigned. No lingering compute job |
| 2026-11-15 | User requested per-layer comparison; source review of native event APIs before instrumentation | `cudaEventRecord` queues a stream event, while the stub only fills `finished_time` during `cudaEventSynchronize`; `cudaEventElapsedTime` returns the raw difference of simulator clock ticks (check µs against a known delay). Record markers inside the sampled step, collect only after all step timing, avoid in-layer fences, and use different verified unit conversions for CUDA and Phantora. These are stream envelopes, not isolated kernels |
| 2026-11-15 | QH-07i matched per-layer marker run, real pass (Medium) | Both GPUs idle before launch; private `gpu01:/home/dathar/Phantora/qh07i-gpu01-XVS1U5`. One warmed step, microbatch 0 only, CUDA events and Tensor gradient hooks, no added in-step fences. All 99 real markers per rank present, default stream 0, no missing gradients. Warm sampled step 881.566 ms; replay and 2,500 µs known-delay calibration pending |
| 2026-11-15 | QH-07i matched per-layer replay, known-delay probe and aggregation (Medium; [per-block CSV](.cache/qwen-hybrid/qh07i/evidence-XVS1U5/layers.csv), [summary](.cache/qwen-hybrid/qh07i/evidence-XVS1U5/summary.json)) | Both jobs exit 0, all 24 forward/backward intervals ordered on both ranks, exact GDN counts and unchanged 14 compute misses; stub 2,500 raw ticks for 2,500 injected µs. Sampled microbatch real rank0/1 116.806/106.368 ms vs replay 11.324/11.324. Inside blocks 92.113/90.566 ms of 105.482/95.044 ms total gap (87.3%/95.3%). GDN median forward real 1.97–2.02 vs 0.160 ms replay, backward ~2.25 vs 0.254; full forward 1.84–1.89 vs 0.160, backward 1.90 vs 0.212. Stream envelopes include waits; largest outside-block gap is rank 0 before first layer (11.717 vs 0.007 ms). Sampled step real 881.566 vs adjacent 881.876/878.325 ms; replay 96.461 vs adjacent 97.624/96.247. No lingering jobs or timing approval |
| 2026-11-16 | QH-07j local re-analysis of saved QH-07g CUDA-only traces (parent): pair every kernel/memcpy/memset with its launch call by correlation ID and classify gaps in GPU activity | Per rank per step ~36,300 GPU activities (828 NCCL), mean compute kernel 4.0 µs, compute busy 141.7/135.8 ms. Rank 0 (pace-setting in this trace): GPU empty with next launch not yet issued for 756.5 ms of 1,040.2 ms (72.7%); NCCL residence 119.3 ms. Rank 1 (host ahead): NCCL residence 551.0 ms, host-limited empty 294.7 ms. Launches come from two host threads (21,696 forward/optimizer, ~14,600 backward). Profiled step is 19% slower than unprofiled, so host-limited ms are inflated; kernel time is not. GPU-only prediction cannot exceed ~141 ms at this size |
| 2026-11-16 | QH-07j real sequence-length sweep, unmodified QH-03a launcher/trainer with only `--sequence_length` changed, both idle Blackwells, serial (parent) | Mean max-rank steps 1–3: seq 128 902.04 ms, seq 512 894.29 ms, seq 2048 915.42 ms (1,024/4,096/16,384 tokens per step). Flat within ~2% across 16x tokens: host-bound. Launcher exit 1 at 512/2048 is only the copied `verify.py` 1,024-token assertion. `gpu01:/home/dathar/Phantora/qh07j-seqsweep-MfDAqY` |
| 2026-11-16 | QH-07k real microbatch sweep at seq 2048, same launcher with `--micro_batch_size`/`--global_batch_size` changed (parent) | MB 2/4/8 at GA 8: 1,225.85/2,197.17/4,475.98 ms; MB 8 GA 1: 593.53 ms. 8,192→16,384 tokens per microbatch scales 2.04x (GPU-bound); 2,048→4,096 scales 1.34x. Fixed host cost ~110 ms per microbatch. No OOM. `gpu01:/home/dathar/Phantora/qh07k-mbsweep-BsqJEv` |
| 2026-11-16 | QH-07l real long-sequence sweep, MB 1 GA 8, `nvidia-smi` memory sampled every 2 s (parent) | seq 4096: 1,357.42 ms, 24,196 MiB; seq 8192: 3,191.61 ms, 41,534 MiB; seq 16384: 9,074.70 ms, 95,614 MiB; seq 32768: CUDA OOM on both ranks. Memory is device-level peak including allocator cache. Superlinear time from the six unfused full-attention layers. `gpu01:/home/dathar/Phantora/qh07l-longseq-zTevfm` |
| 2026-11-16 | QH-07m GPU-free discovery at seq 8192 with QH-07c native build and QH-07d DB; `discover.py` changed only in sequence length and deadline (parent) | Both ranks complete four steps, worker exit 0, `valid=false`. Missing 161 compute, 7 memcpy and 6 fused keys at the new shapes; 0.403 s/step from this pass is not a result. `gpu01:/home/dathar/Phantora/qh07m-seq8192-fGizjn/discovery/discover-9t7y1qyk` |
| 2026-11-16 | QH-07m CUPTI profiling on idle gpu01 GPU 0 (exact stack, QH-07d `exp_out` handling), merge into a DB copy, GPU-free replay (parent) | Profile and replay exit 0. 161 compute profiled, 147 positive rows added, 14 zero-activity rows excluded; 7 memcpy added; six fused keys committed (ns fwd/bwd: conv 13,764/282,489; L2 5,068/5,967; chunk 252,836/662,524). Replay steps 1–3 2.540430 s on both ranks vs real 3.191606 s: −20.40%. 14 compute keys still missing, native `valid=false`, CPU time ignored, same network rate. Prior DB unchanged; merged copy `qh07m-seq8192-fGizjn/db-8192`, replay `replay/discover-5trw483_`. Both GPUs idle and scratch removed afterwards |
| 2026-11-16 | QH-07n same pipeline at seq 16384, starting from the QH-07m DB copy; `exp_out` profiled separately with 1 warmup / 3 measured calls to bound memory (parent) | Discovery, profile and replay exit 0. Discovery missed 161 compute / 3 memcpy / 6 fused keys. 147 positive compute rows and 3 memcpy rows added, 14 zero-activity rows excluded; fused ns fwd/bwd: conv 127,187/538,143; L2 12,062/14,009; chunk 616,777/1,527,001. Replay steps 1–3 7.410806 s on both ranks vs real 9.074697 s: −18.34%. Simulated peak 78,746 MiB (unvalidated) vs 95,614 MiB device-level real. 14 compute keys still missing, native `valid=false`. `gpu01:/home/dathar/Phantora/qh07n-seq16384-wUjCQk` (`db-16384`, `replay/discover-recnhg2e`). Both GPUs idle and scratch removed afterwards |
| 2026-11-16 | QH-07o same pipeline at seq 4096, starting from the QH-07n DB copy (parent) | Discovery, profile and replay exit 0. 147 positive compute and 4 memcpy rows added, 14 zero-activity rows excluded; six fused keys committed. Replay steps 1–3 1.018278 s on both ranks vs real 1.357420 s: −24.98%. Across seq 4096/8192/16384 the absolute shortfall is 0.339/0.651/1.664 s, which is close to 0.13 s + 0.207 × replay time (two-point fit predicts 0.654 s at 8192); that is a three-point observation, not an attribution. 14 compute keys still missing, native `valid=false`. `gpu01:/home/dathar/Phantora/qh07o-seq4096-wXFwuQ` (`db-4096`, `replay/discover-jt57yhh2`). Both GPUs idle and scratch removed afterwards |
| 2026-11-16 | QH-07p add `0.8b-l4` preset (0.8B layer shapes, 4 layers: 3 GDN + 1 full attention) with a config test; real TP2 run at seq 32768, MB 1, GA 8 (parent) | Local config tests pass (other hybrid test modules need torch, absent on the Mac). Attempt 1 OOM in backward on both ranks: 16.00 GiB request with 64.79 GiB allocated and 23.87 GiB reserved-unallocated. Attempt 2 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` completed four steps on both ranks: steps 1–3 10.36/10.34/9.79 s, device peak 96,442 MiB. One unfused full-attention layer at 32768 needs several 16 GiB score-sized tensors, so the earlier 55–60 GiB projection was wrong and there is no headroom. No replay yet. `gpu01:/home/dathar/Phantora/qh07p-l4-seq32768-Y590h1` (`real`, `real-expandable`) |
| 2026-11-16 | QH-07q unattended `0.8b-l4` TP2 sweep, MB 1, GA 8: per size a real run (expandable segments), GPU-free discovery, one-GPU CUPTI profile from a copy of the QH-07o DB, and replay (parent) | Real/replay ms and difference: 128 184.4/29.7 (−83.9%); 1024 187.1/92.8 (−50.4%); 2048 273.2/169.2 (−38.1%); 4096 494.5/350.6 (−29.1%); 8192 1,102.9/785.7 (−28.8%); 12288 1,921.8/1,328.0 (−30.9%); 16384 2,932.0/1,987.5 (−32.2%); 24576 5,520.7/3,778.2 (−31.6%); 32768 10,164.1/5,908.4 (−41.9%). All replays exit 0 with 14 compute keys missing, `valid=false`. At 32768 the replay with `PHANTORA_VRAM_MIB=97887` failed with the same OOM as the real default allocator (16 GiB request, 64.68 GiB allocated, 23.88 GiB reserved); rerun with 131072. Real peak at 32768 is 96,442 MiB, so that point may include memory-pressure overhead; not established. CSV in `results/qwen35_08b_sequence_sweep/`. `gpu01:/home/dathar/Phantora/qh07q-l4-sweep-3CPzh4` |
| 2026-11-16 | QH-07r attribution at 0.8B seq 8192: one real run tracing step 2 (CUDA-only Kineto) and step 3 (CPU+CUDA, `record_shapes`), and a replay of the QH-07m DB with simulator `PHANTORA_LOG=debug`; join real GPU activity per outermost ATen op (via Kineto external id and op ancestry) with Phantora's per-event charged time (parent; scripts in `.cache/qwen-hybrid/qh07r/`) | Traced steps 3.200/3.210 s vs untraced 3.18–3.19 s; replay 2.540430 s unchanged. Compute-kernel totals agree across trace kinds (rank 0 2,607.6/2,610.0 ms; rank 1 2,572.5/2,572.6). Rank 0 per step: real compute 2,610.0, NCCL residence 331.1, GPU empty 269.0 (189.9 no launch pending, 79.1 launch latency); Phantora charged compute 2,260.0, remainder 280.4 (its all-reduce flows total 386.5 ms per rank). Compute delta 350.0 ms = never charged +203.2 (unmapped: contiguous 64.7, cat 53.4, sub_ 43.7, max 21.3, foreach and others) + `to` +120.5 (96 real vs 48 charged conversions at `[1,4,8192,8192]` each direction; rejected call form) + other undercharged +175.9 (mul 38.1, add_ 37.4, softmax backward 31.3, fused GDN 14.5, memcpy 12.8, matmul 12.6) + compile/memset +6.3 − overcharged 156.0 (masked_fill 70.2 double-charged with its clone memcpy, baddbmm 52.4, add 14.2, bmm 11.1). Zero-key events per step: 8 ConvDType, 440 Dropout; negligible. Durations are also truncated to whole µs per event (≤30 ms over ~30k events; not separated). CSV `results/qwen35_08b_sequence_sweep/attribution_seq8192.csv`. `gpu01:/home/dathar/Phantora/qh07r-attrib-x6LKFo` |
| 2026-11-16 | QH-07s harness change: `tests/test_megatron.py` prefetches a step's batches before the timer (`--no_prefetch_batches` restores the old path). Cause: QH-07r trace showed one 163 ms GPU-empty gap 1 ms into every step on both ranks, before the first host-to-device copy, while later microbatches' uploads land at even 376 ms intervals. Real-only reruns, same launcher, 4 iterations (parent) | 24-layer real ms no-prefetch → prefetch: 128 897.5→902.1; 512 894.3→901.7; 2048 915.4→873.8; 4096 1,357.4→1,317.5; 8192 3,191.6→3,030.8 (traced rerun 3,039.2); 16384 9,074.7→8,444.1. 4-layer: 128 184.4→186.1; 1024 187.1→185.6; 2048 273.2→260.8; 4096 494.5→461.4; 8192 1,102.9→942.8; 12288 1,921.8→1,561.3; 16384 2,932.0→2,291.5; 24576 5,520.7→4,086.2; 32768 10,164.1→7,608.7. Replay unchanged (host time never charged); differences vs prefetch real: 24-layer 4096 −22.7%, 8192 −16.2%, 16384 −12.2%; 4-layer 8192 −16.7%, 16384 −13.3%, 24576 −7.5%, 32768 −22.3%. Earlier values kept as separate columns. Also added `missing_compute_ledger.csv` (per-op vanilla-handler and scope classification). `gpu01:/home/dathar/Phantora/qh07s-prefetch-3ztcW0` |
| 2026-11-16 | QH-07t GDN-specific compute, ablation at 0.8B seq 8192: opt-in hybrid handlers `aten::cat` → `QwenOp cat` and `aten::contiguous` → `QwenOp contiguous` (`torch_call.rs`, `qwen_ops.py`), simulation-only `Tensor.contiguous` no-op filter (`tests/qwen_hybrid_kernels.py`), tests. Private gpu01 build on the QH-07c source, one native binary; ablation rows differ only in which new DB rows are present (parent) | Build exit 0: 15 Rust tests pass (1 ignored), 16 Python ops/kernels tests pass in the image. Discovery: `cat` 6,530 and `contiguous` 8,832 events move to `modeled_compute` (unmapped names 38→36); 13 new keys (9 cat, 4 contiguous) profiled on GPU 0, all positive; the 14 remaining zero keys are 12 same-dtype `ConvDType` and 2 `Dropout` p=0. Replay ms: baseline 2,540.430; +cat 2,565.726 (+25.30); +cat+contiguous 2,605.662 (+39.94). Charged vs real: cat 25.3/53.4 ms, contiguous 39.9/64.7 ms. Layout gate (profiled within 15% of in-situ) passes only for contiguous `[1,3072,8192]` (212.0 vs 212.8 µs); fails for `[1,8192,8,128]` (13 vs 55.6 µs), GDN split-backward cat (81 vs 233 µs; real uses generic `CatArrayBatchedCopy` on strided gradients) and aligned cats (23–29% low). Contiguous count unchanged at 1,104/rank-step: the 48 not seen as real kernels are the attention output-gate copy that Megatron's `jit_fuser` fuses in real runs, not no-ops. Fused analysis: real/charged ms conv 2.8/2.0 and 41.4/40.7, l2norm 2.1/1.5 and 4.8/1.7, chunk 37.4/36.4 and 103.0/95.4. Handlers not applied to the sweeps pending a decision on carrying strides. `gpu01:/home/dathar/Phantora/qh07t-gdnfix-eoqL0c`; tables in `results/qwen35_08b_sequence_sweep/ablation_seq8192*.csv` |
| 2026-11-17 | QH-07u layout-aware trace, same configuration as QH-07t (0.8B seq 8192 TP2, QH-07m base DB); one change: the tracer sends each strided tensor's `stride` and `offset` (elements). Tracer patch `patches/pytorch-function-tracer-layout.patch` (torch `bbb19f8c`, `jsonIValue`), rebuilt into a separate prefix `gpu01:/tmp/phantora-qh07u-keep/torch-deps` (QH-02 prefix untouched). Native `patches/layout-native.patch` on top of QH-07t: optional JSON fields (older traces still parse), `TorchValue::Tensor` gains `Option<Layout>` (size-1 strides stored as 0), `QwenOp` gains `layouts` only for `cat` and `contiguous`, all members or none; empty `layouts` is omitted so every existing key is byte-identical. `qwen_ops.py` rebuilds each input on its own storage with `as_strided` when layouts are present; old keys keep the guessed source. Rust and Python regressions added. Expected to move: charged `cat`/`contiguous` time and the per-shape layout gate. Expected unchanged: every other key and its time. Run script `patches/qh07u-layout.sh` (parent) | Native build exit 0: 16 Rust tests pass (1 ignored; new `traced_layouts_reach_copy_keys_and_legacy_traces_keep_old_keys`), 17 Python ops/kernels tests pass (new `test_traced_layouts_are_rebuilt_exactly`). Torch wheel (18-step incremental, no CUDA recompiles) exit 0 in 73 s; tracer string verified in `libtorch_cuda.so`. Discovery exit 0, 2,540.430 ms (unchanged, as expected); 34 missing keys = 20 layout-carrying cat/contiguous + the same 14 zero keys; no large-message warnings. Profile on GPU 0 exit 0, all 20 layout keys positive. Replay 2,651.694 ms (+46.03 vs QH-07t's 2,605.662); vs real prefetch 3,030.833 −12.51% (was −14.03), no-prefetch −16.92%. Charged per rank-step cat 50.4/53.4 ms, contiguous 60.9/64.7 ms (was 25.3, 39.9). Layout gate 8/12 pass (was 1/12): GDN split-backward cat 230/233.2 µs, contiguous `[1,8192,8,128]` 51.8/55.6, all rotary cats, SwiGLU cat 43/44.4. Remaining fails: row-order GDN `[1,8192,1024]x3` cat 0.71 and gated-attention cat 0.73 (not layout; A9), contiguous `[8192,1,4,256]` 0.58 (fused-count mismatch), `[1,8192,8]` 0.83 (2–3 µs). CSV columns/row added in `results/qwen35_08b_sequence_sweep/ablation_seq8192*.csv`. Handlers still not applied to the sweeps. `gpu01:/home/dathar/Phantora/qh07u-layout-IxfR7j` |
| 2026-11-17 | QH-07v: rerun the two sweep points closest to real with the QH-07u build (cat/contiguous handlers + traced layouts, patched torch prefix). Each starts from its own sweep's final DB and real numbers are reused (real runs do not depend on the simulator build): 4-layer seq 24576 (QH-07q DB, sweep replay 3,778.2 ms, real prefetch 4,086.2) and 24-layer seq 16384 (QH-07n DB, sweep replay 7,410.8 ms, real prefetch 8,444.1). Tests = QH-07t tests (contiguous no-op filter) + QH-07q `qwen_hybrid_config.py` (adds `0.8b-l4`). Expected to move: Phantora step time, up by the newly charged cat/contiguous; nothing else. Script `patches/qh07v-points.sh` (parent) | Both points: discovery reproduces the sweep replay exactly (3,778.226 / 7,410.806 ms), 34 missing keys = 20 layout + the same 14 zero keys, no large-message warnings, profile exit 0, no other positive keys. Replays: 4-layer 24576 3,843.538 ms (+65.31; cat 33.21 + contiguous 32.10 per rank-step), −5.94% vs real prefetch 4,086.193 (was −7.54%); 24-layer 16384 7,660.502 ms (+249.70; cat 128.26 + contiguous 121.44), −9.28% vs 8,444.144 (was −12.24%). Per-op real times not attributed at these sizes. Columns `*_qh07v` added to both sweep CSVs. `gpu01:/home/dathar/Phantora/qh07v-points-aQv4GH` |
| 2026-11-17 | QH-07w MTP. Checkpoint inspection: Qwen3.5-0.8B (pinned revision 2fc0636) and 9B (c202236) `config.json` both give `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`; 0.8B weight names `mtp.{pre_fc_norm_embedding,pre_fc_norm_hidden,fc,norm}` and `mtp.layers.0.{self_attn.{q,k,v,o}_proj,q_norm,k_norm,mlp.{gate,up,down}_proj,input_layernorm,post_attention_layernorm}`, i.e. one full-attention layer. Source review of Megatron core in `upgraded-deps` (`transformer/multi_token_prediction.py`, `models/gpt/gpt_model.py`, `pipeline_parallel/schedules.py`): with CP=1 and no packed sequences, MTP has no host reads of tensor values and no data-dependent shapes (`roll_tensor` = `torch.roll` + `fill_` + `sum`; loss scale from `torch.ones(1, device)`), so Phantora's data-independence rule holds. Packed-sequence roll (`_roll_tensor_packed_seq`) walks `cu_seqlens` and is excluded with packing. Change: `ModelSpec.mtp_num_hidden_layers`, `HybridConfig.mtp_num_layers` (replaces the rejected `mtp` flag), `build_hybrid_mtp_spec`, MTP structure checks, `--mtp-layers` in `test_megatron.py`, `--mtp` in gpu01 `run.sh`/`simulate.py`/`real_inside.sh`. Unit tests: `python3 -m unittest discover -p 'test_qwen_hybrid_*.py'` in `phantora-gpu01.sif` on a temporary copy | 72 passed (HEAD: 68 passed); no GPU run yet |

## Implementation map and remaining execution gates

| Area | Reviewable files |
|---|---|
| CPU config/global layers/batch contracts | `tests/qwen_hybrid_config.py`, config/model tests |
| Local dense model and training integration | `tests/qwen_hybrid_model.py`, `tests/test_megatron.py`, `tests/qwen_hybrid_runtime_check.py` |
| Simulated fused autograd and coverage observation | `tests/qwen_hybrid_kernels.py`, `tests/qwen_hybrid_control.py`, `tests/qwen_hybrid_observer.py` |
| Native transport, timing and strict status | `phantora/cuda_call/src/{lib,capi}.rs`, `stub/cudart.c`, `phantora/phantora/src/{gdn,coverage,simulator,main}.rs` |
| Ordinary-op hybrid mappings | `phantora/phantora/src/{torch_call,torch_estimate}.rs`, `phantora/phantora/src/qwen_ops.py`, `tests/perfdb/bench.py` |
| Real-only fused profiling/reference equations | `tests/perfdb/gdn_{common,profile,references,validate}.py`, `tests/perfdb/GDN.md` |
| All-host replay/matrix/metrics | `tests/docker/megatron/hybrid_{run,matrix}.py`, `tests/qwen_hybrid_report.py`, `tests/docker/megatron/HYBRID.md` |
| Isolated dependency candidate stack | `Dockerfile` target `phantora-hybrid`; untargeted final alias remains `phantora-default` |

Continue in this order, with no promotion of a gate without executed evidence:

1. **QH-02:** Linux native build/tests/format, exact runtime import/signatures and
   the existing dense Llama recorded workload on both original and upgraded
   images. Revise candidate pins only if measured compatibility evidence requires it.
2. **QH-03/04/05a:** Actual initialized debug forward/backward and shape guards;
   native serialization, DB lookup, translated stream ordering, observer CUDA/TLS
   behavior, deliberately missing/dropped events and worker failure cleanup.
   Quantify/correct A32 before accepting whole-model HF numerical equivalence.
3. **QH-07 discovery:** Obtain actual debug traces, then every full-depth case.
   Resolve unsupported outermost operations (including copy/layout/composite and
   initialized factories) with accurate keys/timings; do not add broad exemptions.
   This is remaining implementation work informed by the Linux trace, not a
   claim that current strict replay already has complete operator coverage.
4. **QH-06/07:** Run actual fused numerical checks, profile all ordinary/fused
   missing signatures on the matching GPU/stack, and repeat strict all-rank replay
   until each accepted configuration has no misses or unclassified compute.
5. **QH-08:** Collect successful all-rank hardware runs and observed matching
   metadata, compare each case at <=10% error, investigate discrepancies and retain
   explicit CPU/network/layout/memory limits. No timing multipliers to conceal error.
6. **QH-09:** Only after the Qwen3.5 milestone passes, implement and separately
   validate a clearly labeled dense Next-style adaptation.

**Current evidence:** 63 passing local CPU tests, Linux native build plus 7 Rust
unit tests, scoped Blackwell Llama timings, exact-wheel no-miss Llama replay on
both Core versions, production runtime probes and four real steps each for
single-GPU debug and the 24-layer two-GPU 0.8B preset.
**Still pending:** Docker build, fused numerical validation, whole-model fidelity,
strict hybrid replay, distributed 9B and hardware calibration. No GDN timing rows
have been produced. Peak simulation memory remains unvalidated.

## Decision and change log

- 2026-10-29: User approved the full plan and requested implementation. Created
  this file before changing code. QH-01 starts next; later platform gates remain pending.
- 2026-10-29: QH-01 is an isolated stdlib-only configuration module and unittest
  suite first. Model/runtime imports remain unchanged until the contracts pass.
  Resolved the official model config revision for reproducible preset provenance.
- 2026-10-29: Accepted QH-01's initial 10 passing CPU tests; requested explicit
  causal-convolution-width and finite-RoPE guards in the next batch. Started
  QH-02/03 code preparation; their Linux integration gates remain pending.
- 2026-10-29: Parent found a TP correctness requirement in Core 0.16 source:
  head-shared GDN output-norm gradients need TP SUM, but the default finalizer's
  Q/K/sequence-parallel predicates do not select them. Requested a focused numeric
  shard-gradient test and minimal hybrid synchronization integration.
- 2026-10-29: Reviewed QH-02/03 prepared code and 19 passing CPU tests. The
  source-built causal convolution is pinned to patched torch; helper pins are
  packaging 24.2, einops 0.8.1 and ninja 1.11.1.4. Installed runtime compatibility
  remains pending. Started QH-04 with explicit native events rather than latency-only injection.
- 2026-10-29: Reviewed QH-04 sources and 28 passing CPU tests. Added measured-only
  `gdn.csv` and separate `gdn.missing.csv` / `gdn.status.json` discovery artifacts.
  Float options canonicalize through IEEE-754 bits in Rust; incoming gradient
  layout is part of backward keys. Native serialization/DB tests are prepared,
  while executed stream-order validation remains pending Linux. Started QH-05.
- 2026-10-29: QH-05 review found new ordinary-op mappings would change legacy
  replay coverage requirements. Gate them behind opt-in hybrid coverage while
  preserving legacy parsing and exponent-2 timing keys. Reviewed the isolated
  runner's per-run Compose project, bounded worker waits and final status contract;
  actual Compose/network/exit integration remains pending Linux.
- 2026-10-29: Pinned upstream revisions and corrected A22 against the pinned
  Qwen equation source. The earlier FP32-through-gate output-norm implementation
  introduces a different rounding path; requested cast-before-scale/gate and a
  BF16 regression. Zero-centered pre/post/QK normalization is a separate rule.
- 2026-10-29: Resumed the requested Medium agent after a service usage-limit
  interruption. Preserve earlier passing evidence and rerun only after remaining
  coverage/runtime review changes. Added A28/A29 from ordinary DB source review;
  strict zero-placeholder validation is required before a valid hybrid report.
- 2026-10-29: Split image-build metadata checks from real/replay imports and API
  checks; a GPU-free Docker build must not claim real FLA initialization passed.
  Forty CPU tests pass after the normalization correction. Native-source changes
  for zero-placeholder rediscovery are the remaining QH-05 review batch.
- 2026-10-29: Reviewed final QH-05 implementation and 41 passing CPU tests.
  Strict all-rank acceptance is implemented, not Linux validated. Start QH-06
  code preparation: separate actual forward/backward timing with strict provenance,
  independent reference equations and max-rank reports. Preserve failed measurements
  as failures, not usable DB rows. Prepare all five debug/full-depth configurations;
  QH-07/08 execution and QH-09 remain behind their environment/fidelity gates.
- 2026-10-29: Pinned tracer source became accessible via a small approved
  read-only fetch. This resolves A30 and exposes A34: server-side device parsing
  cannot recover events never sent. Add QH-05a after QH-06's independent tooling
  batch, with per-rank activation and conservative factory coverage failures.
- 2026-10-29: QH-06 source/local batch reached 53 passing tests. Parent reviewed
  real-only profiling, atomic measurement retention and report math; requested
  stronger numerical anti-zero checks, all-rank lifecycle/effective-token report
  validation and broader actual-model parameter-shape guards. Start QH-05a and
  these focused review corrections before the final combined local gate.
- 2026-10-29: Corrected the report policy contract: matched hardware/software
  settings do not imply hardware CPU time was excluded. Preserve the simulator's
  CPU-time assumption as a visible comparison limit and record both measurement
  policies explicitly (A36).
- 2026-10-29: Final combined local gate reached 61 passing tests after factory
  metadata, stronger status/workload evidence, FP32 convolution/relative-error
  validation, CUDA/u64 provenance and verified parameter-shape guards. Reviewed
  implementation and preserved the pre-existing untracked `AGENTS.md`. All code
  and test orchestration stayed with the requested Medium subagent; parent handled
  source review and this tracker. Remaining Linux/GPU/coverage/fidelity/Next work is
  explicitly ordered above; the end-to-end objective is not marked complete.
- 2026-11-07: User chose to keep vanilla separate and publish their changes on
  `Phantora-Het/main`, then authorized proceeding. Start QH-P01 with the requested
  Sol Medium agent executing Git checks/commit/push and the parent recording
  evidence. Preserve all pending Linux/GPU gates; publication is a handoff, not
  validation of the simulation objective.
- 2026-11-07: Accepted the Medium agent's publication preflight for the 43 task
  files. Proceed with a normal commit and non-force push to the verified origin
  on main, then record the actual implementation commit and remote result here.
- 2026-11-07: Implementation commit `9adeef71f58c20b8210d387bd20d84405062d177`
  was pushed and independently checked against the remote main ref. Parent
  reviewed the 43-file commit, clean task worktree and publication log. Record
  this result in a tracker-only follow-up commit; resume QH-02 on Linux next.
  The source is now transferable through GitHub; Linux/GPU validation and the
  remaining operator/fidelity work remain pending.
- 2026-11-14: User authorized `gpu01` for the upgrade. Start QH-02 with read-only
  host and capacity checks, then use a scoped checkout, distinct image tags and
  separate run artifacts. Compare the same Llama baseline before and after the
  candidate upgrade. Keep QH-03 and later gates pending until compatibility is
  demonstrated; do not change the system Python or the default image tag.
- 2026-11-14: Prefer the existing Apptainer SIF on gpu01 if it passes revision
  and Blackwell arch checks. The Docker target is separate and still unbuilt;
  its inherited CUDA arch settings (8.0/9.0) do not include gpu01's sm_120.
  Track the host-specific procedure and verify the same dependency pins.
- 2026-11-14: Continue the independent native compile and Core 0.13.1 Llama
  baseline on the existing SIF while preparing a revision-bearing Torch build.
  Keep the `git_version` guard intact. Historical source checkout and compiled
  CUDA arch are supporting evidence, not substitutes for the exact runtime pin.
- 2026-11-14: Start the independent packaging correction (A41) while native
  compilation runs: add the missing FLA core pin, record the actual kernel
  distribution in timing provenance, and include the required hybrid helpers
  in the build context. Scope the change to opt-in paths. Do not weaken the
  Torch identity gate or claim Docker support from source checks.
- 2026-11-14: After the original Llama replay passed, start the candidate install
  into a user-owned Apptainer package prefix activated only for the candidate.
  Keep the existing SIF and Torch binaries fixed for the comparison. Diagnostic
  imports and Llama replay can isolate package/API compatibility before running
  the exact provenance gate. Causal-conv1d 1.5.3.post1's source builds for sm_120
  with nvcc >=12.8; request a source build and record the actual result.
- 2026-11-14: Accepted the opt-in packaging fix after 62 passing CPU tests.
  Retain the versioned `fla` timing field, now based on the actual `fla-core`
  distribution; require the wrapper pin too. Candidate wheel installation on
  gpu01 succeeded. Handle the observed tokenizer API rename without changing
  default-image behavior, then rerun the controlled comparison.
- 2026-11-14: Use the existing CPU metadata group for the newly observed
  optimizer setup `all_gather_object` on simulated ranks. Never reroute numeric
  training collectives. In parallel, check byte identity of the SIF's retained
  Torch source to the pinned Git tree; only reuse a retained build cache if
  inputs match and the unmodified upstream build emits the expected revision.
- 2026-11-14: Candidate app executes both Llama ranks after the API fixes, but
  original L40S timing coverage is incomplete. Do not fill the missing SwiGLU
  add with a placeholder or a Blackwell timing in the L40S DB. Use one idle
  gpu01 device to record both versions into a separate Blackwell DB and replay
  from it without real GPU devices. This is ordinary Llama compatibility, not
  Qwen or calibrated performance evidence; Torch provenance remains independent.
- 2026-11-14: Verified the retained source against the exact pinned Git and
  recursive submodule objects, allowing only Git-declared line endings. Reuse
  the cache in a user-owned copy at the original container paths, rerun the
  dry-run check, then use upstream `pip wheel --no-index --no-deps
  --no-build-isolation` to generate a revision-bearing wheel. Verify the wheel's
  import origin, loaded libraries, exact SHA, CUDA 12.8, sm_120 and tracer.
  Preserve the old SIF and publish no timing from it as exact hybrid evidence.
- 2026-11-14: Accept the controlled Llama compatibility comparison on the
  provisional SIF (one Blackwell profiler GPU, two simulated ranks, identical
  replay DB; both workloads complete with no misses). Retain the L40S failure
  as historical evidence. Keep exact runtime and Qwen gates pending separately.
- 2026-11-14: The isolated Torch wheel now carries the verified revision from
  upstream's build, and loaded libraries were verified to come from that wheel.
  Use it for both sides of the final Llama gate and rebuild causal-conv1d against
  it to remove the provisional extension ABI assumption. The candidate checker
  must pass unmodified in replay and real modes before starting real Qwen.
- 2026-11-14: Queue the first QH-03 smoke after QH-02 passes: on one idle gpu01
  device, no PHANTORA/preload, `PHANTORA_RUN_ID=real-debug-single torchrun
  --standalone --nproc-per-node=1 tests/test_megatron.py --hybrid-preset debug
  --tensor_parallel_size 1 --data_parallel_size 1 --pipeline_model_parallel_size 1
  --iterations 4`. Require the expected 3 GDN / 1 full layer shape assertions,
  four completed optimizer iterations and 1,024 tokens/iteration. Smoke evidence
  does not pass model fidelity, distributed replay, or the 10% target.
- 2026-11-14: Accept QH-02 on gpu01 Apptainer after both exact candidate probes
  and controlled Llama replay gates passed. Preserve the original Core 0.13.1
  runtime. Start the bounded single-GPU QH-03 debug smoke. The separate Docker
  build, real model numerical checks and performance calibration remain open.
- 2026-11-14: First real debug identified a Triton binary-load failure. Keep
  dependencies and driver fixed for one diagnostic retry. Triton 3.7.0's
  [NVIDIA compiler](https://raw.githubusercontent.com/triton-lang/triton/v3.7.0/third_party/nvidia/backend/compiler.py)
  selects a separate Blackwell assembler for compute capability >=10.0; the
  [knob](https://raw.githubusercontent.com/triton-lang/triton/v3.7.0/python/triton/knobs.py)
  is `TRITON_PTXAS_BLACKWELL_PATH`. Use `/usr/local/cuda/bin/ptxas` and a fresh
  private cache; measure the actual outcome before attributing cause.
- 2026-11-14: The scoped CUDA 12.8 assembler retry passed the full four-step
  single-GPU debug smoke. Adopt this run setting on the verified gpu01 stack
  and keep it in future measurement provenance. Do not interpret the four steps
  as numerical equivalence, TP/DP/PP validation or accepted performance data.
- 2026-11-14: Add QH-03a at the user's request. Inspect the [official 0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json),
  honor tied embeddings and keep the debug and 9B choices unchanged. Start
  with a short real two-GPU BF16 training run; keep simulation and checkpoint
  fidelity separate. The full 9B objective remains open.
- 2026-11-14: QH-03a passed on both Blackwell GPUs, using one KV head per TP
  rank, the existing verified runtime and pinned official text config. Retain
  this short workload as a real training smoke; do not treat it as calibrated
  throughput across contexts or simulated 0.8B coverage. Local changes unpushed.
- 2026-11-15: Start only QH-07a at the user's request. Reuse exact one-host
  TP2/DP1/PP1 real workload and compare to a separate Phantora run with complete
  coverage. Start with prior same-host 8 MiB NCCL measurement (271.916 Gbit/s)
  as a constant-rate assumption; it does not measure 0.8B's smaller collectives.
  Replay excludes CPU time, real wall clock includes it. Do not tune links or
  compute times to the training target. Discovery output is not a comparison.
- 2026-11-15: User accepts leaving original coverage gaps for an exploratory
  comparison. Keep all native diagnostics and strict validation unchanged. A
  partially covered estimate is distinct from a valid replay; measure fused GDN
  rather than treating the defining operations as free. The discovery set is not
  proof that every missing signature also occurred in original Phantora.
- 2026-11-15: QH-07c starts with the [original design paper, §3–4](https://danyangzhuo.com/papers/NSDI26-Phantora.pdf)
  and the original parser, then the layer-boundary ledger in HYBRID.md. The
  exact replay used 144 conv/chunk and 288 L2 calls per direction per rank per
  step. Summing their measured DB entries charges 13.133 ms per rank per step
  for the three fused boundaries; it is already in the model and is not the
  total cost of 18 GDN layers or a measurement of critical-path contribution.
  Start with `square` shared across the Qwen RMSNorm paths, and the recorded
  `exp` out overload. [Core 0.16 cross-entropy source](https://raw.githubusercontent.com/NVIDIA/Megatron-LM/core_v0.16.0/megatron/core/tensor_parallel/cross_entropy.py)
  aliases that call's input/output. The existing shape/dtype profiler already
  aliases equal-size/type buffers; retain and disclose its alias/stride limit.
  No blanket changes to inherited ops or CPU policy during this patch.
- 2026-11-15: QH-07d starts with five FP32 keys extracted from QH-07c's
  `perfdb.missing/compute.csv`. Four `square` inputs: `[128,1,1024]` (block/final
  RMSNorm), `[1024,128]` (consistent with flattened per-head GDN output norm),
  `[128,1,4,256]` and `[128,1,1,256]` (rank-local Q and K norms). One `exp_out`
  input/output `[128,1,124160]` (half the vocabulary, TP2 cross-entropy).
  All are ATen timing events, separate from the six already-measured fused GDN
  keys. Profile only these five under the exact Blackwell runtime. Keep fixed
  network/CPU settings to isolate the change and preserve the before/after DBs.
- 2026-11-16: QH-07j–m. The seq-128 0.8B case is host-bound, so its 9x gap is
  outside what a GPU-work replay with CPU time ignored can predict; stop adding
  timing keys to close it. Use seq 8192 (MB 1, GA 8) as the 0.8B timing
  comparison point: GPU-bound, 42 GiB of 98 GiB. Seq 16384 fits but with ~2 GiB
  headroom. Modeling host launch time is a separate, undecided option.
- Corrections to pasted specification: dense Qwen3.5 is an official target;
  dense Next is an adaptation; gated full attention and exact normalization matter;
  timing coverage must include surrounding ops and all rank-local/full-depth keys;
  cuda_add_latency is in microseconds; strict failure needs runner coordination;
  placeholder autograd does not establish activation-memory accuracy.

## Sources

- Qwen3.5-9B config: https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json
- Qwen3.5 reference equations: https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- Qwen3-Next config: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json
- Core 0.16 GDN: https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/ssm/gated_delta_net.py
- Core 0.16 builder: https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/models/gpt/experimental_attention_variant_module_specs.py
- FLA 0.4.0: https://github.com/fla-org/flash-linear-attention/tree/v0.4.0
- Version identities: https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.16.0 ; https://github.com/fla-org/flash-linear-attention/releases/tag/v0.4.0 ; https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.3.post1
- Patched tracer source: https://github.com/QDelta/pytorch/blob/bbb19f8c856fcf8d7c60267fecb976b4749e2cff/torch/csrc/profiler/standalone/function_tracer.cpp

Mutable upstream sources must be recorded at their actual revision when vendoring
reference fixtures or accepting runtime/timing evidence.
