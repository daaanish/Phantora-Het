# Qwen hybrid training simulation: plan and evidence

Last updated: 2026-11-07 (America/Indiana/Indianapolis).

## Objective and current position

Simulate dense Qwen3.5-9B text-decoder training with TP, DP and PP using local
Megatron projections/MLPs and actual fused GDN kernel measurements. Add a clearly
labeled dense Next-style adaptation after the Qwen3.5 milestone is validated.

**Current phase:** local implementation/review batch finished; QH-02 Linux gate pending.
**Immediate handoff task:** QH-P01 is in progress: review and publish this local
implementation directly to `origin/main`, as the user requested. Vanilla remains
in its separate checkout; a feature branch is not required for this workflow.
**Next validation task:** on Linux x86_64 with the required CUDA stack, build the
opt-in image, run native tests and runtime checks, then validate the existing
recorded dense Llama workload in both default and upgraded images before Qwen.
Follow the ordered commands in `tests/docker/megatron/HYBRID.md` and append actual
environment/command/results here. No Linux/GPU gate has passed in this session.
**Completion:** not achieved. Configuration contracts pass locally; native
integration and GPU timing have not been validated. Pending hardware gates are not passing gates. Independent
code and CPU checks may be prepared while Linux/GPU validation remains pending.

Requested roles: Sol Max parent owns planning, assumptions, review and this file;
Sol Medium subagent writes code and orchestrates implementation/tests. Keep work
in small reviewed batches. Preserve user changes and existing runtime defaults.

## Baseline and environment

- Repository: `301d9d976a13067cc0f5438912395a11efba5f8e` (`main`).
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
- User selected GPU access later. Linux/CUDA build, real fused profiling and
  real multi-GPU calibration are pending suitable infrastructure.
- Publication destination: `https://github.com/daaanish/Phantora-Het.git`, branch
  `main`. On 2026-11-07, a fresh fetch confirmed local HEAD, `origin/main` and
  `FETCH_HEAD` all equal the baseline above, with no divergence.
- Publication scope: Qwen source, tests, Docker changes, runbooks and this tracker.
  The pre-existing untracked `AGENTS.md` stays local. Raw evidence under ignored
  `.cache/qwen-hybrid/` also stays local; its results are summarized below, and a
  Git clone will not contain those log files. No measured GPU timing DB exists yet.

## Tasks and gates

States: planned / in progress / implemented / passed / failed / pending environment.
The four evidence columns are independent; implementation does not imply validation.

| ID | Deliverable and dependencies | Owner | Implementation | Local tests | Linux tests | GPU validation | Acceptance gate |
|---|---|---|---|---|---|---|---|
| QH-00 | Central tracker and baseline | Parent | implemented | n/a | n/a | n/a | Plan, decisions, assumptions and evidence recorded |
| QH-01 | Config, global pattern, TP/DP/PP and batch contracts; QH-00 | Medium | implemented | passed (10 tests, hardened guards) | pending environment | n/a | Positive/negative CPU tests; exact effective batch |
| QH-02 | Isolated pinned hybrid Docker target; QH-01 | Medium | implemented | static/source checks passed | pending environment | pending environment | Imports/signatures/tracer and existing Llama baseline pass |
| QH-03 | Local hybrid layer spec, debug and 9B presets; QH-01/02 | Medium | implemented | CPU equation/wiring/CLI tests passed; BF16 correction checked | pending environment | pending environment | Correct global PP ownership and reference forward/backward |
| QH-04 | Autograd-safe fused boundaries, native events and timing DB; QH-01/03 | Medium | implemented | passed (28 cumulative CPU tests) | native tests prepared, pending environment | pending environment | Graph, serialization, key, stream ordering and no-double-count tests |
| QH-05 | Coverage/discovery/strict status and all-host runner; QH-04 | Medium | implemented | passed (41 cumulative CPU tests) | native/Compose tests pending environment | n/a | Missing events/timings and crashed ranks fail without hanging |
| QH-05a | Observe device-only CUDA factories omitted by pinned tracer; QH-05/source review | Medium | implemented | passed (61 cumulative CPU tests with review fixes) | CUDA/TLS/native tests pending environment | n/a | Per-rank observer active; allocation observations classified and initialized/unknown compute invalidates strict status |
| QH-06 | Actual fused profiler and provenance checks; QH-04/05/05a | Medium | implemented | passed (61 cumulative CPU tests) | pending environment | pending environment | Separate fwd/bwd measured; invalid or mismatched timings rejected |
| QH-07 | Debug and full-depth replay matrix; QH-02 through 06/05a | Medium + parent review | executable commands prepared; execution pending | 10-case command preparation passed | pending environment | pending environment | Complete per-config coverage and expected computation/communication |
| QH-08 | Hardware-matched calibration; QH-07 | Medium + parent review | report/compare tools prepared; execution pending | report math/mismatch cases passed | pending environment | pending environment | At most 10% steady-state iteration error in each accepted case |
| QH-09 | Dense Next-style adaptation; QH-08 | Medium | planned | planned | pending environment | pending environment | Explicit dense adaptation; separate validated configurations |
| QH-P01 | Publish the reviewed local batch for Linux handoff; QH-00 and local batch | Medium Git execution + parent review | in progress; ready to commit/push | preflight passed; existing CPU evidence retained | n/a | n/a | Reviewed task files committed and pushed to origin/main; remote commit verified; user-local instructions/logs preserved |

Before each task: record scope, dependencies, assumptions and acceptance checks.
After each test/change: append command, result and artifact location; update the
task's relevant evidence column and next action. Record failed attempts as well
as successes. Do not change a pending environment gate to passed without evidence.

## Implementation decisions

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
| A01 | verified user choice | Dense Qwen3.5 first; GPU access later | Hardware-dependent gates remain pending |
| A02 | deliberate simplification | BF16, fixed-length text, synthetic data | No numerical training-quality or dataset-throughput claim |
| A03 | deliberate simplification | CP=1, EP=1, no MoE | World size is TP*DP*PP; reject incompatible hybrid flags |
| A04 | deliberate simplification | No SP, virtual PP, recomputation, packing, caches, vision or MTP initially | Reject unsupported combinations rather than silently approximate |
| A05 | verified code behavior | Distributed optimizer and overlap are enabled | Preserve both DDP and optimizer settings; validate actual collectives |
| A06 | provisional dependency choice | Core 0.16.0, FLA 0.4.0, causal-conv1d 1.5.3.post1 | Verify imports/signatures/runtime before claiming support; record revisions |
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

**Final local evidence:** 61 passing CPU tests; 22 files parse; command preparation
and whitespace checks pass. **Not executed:** native Rust/C build/tests, Docker
image/runtime integration, real CUDA kernels or model, any distributed replay or
hardware calibration. **Not achieved:** validated Qwen3.5 distributed simulation,
10% accuracy milestone or dense Next preset. No measured timing rows were fabricated.

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
