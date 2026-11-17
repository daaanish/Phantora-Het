# Opt-in hybrid replay runner

## Migration record

### 1. Original Phantora

**Existed:** Megatron Core 0.13.1, patched PyTorch 2.9.1/CUDA 12.8, ATen and
CUDA/NCCL interception, and timing DBs for recorded workloads. The two-rank
Llama replay passed. **Limit:** No Qwen3.5 GDN timing events, and the original
ATen parser had no dispatch case for any of the 39 compute names later reported
as unmapped in 0.8B discovery (e.g. `embedding`, `clone`, `zero_`). That is
source evidence of inherited parser gaps, not evidence Llama hit all 39.
**Changed:** Nothing in the original checkout or default runtime.

### 2. Megatron update: Core 0.13.1 → 0.16.0

**Existed:** Same Phantora trainer, PyTorch patch and Llama replay workload.
Core 0.16.0 was installed in a separate environment. The Llama control isolated
what broke before adding GDN:

| What broke | What changed |
|---|---|
| Trainer import: `datasets.megatron_tokenizer.MegatronTokenizer` was renamed. | Keep the 0.13.1 import; fall back to `MegatronLegacyTokenizer` on 0.16.0. |
| Optimizer setup: `_get_param_groups` calls `all_gather_object`; simulated NCCL moves no payload, so unpickling raised `EOFError`. | Route the default-group object gather through CPU Gloo only during simulated optimizer construction. Real training and numerical NCCL traffic are unchanged. |
| L40S timings: Core 0.16.0's unfused SwiGLU evaluates `x_linear + glu_linear_offset` even at zero offset. Llama TP2 missed `Add(BF16 [1024,1,7168], FP64 scalar)`. | Measure the new key on the target GPU. Both versions passed the same Llama replay with a fresh Blackwell DB. Keep L40S and Blackwell timings separate. |

**Packaging:** The original gpu01 SIF reported `git_version='Unknown'` because
its copied PyTorch submodule Git pointer was dangling. Rebuilt the pinned Torch
wheel with verified Git metadata; the revision check stayed intact.

### 3. Qwen3.5 GDN run

**Existed:** Core 0.16.0 supplies GDN, but its stock hybrid builder requires
Transformer Engine and the original Phantora trace does not time fused GDN
kernels. **Changed:** Use Phantora's local hybrid spec, and add explicit forward
and backward events for causal convolution, L2 norm and chunked GDN. Install
`flash-linear-attention==0.4.0` and `fla-core==0.4.0` explicitly with `--no-deps`;
build `causal-conv1d==1.5.3.post1` against the pinned Torch.

**What broke on gpu01:** Triton 3.7.0's bundled Blackwell `ptxas` 13.1 produced
`device kernel image is invalid` on driver 570.195.03. Set
`TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` (CUDA 12.8) inside
Apptainer. Give contained replay a writable `HOME` for DeepSpeed imports.

**Verified:** Real Qwen3.5-0.8B TP2 training passed. Phantora discovery finished
both ranks with all expected GDN events. After measuring all six GDN timing keys,
the partial replay estimated **0.095 s/step vs 0.898 s/step real**. It still omits
operator work and CPU time, so strict timing acceptance remains open. See the
[tracker](../../../QWEN_HYBRID_TRACKER.md) for the timing DB and remaining misses.

## What to model next

[Original Phantora](https://danyangzhuo.com/papers/NSDI26-Phantora.pdf) (§3–4)
runs the framework's actual layer schedule and models device work at operation
boundaries. Reuse the layer code, profile real GPU work once at each supported
boundary, and preserve stream and communication ordering. This does not call
for hand-written latency models for each layer or numerical tensor payloads.

| Boundary in the 0.8B case | What exists and what remains |
|---|---|
| GDN layers (18) | All six fused forward/backward timings were measured and used, with the expected counts on both ranks. Their sum is **13.13 ms of charged fused work per rank per step**; this excludes projections, norms, MLPs and communication and is not an isolated wall-time contribution. The fused GDN triplet is not missing. |
| Full-attention layers (6) and shared layer work | Megatron's local attention and projections already run; existing matmul and softmax times are reused. RMSNorm `square` is now recognized (5,056 calls); measure the four new keys. Device/layout `to`, rotary `sin`/`cos` and layout copies still need classification. Distinguish real copies from views; charge each once. |
| Model-wide training | The recorded two-tensor `exp` out overload at the vocab-shard logits shape is now recognized (one key to measure). Embedding/backward, optimizer foreach and initialized factories expose inherited gaps; leave them visible and prioritize by measured cost. |
| End-to-end comparison | Measuring the five newly recognized keys brought the missing compute count from 19 to 14. No copy or GDN timing misses. Real includes CPU time; replay excluded it and used a fixed measured network rate. CPU-on replay is a sensitivity check, not a measurement of real CPU time. |

The five new FP32 keys are four `square` shapes (`[128,1,1024]` block/final
RMSNorm, `[1024,128]` GDN head norm, `[128,1,4,256]` Q norm and
`[128,1,1,256]` K norm) and `exp_out` at `[128,1,124160]` for the rank's
vocabulary shard. `square` feeds each RMSNorm's mean-square reduction;
`exp_out` exponentiates the cross-entropy logits in place. Profile these five
on the same Blackwell and add them to a copy of the previous DB. Reuse the
same workload, network and CPU-time settings to isolate their effect.

**Result:** All five keys hit. The partial replay rose from **95.02 to 96.52
ms/step** on the same native build; real training took **897.54 ms/step**.
The added **1.50 ms/step** explains very little of the difference. Strict
timing validation remains open.

**Phase check (same Megatron timers as GPT/Llama):** Real/replay were
**907.78/96.21 ms/step** with instrumentation. Forward/backward accounted for
**857.91/90.71 ms**, about 95% of the gap; inner optimizer was **21.04/4.21 ms**.
In a separate real CUDA trace, TP all-reduce kernels on rank 0 often waited for
rank 1 to arrive (median start skew 0.696 ms, median finish skew 0.0005 ms).
The trace added overhead; it localizes a rank-progress problem but does not tell
us the unprofiled wait duration or its cause. CPU-on replay overshot at 3.33 s;
it counts replay process CPU time, so it does not fix the comparison.

**Cause (2026-11-16):** At sequence length 128 the real step is host-bound.
Each rank issues about 36,000 GPU activities per step averaging 4 µs, about
141 ms of GPU compute; on the pace-setting rank the GPU is empty with no launch
pending for 73% of a traced step. The peer waits at TP all-reduces are the
other rank's GPU parked in NCCL until the slower host arrives. Real step time
is flat as tokens grow 16x, then tracks GPU work:

| Sequence length (MB 1, GA 8) | Real s/step | Replay s/step | Difference | Peak device memory |
|---|---:|---:|---:|---:|
| 128 | 0.898 | 0.097 | −89.2% | ~9 GiB |
| 512 | 0.894 | not run | | not recorded |
| 2048 | 0.915 | not run | | not recorded |
| 4096 | 1.357 | 1.018 | −25.0% | 24 GiB |
| 8192 | 3.192 | 2.540 | −20.4% | 42 GiB |
| 16384 | 9.075 | 7.411 | −18.3% | 95.6 of 97.9 GiB |
| 32768 | OOM | not run | | — |

At 8192 the partial replay is 20.4% below real after measuring 161 compute,
7 copy and 6 fused keys at the new shapes, and at 16384 it is 18.3% below; 14 zero-activity compute keys are
still missing and strict validation remains open. The difference flattens
rather than falling linearly: for 4096–16384 the shortfall is close to
0.13 s + 0.207 × replay time, which would level off near −17%. The CSV is in
`results/qwen35_08b_sequence_sweep/` at the repository root.

**Running on gpu01:** from 2026-11-17 every gpu01 run goes through
[gpu01/README.md](gpu01/README.md): runs start from a pushed commit and record it.
The QH-02 to QH-07v run directories predate this and stay as records.

**Attribution at 8192 (2026-11-16):** A traced real step against a replay with
per-event logging splits the 670 ms gap into 350 ms of GPU compute Phantora does
not charge, 269 ms with the GPU empty between kernels, and 51 ms of all-reduce.
The compute part is 203 ms of unmapped operations (`contiguous`, `cat`, `sub_`,
`max`, `_foreach_*`), 121 ms of `aten::to` conversions in a call form the parser
rejects, and 176 ms of charged operations that come out low, less 156 ms that
come out high (`masked_fill` is charged once with its clone and again for the
intercepted memcpy). The 14 zero-activity keys contribute almost nothing.

**Batch prefetching (2026-11-16):** the trainer now fetches a step's batches
before the timer starts (`--no_prefetch_batches` restores the old path), which
removes a host stall at the start of every real step. Replay times do not
change. Against prefetch real times the replay is 22.7% low at 4096, 16.2% at
8192 and 12.2% at 16384; the table above keeps the earlier no-prefetch values. Use 8192 as the 0.8B timing
comparison point. A GPU-work replay with CPU time ignored cannot predict the
128 case; measuring more keys there will not close it.

## Docker replay

First run the existing Llama baseline on the original/default image with the
exact recorded replay configuration in the repository README. For example, the
existing preset entry point is `tests/docker/megatron/llama/run_llama3_8b.sh` after
`tests/docker/megatron/config_gen.py` has generated its matching configuration.
The Docker baseline remains pending; the `gpu01` compatibility result used
Apptainer. Then build the separate hybrid image:

```bash
docker build --target phantora-hybrid -t phantora:hybrid .
python3 tests/docker/megatron/hybrid_run.py --image phantora:hybrid \
  --perf-db tests/perfdb/l40s --preset debug --mode discovery \
  --tensor-parallel-size 2 --data-parallel-size 2 --pipeline-parallel-size 2 --ngpu 4
```

This is a GPU-free **simulation/replay** command using linked CUDA libraries, not
real distributed calibration. The example legacy DB has no accepted hybrid
coverage; its output must be INVALID until discovery is profiled and completed.
Use `--mode strict` for acceptance: any timing miss, unmapped/malformed CUDA
operation, stale/duplicate/missing rank, or fused count mismatch fails the run.
Discovery can complete with exit zero while `valid:false`; it is not evidence of
timing accuracy. Worker failures, missing rank exits and timeouts fail either mode.

Every invocation creates a UUID directory under `.cache/qwen-hybrid/runs/`
(`--output-dir` overrides its parent), a unique Compose project and scoped socket
volume. It waits for simulator readiness, starts and checks every host, then
requires all expected rank finish/exit notifications and final status with the
same run ID. It keeps the server running until workers exit. Timeout/failure
terminates client process groups and removes only this Compose stack. Each host,
the simulator, setup and cleanup have separate logs. `runner.json`, `status.json`,
`plan.json`, `workload.json`, and `perfdb.missing/` persist in that UUID directory.

The measured region starts after model/optimizer setup and ends after all training
iterations synchronize. Expected counts include warmup iterations:
`local_GDN_layers * accumulation * iterations` for conv/chunk forward and backward,
and twice that for L2norm. Initialization/teardown operator counts are outside this
region; missing timing shapes across the entire run still invalidate acceptance.
Core's default TP-DP-PP rank order determines the runner's global PP manifest;
registration must agree with the actual process group's ownership.

Only explicit metadata views and uninitialized `empty`/`empty_strided`/`empty_like`
allocations are exempt from compute lookup. The latter still use native allocation
accounting. Copy, clone, contiguous, reshape, zero-fill and arbitrary composite
parents are not blanket exemptions. Unmapped signatures are reported for review;
actual Linux trace validation must establish any additional classification.

Qwen-specific exp/sigmoid/SiLU/softplus/rsqrt and backward/reduction timing mappings
are enabled only with coverage. Dimension-specific sum/mean and scalar power
exponents are keyed accurately there. Exponent 2 retains its existing `Pow` key.
Legacy mode retains its inherited exponent/reduction simplifications and unmapped
operations so its existing DB/default behavior is unchanged. Ordinary timing keys
remain shape/dtype based and the ordinary profiler retains its allocation aliasing;
these are separate limitations from fused stride/provenance keys.

The image build runs distribution metadata checks only, without importing
torch/FLA/Triton or probing a driver. Real-kernel and replay-alias import/signature
checks run at workload startup and remain separate Linux gates. The runner requires
one nonempty DB GPU identity, passes it to simulated hosts, and records it with the
image, expected software stack, per-host virtual GPU count, full network config,
CPU-time policy and VRAM assumption in workload/runner artifacts. Software expected
pins are assumptions until `HYBRID_RUNTIME` startup evidence confirms them.

Opt-in startup discards unmeasured zero compute/transfer/flash rows so used keys
are rediscovered and invalidate strict acceptance. Narrow zero-cost exceptions are
identity dtype conversions, disabled/probability-zero functional dropout, zero-byte
transfers and host-only transfers. Legacy replay caches retain their prior behavior.
The stock CPU surrounding-op audit also sees `_to_copy`/`copy_`, `square` and
`ones_like`/`fill_` parents/children. These remain visible unmapped operations until
patched-tracer evidence establishes their correct boundaries; strict mode cannot
accept them silently. `neg` and in-place `div_` have explicit hybrid mappings.

The runner does not validate memory capacity. Hybrid peak-memory output is labeled
unvalidated. Real GPU counts/topology are still required for distributed timing
calibration; virtual ranks are not real multi-GPU evidence.

Prepared end-to-end Linux validation sequence (these hardware/native gates have
not run on the development Mac):

1. Run the unchanged Llama baseline above on its matching committed DB/image.
   Compile and run native tests, then build/check the isolated hybrid image:

   ```bash
   (cd phantora && cargo test -p phantora)
   docker build --target phantora-hybrid -t phantora:hybrid .
   python3 tests/qwen_hybrid_runtime_check.py
   ```

   The last command runs inside the real GPU image without PHANTORA/preload. For
   replay, the same checker runs at startup with simulation aliases installed.
   Build-time metadata checks alone do not establish either gate.

   Before any hybrid model execution, run the same existing dense Llama workload
   in the upgraded hybrid image as a second compatibility gate. This isolated
   Compose override leaves the default/latest image tags unchanged. From the
   repository root, reproduce the committed `record_all.sh` Llama configuration:

   ```bash
   cd tests/docker/megatron
   python3 config_gen.py --nhost 1 --ngpu 2 --vram_mib 81920 --perf-db l40s
   cat > /tmp/phantora-qh-llama-images.json <<'JSON'
   {"services":{"simulator":{"image":"phantora:hybrid"},"host-1":{"image":"phantora:hybrid"}}}
   JSON
   docker compose -p phantora-qh-llama -f compose.yaml -f /tmp/phantora-qh-llama-images.json up -d
   docker compose -p phantora-qh-llama -f compose.yaml -f /tmp/phantora-qh-llama-images.json exec -T host-1 \
     /phantora/dist/phantora_run torchrun --nproc_per_node 2 --nnodes 1 \
     --rdzv_backend c10d --rdzv_endpoint host-1:12345 /phantora/tests/test_megatron.py \
     --num_layers 4 --sequence_length 1024 --iterations 3 --tensor_parallel_size 2 \
     --hidden_size 4096 --ffn_hidden_size 14336 --num_attention_heads 32 \
     --num_query_groups 8 --vocab_size 128256 --swiglu --position_embedding_type rope \
     --rotary_base 500000 --normalization RMSNorm --disable_bias_linear
   docker compose -p phantora-qh-llama -f compose.yaml -f /tmp/phantora-qh-llama-images.json logs --no-color simulator
   docker compose -p phantora-qh-llama -f compose.yaml -f /tmp/phantora-qh-llama-images.json down --remove-orphans
   cd ../../..
   ```

   Archive worker exit status and simulator logs; missing timings invalidate the
   legacy timing gate. Both original-image and upgraded-image Llama gates remain
   pending in this Mac environment. The legacy Compose socket prefix is shared:
   finish/stop the original baseline before this isolated project runs.

2. Validate real fused equations and run initialized real debug training before
   replay. Use an actual GPU allocation and the same image/runtime, no stubs:

   ```bash
   python3 tests/perfdb/gdn_validate.py --gpu-name 'ACTUAL GPU NAME' \
     --out .cache/qwen-hybrid/fused-real.json
   PHANTORA_RUN_ID=real-debug-single torchrun --standalone --nproc-per-node=1 \
     tests/test_megatron.py --hybrid-preset debug --tensor_parallel_size 1 \
     --data_parallel_size 1 --pipeline_model_parallel_size 1 --iterations 4
   ```

   The trainer checks local/global layer ownership and GDN norm/projection
   shapes and emits unrounded `HYBRID_METRICS` for every rank. This exercises the
   initialized local backend, not whole-model HF numerical equivalence. Packed
   projection/L2 composition and BF16 A_log exponentiation differences remain
   explicit fidelity limits. Repeat actual training for each distributed case,
   using the allocation's real host/GPU topology and matching TP/DP/PP flags.

3. Prepare debug discovery then full-depth 9B five-case matrices:

   ```bash
   python3 tests/docker/megatron/hybrid_matrix.py --perf-db tests/perfdb/MATCHED_GPU \
     --preset both --mode discovery --gpus-per-host 1,2,1,1,4 \
     --out .cache/qwen-hybrid/discovery-matrix --execute
   ```

   Case order is TP/DP/PP 1/1/1, 2/1/1, 1/2/1, 1/1/2, 2/2/2.
   Values explicitly place cases on 1x1, 1x2, 2x1, 2x1, 2x4 virtual hosts/GPUs.
   They are replay assumptions, not evidence of the real hardware topology.
   Omit `--execute` to archive executable commands only. Debug precedes 9B.
   Discovery remains invalid and may expose unsupported outermost operators;
   resolve their real trace boundaries before claiming complete coverage.
   The pinned tracer sends only outermost ATen operators and drops device-only
   factories before native dispatch. An opt-in TorchDispatchMode observer now
   supplements those device-only operations during the region: uninitialized
   empty factories are allocations; CPU-to-CUDA `_to_copy` is an observation of
   the already modeled native transfer; initialized factories remain unsupported
   and invalidate strict acceptance. Every rank must activate/deactivate this
   observer in native lifecycle status. No semantic transfer timing is duplicated.
   Actual CUDA dispatch/backward/worker-thread visibility remains a Linux gate;
   mode activation alone does not prove complete coverage.

4. Profile discovered ordinary tables with the existing GPU profiler, then fused
   keys using the separate tool. Use keys from every case, including 32-layer
   foreach/optimizer shapes; a four-layer DB is not a full-depth DB:

   ```bash
   python3 tests/perfdb/bench.py --ref RUN/perfdb.missing --out tests/perfdb/MATCHED_GPU --merge
   python3 tests/perfdb/gdn_profile.py --keys RUN/perfdb.missing/gdn.missing.csv \
     --out tests/perfdb/MATCHED_GPU --gpu-name 'ACTUAL GPU NAME'
   python3 tests/docker/megatron/hybrid_matrix.py --perf-db tests/perfdb/MATCHED_GPU \
     --preset both --mode strict --gpus-per-host 1,2,1,1,4 \
     --out .cache/qwen-hybrid/strict-matrix --execute
   ```

   The ordinary profiler retains operand aliasing and shape/dtype-only keys;
   ordinary provenance remains inherited GPU-name-only. These are reported
   limitations. Fused timings have separate strict stack/GPU/CUDA/method pins.

5. Archive an observed `match.json` for each hardware/replay pair with exactly
   `workload`, `software`, `gpu`, `gpus_per_host`, `network`, `cpu_time_policy`.
   `workload` is the complete archived HybridConfig including its model snapshot;
   token count, TP*DP*PP and per-rank fused counts are checked against it.
   `gpus_per_host` is a list with one positive GPU count per actual/virtual host.
   CPU policy must explicitly be
   `{"replay":"ignored","hardware":"wall_clock_elapsed"}`. Hardware durations
   include CPU wall time; replay excludes modeled CPU time. This asymmetry is a
   comparison assumption and remains visible as each report's effective policy.
   These must reflect actual runtime versions and allocated topology. The runner's
   workload/assumptions and startup logs supply replay evidence; do not substitute
   expected software pins for observed versions. Hardware completion status must
   explicitly record its run ID, expected world, `phase:"complete"`, `valid:true`
   only after every real rank exited successfully. Build bundles and compare:

   ```bash
   python3 tests/qwen_hybrid_report.py --kind replay --status RUN/status.json --runner RUN/runner.json \
     --logs RUN/host-*.log --match match.json --save-bundle replay.json --out replay-report.json
   python3 tests/qwen_hybrid_report.py --kind hardware --status real-status.json \
     --logs real-host-*.log --match match.json --save-bundle hardware.json --out hardware-report.json
   python3 tests/qwen_hybrid_report.py --bundle replay.json --hardware-bundle hardware.json \
     --out comparison.json
   ```

   Reports require native versioned complete status, unique rank identities,
   observer/training/exit lifecycle, exact expected and actual fused counts, no
   unknown/malformed operators or timing misses, and matching successful all-host
   runner exit codes for replay. They require complete positive unrounded duration arrays from exactly all
   ranks, matching IDs/world/iterations/tokens and valid complete status. Replay
   additionally requires strict mode. They discard iteration 0, take the maximum
   rank duration for each remaining iteration, then average those maxima and
   divide global tokens by that time. Each matched case passes only at <=10%
   relative time error; no cross-case average can hide a failure. Network and
   CPU-time assumptions remain in output; memory is always **unvalidated**.
