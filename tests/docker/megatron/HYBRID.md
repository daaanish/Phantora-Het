# Opt-in hybrid replay runner

Prepared Linux command (native/runtime and GPU gates remain pending):

First run the existing Llama baseline on the original/default image with the
exact recorded replay configuration in the repository README. For example, the
existing preset entry point is `tests/docker/megatron/llama/run_llama3_8b.sh` after
`tests/docker/megatron/config_gen.py` has generated its matching configuration.
This baseline gate remains pending here; do not substitute a successful hybrid
CPU test for it. Then build the separate hybrid image:

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
