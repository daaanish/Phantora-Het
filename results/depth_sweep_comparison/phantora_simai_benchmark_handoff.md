# Phantora and SimAI Benchmark Handoff

gpu01 depth sweep · November 12, 2026

We completed six dense Megatron training cases on gpu01, spanning Llama-style and GPT-style models at 8, 16, and 32 layers. Both SimAI backends were closer to the measured iteration time in all six cases. The generated LayerNorm all-reduce contributes more time than SimAI’s lead over Phantora in each case.

## Iteration Time Results

| Model | Real Megatron | Phantora | SimAI analytical | SimAI ns-3 | Analytical less LayerNorm | ns-3 less LayerNorm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama 8 | 229.0 | 165.3 | 198.0 | 199.1 | 145.3 | 145.9 |
| Llama 16 | 432.5 | 328.2 | 373.2 | 375.1 | 283.5 | 284.6 |
| Llama 32 | 838.1 | 621.7 | 721.0 | 724.7 | 557.3 | 559.6 |
| GPT 8 | 199.8 | 143.7 | 179.6 | 180.6 | 126.9 | 127.5 |
| GPT 16 | 373.4 | 275.5 | 335.2 | 337.2 | 245.5 | 246.7 |
| GPT 32 | 720.7 | 538.4 | 650.5 | 654.1 | 486.7 | 489.0 |

Times are milliseconds per training step. The last two columns subtract each backend’s reported LayerNorm communication time from its saved total. These are arithmetic estimates.

Across the six cases, Phantora was 24.1–28.1% below real and SimAI was 9.2–14.0% below real. Subtracting the reported collective cost increases the SimAI shortfall to 32.1–36.5%.

## Optimizer Phase Measurements

| Model | Real AdamW | Phantora AdamW | Real FP32 to BF16 parameter copy | Phantora FP32 to BF16 parameter copy |
| --- | ---: | ---: | ---: | ---: |
| Llama 8 | 51.1 | 20.2 | 3.8 | 0.0 |
| Llama 16 | 95.1 | 37.6 | 7.0 | 0.0 |
| Llama 32 | 182.8 | 72.5 | 13.5 | 0.0 |
| GPT 8 | 41.6 | 12.8 | 3.2 | 0.0 |
| GPT 16 | 75.8 | 22.7 | 5.8 | 0.0 |
| GPT 32 | 144.0 | 42.7 | 10.8 | 0.0 |

Separate instrumented passes, in milliseconds per step. A zero means the timer printed 0.000 ms. The AdamW gap grows from 31 to 110 ms for Llama and from 29 to 101 ms for GPT.

## Benchmark Setup and Data

### Shared Settings

Both NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs were on gpu01. Each case used BF16, tensor parallelism 2, pipeline and data parallelism 1, sequence length 1,024, microbatch 1, gradient accumulation 4, and global batch 4. The sweep held hidden size 4,096, FFN size 11,008, 32 attention heads, and vocabulary 32,000 fixed. Linear bias, activation recomputation, and sequence parallelism were off.

Llama used RMSNorm, RoPE with base 10,000, and SwiGLU. SimAI profiled it through the existing November 9 RMSNorm and RoPE patch. GPT used LayerNorm, learned absolute position embeddings, and GELU, with the built-in AICB profile. The installed Megatron local layer spec uses dot-product attention. Both AICB launchers used the non-Flash path.

Phantora excluded CPU timing and enabled CUPTI. Both simulators used the measured 8 MiB NCCL all-reduce rate of 271.916 Gbit/s. The SimAI analytical input was 31.655 GiB/s. SimAI used an H100 communication-model class and profiled compute on Blackwell. Both backends read the generated workload directly.

### Timing Windows

Headline Megatron and Phantora times use the slower rank’s mean over zero-based iterations 10–19 of 20. Separate phase passes use the same workload settings and report phase averages over iterations 1–19. The phase CSV flags the rank selected by its iterations 10–19 mean. Megatron phase timers synchronize CUDA and can change iteration time. Forward, backward, and communication timers can nest.

### Local Results Files

The CSVs and PNGs are beside this handoff in results/depth_sweep_comparison. The CSVs give the source paths on gpu01.

- [training_iteration_time_comparison.csv](training_iteration_time_comparison.csv) — Headline times, signed error, and LayerNorm subtraction estimates.

- [megatron_and_phantora_phase_timings_by_rank.csv](megatron_and_phantora_phase_timings_by_rank.csv) — Both ranks’ phase timers and instrumented iteration means.

- [simai_compute_breakdown.csv](simai_compute_breakdown.csv) — Modeled compute by direction and workload component.

- [simai_communication_breakdown.csv](simai_communication_breakdown.csv) — Total and exposed communication, LayerNorm payload, and collective costs.

- [simai_per_layer_compute_and_communication.csv](simai_per_layer_compute_and_communication.csv) — All 1,984 workload rows across the six cases and two backends.

Companion plots cover scaling and simulator error, Megatron and Phantora phases, and SimAI compute and exposed communication.

### Software Provenance

The gpu01 Phantora checkout is at 301d9d976a13067cc0f5438912395a11efba5f8e. The launchers use the saved phantora-gpu01.sif and the installed SimAI analytical and ns-3 binaries under /home/dathar/simai-vanilla. The exact build revisions embedded in the binaries are unverified.

## Rerun Commands

Connect to gpu01 with access to the saved launchers and images and write access to the benchmark directory. Run the sweep when both GPUs are idle. The commands below copy the verified templates into fresh directories and run the six cases sequentially.

```bash
ssh gpu01.pronet.rcac.purdue.edu
```

On gpu01, enter Bash and check the devices before pasting the rest of the block.

```bash
bash
nvidia-smi
```

```bash
base=/home/dathar/Phantora/Phantora/tests/docker/megatron
sweep="$base/depth-sweep-20261112-215835-8rO3ua"
llama32="$base/results-llama-existing-aicb-patch-20261112-135914-KJyjKh"
gpt32="$base/results-gpt-native-simai-20261112-135843-7XAfEX"

rerun() (
    set -euo pipefail
    name=$1
    template=$2
    run=$(mktemp -d "$base/rerun-$name-$(date +%Y%m%d-%H%M%S)-XXXXXX")
    cp -p "$template"/{run_comparison.sh,run_phantora.sh} "$run/"
    cp -p "$template"/{nccl.json,netconfig.toml,gpu_topology.txt} "$run/"
    printf 'Results directory %s\n' "$run"
    bash "$run/run_comparison.sh" 2>&1 | tee "$run/comparison.log"
    cat "$run/summary.csv"
)

rerun llama-8  "$sweep/llama-8" &&
rerun gpt-8    "$sweep/gpt-8" &&
rerun llama-16 "$sweep/llama-16" &&
rerun gpt-16   "$sweep/gpt-16" &&
rerun llama-32 "$llama32" &&
rerun gpt-32   "$gpt32"
```

To rerun one case, define the variables and function, then run its line without the trailing double ampersand.

### Run Outputs

Each launcher writes summary.csv, model_args.txt, the backend commands, real.log, phantora_cpu1.log, both SimAI EndToEnd CSVs, and the generated workload under simai/aicb/results/workload. Verify that summary.csv contains all four methods and that the phases directory contains completed real and Phantora logs with timers for both ranks.

The existing 8- and 16-layer runs are the four model subdirectories under the saved sweep path in the block above. The 32-layer baselines are the two named run directories. Their original phase measurements are in phases-J5LbYR for Llama and phases-zbJj6g for GPT.

## Interpretation and Follow Up

### SimAI Collective and Compute Model

AICB emits the row named layernorm on this Megatron path when sequence parallelism is off. The row schedules a backward TP all-reduce at two bytes per modeled parameter. Both families receive the same payload at each depth despite different measured parameter counts in Megatron. The payload grows from 1.79 to 3.05 to 5.57 GB, and the reported cost grows from about 53 to 90 to 165 ms.

Megatron uses sequence-parallel gradient synchronization for selected replicated parameters when sequence parallelism is enabled. The full-parameter collective is absent from the dense Megatron path used here. AICB also uses forward timings as backward proxies, a synthetic gradient-parameter compute row, and an approximate collective schedule. The depth sweep shows that the saved SimAI estimate tracks the scaling trend. The match in totals leaves the contribution of these approximations unresolved.

### Phantora Operator Coverage and Profiling

Phantora runs the Megatron harness with intercepted GPU calls and schedules simulated work for recognized operations. The inspected dispatcher lacks handlers for several bulk AdamW operations and the FP32 to BF16 parameter conversion. The profiler also reuses same-size, same-dtype allocations across distinct operands in some supported bulk operations. The individual timing impact of coverage and allocation aliasing remains unmeasured.

Source notes are in Phantora-Vanilla/gpu01_timing_investigation_2026-11-08.md.

### Next Checks

Capture one warmed-up real iteration with optimizer kernel and NCCL records. Match the optimizer overloads, storage layouts, and collective sizes against the Phantora dispatcher and AICB workload. The local helper is Phantora-Vanilla/gpu01-apptainer/diagnose_timing.py. Keep traces in separate passes.

At one fixed depth, vary gradient accumulation while holding sequence length, microbatch, parallelism, and bandwidth fixed. This varies work per optimizer step while the LayerNorm payload stays tied to the model size. Preserve the generated workloads and collect phase timings for each case.

### Earlier Comparison Provenance

The original screenshot combined November 8 local Megatron and Phantora logs (893.144 and 689.071 ms) with November 9 SimAI estimates (630.615 ms analytical and 643.921 ms ns-3). The local log files are in Phantora-Vanilla with the prefix gpu01_llama2_7b_tp2_seq1024_ga4. The original SimAI CSVs are prediction-EndToEnd.csv and ns3/ncclFlowModel_EndToEnd.csv under the gpu01 directory below.

```text
/home/dathar/simai-vanilla/results/llama2-7b-approx-tp2-seq1024-ga4
```

A bandwidth-only rerun of that original workload produced 723.017 ms analytical and 726.695 ms ns-3. Its summary.csv is in the SimAI results directory named original-workload-measured-bw-20261112-133931-UbRZVt.

The earlier calibrated Llama run used activation recomputation and removed the LayerNorm row from the workload. Its summary.csv is in the Megatron benchmark directory named results-calibrated-20261112-131309-c3re1F.
