#!/usr/bin/env bash
# Run one configuration against the commit checked out in $PH_CLONE. See README.md.
#
# usage: run.sh <name> --preset P --seq N [options]
#   --stages LIST     comma list from: discovery,profile,replay,real,phases-sim,phases-real
#                     (default discovery,profile,replay)
#   --phases          append phases-sim,phases-real: separate passes with Megatron phase timers
#                     (forward/backward/optimizer/grad sync). The timers synchronize CUDA, so they
#                     never run in the headline discovery/replay/real stages. phases-sim replays with
#                     the run's db (or --base-db); phases-real is a second real run.
#   --base-db DIR     perf DB to start from (required for discovery/replay); copied, never modified
#   --mb N            micro-batch size (default 1)
#   --ga N            gradient accumulation (default 8)
#   --tp N            tensor-parallel size (default 2)
#   --pp N            pipeline-parallel size (default 1)
#   --dp N            data-parallel size (default 1); GPUs = TP * PP * DP. --ga stays per replica,
#                     so the global batch is mb * ga * dp
#   --stage-layers L  uneven PP split, decoder layers per stage, e.g. 1,3 (default: Megatron's even split)
#   --iterations N    training iterations (default 4; step time = mean of iterations 2..N, slower rank)
#   --mtp N           train N multi-token-prediction layers (default: the preset's own depth, 1 for
#                     0.8b*/9b, 0 for debug; --mtp 0 = off, as in every run before
#                     2026-11-22 except QH-07w)
#   --vram-mib M      simulated GPU memory (default 97887; seq 32768 needed 131072)
#   --sim-log LEVEL   replay simulator log level (default debug: keeps per-event charges in events.jsonl.gz)
#   --expandable / --no-expandable   PYTORCH_CUDA_ALLOC_CONF=expandable_segments for the real run
#                     (default: on for the reduced-depth 0.8b-l* presets, off otherwise)
#   --no-compile-cache  real runs compile into a fresh per-run cache (as before 2026-11-23) instead
#                     of $PH_CACHE/real-compile/<torch build>/ (Triton, Inductor, CUDA JIT), which is
#                     kept across runs. Step time excludes iteration 0, where compilation happens.
#   --allow-unpushed  run even if the clone's HEAD is not on its upstream branch
#
# Output: $PH_RUNS/<YYYYmmdd-HHMMSS>-<name>-<commit8>/  (commit.txt, config.json, build.json,
# progress.log, one directory per stage, summary.json). Nothing outside it is written,
# except the build of the commit when it does not exist yet.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/paths.sh"

name=${1:?usage: run.sh <name> --preset P --seq N [options]}; shift
[[ $name =~ ^[A-Za-z0-9._-]+$ ]] || ph_die "name may use letters, digits, . _ - only"
preset="" seq="" stages="discovery,profile,replay" base_db="" mb=1 ga=8 tp=2 iters=4 vram=97887
sim_log=debug expandable="" allow_unpushed=0 mtp="" pp=1 dp=1 stage_layers="" phases=0 compile_cache=1
while (( $# )); do
    case $1 in
        --preset) preset=$2; shift 2;; --seq) seq=$2; shift 2;; --stages) stages=$2; shift 2;;
        --base-db) base_db=$(cd "$2" && pwd); shift 2;; --mb) mb=$2; shift 2;; --ga) ga=$2; shift 2;;
        --tp) tp=$2; shift 2;; --iterations) iters=$2; shift 2;; --vram-mib) vram=$2; shift 2;;
        --sim-log) sim_log=$2; shift 2;; --expandable) expandable=1; shift;; --no-expandable) expandable=0; shift;;
        --allow-unpushed) allow_unpushed=1; shift;; --mtp) mtp=$2; shift 2;;
        --phases) phases=1; shift;;
        --no-compile-cache) compile_cache=0; shift;;
        --pp) pp=$2; shift 2;; --dp) dp=$2; shift 2;; --stage-layers) stage_layers=$2; shift 2;;
        *) ph_die "unknown option $1";;
    esac
done
[[ -n $preset && -n $seq ]] || ph_die "--preset and --seq are required"
(( phases )) && stages="$stages,phases-sim,phases-real"
[[ -z $mtp ]] && { [[ $preset == debug ]] && mtp=0 || mtp=1; }   # the preset's released MTP depth
[[ $mtp =~ ^[0-9]+$ ]] || ph_die "--mtp takes a non-negative integer"
[[ $tp =~ ^[1-9][0-9]*$ && $pp =~ ^[1-9][0-9]*$ && $dp =~ ^[1-9][0-9]*$ ]] || ph_die "--tp, --pp and --dp take positive integers"
[[ -z $stage_layers || $stage_layers =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || ph_die "--stage-layers takes e.g. 1,3"
[[ -z $stage_layers ]] || (( $(tr ',' '\n' <<< "$stage_layers" | wc -l) == pp )) || ph_die "--stage-layers needs $pp entries"
world=$((tp * pp * dp))
[[ -z $expandable ]] && { [[ $preset == 0.8b-l* ]] && expandable=1 || expandable=0; }
has() { [[ ",$stages," == *",$1,"* ]]; }
for s in ${stages//,/ }; do [[ $s =~ ^(discovery|profile|replay|real|phases-sim|phases-real)$ ]] || ph_die "unknown stage $s"; done
if has discovery || has replay || { has phases-sim && ! has profile; }; then [[ -f $base_db/compute.csv ]] || ph_die "--base-db with compute.csv required"; fi
if has profile && ! has discovery; then ph_die "profile needs discovery in the same run"; fi

# ---- the code must be a clean, pushed commit ----
[[ -z "$(git -C "$PH_CLONE" status --porcelain)" ]] || ph_die "$PH_CLONE has local changes; commit and push from your machine, then git pull here"
commit=$(git -C "$PH_CLONE" rev-parse HEAD)
if (( ! allow_unpushed )); then
    git -C "$PH_CLONE" fetch -q || ph_die "git fetch failed"
    upstream=$(git -C "$PH_CLONE" rev-parse '@{u}' 2>/dev/null) || ph_die "clone branch has no upstream"
    [[ $commit == "$upstream" ]] || ph_die "clone HEAD ${commit:0:8} differs from upstream ${upstream:0:8}; run git pull (or --allow-unpushed)"
fi
build=$("$here/build.sh" "$commit")
src="$build/src" dist="$build/dist"
torch_dir=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["torch_dir"])' "$build/build.json")
torch_rev=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["torch_reported_git_version"])' "$build/build.json")
g="$src/tests/docker/megatron/gpu01"   # this script's files as of the commit (used for everything below)

run="$PH_RUNS/$(date +%Y%m%d-%H%M%S)-$name-${commit:0:8}"
mkdir -p "$run"
export PH_LOG_FILE="$run/progress.log"
echo "$commit" > "$run/commit.txt"
cp "$build/build.json" "$run/build.json"
python3 - "$run/config.json" <<P
import json, sys
json.dump(dict(name="$name", commit="$commit", preset="$preset", sequence_length=$seq, micro_batch_size=$mb,
               gradient_accumulation=$ga, global_batch_size=$((mb * ga * dp)), tensor_parallel_size=$tp,
               pipeline_parallel_size=$pp, data_parallel_size=$dp,
               pipeline_stage_layers=[int(x) for x in "$stage_layers".split(",")] if "$stage_layers" else None,
               world_size=$world, iterations=$iters, vram_mib=$vram, mtp_layers=$mtp,
               stages="$stages".split(","), base_db="$base_db" or None, sim_log_level="$sim_log",
               real_expandable_segments=bool($expandable), real_compile_cache=bool($compile_cache),
               host="$(hostname)"), open(sys.argv[1], "w"), indent=1)
P
ph_log "run $run: $preset seq $seq, tp $tp pp $pp dp $dp${stage_layers:+ split $stage_layers}, mtp $mtp, stages $stages, commit ${commit:0:8}"
scratch=$(mktemp -d "$PH_SCRATCH/ph-run-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT

simulate() {  # <perf-db> <stage> <log-level> [phases]
    local out="$scratch/$2" rc=0; mkdir -p "$out"
    timeout -k 20 2400 python3 "$g/simulate.py" --tests "$src/tests" --image "$PH_SIF" --native-dist "$dist" \
        --torch-prefix "$torch_dir/torch-deps" --causal-prefix "$PH_ENV/causal-deps" --upgraded-prefix "$PH_ENV/upgraded-deps" \
        --perf-db "$1" --calibrated-netconfig "$g/netconfig.toml" --output-dir "$out" \
        --preset "$preset" --sequence-length "$seq" --micro-batch-size "$mb" --gradient-accumulation "$ga" \
        --tensor-parallel-size "$tp" --pipeline-parallel-size "$pp" --data-parallel-size "$dp" ${stage_layers:+--pipeline-stage-layers "$stage_layers"} --iterations "$iters" --vram-mib "$vram" --sim-log-level "$3" --mtp-layers "$mtp" ${4:+--phase-timings} \
        > "$run/$2.log" 2>&1 || rc=$?
    local o; o=$(ls -d "$out"/discover-* 2>/dev/null | head -1)
    mkdir -p "$run/$2"
    if [[ -n $o ]]; then
        cp -a "$o"/{status.json,runner.json,worker.log,workload.json,commands.json,input-hashes.json} "$run/$2/" 2>/dev/null || true
        [[ -d $o/perfdb.missing ]] && cp -a "$o/perfdb.missing" "$run/$2/"
        [[ $3 == debug ]] && python3 "$g/events_agg.py" "$o/simulator.log" "$run/$2/events.jsonl.gz" > "$run/$2/events_agg.log" 2>&1
    fi
    echo "$rc" > "$run/$2/exit"
    [[ -n ${4:-} ]] && python3 "$g/phases.py" --ranks "$world" "$run/$2/worker.log" > "$run/$2/phases.json"
    ph_log "$2 exit $rc: $(python3 "$g/metrics.py" --ranks "$world" "$run/$2/worker.log") ms; missing compute keys $([[ -f $run/$2/perfdb.missing/compute.csv ]] && echo $(($(wc -l < "$run/$2/perfdb.missing/compute.csv") - 1)) || echo 0)"
    rm -rf -- "$out"
}

binds_common="$torch_dir/torch-deps:/qh-torch:ro,$PH_ENV/causal-deps:/qh-causal:ro,$PH_ENV/upgraded-deps:/qh-upgraded:ro,$src:/qh-src:ro,$run:/qh-run"

if has discovery; then
    simulate "$base_db" discovery info
    miss="$run/discovery/perfdb.missing"
    [[ -f $run/discovery/runner.json ]] || ph_die "discovery produced no result; see discovery.log"
fi

if has profile; then
    p="$run/profile"; mkdir -p "$p/keys" "$p/keys-exp" "$scratch/cache"
    cp -a "$base_db" "$run/db"
    if [[ -f $miss/compute.csv ]]; then
        cp "$miss"/*.csv "$p/keys/"
        [[ -f $p/keys/memcpy.csv ]] || head -1 "$base_db/memcpy.csv" > "$p/keys/memcpy.csv"
        head -1 "$p/keys/memcpy.csv" > "$p/keys-exp/memcpy.csv"
        head -1 "$p/keys/compute.csv" > "$p/keys-exp/compute.csv"
        grep 'exp_out' "$p/keys/compute.csv" >> "$p/keys-exp/compute.csv" || true
        grep -v 'exp_out' "$p/keys/compute.csv" > "$p/keys/compute.rest"; mv "$p/keys/compute.rest" "$p/keys/compute.csv"
        ph_log "profile: $(($(wc -l < "$p/keys/compute.csv") - 1)) compute keys, $(($(wc -l < "$p/keys-exp/compute.csv") - 1)) exp_out keys; waiting for GPU 0"
        ph_wait_idle 0 || ph_die "GPU 0 not idle after waiting"
        gpu=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
        rc=0
        APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu" APPTAINERENV_QH_TORCH_REVISION="$torch_rev" \
        APPTAINERENV_QH_GPU_NAME="$PH_GPU_NAME" APPTAINERENV_QH_EXP_ITERS=$([[ $seq -gt 16384 ]] && echo 2 || echo 3) \
            timeout -k 15 2400 apptainer exec --containall --cleanenv --nv \
            --bind "$binds_common,$scratch/cache:/qh-cache" --pwd /qh-run "$PH_SIF" bash "/qh-src/tests/docker/megatron/gpu01/profile_inside.sh" \
            > "$p/profile.log" 2>&1 || rc=$?
        echo "$rc" > "$p/exit"
        ph_log "profile exit $rc"
        python3 "$g/merge.py" "$run" > "$p/merge.log" 2>&1 || ph_die "merge failed; see profile/merge.log"
        ph_log "merge: $(cat "$p/merge.log")"
        [[ $rc == 0 ]] || ph_die "profiling failed; see profile/profile.log"
    else
        ph_log "profile: nothing missing"
    fi
fi

if has replay; then
    db="$base_db"; [[ -d $run/db ]] && db="$run/db"
    simulate "$db" replay "$sim_log"
fi

real_pass() {  # <stage dir: real | phases-real> [phases]
    local r="$run/$1" rc=0 cbind="" cenv="" t0=$SECONDS; mkdir -p "$r" "$scratch/rcache"
    if (( compile_cache )); then
        local cdir="$PH_CACHE/real-compile/$(basename "$torch_dir")"; mkdir -p "$cdir"
        local state=cold; [[ -n "$(ls -A "$cdir" 2>/dev/null)" ]] && state=warm
        echo "$cdir $state" > "$r/compile_cache"
        cbind=",$cdir:/qh-compile" cenv=/qh-compile
        ph_log "$1: compile cache $state ($cdir)"
    else
        echo "per-run" > "$r/compile_cache"
    fi
    idx=$(seq -s, 0 $((world - 1)))
    (( $(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l) >= world )) || ph_die "real needs $world GPUs"
    ph_log "$1: waiting for GPUs $idx"
    ph_wait_idle "$idx" || ph_die "GPUs not idle after waiting"
    gpus=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -"$world" | paste -sd, -)
    ( while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -; done > "$r/mem.csv" ) &
    mon=$!
    APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpus" APPTAINERENV_QH_TORCH_REVISION="$torch_rev" APPTAINERENV_QH_TP=$tp \
    APPTAINERENV_QH_PRESET=$preset APPTAINERENV_QH_SEQ=$seq APPTAINERENV_QH_MB=$mb APPTAINERENV_QH_GBS=$((mb * ga * dp)) \
    APPTAINERENV_QH_ITERS=$iters APPTAINERENV_QH_EXPANDABLE=$expandable APPTAINERENV_QH_MTP=$mtp \
    APPTAINERENV_QH_PP=$pp APPTAINERENV_QH_DP=$dp APPTAINERENV_QH_STAGE_LAYERS="$stage_layers" \
    APPTAINERENV_QH_REAL_DIR="$1" APPTAINERENV_QH_PHASES="${2:-}" APPTAINERENV_QH_COMPILE_CACHE="$cenv" \
        timeout -k 25 3600 apptainer exec --containall --cleanenv --nv \
        --bind "$binds_common,$scratch/rcache:/qh-cache$cbind" --pwd /qh-run "$PH_SIF" bash "/qh-src/tests/docker/megatron/gpu01/real_inside.sh" \
        > "$r/trainer.log" 2>&1 || rc=$?
    kill "$mon" 2>/dev/null || true
    echo "$rc" > "$r/exit"
    [[ -n ${2:-} ]] && python3 "$g/phases.py" --ranks "$world" "$r"/ranks/*/*/*/stdout.log > "$r/phases.json"
    echo $((SECONDS - t0)) > "$r/wall_s"   # whole stage incl. container start and compilation
    ph_log "$1 exit $rc ($((SECONDS - t0)) s): $(python3 "$g/metrics.py" --ranks "$world" "$r"/ranks/*/*/*/stdout.log) ms; peak $(sort -t, -k1 -n "$r/mem.csv" | tail -1) MiB"
}

if has real; then real_pass real; fi

if has phases-sim; then
    db="$base_db"; [[ -d $run/db ]] && db="$run/db"
    simulate "$db" phases-sim info phases
fi

if has phases-real; then real_pass phases-real phases; fi

python3 - "$run" "$g/metrics.py" "$world" <<'P'
import json, subprocess, sys
from pathlib import Path
run, metrics, world = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def ms(files):
    files = [str(f) for f in files if f.exists()]
    if not files: return None
    out = subprocess.run([sys.executable, metrics, "--ranks", world, *files], capture_output=True, text=True).stdout.strip()
    return float(out) if out else None
s = dict(commit=(run / "commit.txt").read_text().strip(), config=json.load(open(run / "config.json")))
for stage in ("discovery", "replay"):
    d = run / stage
    if d.exists():
        runner = json.loads((d / "runner.json").read_text()) if (d / "runner.json").exists() else {}
        miss = d / "perfdb.missing" / "compute.csv"
        s[stage] = dict(exit=(d / "exit").read_text().strip(), step_ms=ms([d / "worker.log"]), valid=runner.get("valid"),
                        missing_compute_keys=(len(miss.read_text().splitlines()) - 1) if miss.exists() else 0)
if (run / "profile" / "merge.json").exists():
    s["profile"] = json.loads((run / "profile" / "merge.json").read_text())
    for v in s["profile"].values():
        if isinstance(v, dict): v.pop("nonpositive_keys", None)
if (run / "real").exists():
    s["real"] = dict(exit=(run / "real" / "exit").read_text().strip(),
                     step_ms=ms(sorted((run / "real").glob("ranks/*/*/*/stdout.log"))))
    for name, f in (("wall_s", "wall_s"), ("compile_cache", "compile_cache")):
        if (run / "real" / f).exists():
            s["real"][name] = (run / "real" / f).read_text().strip()
for stage in ("phases-sim", "phases-real"):
    f = run / stage / "phases.json"
    if f.exists():
        s[stage] = dict(exit=(run / stage / "exit").read_text().strip(), **json.loads(f.read_text() or "{}"))
if (s.get("real") or {}).get("step_ms") and (s.get("replay") or {}).get("step_ms"):
    s["replay_vs_real_pct"] = round(100 * (s["replay"]["step_ms"] - s["real"]["step_ms"]) / s["real"]["step_ms"], 2)
(run / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
print(json.dumps(s, indent=1))
P
ph_log "done: $run"
