#!/usr/bin/env bash
# Re-time every key that earlier profile sessions measured, in one session on GPU 0, with the old
# and the fixed bench.py back to back, then compare each earlier session against the fixed timings
# (A15: some sessions recorded 4 of 5 iterations, so their keys are 0.8x).
#
# usage: reprofile_sessions.sh <old-commit> <out-dir> <run-dir-glob>...
#   old-commit    a commit with the pre-fix bench.py (built if needed)
#   run-dir-glob  run directories whose profile/measured*/ keys to collect, e.g.
#                 "$PH_RUNS/2026112[23]-*"
# Output in <out-dir>: keys/, keys-exp/ (union of keys), old/, new/ (+ -exp), clocks.csv,
# sessions.json (per session: median ratio session/new for bench keys, verdict).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/paths.sh"
old_commit=${1:?old commit}; out=${2:?out dir}; shift 2
new_build=$("$here/build.sh" HEAD) old_build=$("$here/build.sh" "$old_commit")
torch_dir=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["torch_dir"])' "$new_build/build.json")
torch_rev=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["torch_reported_git_version"])' "$new_build/build.json")
mkdir -p "$out"; out=$(cd "$out" && pwd)
export PH_LOG_FILE="$out/progress.log"
ph_log "reprofile: new $(basename "$new_build"), old $(basename "$old_build"), sessions from $*"
python3 "$here/reprofile_compare.py" collect "$out" "$@"
ph_wait_idle 0 || ph_die "GPU 0 not idle"
gpu=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
mkdir -p "$out/cache"; glog=$(ph_gpu_log "$out/clocks.csv")
rc=0
APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu" APPTAINERENV_QH_TORCH_REVISION="$torch_rev" \
    timeout -k 15 5400 apptainer exec --containall --cleanenv --nv \
    --bind "$torch_dir/torch-deps:/qh-torch:ro,$PH_ENV/causal-deps:/qh-causal:ro,$PH_ENV/upgraded-deps:/qh-upgraded:ro,$new_build/src:/qh-src:ro,$old_build/src:/qh-old:ro,$out:/qh-run,$out/cache:/qh-cache" \
    --pwd /qh-run "$PH_SIF" bash -c '
set -uo pipefail
unset PHANTORA LD_PRELOAD
export HOME=/qh-cache/home TMPDIR=/qh-cache/tmp TRITON_CACHE_DIR=/qh-cache/triton PYTHONDONTWRITEBYTECODE=1
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/qh-torch/torch/lib:$LD_LIBRARY_PATH" PYTHONPATH=/qh-torch:/qh-causal:/qh-upgraded:/qh-src/tests
mkdir -p "$HOME" "$TMPDIR" "$TRITON_CACHE_DIR"
w=/qh-src/tests/docker/megatron/gpu01/profile_keys.py rc=0
for v in old new; do
    src=/qh-src; [[ $v == old ]] && src=/qh-old
    QH_SRC=$src QH_KEYS=/qh-run/keys QH_OUT=/qh-run/$v python3 -u $w > /qh-run/$v.log 2>&1 || rc=1
    if [[ $(wc -l < /qh-run/keys-exp/compute.csv) -gt 1 ]]; then
        QH_SRC=$src QH_KEYS=/qh-run/keys-exp QH_OUT=/qh-run/$v-exp QH_WARMUP=1 QH_ITERS=3 \
            python3 -u $w > /qh-run/$v-exp.log 2>&1 || rc=1
    fi
done
exit $rc' > "$out/profile.log" 2>&1 || rc=$?
kill "$glog" 2>/dev/null || true
ph_log "reprofile exit $rc"
python3 "$here/reprofile_compare.py" compare "$out" "$@" | tee "$out/sessions.txt"
ph_log "reprofile done: $out"
