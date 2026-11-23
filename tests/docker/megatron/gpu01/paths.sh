# Locations for the gpu01 workflow (see README.md). Sourced by the other scripts.
# Override PH_ROOT to use another base directory.
PH_ROOT="${PH_ROOT:-$HOME/Phantora}"
PH_CLONE="$PH_ROOT/Phantora-Het"              # git clone of daaanish/Phantora-Het; never edited by hand
PH_ENV="$PH_ROOT/env"                         # one-time copies of external dependencies (setup_env.sh)
PH_BUILDS="$PH_ROOT/builds"                   # <commit>/ native builds, torch-<commit>/ torch builds
PH_RUNS="$PH_ROOT/runs"                       # one directory per run, named <date>-<name>-<commit8>
PH_CACHE="$PH_ROOT/cache"                     # real-compile/<torch build>/: Triton/Inductor/CUDA JIT caches kept across real runs
PH_SIF="$PH_ROOT/Phantora/phantora-gpu01.sif" # base image: CUDA 12.8, Python 3.11, Megatron, Rust
PH_HOST="gpu01.pronet.rcac.purdue.edu"
PH_GPU_NAME="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
PH_SCRATCH="${PH_SCRATCH:-/tmp}"              # fast local disk for build trees and simulator logs

ph_log() { printf '%s %s\n' "$(date +%F_%T)" "$*" | tee -a "${PH_LOG_FILE:-/dev/null}" >&2; }
ph_die() { ph_log "ERROR: $*"; exit 1; }

# Wait until the given GPU indices (e.g. "0" or "0,1") have no process, <=16 MiB used and 0% utilization.
ph_wait_idle() {
    local i
    for i in $(seq 1 "${PH_WAIT_MINUTES:-480}"); do
        if [[ -z "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader)" ]] &&
           nvidia-smi -i "$1" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
               awk -F, '$1 > 16 || $2 != 0 {bad=1} END {exit bad}'; then return 0; fi
        [[ $i == 1 ]] && ph_log "GPUs $1 busy; waiting (checks every minute)"
        sleep 60
    done
    return 1
}
