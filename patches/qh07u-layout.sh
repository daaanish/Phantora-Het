#!/usr/bin/env bash
# QH-07u: carry tensor layouts (strides, storage offset) from the tracer to the
# cat/contiguous profiling keys. Same configuration as QH-07t (0.8B, seq 8192,
# TP2, QH-07m base DB); the only change is the traced layout.
#   1. native build with layout-native.patch on top of QH-07t, Rust + Python tests
#   2. torch wheel rebuilt with pytorch-function-tracer-layout.patch -> new prefix
#   3. GPU-free discovery with the new torch prefix; check layouts reached the keys
#   4. profile the new keys on GPU 0 (waits for it to be free), replay
set -uo pipefail
umask 077
run=$(cd "$(dirname "$0")" && pwd)
base="$HOME/Phantora"
qh="$base/qh02-gpu01-20261114-NRVgKQ"; exact="$qh/exact-gate"; stack=$(cat "$exact/scratch.txt")
old="$base/qh07b-gpu01-20261115-FGikEJ"; prev="$base/qh07c-gpu01-20261115-KiNNh8"; t="$base/qh07t-gdnfix-eoqL0c"
repo="$base/Phantora"; image="$repo/phantora-gpu01.sif"
cal="$repo/tests/docker/megatron/depth-sweep-20261112-215835-8rO3ua/llama-8"
db0="$base/qh07m-seq8192-fGizjn/db-8192"
log() { printf '%s %s\n' "$(date +%F_%T)" "$*" | tee -a "$run/progress.log"; }
keep="/tmp/phantora-qh07u-keep"; mkdir -p "$keep"          # new torch prefix (large, kept)
scratch=$(mktemp -d /tmp/phantora-qh07u-XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
mkdir "$scratch"/{home,cache,tmp,out}
sha256sum "$run"/*.patch > "$run/patches.sha256"

# ---- 1. native ----
if [[ ! -f "$run/native-dist.tar" ]]; then
    cp -a "$stack/source" "$scratch/source"
    cp "$t/qwen_hybrid_kernels.py" "$t/test_qwen_hybrid_kernels.py" "$t/test_qwen_hybrid_ops.py" "$scratch/source/tests/"
    (cd "$scratch/source" && patch --batch -p1 < "$prev/native.patch" && patch --batch -p1 < "$t/gdnfix-native.patch" \
        && patch --batch -p1 < "$run/layout-native.patch") > "$run/patch.log" 2>&1 || { log "native patch failed"; exit 1; }
    cp -a /tmp/phantora-qh02-V8gf5y/cache/{registry,git} "$scratch/cache/"
    sha256sum "$scratch/source/phantora/phantora/src/"{torch_call.rs,qwen_ops.py,perf_db.rs} > "$run/source.sha256"
    rc=0
    timeout -k 20 900 apptainer exec --containall --cleanenv \
        --bind "$scratch:/qhbuild,$stack/torch-deps:/qhtorch:ro" --pwd /qhbuild/source "$image" bash -c '
            set -euxo pipefail
            export HOME=/qhbuild/home TMPDIR=/qhbuild/tmp CARGO_HOME=/qhbuild/cache RUSTUP_HOME=/usr/local/rustup
            export CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=4 LIBTORCH=/qhtorch/torch
            export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
            mkdir -p build
            (cd phantora; cargo test --release -p phantora --lib)
            make dist
            cd tests && PYTHONPATH=/qhtorch PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_qwen_hybrid_ops test_qwen_hybrid_kernels
        ' > "$run/native.log" 2>&1 || rc=$?
    echo "$rc" > "$run/native.exit"; log "native build+tests exit $rc"
    [[ $rc == 0 ]] || exit 1
    tar -C "$scratch/source" -cf "$run/native-dist.tar" dist
    cp "$scratch/source/phantora/phantora/src/qwen_ops.py" "$run/qwen_ops.patched.py"
    sha256sum "$run/native-dist.tar" > "$run/native-dist.sha256"
fi

# ---- 2. torch with the layout-aware tracer ----
if [[ ! -f "$keep/torch-deps/torch/__init__.py" ]]; then
    tsrc="$keep/pytorch-src"; rm -rf "$tsrc" "$keep/wheels" "$keep/torch-deps"; mkdir -p "$tsrc" "$keep/wheels" "$keep/ccache"
    apptainer exec --cleanenv -B "$keep:$keep" "$image" cp -a --no-preserve=ownership /phantora/pytorch/. "$tsrc/"
    log "torch source copied"
    (cd "$tsrc" && patch --batch -p1 < "$run/pytorch-function-tracer-layout.patch") >> "$run/patch.log" 2>&1 || { log "torch patch failed"; exit 1; }
    timg=(apptainer exec --cleanenv -B "$keep:$keep" -B "$tsrc:/phantora/pytorch" -B "$repo/.git/modules/pytorch:/phantora/.git/modules/pytorch:ro" "$image")
    "${timg[@]}" ninja -C /phantora/pytorch/build -n install > "$run/torch-ninja-dry-run.log" 2>&1
    grep -E '^\[[0-9]+/[0-9]+\]' "$run/torch-ninja-dry-run.log" | grep -c 'Building CUDA' > "$run/torch-cuda-steps.txt" || true
    if [[ "$(cat "$run/torch-cuda-steps.txt")" != 0 ]]; then log "torch dry run wants CUDA recompiles; stopping"; exit 1; fi
    log "torch dry run: $(grep -cE '^\[[0-9]+/[0-9]+\]' "$run/torch-ninja-dry-run.log") steps; building"
    rc=0
    timeout -k 20 3600 "${timg[@]}" env -u PYTHONPATH GIT_OPTIONAL_LOCKS=0 CCACHE_DIR="$keep/ccache" TMPDIR="$scratch/tmp" \
        MAX_JOBS=8 CUDA_HOME=/usr/local/cuda USE_CUDA=1 USE_CUDNN=0 USE_CUSPARSELT=0 \
        BUILD_TEST=1 TORCH_CUDA_ARCH_LIST=12.0 PYTORCH_BUILD_VERSION=2.9.1 PYTORCH_BUILD_NUMBER=1 \
        /bin/bash -c 'cd /phantora/pytorch && exec python3 -m pip wheel --no-index --no-deps --no-build-isolation --no-cache-dir -v --wheel-dir "$1" .' \
        bash "$keep/wheels" > "$run/torch-wheel.log" 2>&1 || rc=$?
    echo "$rc" > "$run/torch-wheel.exit"; log "torch wheel exit $rc"
    [[ $rc == 0 ]] || exit 1
    apptainer exec --cleanenv -B "$keep:$keep" "$image" env -u PYTHONPATH TMPDIR="$scratch/tmp" HOME="$scratch/home" \
        python3 -m pip --isolated install --no-index --no-deps --no-cache-dir --target "$keep/torch-deps" "$keep"/wheels/torch-*.whl \
        > "$run/torch-install.log" 2>&1 || { log "torch install failed"; exit 1; }
    sha256sum "$keep"/wheels/torch-*.whl > "$run/torch-wheel.sha256"
    rm -rf "$tsrc"
    log "torch prefix ready at $keep/torch-deps"
fi
grep -q '"stride"' "$keep/torch-deps/torch/lib/libtorch_cuda.so" && log "tracer string present in libtorch_cuda.so" \
    || { log "patched tracer string missing from libtorch_cuda.so"; exit 1; }

# ---- 3. discovery ----
tar -xf "$run/native-dist.tar" -C "$scratch"
mkdir -p "$scratch/psrc/phantora/phantora/src"
cp "$run/qwen_ops.patched.py" "$scratch/psrc/phantora/phantora/src/qwen_ops.py"
[[ -d "$run/tests" ]] || cp -a "$t/tests" "$run/tests"
sim() {  # <perf-db> <label> <log-level>
    local out="$scratch/out/$2"; mkdir "$out"
    sed "s/\"PHANTORA_LOG\": \"debug\"/\"PHANTORA_LOG\": \"$3\"/" "$base/qh07r-attrib-x6LKFo/discover.py" > "$run/discover-$2.py"
    local rc=0
    timeout -k 20 2400 python3 "$run/discover-$2.py" --tests "$run/tests" --image "$image" --native-dist "$scratch/dist" \
        --torch-prefix "$keep/torch-deps" --causal-prefix "$stack/causal-deps" \
        --upgraded-prefix "$qh/candidate-artifacts/upgraded-deps" --perf-db "$1" \
        --calibrated-netconfig "$cal/netconfig.toml" --output-dir "$out" > "$run/$2.log" 2>&1 || rc=$?
    local o; o=$(ls -d "$out"/discover-* | head -1)
    mkdir "$run/$2"
    cp -a "$o"/{status.json,runner.json,worker.log,workload.json} "$run/$2/" 2>/dev/null
    [[ -d "$o/perfdb.missing" ]] && cp -a "$o/perfdb.missing" "$run/$2/"
    if [[ $3 == debug ]]; then python3 "$base/qh07r-attrib-x6LKFo/phantora_agg.py" "$o/simulator.log" "$run/$2/events.jsonl.gz" > "$run/$2/agg.log" 2>&1; fi
    rm -rf -- "$out"
    log "$2 exit $rc: $(grep -ho 'Rank 0 Time: \[[^]]*\]' "$run/$2/worker.log" | head -1)"
}
[[ -d "$run/discovery" ]] || sim "$db0" discovery info
miss="$run/discovery/perfdb.missing"
[[ -f "$miss/compute.csv" ]] || { log "no missing compute keys"; exit 1; }
log "missing keys: $(($(wc -l < "$miss/compute.csv") - 1)); with layouts: $(grep -c '"layouts"' "$miss/compute.csv"); large-message warnings: $(grep -c 'Large message' "$run/discovery/worker.log")"
grep -q '"layouts"' "$miss/compute.csv" || { log "no layout keys; tracer change not effective"; exit 1; }

# ---- 4. profile on GPU 0, then replay ----
if [[ ! -f "$run/profile.exit" ]]; then
    mkdir -p "$run/keys" "$scratch/pcache"
    cp "$miss"/compute.csv "$run/keys/"; if [[ -f "$miss/memcpy.csv" ]]; then cp "$miss/memcpy.csv" "$run/keys/"; else head -1 "$db0/memcpy.csv" > "$run/keys/memcpy.csv"; fi
    cp "$t/profile-all.py" "$t/inside.sh" "$run/"
    log "waiting for GPU 0"
    for i in $(seq 1 480); do
        if [[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]]; then break; fi; sleep 60
    done
    log "GPU 0 free; profiling"
    gpu=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
    rc=0
    APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu" timeout -k 15 1200 apptainer exec --containall --cleanenv --nv \
        --bind "$stack:/qhexact:ro,$exact:/qhgate:ro,$qh/candidate-artifacts:/qhcandidate:ro,$run:/qhrun,$scratch/pcache:/qhcache,$scratch/psrc:/qhsrc:ro,$old/tests:/qhsrc/tests:ro" \
        --pwd /qhrun "$image" bash /qhrun/inside.sh > "$run/profile.log" 2>&1 || rc=$?
    echo "$rc" > "$run/profile.exit"; log "profile exit $rc"
    [[ $rc == 0 ]] || exit 1
fi
python3 - "$run" "$db0" <<'P'
import csv, shutil, sys, json
from pathlib import Path
run, db0 = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.reader((run / "measured/compute.csv").open(newline="")))[1:]
out = run / "db-layout"
if not out.exists():
    shutil.copytree(db0, out)
    sel = [r for r in rows if int(r[-1]) > 0 and '"layouts"' in r[0]]
    with (out / "compute.csv").open("a", newline="") as f: csv.writer(f).writerows(sel)
info = {"profiled": len(rows), "added_layout_rows": sum(int(r[-1]) > 0 and '"layouts"' in r[0] for r in rows),
        "nonpositive": [r[0] for r in rows if int(r[-1]) <= 0]}
(run / "merge.json").write_text(json.dumps(info, indent=1)); print(json.dumps(info)[:600])
P
[[ -d "$run/replay-layout" ]] || sim "$run/db-layout" replay-layout debug
log "ALL DONE"; touch "$run/ALL_DONE"
