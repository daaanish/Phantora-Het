#!/usr/bin/env bash
# Build one commit of $PH_CLONE. Usage: build.sh [commit-ish]   (default HEAD)
# Prints the build directory on stdout; progress goes to stderr and build.log.
#
#   $PH_BUILDS/torch-<pytorch commit>/   torch for the commit the pytorch submodule pins,
#                                        shared by every Phantora-Het commit with that pin
#   $PH_BUILDS/<commit>/                 src/ (git archive, read-only), dist/ (native
#                                        simulator), build.json, logs, BUILD_OK
#
# A finished build is never rebuilt or modified. Torch is built incrementally from the
# base image's compiled tree: only files that differ from the pinned commit are
# replaced, and the build stops if that would recompile CUDA code.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/paths.sh"

commit=$(git -C "$PH_CLONE" rev-parse --verify "${1:-HEAD}^{commit}")
out="$PH_BUILDS/$commit"
if [[ -f "$out/BUILD_OK" ]]; then echo "$out"; exit 0; fi
mkdir -p "$PH_BUILDS"
lock="$PH_BUILDS/.lock-$commit"
mkdir "$lock" 2>/dev/null || ph_die "another build of $commit is running (or stale lock $lock)"
work=$(mktemp -d "$PH_SCRATCH/ph-build-XXXXXX")
trap 'chmod -R u+w "$work" 2>/dev/null; rm -rf -- "$work"; rmdir "$lock" 2>/dev/null || true' EXIT
clear() { [[ -e "$1" ]] && chmod -R u+w "$1" && rm -rf -- "$1"; return 0; }
stage="$PH_BUILDS/.partial-$commit"; clear "$stage"; mkdir -p "$stage/src"
export PH_LOG_FILE="$stage/build.log"
ph_log "build $commit"
git -C "$PH_CLONE" archive "$commit" | tar -x -C "$stage/src"
chmod -R a-w "$stage/src"

# ---- torch for the pinned pytorch commit ----
torch_commit=$(git -C "$PH_CLONE" ls-tree "$commit" pytorch | awk '$2 == "commit" {print $3}')
[[ -n "$torch_commit" ]] || ph_die "no pytorch submodule pin in $commit"
torch_url=$(git -C "$PH_CLONE" config --blob "$commit:.gitmodules" submodule.pytorch.url)
# The code requires one base torch revision for provenance (tests/perfdb/gdn_common.py,
# qwen_hybrid_runtime_check.py, gdn.rs). The built torch reports that base revision as
# torch.version.git_version, exactly as the QH-02/QH-07u builds did; the true source commit
# is recorded in torch.json.
base_rev=$(sed -n 's/^TORCH_REVISION = "\([0-9a-f]\{40\}\)"$/\1/p' "$stage/src/tests/perfdb/gdn_common.py")
[[ ${#base_rev} == 40 ]] || ph_die "cannot read TORCH_REVISION from tests/perfdb/gdn_common.py"
tdir="$PH_BUILDS/torch-$torch_commit"
if [[ ! -f "$tdir/TORCH_OK" ]]; then
    ph_log "torch $torch_commit from $torch_url: copying the image's compiled tree"
    tsrc="$work/pytorch"; mkdir "$tsrc"
    apptainer exec --cleanenv -B "$work:$work" "$PH_SIF" cp -a --no-preserve=ownership /phantora/pytorch/. "$tsrc/"
    rm -rf -- "$tsrc/.git"
    git -C "$tsrc" init -q
    git -C "$tsrc" remote add origin "$torch_url"
    git -C "$tsrc" fetch -q --filter=blob:none --depth=1 origin "$torch_commit" "$base_rev"
    git -C "$tsrc" read-tree "$torch_commit"
    git -C "$tsrc" update-ref HEAD "$torch_commit"
    mapfile -t gitlinks < <(git -C "$tsrc" ls-files -s | awk '$1 == "160000" {print $4}')
    mapfile -t changed < <(git -C "$tsrc" diff --name-only --diff-filter=MD -- . "${gitlinks[@]/#/:!}")
    ph_log "files differing from $torch_commit: ${#changed[@]} (${changed[*]:0:10})"
    (( ${#changed[@]} <= 20 )) || ph_die "image tree differs from the pin in too many files; not an incremental build"
    (( ${#changed[@]} == 0 )) || git -C "$tsrc" checkout -q "$torch_commit" -- "${changed[@]}"
    [[ -z "$(git -C "$tsrc" diff --name-only -- . "${gitlinks[@]/#/:!}")" ]] || ph_die "tree still differs after checkout"
    git -C "$tsrc" update-ref HEAD "$base_rev"   # what torch.version.git_version will report
    timg=(apptainer exec --cleanenv -B "$work:$work" -B "$tsrc:/phantora/pytorch" "$PH_SIF")
    "${timg[@]}" ninja -C /phantora/pytorch/build -n install > "$stage/torch-ninja-dry-run.log" 2>&1
    steps=$(grep -cE '^\[[0-9]+/[0-9]+\]' "$stage/torch-ninja-dry-run.log" || true)
    cuda=$(grep -E '^\[[0-9]+/[0-9]+\]' "$stage/torch-ninja-dry-run.log" | grep -c 'Building CUDA' || true)
    ph_log "torch dry run: $steps steps, $cuda CUDA compiles"
    [[ $cuda == 0 ]] || ph_die "torch build would recompile CUDA code; see torch-ninja-dry-run.log"
    mkdir -p "$work/wheels" "$work/ccache" "$work/tmp" "$work/home"
    "${timg[@]}" env -u PYTHONPATH GIT_OPTIONAL_LOCKS=0 CCACHE_DIR="$work/ccache" TMPDIR="$work/tmp" \
        MAX_JOBS=8 CUDA_HOME=/usr/local/cuda USE_CUDA=1 USE_CUDNN=0 USE_CUSPARSELT=0 \
        BUILD_TEST=1 TORCH_CUDA_ARCH_LIST=12.0 PYTORCH_BUILD_VERSION=2.9.1 PYTORCH_BUILD_NUMBER=1 \
        /bin/bash -c 'cd /phantora/pytorch && exec python3 -m pip wheel --no-index --no-deps --no-build-isolation --no-cache-dir -v --wheel-dir "$1" .' \
        bash "$work/wheels" > "$stage/torch-wheel.log" 2>&1 || ph_die "torch wheel build failed; see torch-wheel.log"
    tpart="$PH_BUILDS/.partial-torch-$torch_commit"; clear "$tpart"; mkdir -p "$tpart"
    "${timg[@]}" env -u PYTHONPATH TMPDIR="$work/tmp" HOME="$work/home" python3 -m pip --isolated install \
        --no-index --no-deps --no-cache-dir --target "$tpart/torch-deps" "$work"/wheels/torch-*.whl \
        > "$stage/torch-install.log" 2>&1 || ph_die "torch install failed"
    cp "$stage"/torch-*.log "$tpart/"
    git_version=$("${timg[@]}" env -u PYTHONPATH PYTHONPATH="$tpart/torch-deps" python3 -c 'import torch; print(torch.version.git_version)' 2>/dev/null | tail -1)
    [[ $git_version == "$base_rev" ]] || ph_die "built torch reports git_version $git_version, expected $base_rev"
    python3 - "$tpart/torch.json" "$torch_commit" "$torch_url" "$base_rev" "$(sha256sum "$work"/wheels/torch-*.whl | cut -d' ' -f1)" "${changed[@]}" <<'P'
import json, sys
path, commit, url, base, wheel, *changed = sys.argv[1:]
json.dump(dict(source_commit=commit, url=url, reported_git_version=base, wheel_sha256=wheel,
               files_taken_from_source_commit=changed,
               note="compiled tree of the base image (base revision) with these files replaced from source_commit"),
          open(path, "w"), indent=1)
P
    touch "$tpart/TORCH_OK"
    mv "$tpart" "$tdir"
    rm -rf -- "$tsrc"
    ph_log "torch ready: $tdir"
fi

# ---- native simulator ----
ph_log "native build and unit tests"
cp -a "$stage/src" "$work/src"; chmod -R u+w "$work/src"
cp -a "$PH_ENV/cargo-home" "$work/cargo"
mkdir -p "$work/home" "$work/tmp"
apptainer exec --containall --cleanenv \
    --bind "$work:/qhbuild,$tdir/torch-deps:/qhtorch:ro" --pwd /qhbuild/src "$PH_SIF" bash -c '
        set -euxo pipefail
        export HOME=/qhbuild/home TMPDIR=/qhbuild/tmp CARGO_HOME=/qhbuild/cargo RUSTUP_HOME=/usr/local/rustup
        export CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=8 LIBTORCH=/qhtorch/torch
        export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
        mkdir -p build
        (cd phantora; cargo test --release -p phantora --lib)
        make dist
        cd tests && PYTHONPATH=/qhtorch PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_qwen_hybrid_ops test_qwen_hybrid_kernels
    ' > "$stage/native.log" 2>&1 || ph_die "native build or tests failed; see $stage/native.log"
cp -a "$work/src/dist" "$stage/dist"
rust=$(grep -E '^test result:' "$stage/native.log" | tail -1)
py=$(grep -E '^Ran [0-9]+ tests' "$stage/native.log" | tail -1)
python3 - "$stage/build.json" "$commit" "$torch_commit" "$tdir" "$rust" "$py" "$base_rev" <<'P'
import json, sys, datetime
path, commit, torch_commit, tdir, rust, py, base = sys.argv[1:]
json.dump(dict(commit=commit, torch_commit=torch_commit, torch_reported_git_version=base, torch_dir=tdir,
               rust_tests=rust, python_tests=py,
               built=datetime.datetime.now().isoformat(timespec="seconds")), open(path, "w"), indent=1)
P
(cd "$stage" && find dist -type f -print0 | sort -z | xargs -0 sha256sum > dist.sha256)
touch "$stage/BUILD_OK"
chmod -R a-w "$stage/dist"
mv "$stage" "$out"
PH_LOG_FILE="$out/build.log" ph_log "build ready: $out ($rust; $py)"
echo "$out"
