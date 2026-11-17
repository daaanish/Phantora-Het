#!/usr/bin/env bash
# One-time: copy the external dependencies the runs need into $PH_ENV, so nothing
# depends on /tmp. Never overwrites: an existing destination is left as it is.
#   causal-deps    causal-conv1d build (from the QH-02 exact gate)
#   upgraded-deps  candidate Python packages (Megatron Core 0.16, FLA, ...)
#   cargo-home     offline Cargo registry/git cache for the native build
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/paths.sh"

QH02="$PH_ROOT/qh02-gpu01-20261114-NRVgKQ"
stack=$(cat "$QH02/exact-gate/scratch.txt")
declare -A SOURCES=(
    [causal-deps]="$stack/causal-deps"
    [upgraded-deps]="$QH02/candidate-artifacts/upgraded-deps"
    [cargo-home]="/tmp/phantora-qh02-V8gf5y/cache"
)
mkdir -p "$PH_ENV" "$PH_BUILDS" "$PH_RUNS"
for name in causal-deps upgraded-deps cargo-home; do
    dest="$PH_ENV/$name" src="${SOURCES[$name]}"
    if [[ -e "$dest" ]]; then echo "keep   $dest"; continue; fi
    [[ -d "$src" ]] || ph_die "source for $name not found: $src"
    tmp="$PH_ENV/.tmp-$name"; rm -rf -- "$tmp"; mkdir "$tmp"
    if [[ $name == cargo-home ]]; then cp -a "$src/registry" "$src/git" "$tmp/"; else cp -a "$src/." "$tmp/"; fi
    mv "$tmp" "$dest"
    echo "copied $src -> $dest"
done
[[ -f "$PH_SIF" ]] || ph_die "base image missing: $PH_SIF"
(cd "$PH_ENV" && find causal-deps upgraded-deps -type f -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
echo "environment ready in $PH_ENV ($(wc -l < "$PH_ENV/MANIFEST.sha256") files recorded in MANIFEST.sha256)"
