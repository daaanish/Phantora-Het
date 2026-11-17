#!/usr/bin/env bash
# QH-07v: rerun the two sweep points closest to real with the QH-07u build
# (cat/contiguous handlers + traced layouts). Each point starts from its own
# sweep's final DB, so the only change vs the sweep row is the build.
#   l4-24576: 4-layer preset, seq 24576 (QH-07q, replay 3,778.2 ms)
#   24-16384: 24-layer preset, seq 16384 (QH-07n, replay 7,410.8 ms)
# Per point: discovery -> profile new keys on GPU 0 -> replay (debug, for per-op charges).
set -uo pipefail
umask 077
run=$(cd "$(dirname "$0")" && pwd)
base="$HOME/Phantora"
qh="$base/qh02-gpu01-20261114-NRVgKQ"; exact="$qh/exact-gate"; stack=$(cat "$exact/scratch.txt")
old="$base/qh07b-gpu01-20261115-FGikEJ"; t="$base/qh07t-gdnfix-eoqL0c"; u="$base/qh07u-layout-IxfR7j"
image="$base/Phantora/phantora-gpu01.sif"
cal="$base/Phantora/tests/docker/megatron/depth-sweep-20261112-215835-8rO3ua/llama-8"
torch="/tmp/phantora-qh07u-keep/torch-deps"
log() { printf '%s %s\n' "$(date +%F_%T)" "$*" | tee -a "$run/progress.log"; }
sha256sum -c "$u/native-dist.sha256" > /dev/null || { log "QH-07u native checksum failed"; exit 1; }
[[ -f "$torch/torch/__init__.py" ]] || { log "QH-07u torch prefix missing"; exit 1; }
scratch=$(mktemp -d /tmp/phantora-qh07v-XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
mkdir "$scratch"/{out,pcache}
tar -xf "$u/native-dist.tar" -C "$scratch"
mkdir -p "$scratch/psrc/phantora/phantora/src"
cp "$u/qwen_ops.patched.py" "$scratch/psrc/phantora/phantora/src/qwen_ops.py"
# QH-07t tests (contiguous no-op filter) + the 4-layer preset from the QH-07q sweep.
if [[ ! -d "$run/tests" ]]; then
    cp -a "$t/tests" "$run/tests"
    cp "$base/qh07q-l4-sweep-3CPzh4/tests/qwen_hybrid_config.py" "$run/tests/"
fi

point() {  # <label> <seq> <preset> <base-db>
    local label=$1 seq=$2 preset=$3 db0=$4 d="$run/$1"
    [[ -f "$d/finished" ]] && return 0
    mkdir -p "$d"
    sed -e "s/sequence_length=8192,/sequence_length=$seq,/" -e "s/\"--sequence_length\", \"8192\",/\"--sequence_length\", \"$seq\",/" \
        "$base/qh07r-attrib-x6LKFo/discover.py" > "$d/discover.base.py"
    if [[ $preset == 0.8b-l4 ]]; then
        sed -i -e 's/QWEN35_08B/QWEN35_08B_L4/g' -e 's/\[576, 576, 576, 576, 1152, 1152\]/[96, 96, 96, 96, 192, 192]/' \
            -e 's/"--hybrid-preset", "0.8b",/"--hybrid-preset", "0.8b-l4",/' "$d/discover.base.py"
        grep -q '"0.8b-l4"' "$d/discover.base.py" && grep -q '\[96, 96' "$d/discover.base.py" || { log "$label: discover edit failed"; return 1; }
    fi
    [[ $(grep -c "$seq" "$d/discover.base.py") -ge 2 ]] || { log "$label: discover seq edit failed"; return 1; }
    sim() {  # <perf-db> <stage> <log-level>
        local out="$scratch/out/$label-$2"; mkdir "$out"
        sed "s/\"PHANTORA_LOG\": \"debug\"/\"PHANTORA_LOG\": \"$3\"/" "$d/discover.base.py" > "$d/discover-$2.py"
        local rc=0
        timeout -k 20 2400 python3 "$d/discover-$2.py" --tests "$run/tests" --image "$image" --native-dist "$scratch/dist" \
            --torch-prefix "$torch" --causal-prefix "$stack/causal-deps" \
            --upgraded-prefix "$qh/candidate-artifacts/upgraded-deps" --perf-db "$1" \
            --calibrated-netconfig "$cal/netconfig.toml" --output-dir "$out" > "$d/$2.log" 2>&1 || rc=$?
        local o; o=$(ls -d "$out"/discover-* 2>/dev/null | head -1)
        mkdir "$d/$2"
        [[ -n "$o" ]] && cp -a "$o"/{status.json,runner.json,worker.log,workload.json} "$d/$2/" 2>/dev/null
        [[ -n "$o" && -d "$o/perfdb.missing" ]] && cp -a "$o/perfdb.missing" "$d/$2/"
        if [[ $3 == debug && -n "$o" ]]; then python3 "$base/qh07r-attrib-x6LKFo/phantora_agg.py" "$o/simulator.log" "$d/$2/events.jsonl.gz" > "$d/$2/agg.log" 2>&1; fi
        rm -rf -- "$out"
        echo "$rc" > "$d/$2.exit"
        log "$label $2 exit $rc: $(grep -ho 'Rank 0 Time: \[[^]]*\]' "$d/$2/worker.log" 2>/dev/null | head -1)"
    }
    [[ -d "$d/discovery" ]] || sim "$db0" discovery info
    local miss="$d/discovery/perfdb.missing"
    [[ -f "$miss/compute.csv" ]] || { log "$label: no missing compute keys"; return 1; }
    log "$label missing keys: $(($(wc -l < "$miss/compute.csv") - 1)); with layouts: $(grep -c '"layouts"' "$miss/compute.csv"); large-message warnings: $(grep -c 'Large message' "$d/discovery/worker.log")"
    if [[ ! -f "$d/profile.exit" ]]; then
        mkdir -p "$d/keys"
        cp "$miss/compute.csv" "$d/keys/"
        if [[ -f "$miss/memcpy.csv" ]]; then cp "$miss/memcpy.csv" "$d/keys/"; else head -1 "$db0/memcpy.csv" > "$d/keys/memcpy.csv"; fi
        cp "$u/profile-all.py" "$d/"
        sed 's/^unset PHANTORA LD_PRELOAD$/unset PHANTORA LD_PRELOAD\nexport PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True/' "$u/inside.sh" > "$d/inside.sh"
        for i in $(seq 1 480); do
            [[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]] && break
            [[ $i == 1 ]] && log "$label: waiting for GPU 0"; sleep 60
        done
        local gpu rc=0; gpu=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
        APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu" timeout -k 15 1800 apptainer exec --containall --cleanenv --nv \
            --bind "$stack:/qhexact:ro,$exact:/qhgate:ro,$qh/candidate-artifacts:/qhcandidate:ro,$d:/qhrun,$scratch/pcache:/qhcache,$scratch/psrc:/qhsrc:ro,$old/tests:/qhsrc/tests:ro" \
            --pwd /qhrun "$image" bash /qhrun/inside.sh > "$d/profile.log" 2>&1 || rc=$?
        echo "$rc" > "$d/profile.exit"; log "$label profile exit $rc"
        [[ $rc == 0 ]] || return 1
    fi
    python3 - "$d" "$db0" <<'P'
import csv, shutil, sys, json
from pathlib import Path
d, db0 = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.reader((d / "measured/compute.csv").open(newline="")))[1:]
out = d / "db"
if not out.exists():
    shutil.copytree(db0, out)
    sel = [r for r in rows if int(r[-1]) > 0 and '"layouts"' in r[0]]
    with (out / "compute.csv").open("a", newline="") as f: csv.writer(f).writerows(sel)
info = {"profiled": len(rows), "added_layout_rows": sum(int(r[-1]) > 0 and '"layouts"' in r[0] for r in rows),
        "other_positive_not_added": [r[0] for r in rows if int(r[-1]) > 0 and '"layouts"' not in r[0]],
        "nonpositive": [r[0] for r in rows if int(r[-1]) <= 0]}
(d / "merge.json").write_text(json.dumps(info, indent=1)); print(json.dumps(info)[:400])
P
    [[ -d "$d/replay" ]] || sim "$d/db" replay debug
    touch "$d/finished"
}
point l4-24576 24576 0.8b-l4 "$base/qh07q-l4-sweep-3CPzh4/seq-24576/db"
point 24-16384 16384 0.8b "$base/qh07n-seq16384-wUjCQk/db-16384"
log "ALL DONE"; touch "$run/ALL_DONE"
