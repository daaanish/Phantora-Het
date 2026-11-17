# Runs inside the container (see run.sh, stage "profile"). Profiles the keys discovery
# found missing on one GPU and writes timings under /qh-run/profile; fused GDN timings go
# straight into /qh-run/db through gdn_profile.py.
set -uo pipefail
unset PHANTORA LD_PRELOAD
export HOME=/qh-cache/home TMPDIR=/qh-cache/tmp TRITON_CACHE_DIR=/qh-cache/triton
export PYTHONDONTWRITEBYTECODE=1 PIP_CONFIG_FILE=/dev/null
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/qh-torch/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/qh-torch:/qh-causal:/qh-upgraded:/qh-src/tests PATH="/qh-upgraded/bin:$PATH"
mkdir -p "$HOME" "$TMPDIR" "$TRITON_CACHE_DIR"
g=/qh-src/tests/docker/megatron/gpu01 p=/qh-run/profile
python3 "$g/verify_torch.py" || exit 1
python3 /qh-src/tests/qwen_hybrid_runtime_check.py || exit 1
rc=0
if [[ -f $p/keys/gdn.missing.csv && $(wc -l < $p/keys/gdn.missing.csv) -gt 1 ]]; then
    python3 -u /qh-src/tests/perfdb/gdn_profile.py --keys $p/keys/gdn.missing.csv --out /qh-run/db \
        --gpu-name "$QH_GPU_NAME" > $p/gdn.log 2>&1 || rc=11
fi
QH_KEYS=$p/keys QH_OUT=$p/measured python3 -u "$g/profile_keys.py" > $p/bench.log 2>&1 || rc=12
if [[ $(wc -l < $p/keys-exp/compute.csv) -gt 1 ]]; then
    QH_KEYS=$p/keys-exp QH_OUT=$p/measured-exp QH_WARMUP=1 QH_ITERS=$QH_EXP_ITERS \
        python3 -u "$g/profile_keys.py" > $p/bench-exp.log 2>&1 || rc=13
fi
exit $rc
