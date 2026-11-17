# Runs inside the container (see run.sh, stage "real"): one real training run on both GPUs.
set -euxo pipefail
unset PHANTORA LD_PRELOAD PHANTORA_COVERAGE
export HOME=/qh-cache/home TMPDIR=/qh-cache/tmp PYTHONDONTWRITEBYTECODE=1 PIP_CONFIG_FILE=/dev/null
export TRITON_CACHE_DIR=/qh-cache/triton TORCHINDUCTOR_CACHE_DIR=/qh-cache/inductor CUDA_CACHE_PATH=/qh-cache/cuda
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
export LD_LIBRARY_PATH="/qh-torch/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/qh-torch:/qh-causal:/qh-upgraded:/qh-src/tests PATH="/qh-upgraded/bin:$PATH"
export PHANTORA_RUN_ID=real
[[ "${QH_EXPANDABLE:-0}" == 1 ]] && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
python3 -c 'import triton; tool = triton.knobs.nvidia.ptxas_blackwell; print("selected Blackwell ptxas:", tool); assert tool.path == "/usr/local/cuda/bin/ptxas"'
python3 /qh-src/tests/docker/megatron/gpu01/verify_torch.py
cd /qh-run/real
exec torchrun --standalone --nproc-per-node="$QH_TP" --log-dir=/qh-run/real/ranks --redirects=3 /qh-src/tests/test_megatron.py \
    --hybrid-preset "$QH_PRESET" --tensor_parallel_size "$QH_TP" --data_parallel_size 1 \
    --pipeline_model_parallel_size 1 --sequence_length "$QH_SEQ" --micro_batch_size "$QH_MB" \
    --global_batch_size "$QH_GBS" --iterations "$QH_ITERS"
