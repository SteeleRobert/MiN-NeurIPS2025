#!/bin/bash
# Exact-replay of one disputed H cell in the qz harness, inside the holder
# allocation. Mirrors slurm/run_sweep_superpod.sbatch's container invocation
# (same mounts, same env) with the later lock-contention fixes (no -w,
# per-run ENROOT_RUNTIME_PATH).
# Usage: h_replay_run.sh <gpu> <run_index>
set -euo pipefail

GPU="$1"; IDX="$2"

LOCAL_DIR=/gpfs/data/oermannlab/users/steelr04
REPO_ROOT=${LOCAL_DIR}/qz-compcont-learning
MIN_REPO=${LOCAL_DIR}/MiN-NeurIPS2025
WDS_ROOT_HOST=/gpfs/data/oermannlab/public_data/continual_learning/webdatasets
S=${LOCAL_DIR}/h_replay

module load enroot 2>/dev/null || true
export XDG_RUNTIME_DIR=${LOCAL_DIR}
export XDG_CONFIG_HOME=${LOCAL_DIR}/.config
export XDG_CACHE_HOME=${LOCAL_DIR}/.cache
export XDG_DATA_HOME=${LOCAL_DIR}/.local/share
export ENROOT_RUNTIME_PATH=${LOCAL_DIR}/enroot_runtime_hreplay_${GPU}

HF_TOKEN=$(sed -n '/\[nyu\]/,/^$/p' /gpfs/home/steelr04/.cache/huggingface/stored_tokens | grep hf_token | awk '{print $3}')

echo "[h_replay] host=$(hostname) gpu=$GPU run_index=$IDX $(date -Is)"

enroot start -r \
    --mount ${REPO_ROOT}:/workspace/qz-compcont-learning \
    --mount ${MIN_REPO}:/workspace/MiN-NeurIPS2025 \
    --mount ${WDS_ROOT_HOST}:/workspace/wds \
    --mount ${S}:/workspace/h_replay \
    -e CUDA_VISIBLE_DEVICES="$GPU" \
    -e OMP_NUM_THREADS=16 \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HF_HOME=${LOCAL_DIR}/.cache/huggingface \
    -e PYTHONPATH=/workspace/qz-compcont-learning/src \
    -e WDS_ROOT=/workspace/wds \
    -e MIN_REPO=/workspace/MiN-NeurIPS2025 \
    pytorch_min \
    -- bash -c "cd /workspace/qz-compcont-learning && python -m kanerva_sdm.cli.main sweep /workspace/h_replay/replay_h_cells.yaml --results-dir /workspace/qz-compcont-learning/results --device cuda --run-index ${IDX}"
rc=$?
echo "[h_replay] exit=$rc"
exit $rc
