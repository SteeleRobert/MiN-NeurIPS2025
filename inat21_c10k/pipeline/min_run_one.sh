#!/bin/bash
# Run one MiN benchmark inside the pytorch_min enroot container on a pinned GPU.
# Usage: min_run_one.sh <base_config> <model_config> <gpu> <seed> [results_dir]
set -euo pipefail

BASE="$1"; MODEL="$2"; GPU="$3"; SEED="$4"
RESULTS_DIR="${5:-/gpfs/data/oermannlab/users/steelr04/min_sweep/results}"

U=/gpfs/data/oermannlab/users/steelr04
REPO=$U/MiN-NeurIPS2025

module load enroot 2>/dev/null || true
export XDG_RUNTIME_DIR=$U
export XDG_CONFIG_HOME=$U/.config
export XDG_CACHE_HOME=$U/.cache
export XDG_DATA_HOME=$U/.local/share

HF_TOKEN=$(sed -n '/\[nyu\]/,/^$/p' /gpfs/home/steelr04/.cache/huggingface/stored_tokens | grep hf_token | awk '{print $3}')

echo "[min_run_one] host=$(hostname) gpu=$GPU base=$BASE model=$MODEL seed=$SEED -> $RESULTS_DIR"

# enroot does not propagate arbitrary host env — pass everything via --env.
enroot start \
  --mount /gpfs/data/oermannlab:/gpfs/data/oermannlab \
  --env CUDA_VISIBLE_DEVICES="$GPU" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env HF_TOKEN="$HF_TOKEN" \
  --env HF_HOME=$U/.cache/huggingface \
  --env TORCH_HOME=$U/.cache/torch \
  --env XDG_CACHE_HOME=$U/.cache \
  --env COMPCONT_RESULTS_DIR="$RESULTS_DIR" \
  pytorch_min -- bash -c "
    cd $REPO &&
    python3 MiN/main.py \
      --base_configs  MiN/configs/base_configs/${BASE}.json \
      --model_configs MiN/configs/model_configs/${MODEL}.json \
      --seed $SEED
  "
rc=$?
echo "[min_run_one] exit=$rc"
exit $rc
