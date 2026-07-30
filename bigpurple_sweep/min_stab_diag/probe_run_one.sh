#!/bin/bash
# Run one instrumented MiN probe inside the pytorch_min enroot container.
# Usage: probe_run_one.sh <base_config> <model_config> <gpu> <seed> <stages> <flags> <out_dir>
# flags: comma-separated subset of {eig,no-measure} or empty '-'
set -euo pipefail

BASE="$1"; MODEL="$2"; GPU="$3"; SEED="$4"; STAGES="$5"; FLAGS="$6"; OUTDIR="$7"

U=/gpfs/data/oermannlab/users/steelr04
REPO=$U/MiN-NeurIPS2025

module load enroot 2>/dev/null || true
export XDG_RUNTIME_DIR=$U
export XDG_CONFIG_HOME=$U/.config
export XDG_CACHE_HOME=$U/.cache
export XDG_DATA_HOME=$U/.local/share

HF_TOKEN=$(sed -n '/\[nyu\]/,/^$/p' /gpfs/home/steelr04/.cache/huggingface/stored_tokens | grep hf_token | awk '{print $3}')

EXTRA=""
case ",$FLAGS," in *,eig,*) EXTRA="$EXTRA --eig";; esac
case ",$FLAGS," in *,no-measure,*) EXTRA="$EXTRA --no-measure";; esac

PROBE_OUT="$OUTDIR/probe_${MODEL}_s${SEED}_$(date +%s).jsonl"
mkdir -p "$OUTDIR"

echo "[probe_run_one] host=$(hostname) gpu=$GPU model=$MODEL seed=$SEED stages=$STAGES flags=$FLAGS -> $PROBE_OUT"

enroot start \
  --mount /gpfs/data/oermannlab:/gpfs/data/oermannlab \
  --env CUDA_VISIBLE_DEVICES="$GPU" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env HF_TOKEN="$HF_TOKEN" \
  --env HF_HOME=$U/.cache/huggingface \
  --env TORCH_HOME=$U/.cache/torch \
  --env XDG_CACHE_HOME=$U/.cache \
  pytorch_min -- bash -c "
    cd $REPO &&
    python3 MiN/min_stab_probe.py \
      --base_configs  MiN/configs/base_configs/${BASE}.json \
      --model_configs MiN/configs/model_configs/${MODEL}.json \
      --seed $SEED --stages $STAGES --probe_out $PROBE_OUT $EXTRA
  "
rc=$?
echo "[probe_run_one] exit=$rc"
exit $rc
