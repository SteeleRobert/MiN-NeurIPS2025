#!/usr/bin/env bash
# Run MiN on ImageNet-R and ImageNet-A for DINOv2 and DINOv3 backbones.
# Usage:
#   bash run_imagenet_dino.sh
#   GPU0=0 GPU1=1 bash run_imagenet_dino.sh

set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
LOG_DIR="${LOG_DIR:-$HOME/min_benchmark_logs}"
DATA_ROOT="${DATA_ROOT:-$HOME/qz-compcont-learning/data}"
CONFIGS="MiN/configs"

mkdir -p "$LOG_DIR"
cd "$(dirname "$0")"

# Source Slack notifier (no-op if not configured)
source slurm/slack_notify.sh 2>/dev/null || true

_slack_on_exit_failure() {
    local rc=$?
    [[ $rc -eq 0 ]] && return 0
    min_slack_notify_failure "$rc" "*run_imagenet_dino.sh* — imagenet-r/a DINOv2+DINOv3"
}
trap '_slack_on_exit_failure' EXIT

# ── Ensure train/test splits exist ───────────────────────────────────────────
for split in imagenet-r imagenet-a; do
    if [[ ! -d "$DATA_ROOT/$split/train" || ! -d "$DATA_ROOT/$split/test" ]]; then
        echo "Train/test splits missing — running prepare_data.py..."
        .venv/bin/python prepare_data.py --data-root "$DATA_ROOT"
        break
    fi
done

run() {
    local base="$1" model="$2" gpu="$3" tag="$4"
    local log="$LOG_DIR/${tag}.log"
    echo "[$tag] Starting on GPU $gpu -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python MiN/main.py \
        --base_configs  "$CONFIGS/base_configs/${base}.json" \
        --model_configs "$CONFIGS/model_configs/${model}.json" \
        > "$log" 2>&1
    echo "[$tag] Done."
    strings "$log" | grep -E 'FINAL' | tail -3 || true
}

echo "=== Batch 1: ImageNet-R (DINOv2 GPU${GPU0}, DINOv3 GPU${GPU1}) ==="
run imagenetr_ease MiN-inr-10steps-dinov2vitb14 "$GPU0" "imagenetr_ease_20T_dinov2_vitb14" &
run imagenetr_ease MiN-inr-10steps-dinov3vitb16 "$GPU1" "imagenetr_ease_20T_dinov3_vitb16" &
wait || true

echo "=== Batch 2: ImageNet-A (DINOv2 GPU${GPU0}, DINOv3 GPU${GPU1}) ==="
run imageneta_ease MiN-imageneta-dinov2vitb14 "$GPU0" "imageneta_ease_10T_dinov2_vitb14" &
run imageneta_ease MiN-imageneta-dinov3vitb16 "$GPU1" "imageneta_ease_10T_dinov3_vitb16" &
wait || true

echo ""
echo "All done. Results:"
for tag in \
    imagenetr_ease_20T_dinov2_vitb14 \
    imagenetr_ease_20T_dinov3_vitb16 \
    imageneta_ease_10T_dinov2_vitb14 \
    imageneta_ease_10T_dinov3_vitb16; do
    log="$LOG_DIR/${tag}.log"
    [[ -f "$log" ]] || continue
    result=$(strings "$log" | grep 'FINAL' | tail -1 || true)
    echo "  [$tag] ${result:-no FINAL line found}"
done

min_slack_notify "*run_imagenet_dino.sh* — imagenet-r/a DINOv2+DINOv3 — finished OK" 2>/dev/null || true
