#!/bin/bash
# Run all MiN EASE-protocol benchmarks for both DINO backbones as a job array.
# 14 array tasks total: indices 0–6 for dinov2_vitb14, indices 7–13 for dinov3_vitb16.
# Each task runs one benchmark on one GPU.
#
# Array index → (backbone, benchmark) mapping:
#   Task 0–6  (backbone=dinov2_vitb14):
#     0: CIFAR-100, 1: ImageNet-R, 2: ImageNet-A, 3: CUB-200
#     4: OmniBenchmark, 5: VTAB, 6: ObjectNet
#   Task 7–13 (backbone=dinov3_vitb16):
#     7: CIFAR-100, 8: ImageNet-R, 9: ImageNet-A, 10: CUB-200
#     11: OmniBenchmark, 12: VTAB, 13: ObjectNet
#
# Submit from repo root:
#   cd /path/to/MiN-NeurIPS2025
#   sbatch slurm/array_dino_benchmarks.slurm
#
# To run only one backbone:
#   sbatch --array=0-6   slurm/array_dino_benchmarks.slurm   # DINOv2 only
#   sbatch --array=7-13  slurm/array_dino_benchmarks.slurm   # DINOv3 only
#
# To run a single benchmark with one backbone (e.g. DINOv2 CIFAR-100):
#   sbatch --array=0     slurm/array_dino_benchmarks.slurm
#
# Slack on completion (optional): export SLACK_WEBHOOK_URL=... or slurm/.slack_webhook_url
#
# Uncomment if required:
##SBATCH --account=<YOUR_ACCOUNT>
##SBATCH --qos=<YOUR_QOS>

#SBATCH --job-name=min-dino-array
#SBATCH --array=0-13
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=dino-bench-%A_%a.out
#SBATCH --error=dino-bench-%A_%a.err

set -euo pipefail

# ── Resolve backbone and benchmark index from array task ID ──────────────────
BACKBONES=(dinov2_vitb14 dinov3_vitb16)
N_BENCH=7

BACKBONE_IDX=$(( SLURM_ARRAY_TASK_ID / N_BENCH ))
BENCH_IDX=$(( SLURM_ARRAY_TASK_ID % N_BENCH ))
BACKBONE="${BACKBONES[$BACKBONE_IDX]}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/run_benchmarks.sh" ]]; then
    echo "Submit from MiN-NeurIPS2025 root so SLURM_SUBMIT_DIR is set correctly."
    exit 1
fi
cd "$REPO_ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv — run uv sync on a login node first."
    exit 1
fi

export BACKBONE
export DATA_ROOT="${DATA_ROOT:-/gpfs/data/oermannlab/public_data/continual_learning}"
export LOG_DIR="${LOG_DIR:-$HOME/min_benchmark_logs/${BACKBONE}}"
mkdir -p "$LOG_DIR"

# shellcheck source=slack_notify.sh
source "$REPO_ROOT/slurm/slack_notify.sh"

# ── Config suffix ─────────────────────────────────────────────────────────────
case "$BACKBONE" in
    dinov2_vitb14) BSUFFIX="-dinov2vitb14" ; BTAG="_${BACKBONE}" ;;
    dinov3_vitb16) BSUFFIX="-dinov3vitb16" ; BTAG="_${BACKBONE}" ;;
esac
CONFIGS="MiN/configs"

# ── Benchmark definitions (indexed by BENCH_IDX) ─────────────────────────────
BENCH_BASE=(    cifar_ease        imagenetr_ease        imageneta_ease        cub_ease              omnibenchmark_ease       vtab_ease           objectnet_ease        )
BENCH_MODEL=(   MiN-cifar-10steps MiN-inr-10steps       MiN-imageneta         MiN-cub-10steps       MiN-omni-10steps         MiN-vtab-5steps     MiN-inr-10steps       )
BENCH_TAG=(     cifar100_ease_20T imagenetr_ease_20T    imageneta_ease_10T    cub200_ease_20T       omnibenchmark_ease_10T   vtab_ease_5T        objectnet_ease_20T    )
BENCH_TRAIN=(   ""                imagenet-r/train      imagenet-a/train      cub/train             omnibenchmark/train      vtab/train          objectnet/train       )
BENCH_TEST=(    ""                imagenet-r/test       imagenet-a/test       cub/test              omnibenchmark/test       vtab/test           objectnet/test        )

BASE_CFG="${BENCH_BASE[$BENCH_IDX]}"
MODEL_CFG="${BENCH_MODEL[$BENCH_IDX]}${BSUFFIX}"
TAG="${BENCH_TAG[$BENCH_IDX]}${BTAG}"
TRAIN_DIR="${BENCH_TRAIN[$BENCH_IDX]}"
TEST_DIR="${BENCH_TEST[$BENCH_IDX]}"

echo "=== MiN DINO benchmark [array task $SLURM_ARRAY_TASK_ID: backbone=$BACKBONE_IDX bench=$BENCH_IDX] ==="
echo "Host:       $(hostname)"
echo "Start:      $(date -Is)"
echo "BACKBONE:   $BACKBONE"
echo "Benchmark:  $TAG"
echo "DATA_ROOT:  $DATA_ROOT"
echo "LOG_DIR:    $LOG_DIR"
echo "================================================================================="

# ── Ensure train/test splits exist for ImageNet-R and ImageNet-A ──────────────
if [[ "$BENCH_IDX" == "1" || "$BENCH_IDX" == "2" ]]; then
    if [[ ! -d "$DATA_ROOT/${TRAIN_DIR}" || ! -d "$DATA_ROOT/${TEST_DIR}" ]]; then
        echo "Train/test splits missing for $TAG — running prepare_data.py..."
        .venv/bin/python prepare_data.py --data-root "$DATA_ROOT"
    else
        echo "Train/test splits present, skipping prepare_data.py."
    fi
fi

# ── Data check (skip benchmarks whose data is absent) ────────────────────────
if [[ -n "$TRAIN_DIR" ]]; then
    for dir in "$DATA_ROOT/$TRAIN_DIR" "$DATA_ROOT/$TEST_DIR"; do
        if [[ ! -d "$dir" ]]; then
            echo "[$TAG] Skipped — directory not found: $dir"
            exit 0
        fi
        if [[ -z "$(find "$dir" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]]; then
            echo "[$TAG] Skipped — no class subdirectories in: $dir"
            exit 0
        fi
    done
fi

# ── Run benchmark ─────────────────────────────────────────────────────────────
LOG="$LOG_DIR/${TAG}.log"
echo "[$TAG] Starting → $LOG"

set +e
CUDA_VISIBLE_DEVICES=0 .venv/bin/python MiN/main.py \
    --base_configs  "$CONFIGS/base_configs/${BASE_CFG}.json" \
    --model_configs "$CONFIGS/model_configs/${MODEL_CFG}.json" \
    --data_root     "$DATA_ROOT" \
    > "$LOG" 2>&1
bench_rc=$?
set -e

echo "[$TAG] Done (exit $bench_rc)."
strings "$LOG" | grep -E 'FINAL:' | tail -1 || true

# ── Notify ────────────────────────────────────────────────────────────────────
if [[ $bench_rc -eq 0 ]]; then
    min_slack_notify "finished OK — $TAG"
else
    min_slack_notify_failure "$bench_rc"
    exit "$bench_rc"
fi
