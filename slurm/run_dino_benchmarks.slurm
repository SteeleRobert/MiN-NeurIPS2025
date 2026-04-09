#!/bin/bash
# Run MiN EASE-protocol benchmarks with a DINO backbone as a job array.
# One array task per benchmark (7 benchmarks, indices 0–6), each on 1 GPU.
#
# Submit from the MiN-NeurIPS2025 repository root:
#   cd /path/to/MiN-NeurIPS2025
#   sbatch --export=ALL,BACKBONE=dinov2_vitb14 slurm/run_dino_benchmarks.slurm
#   sbatch --export=ALL,BACKBONE=dinov3_vitb16 slurm/run_dino_benchmarks.slurm
#
# To run a single benchmark:
#   sbatch --array=0 --export=ALL,BACKBONE=dinov2_vitb14 slurm/run_dino_benchmarks.slurm
#
# Benchmark index → dataset mapping:
#   0: CIFAR-100        (B0-Inc5,  20T)
#   1: ImageNet-R       (B0-Inc10, 20T)
#   2: ImageNet-A       (B0-Inc20, 10T)
#   3: CUB-200          (B0-Inc10, 20T)
#   4: OmniBenchmark    (B0-Inc30, 10T)
#   5: VTAB             (B0-Inc10,  5T)
#   6: ObjectNet        (B0-Inc10, 20T)
#
# DINOv3: authenticate once on the cluster, e.g. `huggingface-cli login` or
#   export HF_TOKEN=...  (add to ~/.bashrc or use `sbatch --export=ALL,HF_TOKEN,...`)
#
# Optional env overrides:
#   REPO_ROOT     — repo path (default: directory from which sbatch was run)
#   DATA_ROOT     — datasets (default: /gpfs/data/oermannlab/public_data/continual_learning)
#   LOG_DIR       — benchmark logs (default: $HOME/min_benchmark_logs/<BACKBONE>)
# Slack on completion (optional): export SLACK_WEBHOOK_URL=... or create
#   slurm/.slack_webhook_url (see slurm/slack_notify.sh).
#
# Uncomment and set if your site requires:
##SBATCH --account=<YOUR_ACCOUNT>
##SBATCH --qos=<YOUR_QOS>

#SBATCH --job-name=min-dino-bench
#SBATCH --array=0-6
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=dino-bench-%A_%a.out
#SBATCH --error=dino-bench-%A_%a.err

set -euo pipefail

: "${BACKBONE:?Set BACKBONE to dinov2_vitb14 or dinov3_vitb16, e.g. sbatch --export=ALL,BACKBONE=dinov2_vitb14 slurm/run_dino_benchmarks.slurm}"

case "$BACKBONE" in
    dinov2_vitb14|dinov3_vitb16) ;;
    *)
        echo "Invalid BACKBONE='$BACKBONE'. Use dinov2_vitb14 or dinov3_vitb16."
        exit 1
        ;;
esac

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/run_benchmarks.sh" ]]; then
    echo "REPO_ROOT must point to MiN-NeurIPS2025 (containing run_benchmarks.sh). Submit with: cd /path/to/MiN-NeurIPS2025 && sbatch ..."
    exit 1
fi
cd "$REPO_ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv — run: uv sync && source .venv/bin/activate (on a login or interactive node)"
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

# ── Benchmark definitions (indexed by SLURM_ARRAY_TASK_ID) ───────────────────
# Each entry: BASE_CONFIG  MODEL_CONFIG_STEM  TAG  TRAIN_DIR  TEST_DIR
# TRAIN_DIR/TEST_DIR are relative to DATA_ROOT; empty string means no data check (auto-download).
BENCH_BASE=(    cifar_ease        imagenetr_ease        imageneta_ease        cub_ease              omnibenchmark_ease       vtab_ease           objectnet_ease        )
BENCH_MODEL=(   MiN-cifar-10steps MiN-inr-10steps       MiN-imageneta         MiN-cub-10steps       MiN-omni-10steps         MiN-vtab-5steps     MiN-inr-10steps       )
BENCH_TAG=(     cifar100_ease_20T imagenetr_ease_20T    imageneta_ease_10T    cub200_ease_20T       omnibenchmark_ease_10T   vtab_ease_5T        objectnet_ease_20T    )
BENCH_TRAIN=(   ""                imagenet-r/train      imagenet-a/train      cub/train             omnibenchmark/train      vtab/train          objectnet/train       )
BENCH_TEST=(    ""                imagenet-r/test       imagenet-a/test       cub/test              omnibenchmark/test       vtab/test           objectnet/test        )

IDX="$SLURM_ARRAY_TASK_ID"
BASE_CFG="${BENCH_BASE[$IDX]}"
MODEL_CFG="${BENCH_MODEL[$IDX]}${BSUFFIX}"
TAG="${BENCH_TAG[$IDX]}${BTAG}"
TRAIN_DIR="${BENCH_TRAIN[$IDX]}"
TEST_DIR="${BENCH_TEST[$IDX]}"

echo "=== MiN DINO benchmark [array task $IDX] ==="
echo "Host:       $(hostname)"
echo "Start:      $(date -Is)"
echo "BACKBONE:   $BACKBONE"
echo "Benchmark:  $TAG"
echo "REPO_ROOT:  $REPO_ROOT"
echo "DATA_ROOT:  $DATA_ROOT"
echo "LOG_DIR:    $LOG_DIR"
echo "============================================="

# ── Ensure train/test splits exist for ImageNet-R and ImageNet-A ──────────────
if [[ "$IDX" == "1" || "$IDX" == "2" ]]; then
    DSET_DIR="${DATA_ROOT}/${TRAIN_DIR%/train}"
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
