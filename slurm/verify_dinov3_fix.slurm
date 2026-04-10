#!/bin/bash
# Verify that the DINOv3 normalization fix restores performance comparable to DINOv2.
# Runs CIFAR-100 and OmniBenchmark with both DINOv2 (baseline) and DINOv3 (fixed).
#
# Array index → (backbone, benchmark) mapping:
#   0: DINOv2-B/14   CIFAR-100        (B0-Inc5,  20T)
#   1: DINOv2-B/14   OmniBenchmark    (B0-Inc30, 10T)
#   2: DINOv3-B/16   CIFAR-100        (B0-Inc5,  20T)   ← verify fix
#   3: DINOv3-B/16   OmniBenchmark    (B0-Inc30, 10T)   ← verify fix
#
# Submit all 4 tasks (DINOv2 baseline + DINOv3 fixed):
#   cd /path/to/MiN-NeurIPS2025
#   sbatch slurm/verify_dinov3_fix.slurm
#
# Submit only DINOv3 tasks (skip re-running DINOv2 if baseline already exists):
#   sbatch --array=2-3 slurm/verify_dinov3_fix.slurm
#
# Submit a single task (e.g. DINOv3 CIFAR-100 only):
#   sbatch --array=2   slurm/verify_dinov3_fix.slurm
#
# Optional env overrides:
#   REPO_ROOT   — repo path (default: directory from which sbatch was run)
#   DATA_ROOT   — datasets (default: /gpfs/data/oermannlab/public_data/continual_learning)
#   LOG_DIR     — benchmark logs (default: $HOME/min_benchmark_logs/dinov3_fix_verify)
#   HF_TOKEN    — HuggingFace token for DINOv3 model download
#                 (or run `huggingface-cli login` once on a login node)
#
# Slack on completion: export SLACK_WEBHOOK_URL=... or create slurm/.slack_webhook_url
#
# Uncomment and set if your site requires:
##SBATCH --account=<YOUR_ACCOUNT>
##SBATCH --qos=<YOUR_QOS>

#SBATCH --job-name=min-dinov3-verify
#SBATCH --array=0-3
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --output=dinov3-verify-%A_%a.out
#SBATCH --error=dinov3-verify-%A_%a.err

set -euo pipefail

# ── Array index → backbone + benchmark ───────────────────────────────────────
# Layout: first N_BENCH slots = DINOv2, next N_BENCH slots = DINOv3.
# BENCH_IDX 0 = CIFAR-100, BENCH_IDX 1 = OmniBenchmark.

BACKBONES=(dinov2_vitb14 dinov3_vitb16)
N_BENCH=2

BACKBONE_IDX=$(( SLURM_ARRAY_TASK_ID / N_BENCH ))
BENCH_IDX=$(( SLURM_ARRAY_TASK_ID % N_BENCH ))
BACKBONE="${BACKBONES[$BACKBONE_IDX]}"

# ── Repo root ────────────────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/run_benchmarks.sh" ]]; then
    echo "Submit from MiN-NeurIPS2025 root so SLURM_SUBMIT_DIR is set correctly."
    exit 1
fi
cd "$REPO_ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv — run: uv sync (on a login or interactive node)"
    exit 1
fi

# ── HuggingFace auth for DINOv3 ──────────────────────────────────────────────
if [[ "$BACKBONE" == dinov3_* && -n "${HF_TOKEN:-}" ]]; then
    .venv/bin/python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" \
        2>/dev/null || true
fi

export BACKBONE
export DATA_ROOT="${DATA_ROOT:-/gpfs/data/oermannlab/public_data/continual_learning}"
export LOG_DIR="${LOG_DIR:-$HOME/min_benchmark_logs/dinov3_fix_verify}"
mkdir -p "$LOG_DIR"

# shellcheck source=slack_notify.sh
source "$REPO_ROOT/slurm/slack_notify.sh"

# ── Config suffix ─────────────────────────────────────────────────────────────
case "$BACKBONE" in
    dinov2_vitb14) BSUFFIX="-dinov2vitb14" ; BTAG="_${BACKBONE}" ;;
    dinov3_vitb16) BSUFFIX="-dinov3vitb16" ; BTAG="_${BACKBONE}" ;;
esac
CONFIGS="MiN/configs"

# ── Benchmark definitions (BENCH_IDX 0=CIFAR-100, 1=OmniBenchmark) ───────────
BENCH_BASE=(  cifar_ease         omnibenchmark_ease      )
BENCH_MODEL=( MiN-cifar-10steps  MiN-omni-10steps        )
BENCH_TAG=(   cifar100_ease_20T  omnibenchmark_ease_10T  )
BENCH_TRAIN=( ""                 omnibenchmark/train      )
BENCH_TEST=(  ""                 omnibenchmark/test       )

BASE_CFG="${BENCH_BASE[$BENCH_IDX]}"
MODEL_CFG="${BENCH_MODEL[$BENCH_IDX]}${BSUFFIX}"
TAG="${BENCH_TAG[$BENCH_IDX]}${BTAG}"
TRAIN_DIR="${BENCH_TRAIN[$BENCH_IDX]}"
TEST_DIR="${BENCH_TEST[$BENCH_IDX]}"

echo "=== MiN DINOv3 fix verification [array task $SLURM_ARRAY_TASK_ID] ==="
echo "Host:       $(hostname)"
echo "Start:      $(date -Is)"
echo "BACKBONE:   $BACKBONE  (idx=$BACKBONE_IDX)"
echo "Benchmark:  $TAG       (idx=$BENCH_IDX)"
echo "REPO_ROOT:  $REPO_ROOT"
echo "DATA_ROOT:  $DATA_ROOT"
echo "LOG_DIR:    $LOG_DIR"
echo "====================================================================="

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
    min_slack_notify_failure "$bench_rc" "$TAG"
    exit "$bench_rc"
fi
