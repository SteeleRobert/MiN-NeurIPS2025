#!/bin/bash
# Re-run MiN + DINOv3 for CIFAR-100 and OmniBenchmark with latest hyperparameters.
#
# Hyperparameters are baked into the model configs:
#   CIFAR-100:      lr=5e-5, gamma=75,  hidden_dim=96  (MiN-cifar-10steps-dinov3vitb16.json)
#   OmniBenchmark:  lr=1e-5, gamma=25,  hidden_dim=96  (MiN-omni-10steps-dinov3vitb16.json)
#
# Array index → benchmark:
#   0: CIFAR-100
#   1: OmniBenchmark
#
# Submit from MiN-NeurIPS2025 repo root:
#   cd /path/to/MiN-NeurIPS2025
#   sbatch slurm/rerun_dinov3_cifar_omni.slurm
#
# DINOv3 authentication: export HF_TOKEN=... or run `huggingface-cli login` on a
#   login node beforehand.
#
# Optional env overrides:
#   REPO_ROOT   — repo path (default: directory from which sbatch was run)
#   DATA_ROOT   — datasets root (default: /gpfs/data/oermannlab/public_data/continual_learning)
#   LOG_DIR     — log directory (default: $HOME/min_benchmark_logs/dinov3_vitb16)
# Slack on completion: export SLACK_WEBHOOK_URL=... or create slurm/.slack_webhook_url

#SBATCH --job-name=min-dinov3-cifar-omni
#SBATCH --array=0-1
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=min-dinov3-cifar-omni-%A_%a.out
#SBATCH --error=min-dinov3-cifar-omni-%A_%a.err

set -euo pipefail

BACKBONE="dinov3_vitb16"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/run_benchmarks.sh" ]]; then
    echo "Submit from MiN-NeurIPS2025 root so SLURM_SUBMIT_DIR is set correctly."
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

CONFIGS="MiN/configs"

# Array task → (base config, model config stem, tag, train dir, test dir)
# TRAIN_DIR/TEST_DIR relative to DATA_ROOT; empty = no data-presence check (auto-download).
BENCH_BASE=(  cifar_ease         omnibenchmark_ease      )
BENCH_MODEL=( MiN-cifar-10steps  MiN-omni-10steps        )
BENCH_TAG=(   cifar100_ease_20T  omnibenchmark_ease_10T  )
BENCH_TRAIN=( ""                 omnibenchmark/train     )
BENCH_TEST=(  ""                 omnibenchmark/test      )

IDX="$SLURM_ARRAY_TASK_ID"
BASE_CFG="${BENCH_BASE[$IDX]}"
MODEL_CFG="${BENCH_MODEL[$IDX]}-dinov3vitb16"
TAG="${BENCH_TAG[$IDX]}_${BACKBONE}"
TRAIN_DIR="${BENCH_TRAIN[$IDX]}"
TEST_DIR="${BENCH_TEST[$IDX]}"

echo "=== MiN DINOv3 re-run [array task $IDX] ==="
echo "Host:       $(hostname)"
echo "Start:      $(date -Is)"
echo "BACKBONE:   $BACKBONE"
echo "Benchmark:  $TAG"
echo "Base cfg:   $CONFIGS/base_configs/${BASE_CFG}.json"
echo "Model cfg:  $CONFIGS/model_configs/${MODEL_CFG}.json"
echo "REPO_ROOT:  $REPO_ROOT"
echo "DATA_ROOT:  $DATA_ROOT"
echo "LOG_DIR:    $LOG_DIR"
echo "============================================="

# Sanity-check: print the hyperparameters we're actually using
echo "Hyperparameters in use:"
.venv/bin/python -c "
import json, sys
with open('$CONFIGS/model_configs/${MODEL_CFG}.json') as f:
    cfg = json.load(f)
for k in ('lr', 'gamma', 'hidden_dim'):
    print(f'  {k} = {cfg.get(k, \"(not set)\")}')
" || echo "  (could not parse config)"

# ── Data check (skip if OmniBenchmark data is absent) ─────────────────────────
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

# ── Run benchmark ──────────────────────────────────────────────────────────────
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

# ── Notify ─────────────────────────────────────────────────────────────────────
if [[ $bench_rc -eq 0 ]]; then
    final_line="$(strings "$LOG" | grep -E 'FINAL:' | tail -1 || true)"
    min_slack_notify "finished OK — $TAG" "$final_line"
else
    min_slack_notify_failure "$bench_rc" "$TAG"
    exit "$bench_rc"
fi
