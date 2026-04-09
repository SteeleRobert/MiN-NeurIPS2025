#!/bin/bash
# Same as run_dino_benchmarks.slurm, but runs both DINO backbones as a job array.
# Task 0: dinov2_vitb14 — Task 1: dinov3_vitb16
# Requests two A100 GPUs per task (see --constraint in run_dino_benchmarks.slurm).
#
# Submit from repo root:
#   cd /path/to/MiN-NeurIPS2025
#   sbatch slurm/array_dino_benchmarks.slurm
#
# Concurrency: %1 runs one backbone at a time (one 2-GPU node). Change to %2 to run
# both array tasks in parallel on two nodes (if your allocation allows).
#
# Slack on completion (optional): export SLACK_WEBHOOK_URL=... or slurm/.slack_webhook_url
#
# Uncomment if required:
##SBATCH --account=<YOUR_ACCOUNT>
##SBATCH --qos=<YOUR_QOS>

#SBATCH --job-name=min-dino-array
#SBATCH --array=0-1%1
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --output=dino-bench-%A_%a.out
#SBATCH --error=dino-bench-%A_%a.err

set -euo pipefail

BACKBONES=(dinov2_vitb14 dinov3_vitb16)
BACKBONE="${BACKBONES[$SLURM_ARRAY_TASK_ID]}"

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

export GPU0=0
export GPU1=1
export BACKBONE
export DATA_ROOT="${DATA_ROOT:-/gpfs/data/oermannlab/public_data/continual_learning}"
export LOG_DIR="${LOG_DIR:-$HOME/min_benchmark_logs/${BACKBONE}}"

echo "=== MiN DINO benchmarks [array task $SLURM_ARRAY_TASK_ID] ==="
echo "BACKBONE=$BACKBONE"
echo "Start: $(date -Is)"

# shellcheck source=slack_notify.sh
source "$REPO_ROOT/slurm/slack_notify.sh"

set +e
bash ./run_benchmarks.sh
bench_rc=$?
set -euo pipefail

echo "=== Finished: $(date -Is) (exit $bench_rc) ==="

if [[ $bench_rc -eq 0 ]]; then
    min_slack_notify "finished OK"
else
    min_slack_notify "FAILED (exit ${bench_rc})"
    exit "$bench_rc"
fi
