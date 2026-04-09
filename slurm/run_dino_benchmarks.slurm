#!/bin/bash
# Run MiN EASE-protocol benchmarks with a DINO backbone (DINOv2 or DINOv3).
# Uses two A100 GPUs in parallel, matching run_benchmarks.sh (GPU0 / GPU1).
# If your site uses typed GRES instead of --constraint (e.g. --gres=gpu:a100:2), edit below.
#
# Submit from the MiN-NeurIPS2025 repository root:
#   cd /path/to/MiN-NeurIPS2025
#   sbatch --export=ALL,BACKBONE=dinov2_vitb14 slurm/run_dino_benchmarks.slurm
#   sbatch --export=ALL,BACKBONE=dinov3_vitb16 slurm/run_dino_benchmarks.slurm
#
# DINOv3: authenticate once on the cluster, e.g. `huggingface-cli login` or
#   export HF_TOKEN=...  (add to ~/.bashrc or use `sbatch --export=ALL,HF_TOKEN,...`)
#
# Optional env overrides:
#   REPO_ROOT     — repo path (default: directory from which sbatch was run)
#   DATA_ROOT     — datasets (default: /gpfs/data/oermannlab/users/zhouj14 on bigpurple)
#   LOG_DIR       — benchmark logs (default: $HOME/min_benchmark_logs/<BACKBONE>)
# Slack on completion (optional): export SLACK_WEBHOOK_URL=... or create
#   slurm/.slack_webhook_url (see slurm/slack_notify.sh).
#
# Uncomment and set if your site requires:
##SBATCH --account=<YOUR_ACCOUNT>
##SBATCH --qos=<YOUR_QOS>

#SBATCH --job-name=min-dino-bench
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --output=dino-bench-%j.out
#SBATCH --error=dino-bench-%j.err

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

export GPU0=0
export GPU1=1
export BACKBONE
export DATA_ROOT="${DATA_ROOT:-/gpfs/data/oermannlab/users/zhouj14}"
export LOG_DIR="${LOG_DIR:-$HOME/min_benchmark_logs/${BACKBONE}}"

echo "=== MiN DINO benchmarks ==="
echo "Host: $(hostname)"
echo "Start: $(date -Is)"
echo "REPO_ROOT=$REPO_ROOT"
echo "BACKBONE=$BACKBONE"
echo "DATA_ROOT=$DATA_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "==========================="

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
