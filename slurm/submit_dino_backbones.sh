#!/usr/bin/env bash
# Submit per-benchmark job arrays for DINOv2 and DINOv3 sweeps.
# Each backbone gets its own 7-task array (one task per benchmark, one GPU per task).
#
# Usage (from MiN-NeurIPS2025 repository root):
#   ./slurm/submit_dino_backbones.sh
#
# Optional:
#   SLURM_EXTRA="--partition=gpu4_short --time=12:00:00" ./slurm/submit_dino_backbones.sh
#   (override partition/time — edit run_dino_benchmarks.slurm for persistent defaults)
# Slack: put the webhook in slurm/.slack_webhook_url or export SLACK_WEBHOOK_URL before sbatch
#   (sbatch --export=ALL,... passes env; see slurm/slack_notify.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SLURM_SCRIPT="$ROOT/slurm/run_dino_benchmarks.slurm"
if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "Missing $SLURM_SCRIPT"
    exit 1
fi

# shellcheck disable=SC2086
submit_backbone() {
    local backbone="$1"
    echo "Submitting array (0-6) for backbone=$backbone..."
    sbatch ${SLURM_EXTRA:-} --export=ALL,BACKBONE="$backbone" "$SLURM_SCRIPT"
}

submit_backbone dinov2_vitb14
submit_backbone dinov3_vitb16

echo ""
echo "Done. Two 7-task arrays submitted (one per backbone, one task per benchmark)."
echo "SLURM stdout/stderr: dino-bench-<arrayid>_<taskid>.out/.err in $ROOT"
echo "Check: squeue -u \$USER"
