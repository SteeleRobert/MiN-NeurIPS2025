#!/usr/bin/env bash
# Submit separate SLURM jobs for DINOv2 and DINOv3 benchmark sweeps.
#
# Usage (from MiN-NeurIPS2025 repository root):
#   ./slurm/submit_dino_backbones.sh
#
# Optional:
#   SLURM_EXTRA="--partition=gpu4_short --time=12:00:00" ./slurm/submit_dino_backbones.sh
#   (override partition/time — edit run_dino_benchmarks.slurm if you prefer defaults)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SLURM_SCRIPT="$ROOT/slurm/run_dino_benchmarks.slurm"
if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "Missing $SLURM_SCRIPT"
    exit 1
fi

# shellcheck disable=SC2086
submit_one() {
    local backbone="$1"
    sbatch ${SLURM_EXTRA:-} --export=ALL,BACKBONE="$backbone" "$SLURM_SCRIPT"
}

echo "Submitting DINOv2 (dinov2_vitb14)..."
submit_one dinov2_vitb14

echo "Submitting DINOv3 (dinov3_vitb16)..."
submit_one dinov3_vitb16

echo "Done. SLURM stdout/stderr: dino-bench-<jobid>.out/.err in $ROOT"
echo "Check: squeue -u \$USER"
