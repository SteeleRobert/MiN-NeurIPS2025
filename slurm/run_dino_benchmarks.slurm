#!/bin/bash
# Run all MiN EASE-protocol benchmarks with a DINO backbone (DINOv2 or DINOv3).
# All benchmarks launch in parallel — one benchmark per GPU (up to 7 GPUs).
# Benchmarks whose data is missing are skipped automatically.
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
#   DATA_ROOT     — datasets (default: /gpfs/data/oermannlab/public_data/continual_learning)
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
#SBATCH --cpus-per-task=56
#SBATCH --mem=224G
#SBATCH --gres=gpu:7
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

export BACKBONE
export DATA_ROOT="${DATA_ROOT:-/gpfs/data/oermannlab/public_data/continual_learning}"
export LOG_DIR="${LOG_DIR:-$HOME/min_benchmark_logs/${BACKBONE}}"
mkdir -p "$LOG_DIR"

# shellcheck source=slack_notify.sh
source "$REPO_ROOT/slurm/slack_notify.sh"

echo "=== MiN DINO benchmarks (all parallel) ==="
echo "Host: $(hostname)"
echo "Start: $(date -Is)"
echo "REPO_ROOT=$REPO_ROOT"
echo "BACKBONE=$BACKBONE"
echo "DATA_ROOT=$DATA_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "==========================================="

# ── Config suffix ─────────────────────────────────────────────────────────────
case "$BACKBONE" in
    dinov2_vitb14) BSUFFIX="-dinov2vitb14" ; BTAG="_${BACKBONE}" ;;
    dinov3_vitb16) BSUFFIX="-dinov3vitb16" ; BTAG="_${BACKBONE}" ;;
esac
CONFIGS="MiN/configs"

# ── Ensure train/test splits exist for ImageNet-R and ImageNet-A ──────────────
need_prepare=0
for split in imagenet-r imagenet-a; do
    if [[ ! -d "$DATA_ROOT/$split/train" || ! -d "$DATA_ROOT/$split/test" ]]; then
        need_prepare=1
        break
    fi
done
if [[ $need_prepare -eq 1 ]]; then
    echo "Train/test splits missing — running prepare_data.py..."
    .venv/bin/python prepare_data.py --data-root "$DATA_ROOT"
else
    echo "Train/test splits present, skipping prepare_data.py."
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
# data_ok <tag> <train_dir> <test_dir>
data_ok() {
    local tag="$1" train_dir="$2" test_dir="$3"
    for dir in "$train_dir" "$test_dir"; do
        if [[ ! -d "$dir" ]]; then
            echo "[$tag] Skipped — directory not found: $dir"
            return 1
        fi
        if [[ -z "$(find "$dir" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]]; then
            echo "[$tag] Skipped — no class subdirectories in: $dir"
            return 1
        fi
    done
    return 0
}

run_bench() {
    local base="$1" model="$2" gpu="$3" tag="$4"
    local log="$LOG_DIR/${tag}.log"
    echo "[$tag] Starting on GPU $gpu → $log"
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python MiN/main.py \
        --base_configs  "$CONFIGS/base_configs/${base}.json" \
        --model_configs "$CONFIGS/model_configs/${model}.json" \
        > "$log" 2>&1
    local rc=$?
    echo "[$tag] Done (exit $rc)."
    strings "$log" | grep -E 'FINAL:' | tail -1 || true
    return $rc
}

# ── Launch all benchmarks in parallel ─────────────────────────────────────────
gpu=0

# CIFAR-100 B0-Inc5 (20T) — always runs (auto-downloads)
run_bench cifar_ease "MiN-cifar-10steps${BSUFFIX}" $((gpu++)) "cifar100_ease_20T${BTAG}" &

# ImageNet-R B0-Inc10 (20T)
if data_ok "imagenetr_ease_20T${BTAG}" "$DATA_ROOT/imagenet-r/train" "$DATA_ROOT/imagenet-r/test"; then
    run_bench imagenetr_ease "MiN-inr-10steps${BSUFFIX}" $((gpu++)) "imagenetr_ease_20T${BTAG}" &
fi

# ImageNet-A B0-Inc20 (10T)
if data_ok "imageneta_ease_10T${BTAG}" "$DATA_ROOT/imagenet-a/train" "$DATA_ROOT/imagenet-a/test"; then
    run_bench imageneta_ease "MiN-imageneta${BSUFFIX}" $((gpu++)) "imageneta_ease_10T${BTAG}" &
fi

# CUB-200 B0-Inc10 (20T)
if data_ok "cub200_ease_20T${BTAG}" "$DATA_ROOT/cub/train" "$DATA_ROOT/cub/test"; then
    run_bench cub_ease "MiN-cub-10steps${BSUFFIX}" $((gpu++)) "cub200_ease_20T${BTAG}" &
fi

# OmniBenchmark B0-Inc30 (10T)
if data_ok "omnibenchmark_ease_10T${BTAG}" "$DATA_ROOT/omnibenchmark/train" "$DATA_ROOT/omnibenchmark/test"; then
    run_bench omnibenchmark_ease "MiN-omni-10steps${BSUFFIX}" $((gpu++)) "omnibenchmark_ease_10T${BTAG}" &
fi

# VTAB B0-Inc10 (5T)
if data_ok "vtab_ease_5T${BTAG}" "$DATA_ROOT/vtab/train" "$DATA_ROOT/vtab/test"; then
    run_bench vtab_ease "MiN-vtab-5steps${BSUFFIX}" $((gpu++)) "vtab_ease_5T${BTAG}" &
fi

# ObjectNet B0-Inc10 (20T)
if data_ok "objectnet_ease_20T${BTAG}" "$DATA_ROOT/objectnet/train" "$DATA_ROOT/objectnet/test"; then
    run_bench objectnet_ease "MiN-inr-10steps${BSUFFIX}" $((gpu++)) "objectnet_ease_20T${BTAG}" &
fi

# Wait for all parallel benchmarks
bench_rc=0
wait || bench_rc=$?

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Finished: $(date -Is) (exit $bench_rc) ==="
echo ""
echo "Summary (FINAL lines):"
for log in "$LOG_DIR"/*${BTAG}*.log; do
    [[ -f "$log" ]] || continue
    tag=$(basename "$log" .log)
    result=$(strings "$log" | grep -E 'FINAL:' | tail -1 || true)
    echo "  [$tag] ${result:-no FINAL line}"
done

if [[ $bench_rc -eq 0 ]]; then
    min_slack_notify "finished OK — all benchmarks"
else
    min_slack_notify_failure "$bench_rc"
    exit "$bench_rc"
fi
