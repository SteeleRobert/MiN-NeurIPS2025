#!/bin/bash
# Run MiN unlearning experiments across all EASE benchmarks × 3 backbones as a
# job array.  Each task trains MiN on one (backbone, benchmark) pair, then runs
# ablations A1/A2/A3 for every task (--all_tasks), and saves a checkpoint.
#
# 18 array tasks total: indices 0–5 per backbone, three backbone blocks.
#
#   Task  0– 5  backbone=pretrained_vit_b16_224_in21k_min  (original MiN)
#   Task  6–11  backbone=dinov2_vitb14
#   Task 12–17  backbone=dinov3_vitb16   ← requires HF_TOKEN
#
# Within each backbone block, bench index = task_id % 6:
#   0: CIFAR-100      B0-Inc5  (20T)
#   1: ImageNet-R     B0-Inc10 (20T)
#   2: ImageNet-A     B0-Inc20 (10T)
#   3: CUB-200        B0-Inc10 (20T)
#   4: OmniBenchmark  B0-Inc30 (10T)
#   5: VTAB           B0-Inc10  (5T)
#
# Submit from the MiN-NeurIPS2025 root:
#   cd ~/MiN-NeurIPS2025
#   sbatch slurm/unlearn_benchmarks.slurm
#
# DINOv3 requires a HuggingFace token (pre-accepted model license):
#   sbatch --export=ALL,HF_TOKEN=hf_... slurm/unlearn_benchmarks.slurm
#
# Run a single backbone block:
#   sbatch --array=0-5   slurm/unlearn_benchmarks.slurm   # ViT-B/16 only
#   sbatch --array=6-11  slurm/unlearn_benchmarks.slurm   # DINOv2 only
#   sbatch --array=12-17 slurm/unlearn_benchmarks.slurm   # DINOv3 only
#
# Run a single (backbone, benchmark) pair, e.g. DINOv2 × CIFAR-100 (task 6):
#   sbatch --array=6 slurm/unlearn_benchmarks.slurm
#
# Re-run unlearning from saved checkpoints (skip training):
#   sbatch --export=ALL,CKPT_DIR=$HOME/min_unlearn_results slurm/unlearn_benchmarks.slurm
#
# Override data root:
#   sbatch --export=ALL,DATA_ROOT=/gpfs/data/oermannlab/... slurm/unlearn_benchmarks.slurm
#
# Slack notifications (optional):
#   export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
#   or:  echo 'https://...' > slurm/.slack_webhook_url

#SBATCH --job-name=min-unlearn
#SBATCH --array=0-17
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/MiN-NeurIPS2025/slurm/logs/unlearn_%A_%a.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/MiN-NeurIPS2025/slurm/logs/unlearn_%A_%a.err

set -euo pipefail

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/MiN/unlearn.py" ]]; then
    echo "Submit from the MiN-NeurIPS2025 root so SLURM_SUBMIT_DIR is set correctly."
    exit 1
fi
cd "$REPO_ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv — run 'uv sync' on a login node first."
    exit 1
fi

mkdir -p slurm/logs

# ── CUDA module ───────────────────────────────────────────────────────────────
module load cuda/12.4 2>/dev/null \
    || module load cuda/12.1 2>/dev/null \
    || module load cuda/11.8 2>/dev/null \
    || echo "WARNING: no cuda module loaded; relying on runtime PATH"

# ── Slack helper ──────────────────────────────────────────────────────────────
# shellcheck source=slack_notify.sh
source "$REPO_ROOT/slurm/slack_notify.sh"

_slack_on_exit_failure() {
    local rc=$?
    [[ $rc -eq 0 ]] && return 0
    min_slack_notify_failure "$rc" "*min-unlearn* [${TAG:-task${SLURM_ARRAY_TASK_ID}}]"
}
trap '_slack_on_exit_failure' EXIT

# ── Derive backbone and benchmark indices from array task ID ──────────────────
N_BENCH=6
N_BACKBONE=3

BACKBONE_IDX=$(( SLURM_ARRAY_TASK_ID / N_BENCH ))
BENCH_IDX=$(( SLURM_ARRAY_TASK_ID % N_BENCH ))

BACKBONES=(
    pretrained_vit_b16_224_in21k_min
    dinov2_vitb14
    dinov3_vitb16
)
BSUFFIXES=(
    ""
    "-dinov2vitb14"
    "-dinov3vitb16"
)
BTAGS=(
    ""
    "_dinov2_vitb14"
    "_dinov3_vitb16"
)

BACKBONE="${BACKBONES[$BACKBONE_IDX]}"
BSUFFIX="${BSUFFIXES[$BACKBONE_IDX]}"
BTAG="${BTAGS[$BACKBONE_IDX]}"

# ── Benchmark table (index 0–5) ───────────────────────────────────────────────
# TRAIN/TEST_SUBDIR: relative to DATA_ROOT; empty = auto-downloaded (CIFAR).
BENCH_BASE=(  cifar_ease        imagenetr_ease         imageneta_ease         cub_ease              omnibenchmark_ease       vtab_ease       )
BENCH_MODEL=( MiN-cifar-10steps MiN-inr-10steps        MiN-imageneta          MiN-cub-10steps       MiN-omni-10steps         MiN-vtab-5steps )
BENCH_TAG=(   cifar100_20T      imagenetr_20T          imageneta_10T          cub200_20T            omnibenchmark_10T        vtab_5T         )
BENCH_TRAIN=( ""                imagenet-r/train       imagenet-a/train       cub/train             omnibenchmark/train      vtab/train      )
BENCH_TEST=(  ""                imagenet-r/test        imagenet-a/test        cub/test              omnibenchmark/test       vtab/test       )

BASE_CFG="${BENCH_BASE[$BENCH_IDX]}"
MODEL_CFG_STEM="${BENCH_MODEL[$BENCH_IDX]}"
TAG="${BENCH_TAG[$BENCH_IDX]}${BTAG}"
TRAIN_SUBDIR="${BENCH_TRAIN[$BENCH_IDX]}"
TEST_SUBDIR="${BENCH_TEST[$BENCH_IDX]}"

# Append backbone suffix to model config name (empty for default backbone).
MODEL_CFG="${MODEL_CFG_STEM}${BSUFFIX}"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/gpfs/data/oermannlab/public_data/continual_learning}"
LOG_DIR="${LOG_DIR:-$HOME/min_unlearn_logs}"
OUT_DIR="${OUT_DIR:-$HOME/min_unlearn_results}/${TAG}"
CKPT_DIR="${CKPT_DIR:-}"   # set to re-use saved checkpoints (skip training)

mkdir -p "$LOG_DIR" "$OUT_DIR"
LOG="$LOG_DIR/${TAG}.log"

CONFIGS="MiN/configs"

# ── Print job info ────────────────────────────────────────────────────────────
echo "=== MiN unlearn [array task ${SLURM_ARRAY_TASK_ID}: backbone=${BACKBONE_IDX} bench=${BENCH_IDX}] ==="
echo "Host:       $(hostname)"
echo "Start:      $(date -Is)"
echo "Tag:        ${TAG}"
echo "Backbone:   ${BACKBONE}"
echo "Base cfg:   ${BASE_CFG}.json"
echo "Model cfg:  ${MODEL_CFG}.json"
echo "Data root:  ${DATA_ROOT}"
echo "Output dir: ${OUT_DIR}"
echo "Log:        ${LOG}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-assigned by SLURM}"
echo "==========================================="

# ── CUDA sanity-check ────────────────────────────────────────────────────────
.venv/bin/python - <<'PYCHECK'
import torch, sys
print(f"  torch={torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available.")
    sys.exit(1)
print(f"  device={torch.cuda.get_device_name(0)}")
PYCHECK

# ── HuggingFace auth (DINOv3 only) ───────────────────────────────────────────
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
if [[ -n "$HF_TOKEN" ]]; then
    .venv/bin/python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>/dev/null || true
fi

# ── Ensure ImageNet-R/A train/test splits exist ───────────────────────────────
if [[ "$BENCH_IDX" == "1" || "$BENCH_IDX" == "2" ]]; then
    if [[ ! -d "$DATA_ROOT/${TRAIN_SUBDIR}" || ! -d "$DATA_ROOT/${TEST_SUBDIR}" ]]; then
        echo "[${TAG}] Creating train/test splits via prepare_data.py …"
        .venv/bin/python prepare_data.py --data-root "$DATA_ROOT"
    else
        echo "[${TAG}] Train/test splits present."
    fi
fi

# ── Data availability check (skip gracefully if data is absent) ──────────────
if [[ -n "$TRAIN_SUBDIR" ]]; then
    for dir in "$DATA_ROOT/$TRAIN_SUBDIR" "$DATA_ROOT/$TEST_SUBDIR"; do
        if [[ ! -d "$dir" ]] || [[ -z "$(find "$dir" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]]; then
            echo "[${TAG}] Skipped — data not found or empty: $dir"
            min_slack_notify "*min-unlearn* [${TAG}] — Skipped (data absent: $dir)"
            exit 0
        fi
    done
fi

# ── Model config existence check ─────────────────────────────────────────────
MODEL_CFG_PATH="$CONFIGS/model_configs/${MODEL_CFG}.json"
if [[ ! -f "$MODEL_CFG_PATH" ]]; then
    echo "Model config not found: $MODEL_CFG_PATH"
    echo "Available configs for this backbone suffix (${BSUFFIX:-(default)}):"
    ls "$CONFIGS/model_configs/"*"${BSUFFIX}.json" 2>/dev/null || ls "$CONFIGS/model_configs/"*.json
    exit 1
fi

# ── Build unlearn command ─────────────────────────────────────────────────────
UNLEARN_CMD=".venv/bin/python MiN/unlearn.py"
UNLEARN_CMD+=" --base_configs  $CONFIGS/base_configs/${BASE_CFG}.json"
UNLEARN_CMD+=" --model_configs $MODEL_CFG_PATH"
UNLEARN_CMD+=" --all_tasks"
UNLEARN_CMD+=" --save_checkpoint"
UNLEARN_CMD+=" --output_dir    $OUT_DIR"
UNLEARN_CMD+=" --data_root     $DATA_ROOT"

# Re-use saved checkpoint to skip retraining (useful for reruns).
if [[ -n "$CKPT_DIR" && -f "$CKPT_DIR/${TAG}/trained_model.pt" ]]; then
    echo "[${TAG}] Loading checkpoint from $CKPT_DIR/${TAG}/trained_model.pt (skipping training)"
    UNLEARN_CMD+=" --checkpoint $CKPT_DIR/${TAG}/trained_model.pt"
fi

echo "[${TAG}] Command: $UNLEARN_CMD"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
set +e
CUDA_VISIBLE_DEVICES=0 eval "$UNLEARN_CMD" 2>&1 | tee "$LOG"
run_rc=${PIPESTATUS[0]}
set -e

echo ""
echo "=== Finished: $(date -Is) (exit ${run_rc}) ==="

# Extract summary lines for Slack.
summary=$(strings "$LOG" | grep -E 'FINAL:|Forgotten|Mean retained|BWT' | tail -10 || true)

if [[ $run_rc -eq 0 ]]; then
    min_slack_notify "*min-unlearn* [${TAG}] backbone=${BACKBONE} — finished OK
${summary}"
else
    # Disable the EXIT trap before calling min_slack_notify_failure explicitly,
    # so the trap does not fire a second notification when we exit below.
    trap - EXIT
    min_slack_notify_failure "$run_rc" "*min-unlearn* [${TAG}] backbone=${BACKBONE}"
    exit "$run_rc"
fi
