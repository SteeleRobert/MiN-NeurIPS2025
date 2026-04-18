#!/bin/bash
# Cache post-noise MiN embeddings for all benchmarks (one SLURM array task per benchmark).
#
# Runs the full MiN training loop and, after each task T, extracts the CLS-token
# embeddings of the cumulative test set as seen through the backbone + all T
# noise modules.  Embeddings are saved as .npz files for downstream geometry
# analysis (WtB separability, NCM margin).
#
# Submit (one backbone at a time):
#   cd /path/to/MiN-NeurIPS2025
#   sbatch --export=ALL,BACKBONE=pretrained_vit_b16_224_in21k_min slurm/cache_embeddings.slurm
#   sbatch --export=ALL,BACKBONE=dinov2_vitb14                     slurm/cache_embeddings.slurm
#   sbatch --export=ALL,BACKBONE=dinov3_vitb16                     slurm/cache_embeddings.slurm
#
# To run a single benchmark (e.g. ImageNet-R only):
#   sbatch --array=1 --export=ALL,BACKBONE=dinov2_vitb14 slurm/cache_embeddings.slurm
#
# Benchmark array index → dataset:
#   0: CIFAR-100         (B0-Inc5,   20T)
#   1: ImageNet-R        (B0-Inc5,   40T)   ← qz-compcont-learning default protocol
#   2: ImageNet-A        (B0-Inc20,  10T)
#   3: CUB-200           (B0-Inc10,  20T)
#   4: OmniBenchmark     (B0-Inc30,  10T)
#   5: VTAB              (B0-Inc10,   5T)
#
# Output layout (per array task):
#   {EMBED_DIR}/{dataset}/{backbone_type}/task_NN.npz
#
# Optional env overrides (via --export=ALL,...):
#   REPO_ROOT   — MiN-NeurIPS2025 root  (default: dir from which sbatch was called)
#   DATA_ROOT   — datasets root          (default: /gpfs/data/oermannlab/public_data/continual_learning)
#   EMBED_DIR   — embedding output dir   (default: $HOME/min_embeddings)
#
# DINOv3: export HF_TOKEN=... before submitting (or huggingface-cli login on a login node).
#
##SBATCH --account=<YOUR_ACCOUNT>
##SBATCH --qos=<YOUR_QOS>

#SBATCH --job-name=min-embed-cache
#SBATCH --array=0-5
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=embed-cache-%A_%a.out
#SBATCH --error=embed-cache-%A_%a.err

set -euo pipefail

: "${BACKBONE:?Set BACKBONE, e.g. sbatch --export=ALL,BACKBONE=dinov2_vitb14 slurm/cache_embeddings.slurm}"

case "$BACKBONE" in
    pretrained_vit_b16_224_in21k_min|dinov2_vitb14|dinov3_vitb16) ;;
    *)
        echo "Invalid BACKBONE='$BACKBONE'."
        echo "Supported: pretrained_vit_b16_224_in21k_min | dinov2_vitb14 | dinov3_vitb16"
        exit 1
        ;;
esac

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/run_benchmarks.sh" ]]; then
    echo "REPO_ROOT must point to MiN-NeurIPS2025 (containing run_benchmarks.sh)."
    echo "Submit with: cd /path/to/MiN-NeurIPS2025 && sbatch ..."
    exit 1
fi
cd "$REPO_ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv — run: uv sync && source .venv/bin/activate (on a login node)"
    exit 1
fi

export DATA_ROOT="${DATA_ROOT:-/gpfs/data/oermannlab/public_data/continual_learning}"
export EMBED_DIR="${EMBED_DIR:-$HOME/min_embeddings}"
mkdir -p "$EMBED_DIR"

# ── Config suffix by backbone ─────────────────────────────────────────────────
case "$BACKBONE" in
    pretrained_vit_b16_224_in21k_min) BSUFFIX="" ;;
    dinov2_vitb14)                     BSUFFIX="-dinov2vitb14" ;;
    dinov3_vitb16)                     BSUFFIX="-dinov3vitb16" ;;
esac
CONFIGS="MiN/configs"

# ── Benchmark table (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────────
# Columns: BASE_CONFIG  MODEL_CONFIG_STEM  TAG  TRAIN_DIR  TEST_DIR
# TRAIN_DIR/TEST_DIR empty → auto-download (CIFAR-100 only).
BENCH_BASE=(    cifar_ease        imagenetr_default    imageneta_ease       cub_ease              omnibenchmark_ease      vtab_ease           )
BENCH_MODEL=(   MiN-cifar-10steps MiN-inr-10steps      MiN-imageneta        MiN-cub-10steps       MiN-omni-10steps        MiN-vtab-5steps     )
BENCH_TAG=(     cifar100_20T      imagenetr_40T         imageneta_10T        cub200_20T            omnibenchmark_10T       vtab_5T             )
BENCH_TRAIN=(   ""                imagenet-r/train     imagenet-a/train     cub/train             omnibenchmark/train     vtab/train          )
BENCH_TEST=(    ""                imagenet-r/test      imagenet-a/test      cub/test              omnibenchmark/test      vtab/test           )

IDX="$SLURM_ARRAY_TASK_ID"
BASE_CFG="${BENCH_BASE[$IDX]}"
MODEL_CFG="${BENCH_MODEL[$IDX]}${BSUFFIX}"
TAG="${BENCH_TAG[$IDX]}_${BACKBONE}"
TRAIN_DIR="${BENCH_TRAIN[$IDX]}"
TEST_DIR="${BENCH_TEST[$IDX]}"

echo "=== MiN embedding cache [array task $IDX] ==="
echo "Host:       $(hostname)"
echo "Start:      $(date -Is)"
echo "BACKBONE:   $BACKBONE"
echo "Benchmark:  $TAG"
echo "REPO_ROOT:  $REPO_ROOT"
echo "DATA_ROOT:  $DATA_ROOT"
echo "EMBED_DIR:  $EMBED_DIR"
echo "=============================================="

# ── Ensure train/test splits exist for ImageNet-R/A ──────────────────────────
if [[ "$IDX" == "1" || "$IDX" == "2" ]]; then
    if [[ ! -d "$DATA_ROOT/${TRAIN_DIR}" || ! -d "$DATA_ROOT/${TEST_DIR}" ]]; then
        echo "Train/test splits missing for $TAG — running prepare_data.py..."
        .venv/bin/python prepare_data.py --data-root "$DATA_ROOT"
    fi
fi

# ── Data check ───────────────────────────────────────────────────────────────
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

# ── Run embedding caching ─────────────────────────────────────────────────────
set +e
CUDA_VISIBLE_DEVICES=0 .venv/bin/python MiN/cache_embeddings.py \
    --base_configs  "$CONFIGS/base_configs/${BASE_CFG}.json" \
    --model_configs "$CONFIGS/model_configs/${MODEL_CFG}.json" \
    --data_root     "$DATA_ROOT" \
    --output_dir    "$EMBED_DIR"
cache_rc=$?
set -e

echo "[$TAG] Done (exit $cache_rc)."

# ── Print summary of cached files ─────────────────────────────────────────────
dataset_key=$(python3 -c "
import json
with open('$CONFIGS/base_configs/${BASE_CFG}.json') as f:
    print(json.load(f)['dataset'])
" 2>/dev/null || true)

if [[ -n "$dataset_key" && -d "$EMBED_DIR/$dataset_key/$BACKBONE" ]]; then
    echo ""
    echo "Cached embeddings:"
    ls -lh "$EMBED_DIR/$dataset_key/$BACKBONE/"*.npz 2>/dev/null || echo "  (none found)"
fi

[[ $cache_rc -eq 0 ]] || exit "$cache_rc"
