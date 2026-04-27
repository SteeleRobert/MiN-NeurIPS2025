#!/bin/bash
#SBATCH --job-name=min_a0
#SBATCH --array=0-20
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/slurm_logs/min_a0_%A_%a.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/slurm_logs/min_a0_%A_%a.err

# ── Config ────────────────────────────────────────────────────────────────────
REPO=/gpfs/data/oermannlab/users/zhouj14/MiN-NeurIPS2025
CKPT_ROOT=/gpfs/data/oermannlab/users/zhouj14/min_unlearn_results
OUT_ROOT=/gpfs/data/oermannlab/users/zhouj14/min_a0_results
DATA_ROOT=/gpfs/data/oermannlab/users/zhouj14/qz-compcont-learning/data
WDS_BASE="${WDS_BASE:-/gpfs/data/oermannlab/public_data/continual_learning/webdatasets}"

# ── Benchmark lookup (index 0–6) ──────────────────────────────────────────────
BENCHMARKS=(cifar100   imagenetr   imageneta   cub200   omnibenchmark   vtab   objectnet)

BASE_CONFIGS=(
    MiN/configs/base_configs/cifar_ease.json
    MiN/configs/base_configs/imagenetr_ease_40T.json
    MiN/configs/base_configs/imageneta_ease.json
    MiN/configs/base_configs/cub_ease.json
    MiN/configs/base_configs/omnibenchmark_ease.json
    MiN/configs/base_configs/vtab_ease.json
    MiN/configs/base_configs/objectnet_ease.json
)

# Protocols used in checkpoint directory names
PROTOCOLS=(20T 40T 10T 20T 10T 5T 20T)

# WDS subdirectory names (matches _WDS_SUBDIR in unlearn.py)
WDS_SUBDIRS=(cifar_100 imagenet_r imagenet_a cub_200 omnibenchmark vtab objectnet)

# ── Backbone lookup (index 0–2) ───────────────────────────────────────────────
BACKBONES=(pretrained_vit_b16_224_in21k_min   dinov2_vitb14   dinov3_vitb16)

# Model config suffix per backbone (empty = default/ViT-B16, otherwise appended)
MODEL_SUFFIXES=(""   "-dinov2vitb14"   "-dinov3vitb16")

# Checkpoint directory suffix per backbone
CKPT_SUFFIXES=(""   "_dinov2_vitb14"   "_dinov3_vitb16")

# ── Map array task ID → (benchmark, backbone) ─────────────────────────────────
BENCH_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
BB_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

BENCHMARK=${BENCHMARKS[$BENCH_IDX]}
BACKBONE=${BACKBONES[$BB_IDX]}
PROTOCOL=${PROTOCOLS[$BENCH_IDX]}
BASE_CFG=${BASE_CONFIGS[$BENCH_IDX]}
MODEL_SUFFIX=${MODEL_SUFFIXES[$BB_IDX]}
CKPT_SUFFIX=${CKPT_SUFFIXES[$BB_IDX]}
WDS_DIR="${WDS_BASE}/${WDS_SUBDIRS[$BENCH_IDX]}"

# Resolve model config base name per benchmark
case $BENCHMARK in
    cifar100)      MODEL_BASE="MiN-cifar-10steps" ;;
    imagenetr)     MODEL_BASE="MiN-inr-10steps" ;;
    imageneta)     MODEL_BASE="MiN-imageneta" ;;
    cub200)        MODEL_BASE="MiN-cub-10steps" ;;
    omnibenchmark) MODEL_BASE="MiN-omni-10steps" ;;
    vtab)          MODEL_BASE="MiN-vtab-5steps" ;;
    objectnet)     MODEL_BASE="MiN-inr-10steps" ;;
esac

MODEL_CFG="MiN/configs/model_configs/${MODEL_BASE}${MODEL_SUFFIX}.json"
CKPT_DIR="${CKPT_ROOT}/${BENCHMARK}_${PROTOCOL}${CKPT_SUFFIX}"
CHECKPOINT="${CKPT_DIR}/trained_model.pt"
OUT_DIR="${OUT_ROOT}/${BENCHMARK}_${PROTOCOL}${CKPT_SUFFIX}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: benchmark=${BENCHMARK} backbone=${BACKBONE}"
echo "  base_cfg   : ${BASE_CFG}"
echo "  model_cfg  : ${MODEL_CFG}"
echo "  checkpoint : ${CHECKPOINT}"
echo "  wds_dir    : ${WDS_DIR}"
echo "  output_dir : ${OUT_DIR}"

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load cuda/11.8

source "${REPO}/.venv/bin/activate"

mkdir -p /gpfs/data/oermannlab/users/zhouj14/slurm_logs
mkdir -p "${OUT_DIR}"

# ── Guard: skip if checkpoint missing ─────────────────────────────────────────
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found at ${CHECKPOINT} — skipping."
    exit 1
fi

# ── Slack notification helper ─────────────────────────────────────────────────
_slack() {
    local msg="$1"
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"${msg}\"}" \
            "${SLACK_WEBHOOK}" || true
    fi
}

# ── Run ───────────────────────────────────────────────────────────────────────
cd "${REPO}"

_slack "[:arrow_forward:] min_a0 started — array task ${SLURM_ARRAY_TASK_ID} (${BENCHMARK} / ${BACKBONE}) on \`$(hostname)\`"

if python MiN/unlearn.py \
    --base_configs  "${BASE_CFG}" \
    --model_configs "${MODEL_CFG}" \
    --checkpoint    "${CHECKPOINT}" \
    --data_root     "${DATA_ROOT}" \
    --wds_dir       "${WDS_DIR}" \
    --output_dir    "${OUT_DIR}" \
    --a0_only \
    --no_feature_probe; then
    _slack "[:white_check_mark:] min_a0 done — array task ${SLURM_ARRAY_TASK_ID} (${BENCHMARK} / ${BACKBONE})"
else
    _slack "[:x:] min_a0 FAILED — array task ${SLURM_ARRAY_TASK_ID} (${BENCHMARK} / ${BACKBONE}) exit code $?"
    exit 1
fi
