#!/usr/bin/env bash
# Cache MiN embeddings locally (OLab3 / no SLURM).
#
# Runs one benchmark at a time, cycling through all three backbones.
# Two GPUs are used: GPU0 for one backbone, GPU1 for another, sequentially.
#
# Usage:
#   bash cache_embeddings_local.sh                   # all benchmarks, all 3 backbones
#   BENCHMARK=imagenetr bash cache_embeddings_local.sh   # single benchmark, all 3 backbones
#   BACKBONE=dinov2_vitb14 bash cache_embeddings_local.sh  # all benchmarks, one backbone
#
# Optional overrides (env vars):
#   GPU0          — CUDA device for primary GPU     (default: 0)
#   GPU1          — CUDA device for secondary GPU   (default: 1)
#   DATA_ROOT     — datasets root                   (default: ~/qz-compcont-learning/data)
#   EMBED_DIR     — output directory for .npz files (default: ~/min_embeddings)
#   BENCHMARK     — run only this dataset key       (cifar224|imagenetr|imageneta|cub|omnibenchmark|vtab)
#   BACKBONE      — run only this backbone           (pretrained_vit_b16_224_in21k_min|dinov2_vitb14|dinov3_vitb16)

set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DATA_ROOT="${DATA_ROOT:-$HOME/qz-compcont-learning/data}"
EMBED_DIR="${EMBED_DIR:-$HOME/min_embeddings}"
CONFIGS="MiN/configs"

cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
    echo "No .venv found. Run: uv sync && source .venv/bin/activate"
    exit 1
fi

mkdir -p "$EMBED_DIR"

# ── All backbones and their config suffixes ───────────────────────────────────
ALL_BACKBONES=(
    pretrained_vit_b16_224_in21k_min
    dinov2_vitb14
    dinov3_vitb16
)
declare -A BSUFFIX=(
    [pretrained_vit_b16_224_in21k_min]=""
    [dinov2_vitb14]="-dinov2vitb14"
    [dinov3_vitb16]="-dinov3vitb16"
)

# ── Benchmark table ───────────────────────────────────────────────────────────
# Key → (base_config  model_stem  train_subdir  test_subdir)
# Empty train/test → auto-download (CIFAR-100 only).
declare -A BENCH_BASE=(
    [cifar224]=cifar_ease
    [imagenetr]=imagenetr_default
    [imageneta]=imageneta_ease
    [cub]=cub_ease
    [omnibenchmark]=omnibenchmark_ease
    [vtab]=vtab_ease
)
declare -A BENCH_MODEL=(
    [cifar224]=MiN-cifar-10steps
    [imagenetr]=MiN-inr-10steps
    [imageneta]=MiN-imageneta
    [cub]=MiN-cub-10steps
    [omnibenchmark]=MiN-omni-10steps
    [vtab]=MiN-vtab-5steps
)
declare -A BENCH_TRAIN=(
    [cifar224]=""
    [imagenetr]="imagenet-r/train"
    [imageneta]="imagenet-a/train"
    [cub]="cub/train"
    [omnibenchmark]="omnibenchmark/train"
    [vtab]="vtab/train"
)
declare -A BENCH_TEST=(
    [cifar224]=""
    [imagenetr]="imagenet-r/test"
    [imageneta]="imagenet-a/test"
    [cub]="cub/test"
    [omnibenchmark]="omnibenchmark/test"
    [vtab]="vtab/test"
)

ALL_BENCHMARKS=(cifar224 imagenetr imageneta cub omnibenchmark vtab)

# Filter to a single benchmark/backbone if requested
SELECTED_BENCHMARKS=("${ALL_BENCHMARKS[@]}")
if [[ -n "${BENCHMARK:-}" ]]; then
    SELECTED_BENCHMARKS=("$BENCHMARK")
fi

SELECTED_BACKBONES=("${ALL_BACKBONES[@]}")
if [[ -n "${BACKBONE:-}" ]]; then
    SELECTED_BACKBONES=("$BACKBONE")
fi

# ── Helper: check data available ─────────────────────────────────────────────
data_ok() {
    local tag="$1" train_dir="$2" test_dir="$3"
    [[ -z "$train_dir" ]] && return 0   # auto-download datasets always OK
    for dir in "$DATA_ROOT/$train_dir" "$DATA_ROOT/$test_dir"; do
        if [[ ! -d "$dir" ]]; then
            echo "[$tag] Skipped — not found: $dir"
            return 1
        fi
        if [[ -z "$(find "$dir" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]]; then
            echo "[$tag] Skipped — no class subdirs in: $dir"
            return 1
        fi
    done
    return 0
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for backbone in "${SELECTED_BACKBONES[@]}"; do
    suffix="${BSUFFIX[$backbone]}"
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Backbone: $backbone"
    echo "════════════════════════════════════════════════════"

    for bench in "${SELECTED_BENCHMARKS[@]}"; do
        base_cfg="${BENCH_BASE[$bench]}"
        model_cfg="${BENCH_MODEL[$bench]}${suffix}"
        train_dir="${BENCH_TRAIN[$bench]}"
        test_dir="${BENCH_TEST[$bench]}"
        tag="${bench}_${backbone}"

        data_ok "$tag" "$train_dir" "$test_dir" || continue

        # Ensure ImageNet-R/A train-test splits exist
        if [[ "$bench" == "imagenetr" || "$bench" == "imageneta" ]]; then
            dset_base="${DATA_ROOT}/${train_dir%/train}"
            if [[ ! -d "$DATA_ROOT/$train_dir" || ! -d "$DATA_ROOT/$test_dir" ]]; then
                echo "[$tag] Building train/test splits..."
                .venv/bin/python prepare_data.py --data-root "$DATA_ROOT"
            fi
        fi

        echo ""
        echo "[$tag] Starting embedding caching → $EMBED_DIR/$bench/$backbone/"

        CUDA_VISIBLE_DEVICES="$GPU0" .venv/bin/python MiN/cache_embeddings.py \
            --base_configs  "$CONFIGS/base_configs/${base_cfg}.json" \
            --model_configs "$CONFIGS/model_configs/${model_cfg}.json" \
            --data_root     "$DATA_ROOT" \
            --output_dir    "$EMBED_DIR"

        echo "[$tag] Done."
        echo "  Files:"
        ls -lh "$EMBED_DIR/$bench/$backbone/"*.npz 2>/dev/null | awk '{print "   "$NF" "$5}' || true
    done
done

echo ""
echo "All embedding caching complete."
echo "Embeddings root: $EMBED_DIR"
