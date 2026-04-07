#!/usr/bin/env bash
# Run MiN on all benchmarks matching the protocols in qz-compcont-learning/docs/.
#
# Prerequisites:
#   1. uv sync && source .venv/bin/activate  (or: uv run python ...)
#   2. Manual datasets in ~/qz-compcont-learning/data/:
#        cub/train, cub/test, omnibenchmark/train, omnibenchmark/test
#
# GPU assignment: set GPU0 / GPU1 below, or override:
#   GPU0=1 GPU1=0 bash run_benchmarks.sh
#
# Protocols match qz-compcont-learning/docs/running-experiments.md (EASE column):
#   imagenet-r:    B0-Inc10 (20T)
#   cifar-100:     B0-Inc5  (20T)
#   imagenet-a:    B0-Inc20 (10T)
#   cub-200:       B0-Inc10 (20T)
#   omnibenchmark: B0-Inc30 (10T)
#   vtab:          B0-Inc10 (5T)
#   objectnet:     B0-Inc10 (20T)

set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
LOG_DIR="${LOG_DIR:-$HOME/min_benchmark_logs}"
CONFIGS="MiN/configs"

mkdir -p "$LOG_DIR"

cd "$(dirname "$0")"

# ── Ensure train/test splits exist for ImageNet-R and ImageNet-A ─────────────
DATA_ROOT="${DATA_ROOT:-$HOME/qz-compcont-learning/data}"
need_prepare=0
for split in imagenet-r-split imagenet-a-split; do
    if [[ ! -d "$DATA_ROOT/$split/train" || ! -d "$DATA_ROOT/$split/test" ]]; then
        need_prepare=1
        break
    fi
done
if [[ $need_prepare -eq 1 ]]; then
    echo "Train/test splits missing — running prepare_data.py..."
    python prepare_data.py --data-root "$DATA_ROOT"
else
    echo "Train/test splits present, skipping prepare_data.py."
fi

# data_ok <tag> <train_dir> <test_dir>
# Returns 0 if both dirs exist and each contains at least one class subdirectory.
data_ok() {
    local tag="$1" train_dir="$2" test_dir="$3"
    for dir in "$train_dir" "$test_dir"; do
        if [[ ! -d "$dir" ]]; then
            echo "[$tag] Skipped — directory not found: $dir"
            return 1
        fi
        # Expect at least one subdirectory (class folder) inside
        if [[ -z "$(find "$dir" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]]; then
            echo "[$tag] Skipped — no class subdirectories in: $dir"
            return 1
        fi
    done
    return 0
}

run() {
    local base="$1"
    local model="$2"
    local gpu="$3"
    local tag="$4"
    local log="$LOG_DIR/${tag}.log"

    echo "[$tag] Starting on GPU $gpu -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python MiN/main.py \
        --base_configs  "$CONFIGS/base_configs/${base}.json" \
        --model_configs "$CONFIGS/model_configs/${model}.json" \
        > "$log" 2>&1
    echo "[$tag] Done. Final result:"
    strings "$log" | grep -E 'Accuracy|accuracy|final|FINAL|task.*acc' | tail -5 || true
}

# All benchmarks match the EASE protocols in qz-compcont-learning.
# Benchmarks whose data is missing or malformed are skipped automatically.

# ── Batch 1 ──────────────────────────────────────────────────────────────────

# ImageNet-R B0-Inc10 (20T, EASE protocol)  -- GPU 0
if data_ok "imagenetr_ease_20T" "$DATA_ROOT/imagenet-r-split/train" "$DATA_ROOT/imagenet-r-split/test"; then
    run imagenetr_ease MiN-inr-10steps "$GPU0" "imagenetr_ease_20T" &
fi

# CIFAR-100 B0-Inc5 (20T, EASE protocol)    -- GPU 1  (auto-downloads, always run)
run cifar_ease MiN-cifar-10steps "$GPU1" "cifar100_ease_20T" &

wait

# ── Batch 2 ──────────────────────────────────────────────────────────────────

# ImageNet-A B0-Inc20 (10T, EASE protocol)  -- GPU 0
if data_ok "imageneta_ease_10T" "$DATA_ROOT/imagenet-a-split/train" "$DATA_ROOT/imagenet-a-split/test"; then
    run imageneta_ease MiN-imageneta "$GPU0" "imageneta_ease_10T" &
fi

# CUB-200 B0-Inc10 (20T, EASE protocol)     -- GPU 1
if data_ok "cub200_ease_20T" "$DATA_ROOT/cub/train" "$DATA_ROOT/cub/test"; then
    run cub_ease MiN-cub-10steps "$GPU1" "cub200_ease_20T" &
fi

wait

# ── Batch 3 ──────────────────────────────────────────────────────────────────

# OmniBenchmark B0-Inc30 (10T, EASE protocol) -- GPU 0
if data_ok "omnibenchmark_ease_10T" "$DATA_ROOT/omnibenchmark/train" "$DATA_ROOT/omnibenchmark/test"; then
    run omnibenchmark_ease MiN-omni-10steps "$GPU0" "omnibenchmark_ease_10T" &
fi

# VTAB B0-Inc10 (5T, EASE protocol)           -- GPU 1
if data_ok "vtab_ease_5T" "$DATA_ROOT/vtab/train" "$DATA_ROOT/vtab/test"; then
    run vtab_ease MiN-vtab-5steps "$GPU1" "vtab_ease_5T" &
fi

wait

# ── Batch 4 ──────────────────────────────────────────────────────────────────

# ObjectNet B0-Inc10 (20T, EASE protocol)     -- GPU 0
if data_ok "objectnet_ease_20T" "$DATA_ROOT/objectnet/train" "$DATA_ROOT/objectnet/test"; then
    run objectnet_ease MiN-inr-10steps "$GPU0" "objectnet_ease_20T"
fi

echo ""
echo "All benchmarks complete. Logs in $LOG_DIR"
echo ""
echo "Summary (final accuracy lines):"
for log in "$LOG_DIR"/*.log; do
    tag=$(basename "$log" .log)
    result=$(strings "$log" | grep -Ei 'accuracy|final' | tail -3 | tr '\n' '  ' || true)
    echo "  [$tag] $result"
done
