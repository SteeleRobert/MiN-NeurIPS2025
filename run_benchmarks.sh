#!/usr/bin/env bash
# Run MiN on all benchmarks matching the protocols in qz-compcont-learning/docs/.
#
# Prerequisites:
#   1. uv sync && source .venv/bin/activate  (or: uv run python ...)
#   2. python prepare_data.py  (creates train/test splits for imagenet-r and imagenet-a)
#   3. Manual datasets in ~/qz-compcont-learning/data/:
#        cub/train, cub/test, omnibenchmark/train, omnibenchmark/test
#
# GPU assignment: set GPU0 / GPU1 below, or override:
#   GPU0=1 GPU1=0 bash run_benchmarks.sh
#
# Protocols match qz-compcont-learning/docs/running-experiments.md:
#   imagenet-r: B0-Inc10 (20T, EASE) and B0-Inc5 (40T, qz default)
#   cifar-100:  B0-Inc5  (20T, EASE)
#   imagenet-a: B0-Inc20 (10T, EASE)
#   cub-200:    B0-Inc10 (20T, EASE)
#   omnibenchmark: B0-Inc30 (10T, EASE)

set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
LOG_DIR="${LOG_DIR:-$HOME/min_benchmark_logs}"
CONFIGS="MiN/configs"

mkdir -p "$LOG_DIR"

cd "$(dirname "$0")"

run() {
    local base="$1"
    local model="$2"
    local gpu="$3"
    local tag="$4"
    local log="$LOG_DIR/${tag}.log"

    echo "[$tag] Starting on GPU $gpu -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" python MiN/main.py \
        --base_configs  "$CONFIGS/base_configs/${base}.json" \
        --model_configs "$CONFIGS/model_configs/${model}.json" \
        > "$log" 2>&1
    echo "[$tag] Done. Final result:"
    strings "$log" | grep -E 'Accuracy|accuracy|final|FINAL|task.*acc' | tail -5 || true
}

# ── Batch 1: alternating GPU0/GPU1 ──────────────────────────────────────────

# ImageNet-R B0-Inc10 (20T, EASE protocol)  -- GPU 0
run imagenetr_ease MiN-inr-10steps "$GPU0" "imagenetr_ease_20T" &

# CIFAR-100 B0-Inc5 (20T, EASE protocol)   -- GPU 1
run cifar_ease MiN-cifar-10steps "$GPU1" "cifar100_ease_20T" &

wait

# ── Batch 2 ─────────────────────────────────────────────────────────────────

# ImageNet-R B0-Inc5 (40T, qz default protocol)  -- GPU 0
run imagenetr_default MiN-inr-10steps "$GPU0" "imagenetr_default_40T" &

# ImageNet-A B0-Inc20 (10T, EASE protocol)       -- GPU 1
run imageneta_ease MiN-imageneta "$GPU1" "imageneta_ease_10T" &

wait

# ── Batch 3 ─────────────────────────────────────────────────────────────────

# CUB-200 B0-Inc10 (20T, EASE protocol)      -- GPU 0
run cub_ease MiN-cub-10steps "$GPU0" "cub200_ease_20T" &

# OmniBenchmark B0-Inc30 (10T, EASE protocol) -- GPU 1
run omnibenchmark_ease MiN-omni-10steps "$GPU1" "omnibenchmark_ease_10T" &

wait

echo ""
echo "All benchmarks complete. Logs in $LOG_DIR"
echo ""
echo "Summary (final accuracy lines):"
for log in "$LOG_DIR"/*.log; do
    tag=$(basename "$log" .log)
    result=$(strings "$log" | grep -Ei 'accuracy|final' | tail -3 | tr '\n' '  ' || true)
    echo "  [$tag] $result"
done
