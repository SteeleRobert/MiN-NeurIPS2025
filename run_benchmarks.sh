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

# ── Slack notifications ───────────────────────────────────────────────────────
# Source the shared helper (no-op if webhook is not configured).
# Configure via:  export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...'
# or:             echo 'https://...' > slurm/.slack_webhook_url
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm/slack_notify.sh
source "$_SCRIPT_DIR/slurm/slack_notify.sh"

# Failure-only on EXIT: any exit with non-zero (set -e, explicit exit 1, signal) notifies here.
# Success is *not* sent from EXIT — it is sent once at the bottom after all batches finish, so a
# stray exit 0 (duplicate shell, wrapper bug) cannot produce a false "finished OK" while work continues.
_slack_on_exit_failure() {
    local rc=$?
    [[ $rc -eq 0 ]] && return 0
    min_slack_notify_failure "$rc" "*run_benchmarks.sh* — *${BACKBONE}*"
}
trap '_slack_on_exit_failure' EXIT

_slack_notify_success() {
    local summary=""
    if [[ -d "${LOG_DIR:-}" ]]; then
        local log
        for log in "$LOG_DIR"/*${BTAG}*.log; do
            [[ -f "$log" ]] || continue
            local tag final_line
            tag=$(basename "$log" .log)
            final_line=$(strings "$log" 2>/dev/null | grep -i 'FINAL:' | tail -1 || true)
            [[ -n "$final_line" ]] && summary+="${tag}: ${final_line}\n"
        done
    fi
    local msg="*run_benchmarks.sh* — *${BACKBONE}* — finished OK"
    [[ -n "$summary" ]] && msg+=$'\n'"$(printf '%b' "$summary")"
    min_slack_notify "$msg"
}

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
LOG_DIR="${LOG_DIR:-$HOME/min_benchmark_logs}"
CONFIGS="MiN/configs"

# Backbone to use.  Supported values:
#   pretrained_vit_b16_224_in21k_min  (default — original MiN backbone)
#   dinov2_vitb14                     (DINOv2 ViT-B/14, auto-downloaded via torch.hub)
#   dinov3_vitb16                     (DINOv3 ViT-B/16, requires HF auth)
BACKBONE="${BACKBONE:-pretrained_vit_b16_224_in21k_min}"

# Map backbone name → model-config filename suffix
case "$BACKBONE" in
    pretrained_vit_b16_224_in21k_min) BSUFFIX="" ;;
    dinov2_vitb14) BSUFFIX="-dinov2vitb14" ;;
    dinov3_vitb16) BSUFFIX="-dinov3vitb16" ;;
    *) echo "Unknown BACKBONE='$BACKBONE'. Supported: pretrained_vit_b16_224_in21k_min | dinov2_vitb14 | dinov3_vitb16"; exit 1 ;;
esac

echo "Backbone: $BACKBONE (config suffix: '${BSUFFIX:-none}')"

# Short backbone tag appended to log filenames to avoid collisions across runs
case "$BACKBONE" in
    pretrained_vit_b16_224_in21k_min) BTAG="" ;;
    *) BTAG="_${BACKBONE}" ;;
esac

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
if data_ok "imagenetr_ease_20T${BTAG}" "$DATA_ROOT/imagenet-r-split/train" "$DATA_ROOT/imagenet-r-split/test"; then
    run imagenetr_ease "MiN-inr-10steps${BSUFFIX}" "$GPU0" "imagenetr_ease_20T${BTAG}" &
fi

# CIFAR-100 B0-Inc5 (20T, EASE protocol)    -- GPU 1  (auto-downloads, always run)
run cifar_ease "MiN-cifar-10steps${BSUFFIX}" "$GPU1" "cifar100_ease_20T${BTAG}" &

wait || true

# ── Batch 2 ──────────────────────────────────────────────────────────────────

# ImageNet-A B0-Inc20 (10T, EASE protocol)  -- GPU 0
if data_ok "imageneta_ease_10T${BTAG}" "$DATA_ROOT/imagenet-a-split/train" "$DATA_ROOT/imagenet-a-split/test"; then
    run imageneta_ease "MiN-imageneta${BSUFFIX}" "$GPU0" "imageneta_ease_10T${BTAG}" &
fi

# CUB-200 B0-Inc10 (20T, EASE protocol)     -- GPU 1
if data_ok "cub200_ease_20T${BTAG}" "$DATA_ROOT/cub/train" "$DATA_ROOT/cub/test"; then
    run cub_ease "MiN-cub-10steps${BSUFFIX}" "$GPU1" "cub200_ease_20T${BTAG}" &
fi

wait || true

# ── Batch 3 ──────────────────────────────────────────────────────────────────

# OmniBenchmark B0-Inc30 (10T, EASE protocol) -- GPU 0
if data_ok "omnibenchmark_ease_10T${BTAG}" "$DATA_ROOT/omnibenchmark/train" "$DATA_ROOT/omnibenchmark/test"; then
    run omnibenchmark_ease "MiN-omni-10steps${BSUFFIX}" "$GPU0" "omnibenchmark_ease_10T${BTAG}" &
fi

# VTAB B0-Inc10 (5T, EASE protocol)           -- GPU 1
if data_ok "vtab_ease_5T${BTAG}" "$DATA_ROOT/vtab/train" "$DATA_ROOT/vtab/test"; then
    run vtab_ease "MiN-vtab-5steps${BSUFFIX}" "$GPU1" "vtab_ease_5T${BTAG}" &
fi

wait || true

# ── Batch 4 ──────────────────────────────────────────────────────────────────

# ObjectNet B0-Inc10 (20T, EASE protocol)     -- GPU 0
if data_ok "objectnet_ease_20T${BTAG}" "$DATA_ROOT/objectnet/train" "$DATA_ROOT/objectnet/test"; then
    run objectnet_ease "MiN-inr-10steps${BSUFFIX}" "$GPU0" "objectnet_ease_20T${BTAG}"
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

_slack_notify_success
