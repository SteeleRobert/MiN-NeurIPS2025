#!/bin/bash
# Extract per-task noise-learning loss/acc trajectories from MiN run logs.
# The training loop logs lines like:
#   ... Task 3 --> Learning Beneficial Noise!: Epoch 7/10 => Loss 0.123, train_accy 97.66
# Usage: parse_noise_loss.sh <log-file-or-dir> > out.tsv
set -uo pipefail
emit() {
  # file task epoch epochs loss acc
  grep -a "Learning Beneficial Noise" "$1" | \
    sed -E 's/.*Task ([0-9]+) --> Learning Beneficial Noise!: Epoch ([0-9]+)\/([0-9]+) => Loss ([0-9.eE+-]+), train_accy ([0-9.]+).*/\1\t\2\t\3\t\4\t\5/' | \
    awk -v f="$(basename "$1")" '{print f"\t"$0}'
}
if [ -d "$1" ]; then
  for f in "$1"/*.log; do emit "$f"; done
else
  emit "$1"
fi
