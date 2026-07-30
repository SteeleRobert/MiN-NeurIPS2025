#!/bin/bash
#SBATCH --job-name=min_dinov3_paperhp
#SBATCH --partition=superpod
#SBATCH --nodelist=sp-0014
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --dependency=afterany:25969090
#SBATCH --output=/gpfs/data/oermannlab/users/steelr04/min_dinov3_paperhp/slurm_%j.log
# Goal-1 of the original-HP-stability study: paper's per-cell DINOv3 HP winners
# x 5 seeds (18 cells, 90 runs). Chained on sp-0014 behind min_inat_lh per the
# allocation rules -- this node chain is the only one we may use.
# Workers exit when the SHARED queue drains; more work can be appended to
# queue.tsv later without touching this allocation.
set -uo pipefail
S=/gpfs/data/oermannlab/users/steelr04/min_dinov3_paperhp
echo "════ min_dinov3_paperhp on $(hostname) $(date)"
mkdir -p $S/logs
for gpu in 0 1 2 3 4 5 6 7; do
  bash $S/stab_worker_mp.sh $gpu > $S/logs/worker_$(hostname)_gpu${gpu}.log 2>&1 &
done
wait
echo "════ min_dinov3_paperhp drained on $(hostname) $(date)"
