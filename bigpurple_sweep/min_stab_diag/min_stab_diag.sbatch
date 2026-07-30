#!/bin/bash
#SBATCH --job-name=min_stab_diag
#SBATCH --partition=superpod
#SBATCH --nodelist=sp-0014
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --dependency=afterany:25969625
#SBATCH --output=/gpfs/data/oermannlab/users/steelr04/min_stab_diag/slurm_%j.log
# Goal-2 instrumented probe runs (36 x 3-stage cifar/omni probes, ~10 GPU-h).
# Chained behind the Goal-1 paperhp sweep on sp-0014 per allocation rules.
set -uo pipefail
S=/gpfs/data/oermannlab/users/steelr04/min_stab_diag
echo "════ min_stab_diag on $(hostname) $(date)"
mkdir -p $S/logs
for gpu in 0 1 2 3 4 5 6 7; do
  bash $S/diag_worker.sh $gpu > $S/logs/worker_$(hostname)_gpu${gpu}.log 2>&1 &
done
wait
echo "════ min_stab_diag drained on $(hostname) $(date)"
