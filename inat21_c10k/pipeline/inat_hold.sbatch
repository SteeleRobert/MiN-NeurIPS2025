#!/bin/bash
#SBATCH --job-name=min_inat
#SBATCH --partition=superpod
#SBATCH --nodelist=sp-0014
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=20:00:00
#SBATCH --output=/gpfs/data/oermannlab/users/steelr04/min_inat/slurm_%j.log
# Holds sp-0004 for the Places365 domain-shift runs while the dataset pipeline is
# wired up. Workers poll queue.tsv and start runs as soon as it is populated, so
# the allocation converts straight into useful work with no resubmission.
set -uo pipefail
S=/gpfs/data/oermannlab/users/steelr04/min_inat
mkdir -p $S/logs $S/claims
echo "════ min_inat holding node from $(date)"
for gpu in 0 1 2 3 4 5 6 7; do
  bash $S/inat_worker.sh $gpu > $S/logs/worker_gpu${gpu}.log 2>&1 &
done
wait
echo "════ min_inat drained $(date)"
