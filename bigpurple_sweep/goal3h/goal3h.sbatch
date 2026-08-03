#!/bin/bash
#SBATCH --job-name=min_goal3h
#SBATCH --partition=superpod
#SBATCH --nodelist=sp-0014
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/data/oermannlab/users/steelr04/goal3h/slurm_%j.log
# Goal-3-H: task-0-cooling rescue of omnibenchmark-H and imagenet-r-H.
# Screen (24 x 3-stage probes) then confirm (full runs appended to
# confirm_queue.tsv after screen analysis). ~49 GPU-h total, ~6.5h wall.
# SUBMIT ONLY per the manager's allocation instruction (dependency/holder
# set at submit time via CLI); never another node.
set -uo pipefail
S=/gpfs/data/oermannlab/users/steelr04/goal3h
echo "════ min_goal3h on $(hostname) $(date)"
mkdir -p $S/logs
for gpu in 0 1 2 3 4 5 6 7; do
  bash $S/goal3h_worker.sh $gpu > $S/logs/worker_$(hostname)_gpu${gpu}.log 2>&1 &
done
wait
echo "════ min_goal3h drained on $(hostname) $(date)"
