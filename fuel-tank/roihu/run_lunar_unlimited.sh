#!/bin/bash
# Lander study on Roihu: the controller trained without a tank under the benchmark's reward (stage campaign, lunar-unlimited),
# from which the runs of a seed start. One seed per array task: seed = 72000 + task id.
# Usage: sbatch --array=6-10 run_lunar_unlimited.sh
#SBATCH --job-name=bpg-lander
# The account is taken from SBATCH_ACCOUNT (export SBATCH_ACCOUNT=<CSC project> before sbatch).
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%A_%a.out
set -euo pipefail
export PATH=/projappl/${CSC_PROJECT}/${USER}-bpg/numerics313/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /scratch/${CSC_PROJECT}/${USER}-bpg/fuel-tank
SEED=$(( 72000 + SLURM_ARRAY_TASK_ID ))
echo "$(date) $(hostname) seed $SEED"
srun python lunar.py --arm lunar-unlimited --seed "$SEED" --stage campaign
echo "$(date) done"
