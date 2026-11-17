#!/bin/bash
# Benchmark-suite study on Roihu: one run of safety_study_v76.py per array task, on a compute node.
# Usage: sbatch --array=1-N --time=HH:MM:SS run_array.sh RUNS.txt
#   RUNS.txt has one line per run: STAGE CONFIG SEED
# Two cores per run: the learner's updates take 12 ms on one core of these nodes and 7.7 ms on two
# (diagnostic jobs 3177210 and 3177260 of 17 November 2026). Memory: a run peaked at 3.3 GB in the test job 3175983.
#SBATCH --job-name=bpg-safety
# The account is taken from SBATCH_ACCOUNT (export SBATCH_ACCOUNT=<CSC project> before sbatch).
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5000M
#SBATCH --output=logs/slurm-%A_%a.out
set -euo pipefail
export PATH=/projappl/${CSC_PROJECT}/${USER}-bpg/safety310/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 STUDY_TORCH_THREADS=2
cd /scratch/${CSC_PROJECT}/${USER}-bpg/benchmark-suite
read -r STAGE CONFIG SEED < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "$(date) $(hostname) task ${SLURM_ARRAY_TASK_ID}: $STAGE $CONFIG $SEED"
srun python safety_study_v76.py run --stage "$STAGE" --config "$CONFIG" --seed "$SEED"
echo "$(date) done"
