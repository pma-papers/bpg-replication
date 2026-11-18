#!/bin/bash
# Lander study on Roihu: the evaluation episodes of stored policies again, on the machine that produced them, so that
# the new records belong to the same episodes as the stored results (a local evaluation gives other episodes).
# Tasks 1 to 8: the priced risk (failing frames) of the stages named in STAGES, shard i of 8 (lunar_reeval.py).
# Task 9: the flight paths of the stage PATHSTAGE for all its configurations (lunar_problem_eval_v76.py).
# Usage: sbatch --array=1-9 run_lunar_reeval.sh PATHSTAGE STAGE [STAGE ...]
#SBATCH --job-name=bpg-reeval
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
PATHSTAGE=$1; shift
if [ "$SLURM_ARRAY_TASK_ID" -le 8 ]; then
    for STAGE in "$@"; do
        srun python lunar_reeval.py --stage "$STAGE" --shard $((SLURM_ARRAY_TASK_ID - 1))/8
    done
else
    ARMS=$(python -c "import json; print(' '.join(json.load(open('lunar-protocol.json'))['stages']['$PATHSTAGE']['arms']))")
    srun python lunar_problem_eval_v76.py --stage "$PATHSTAGE" $ARMS
fi
echo "$(date) done"
