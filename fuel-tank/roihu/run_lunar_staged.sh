#!/bin/bash
# Lander study on Roihu, the runs that start from the controller trained without a price: one configuration and seed per array task.
# The run continues the controller of the same seed trained on the same problem without a price (stage ledger-const, worth 0),
# first at a constant learning rate (stage ledger-staged-const), then with the rate decayed to zero (stage ledger-staged-anneal).
# The run without a price is continued in the same way. A run whose result exists is skipped by lunar.py.
# Usage: sbatch --array=1-N run_lunar_staged.sh RUNS.txt      (RUNS.txt: one line per run, ARM SEED; N = number of lines)
#SBATCH --job-name=bpg-staged
# The account is taken from SBATCH_ACCOUNT (export SBATCH_ACCOUNT=<CSC project> before sbatch).
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm-%A_%a.out
set -euo pipefail
export PATH=/projappl/${CSC_PROJECT}/${USER}-bpg/numerics313/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /scratch/${CSC_PROJECT}/${USER}-bpg/fuel-tank
read -r ARM SEED < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
START=lunar/ledger-const/ledger-R36-b24-k0__seed${SEED}/final_model.zip
[ -f "$START" ] || { echo "starting model missing: $START"; exit 1; }
echo "$(date) $(hostname) task ${SLURM_ARRAY_TASK_ID}: $ARM $SEED from $START"
srun python lunar.py --arm "$ARM" --seed "$SEED" --stage ledger-staged-const --init "$START"
srun python lunar.py --arm "$ARM" --seed "$SEED" --stage ledger-staged-anneal
echo "$(date) done"
