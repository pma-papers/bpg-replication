#!/bin/bash
# Lander study on Roihu: PER configurations and seeds per array task (default 2), one after the other, on a compute node.
# For each, the phase with the constant learning rate (stage formulations-const) and then the final phase with the
# rate decayed to zero (stage formulations-anneal). A run whose result exists is skipped by lunar.py.
# Usage: sbatch --array=1-N run_lunar_array.sh RUNS.txt [PER [CONST ANNEAL]]     (RUNS.txt: one line per run, ARM SEED; N = lines / PER;
#        CONST and ANNEAL name the two stages, default formulations-const and formulations-anneal; the main runs use ledger-const and ledger-anneal)
# The account may have 200 jobs submitted and 100 running on the partition small, hence several runs per task.
#SBATCH --job-name=bpg-lander
# The account is taken from SBATCH_ACCOUNT (export SBATCH_ACCOUNT=<CSC project> before sbatch).
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%A_%a.out
set -euo pipefail
export PATH=/projappl/${CSC_PROJECT}/${USER}-bpg/numerics313/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /scratch/${CSC_PROJECT}/${USER}-bpg/fuel-tank
PER=${2:-2}
CONST=${3:-formulations-const}
ANNEAL=${4:-formulations-anneal}
for k in $(seq 1 "$PER"); do
    line=$(( (SLURM_ARRAY_TASK_ID - 1) * PER + k ))
    read -r ARM SEED < <(sed -n "${line}p" "$1") || continue
    [ -n "${ARM:-}" ] || continue
    echo "$(date) $(hostname) task ${SLURM_ARRAY_TASK_ID} line ${line}: $ARM $SEED"
    srun python lunar.py --arm "$ARM" --seed "$SEED" --stage "$CONST"
    srun python lunar.py --arm "$ARM" --seed "$SEED" --stage "$ANNEAL"
done
echo "$(date) done"
