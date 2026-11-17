#!/bin/bash
# One short run of the lander study on a compute node of Roihu, to measure speed and memory before the pilot.
#SBATCH --job-name=bpg-lander-test
# The account is taken from SBATCH_ACCOUNT (export SBATCH_ACCOUNT=<CSC project> before sbatch).
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:14:00
#SBATCH --output=logs/slurm-test-%j.out
set -euo pipefail
export PATH=/projappl/${CSC_PROJECT}/${USER}-bpg/numerics313/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /scratch/${CSC_PROJECT}/${USER}-bpg/fuel-tank
echo "$(date) $(hostname)"
python -c "import torch, gymnasium, stable_baselines3, Box2D; print(torch.__version__, gymnasium.__version__, stable_baselines3.__version__, torch.get_num_threads())"
srun python lunar.py --arm ledger-R36-b24-k2 --seed 72001 --stage formulations-test
echo "$(date) done"
