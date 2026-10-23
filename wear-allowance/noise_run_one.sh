#!/bin/zsh
cd "$(dirname "$0")"
PYTHONDONTWRITEBYTECODE=1 "$HOME/.venvs/constrained-rl-numerics/bin/python" wear_param_noise_v60.py --kappa1 $1 --kappa2 $2 --seed $3 --stage campaign >> noise-logs/k1_$1__k2_$2__seed$3.log 2>&1
