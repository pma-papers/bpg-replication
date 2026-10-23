#!/bin/zsh
# One run of the decoupled campaign (PROTOCOL-decoupled-v68.md): kappa_1 kappa_2 seed.
cd "$(dirname "$0")"
mkdir -p noise-decoupled-logs
PYTHONDONTWRITEBYTECODE=1 "$HOME/.venvs/constrained-rl-numerics/bin/python" wear_param_noise_decoupled_v68.py --kappa1 $1 --kappa2 $2 --seed $3 --stage campaign >> noise-decoupled-logs/k1_$1__k2_$2__seed$3.log 2>&1
