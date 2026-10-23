#!/bin/zsh
# One zero-objective run (direct simulation, PROTOCOL-optimizers-v68.md): optimizer lr kappa_1 kappa_2 seed
cd "$(dirname "$0")"
mkdir -p optimizers-logs
PYTHONDONTWRITEBYTECODE=1 "$HOME/.venvs/constrained-rl-numerics/bin/python" regularizer_dynamics_v68.py --optimizer $1 --lr $2 --kappa1 $3 --kappa2 $4 --seed $5 --stage campaign >> optimizers-logs/zero__$1__lr$2__k1_$3__k2_$4__seed$5.log 2>&1
