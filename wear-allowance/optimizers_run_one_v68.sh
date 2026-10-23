#!/bin/zsh
# One run of the three-optimizer campaign (PROTOCOL-optimizers-v68.md): optimizer lr kappa_1 kappa_2 seed [stage] [--zero-grad]
cd "$(dirname "$0")"
mkdir -p optimizers-logs
stage=${6:-campaign}
PYTHONDONTWRITEBYTECODE=1 "$HOME/.venvs/constrained-rl-numerics/bin/python" wear_param_optimizers_v68.py --optimizer $1 --lr $2 --kappa1 $3 --kappa2 $4 --seed $5 --stage $stage $7 >> optimizers-logs/$1__lr$2__k1_$3__k2_$4__seed$5__$stage$7.log 2>&1
