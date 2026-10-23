#!/bin/sh
# Added adaptive arms (dual step 5), warm-started like the others.
cd "$(dirname "$0")"
SEEDS="72001 72002 72003 72004 72005"
for a in lunar-b22-adaptive-a0.05-s5 lunar-b22-adaptive-a0.02-s5; do
  for s in $SEEDS; do echo "$a $s"; done
done | xargs -P 4 -L 1 sh -c '"$HOME/.venvs/constrained-rl-numerics/bin/python" lunar.py --arm "$0" --seed "$1" --stage campaign'
echo ADAPTIVE-DONE
