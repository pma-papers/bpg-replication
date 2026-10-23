#!/bin/sh
cd "$(dirname "$0")"
SEEDS="72001 72002 72003 72004 72005"
for a in lunar-b20-m0 lunar-b20-m0.5; do
  for s in $SEEDS; do echo "$a $s"; done
done | xargs -P 4 -L 1 sh -c '"$HOME/.venvs/constrained-rl-numerics/bin/python" lunar.py --arm "$0" --seed "$1" --stage campaign'
echo B20-DONE
