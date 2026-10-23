#!/bin/sh
# Lunar campaign: unconstrained landers first (5M steps), then the warm-started tank arms (3M steps), 4 at a time.
cd "$(dirname "$0")"
SEEDS="72001 72002 72003 72004 72005"
for s in $SEEDS; do echo "$s"; done | xargs -P 4 -I{} "$HOME/.venvs/constrained-rl-numerics/bin/python" lunar.py --arm lunar-unlimited --seed {} --stage campaign
for a in lunar-b22-m0 lunar-b22-m0.05 lunar-b22-m0.15 lunar-b22-m0.5 lunar-b22-m2 lunar-b22-adaptive-a0.05 lunar-b22-adaptive-a0.02 lunar-b21-m0 lunar-b21-m0.5; do
  for s in $SEEDS; do echo "$a $s"; done
done | xargs -P 4 -L 1 sh -c '"$HOME/.venvs/constrained-rl-numerics/bin/python" lunar.py --arm "$0" --seed "$1" --stage campaign'
echo CAMPAIGN-DONE
