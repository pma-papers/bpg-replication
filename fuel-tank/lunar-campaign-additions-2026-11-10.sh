#!/bin/sh
# Additions of 2026-11-10 (see stages.campaign.additions_2026_11_10 in lunar-protocol.json): 50 runs.
cd "$(dirname "$0")"
SEEDS="72001 72002 72003 72004 72005"
for a in lunar-b21-m0.05 lunar-b21-m0.15 lunar-b21-m2 lunar-b20-m0.05 lunar-b20-m0.15 lunar-b20-m2 lunar-b23-m0 lunar-b23-m0.5 lunar-b24-m0 lunar-b24-m0.5; do
  for s in $SEEDS; do echo "$a $s"; done
done | xargs -P 5 -L 1 sh -c '"$HOME/.venvs/constrained-rl-numerics/bin/python" lunar.py --arm "$0" --seed "$1" --stage campaign'
echo ADDITIONS-DONE
