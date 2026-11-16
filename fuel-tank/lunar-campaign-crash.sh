#!/bin/sh
# Stage crash-campaign of 2026-11-14 (see stages.crash-campaign in lunar-protocol.json): the 23 configurations of the
# study with the loss of the lander charged like an empty tank and the tank limited until touchdown, five seeds each, 115 runs.
cd "$(dirname "$0")"
SEEDS="72001 72002 72003 72004 72005"
for a in lunar-b24-m0 lunar-b24-m0.5 lunar-b23-m0 lunar-b23-m0.5 lunar-b22-m0 lunar-b22-m0.05 lunar-b22-m0.15 lunar-b22-m0.5 lunar-b22-m2 lunar-b21-m0 lunar-b21-m0.05 lunar-b21-m0.15 lunar-b21-m0.5 lunar-b21-m2 lunar-b20-m0 lunar-b20-m0.05 lunar-b20-m0.15 lunar-b20-m0.5 lunar-b20-m2 lunar-b22-adaptive-a0.05 lunar-b22-adaptive-a0.02 lunar-b22-adaptive-a0.05-s5 lunar-b22-adaptive-a0.02-s5; do
  for s in $SEEDS; do echo "$a $s"; done
done | xargs -P 5 -L 1 sh -c '"$HOME/.venvs/constrained-rl-numerics/bin/python" lunar.py --arm "$0" --seed "$1" --stage crash-campaign 2>&1 | grep -v -i warn'
echo CRASH-CAMPAIGN-DONE
