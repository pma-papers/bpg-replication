#!/bin/sh
# Parameter-anchor campaign in the wear example (complete prior: mean network and dispersion):
# kappa_p in {1, 10, 100}, eight seeds, 8M steps, six at a time; then the mean-only arm at kappa_p = 100 (seeds 7-8).
cd "$(dirname "$0")"
for k in 100 10 1; do for s in 94001 94002 94003 94004 94005 94006 94007 94008; do echo "$k $s"; done; done |
  xargs -P 6 -L 1 sh -c '"$HOME/.venvs/constrained-rl-numerics/bin/python" wear_param_anchor_v39.py --kappa "$0" --seed "$1" --stage campaign'
for s in 94007 94008; do echo "$s"; done | xargs -P 2 -I{} "$HOME/.venvs/constrained-rl-numerics/bin/python" wear_param_anchor_v39.py --kappa 100 --seed {} --stage campaign-meanonly --mean-only
echo ANCHOR-DONE
