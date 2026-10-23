#!/bin/sh
# Matched-prior parameter-anchor campaign (revision 41): prior dispersion 0.1 (= incumbent), kappa_p in {100, 10, 1, 0}, eight seeds, 8M steps, six at a time.
cd "$(dirname "$0")"
for k in 100 10 1 0; do for s in 94001 94002 94003 94004 94005 94006 94007 94008; do echo "$k $s"; done; done |
  xargs -P 6 -L 1 sh -c '"$HOME/.venvs/constrained-rl-numerics/bin/python" wear_param_anchor.py --kappa "$0" --seed "$1" --stage campaign-matched --prior-logstd -2.302585092994046'
echo ANCHOR-V41-DONE
