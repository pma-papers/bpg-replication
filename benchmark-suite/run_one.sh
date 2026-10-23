#!/bin/sh
# usage: run_one.sh KIND VALUE SEED  (VALUE = price m for fixed, alpha for adaptive, cost limit for cmdp)
# Amendment 2026-10-12 (launcher only; the runner and its hash are unchanged):
#  - the runner names a run's folder by kind, m, budget and seed, not by alpha, so the two adaptive
#    arms would overwrite each other; the alpha = 0.2 arm therefore writes under stage campaign-a0.2;
#  - a run whose result.json exists is skipped, so the campaign can be relaunched after an interruption;
#  - 2026-10-13: the runtime moved from ./.venv310 to ~/.venvs/constrained-rl-safety310 (same packages,
#    frozen in ../../environments/constrained-rl-safety310.txt).
#  - 2026-10-12 after the campaign: kind cmdp-s0.01 adds the CMDP arm with the dual step 0.01 (added afterwards,
#    reported as such), under stage campaign-cmdp-s0.01.
cd "$(dirname "$0")"
KIND=$1; V=$2; S=$3
STAGE=campaign
case $KIND in
  fixed)    ARGS="--kind fixed --m $V"; DIR=safety/$STAGE/fixed_m${V}_b25__seed$S;;
  adaptive) ARGS="--kind adaptive --alpha $V --dual-step 0.5"
            if [ "$V" != "0.05" ]; then STAGE=campaign-a$V; fi
            DIR=safety/$STAGE/adaptive_m0_b25__seed$S;;
  cmdp)     ARGS="--kind cmdp --cost-limit $V --dual-step 0.5"; DIR=safety/$STAGE/cmdp_m0_b25__seed$S;;
  cmdp-s0.01) ARGS="--kind cmdp --cost-limit $V --dual-step 0.01"; STAGE=campaign-cmdp-s0.01; DIR=safety/$STAGE/cmdp_m0_b25__seed$S;;
esac
if [ -f "$DIR/result.json" ]; then echo "$(date) skip $KIND $V $S: $DIR/result.json exists" >> campaign.log; exit 0; fi
PYTHONDONTWRITEBYTECODE=1 "$HOME/.venvs/constrained-rl-safety310/bin/python" safety_pilot_v63.py $ARGS --seed $S --steps 2000000 --eval 200 --stage $STAGE --n-envs 8 >> logs/campaign_${KIND}_${V}_${S}.log 2>&1
