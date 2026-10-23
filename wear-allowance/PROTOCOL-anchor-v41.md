# Parameter-anchor campaign with a matched prior (revision 41)

Fixed on 2026-10-08 before any run of this campaign was inspected. Runner: `wear_param_anchor.py`
(copy of `numerics-v39/wear_param_anchor.py` with one new option, `--prior-logstd`); runtime
`../numerics-v22/.venv`; environment, PPO settings, evaluation and seeds as in the revision-39 campaign
(`numerics-v39/wear-anchor/campaign`, dispersion 0.37 at theta_0).

Design. Price m = 0.03 (the mispriced failure risk of the action-anchored configurations of the wear
campaign). theta_0 = least-squares fit of the actor's mean to the incumbent rule a = min{1, z/(3b)}, with
log dispersion log(0.1) = -2.3026, so that the prior policy (the actor at theta_0) is the incumbent rule
itself, the action reference of the action-anchored configurations. Anchor strengths kappa_p in {0, 1, 10, 100}
(kappa_p = 0 is the unanchored control from the same start), eight seeds 94001-94008 each, 8M steps,
complete anchor (mean network and log dispersion). The revision-39 campaign (dispersion 0.37) is kept
and reported as the mismatched-dispersion case.

Predictions (Proposition prop:prior, Theorem thm:baseline-performance, Section subsec:prior):
(P12') With the prior policy equal to the incumbent, the strongest anchor (kappa_p = 100) returns the
incumbent's breakdown probability and stopped utility within the evaluation's sampling error, and the
learned policy stays within a small parameter distance of theta_0.
(P12'') The unanchored control (kappa_p = 0) leaves the incumbent and moves toward the exact optimum of
the mispriced problem (breakdown probability of order 0.3 at m = 0.03); the anchored configurations lie
between the control and the incumbent, ordered by kappa_p in risk and in utility.
(P13') None of the configurations escapes through the dispersion (the complete anchor includes log sigma_a).
What falsifies: a kappa_p = 100 policy whose risk or utility differs from the incumbent's by more than a few
standard errors; a non-monotone ordering in kappa_p; a kappa_p = 0 run that does not leave the incumbent.
