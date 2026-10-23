# The parameter pull decoupled from Adam: report (revision 68, first protocol)

Date 2026-10-20. Protocol `PROTOCOL-decoupled-v68.md` (SHA-256 in `PROTOCOL-decoupled-v68.sha256`, fixed at 18:01 EEST);
runs 18:01-18:26 EEST, 32 wear-allowance runs seven in parallel (`decoupled-campaign-v68.txt`, logs in
`noise-decoupled-logs/`; the stdout lines of the first batch of seven did not reach their logs, every record
`wear-noise-decoupled-v68/campaign/*/result.json` is complete) and one application run. Runtime
`~/.venvs/constrained-rl-numerics/bin/python` (Python 3.13.3, Stable-Baselines3 2.7.1, torch 2.8.0) for the wear runs and
`~/.venvs/bpg-toolbox/bin/python` (SB3 2.9.0, torch 2.14.1, barrierpg 0.6.1 from the sibling checkout) for the
application; analysis `analyze_decoupled_v68.py` with `~/.venvs/constrained-rl/bin/python`. The machine was shared
with other work (load 5-13 on 14 cores). This campaign was extended the same afternoon into the three-optimizer
comparison of `PROTOCOL-optimizers-v68.md`, whose report `REPORT-optimizers-v68.md` places these runs as the
decoupled-Adam arm; this report covers the first protocol's checks only.

## What was run

The roles-of-the-regularizers setting of revision 60 with one change: the pull kappa_2 (theta - theta_0) is applied to
the actor parameters after every Adam step, theta <- theta - lr_t kappa_2 (theta - theta_0), with the noise of
variance 2 kappa_1 lr_t added after it, instead of being added to the gradient before the step. Rows: kappa_2 = 10
and 1 with kappa_1 = 0.101, and pull alone at both strengths; eight seeds each. Application: the anchored
configuration (F = 50, kappa_p = 10, seed 1) with the same change and no noise.

## Results (means over eight seeds, between-seed SD in parentheses)

| pull | kappa_1 | kappa_2 | predicted spread sqrt(kappa_1/kappa_2) | spread | q_beta (%) | U | A | sigma_a |
|---|---|---|---|---|---|---|---|---|
| coupled (recorded) | 0 | 10 | -- | 3.2e-4 (2e-5) | 3.35 (0.34) | 2.89 (0.02) | 2.0 | 0.10 |
| coupled (recorded) | 0.101 | 10 | 0.1005 | 0.1277 (0.0015) | 13.8 (11.5) | 3.08 (0.73) | 69 | 0.10 |
| decoupled | 0 | 10 | 0 | 0.0142 (0.0008) | 27.6 (1.2) | 4.16 (0.04) | 175 | 0.10 |
| decoupled | 0.101 | 10 | 0.1005 | 0.0999 (0.0010) | 12.7 (10.4) | 3.22 (0.67) | 60 | 0.10 |
| decoupled | 0 | 1 | 0 | 0.0270 (0.0016) | 53.1 (1.2) | 5.12 (0.07) | 755 | 0.085 |
| decoupled | 0.101 | 1 | 0.318 | 0.3182 (0.0031) | 47.1 (19.6) | 5.03 (0.49) | 3426 | 0.17 |
| unanchored control (recorded) | 0 | 0 | -- | 0.0442 (0.0036) | 35.0 (3.6) | 4.37 (0.10) | 276 | 0.02 |

Per seed: `wear-noise-decoupled-v68.json`; table `wear-noise-decoupled-v68.tex`; figure `wear-noise-decoupled-v68.pdf`.
Every run took 78,160 optimizer steps; the flow time (sum of learning rates) is 11.7.

Checks of the protocol:

* (D1) holds. At kappa_2 = 10 the spread is 0.0999; with the pull-alone displacement 0.0142 removed in quadrature,
  0.0989, a ratio of 0.98 to the predicted 0.1005.
* (D2) holds. The ratio of the spreads at kappa_2 = 1 and 10 is 3.19 (particle form sqrt(10) = 3.16; coupled
  Laplace law 1). At kappa_2 = 1 the spread is 0.3182 against the predicted 0.318 (ratio 1.00).
* (D3) holds. Pull alone decoupled, the objective's own displacement is 0.0142 at kappa_2 = 10 and 0.0270 at
  kappa_2 = 1, below the envelopes 0.1 and 1, above the coupled pull-alone spread 3.2e-4, larger at the weaker pull.
* (O16) does not survive the decoupled form: the spreads 0.0999 and 0.318 are not within 0.9-1.7 kappa_1 (0.091-0.172)
  at both pulls and their ratio is 3.19. The spread proportional to kappa_1 whatever the pull is a property of the
  coupling through Adam.
* (D4), reported. At kappa_2 = 10, kappa_1 = 0.101 the seeds still split (q_beta 0.9-33%, U 1.9-4.0), as the
  coupled runs did (0.0-29%, 2.2-4.1), with the same seed-mean risk within noise (12.7% against 13.8%). The
  splitting is therefore not the optimizer outcome the paper attributes it to (the pull's gradient dominating Adam's
  normalizer): decoupled, the policy gradient takes its full Adam step, and the seeds split the same way. What the
  two forms share at this row is the spread, 0.10-0.13, comparable to the incumbent's dispersion.
* The pull-alone rows are the new finding. Decoupled, the pull that keeps the incumbent in the coupled form
  (kappa_2 = 10: q_beta 3.35%, U 2.89, A 2) no longer does: q_beta 27.6%, U 4.16, A 175, most of the way to the
  unanchored control (35.0%, 4.37, 276), at a parameter spread of only 0.014. The coupled pull holds the policy because
  Adam normalizes the pull's gradient kappa_2 x coordinate by coordinate: where the objective's gradient is small the
  pull's normalized step is about lr whatever the distance, and at the recorded distance 3.2e-4 the pull's gradient
  (3e-3 per coordinate) is of the order of the policy gradient. Decoupled, the pull's step lr kappa_2 x is 3e-4 x
  and balances Adam's objective step (about lr where the gradient is persistent) only at x of order 1/kappa_2 =
  0.1, so a displacement of 0.014 per coordinate, 60 times the coupled one, is where the policy settles, and that
  displacement is enough to move the policy. The mapping from strength to distance to behaviour is therefore
  optimizer-dependent, as the paper's Section 6.5 now says, and the quantitative form of it is this: the same
  kappa_2 = 10 is a hold under the coupled form and a weak pull under the decoupled one.

## The application (seed 1, F = 50, kappa_p = 10)

| run | anchor distance | spread | miss p_51 | cost | U | V | delta (gap) | validated upper | mean-action p_51 | sigma_a |
|---|---|---|---|---|---|---|---|---|---|---|
| prior policy (incumbent rule + noise) | 0 | 0 | 0.0128 | 5.298 | -5.253 | -5.893 | | | | 0.135 |
| coupled (recorded `runs/anchored_F50_k10_s1`) | 0.0070 | 1.0e-4 | 0.0130 | 5.287 | -5.241 | -5.891 | 1.558 (35.9%) | 0.0157, pass | 0.0132 | 0.135 |
| decoupled (`runs-decoupled-v68/anchored-decoupled_F50_k10_s1`) | 0.4185 | 6.1e-3 | 0.0170 | 5.063 | -5.004 | -5.854 | 1.520 (35.1%) | 0.0200, pass | 0.0171 | 0.134 |
| residual control, kappa_p = 0 (recorded `runs/anchored_F50_k0_s1`) | 2.592 | 3.8e-2 | 0.0181 | 4.871 | -4.807 | -5.709 | 1.376 (31.7%) | 0.0211, pass | 0.0180 | 0.096 |

(D5) holds: the decoupled anchor distance 0.4185 lies between the coupled 0.0070 and the control's 2.59, below the
envelope sqrt(4738)/10 = 6.9; the miss probability 0.0170 is within 0.005 of the prior's 0.0128 and the cost 5.063
within 0.4 of 5.30, the (A5) band behind (O27). The decoupled run sits between the coupled run and the control in
every column (distance, miss, cost, value), closer to the control in parameter distance (a sixth of it) and closer to
the prior in behaviour. 20,000 optimizer steps at the constant rate 3e-4; contraction 0.997 per step, half-life 231
steps. Training took 160 s against the record's 275 s (different machine load).

## What this changes in the paper's statements

Numbers for the paper session (Section 6.5 and Appendix L of revision 68):

1. The sentence "That configuration was not run" at the end of Appendix L: it has now been run; the decoupled
   spread is 0.0999 at kappa_2 = 10 and 0.318 at kappa_2 = 1 against sqrt(kappa_1/kappa_2) = 0.1005 and 0.318
   (the particle form's spread when the noise carries no pull of its own; the paper's nominal
   sqrt(kappa_1/(kappa_1+kappa_2)) differs from it by 0.5% and 5% here).
2. "(O16) is a property of the coupled implementation, not of the noise": supported directly; the decoupled spread
   scales as kappa_2^(-1/2) (ratio 3.19 between the two pulls) where the coupled one is independent of kappa_2.
3. "the splitting of the seeds in (O17) is an outcome of the optimizer": not supported; the seeds split the same way
   under the decoupled pull at the same spread. The splitting goes with the spread, 0.10-0.13 against the
   incumbent's dispersion 0.1, not with the share of the policy gradient in Adam's step.
4. "The dependence of the spread on kappa_2 in (O19) was not tested": now tested under the decoupled form; it holds
   as the particle form predicts.
5. New: under the decoupled form kappa_2 = 10 does not keep the incumbent (q_beta 27.6% against 3.35% coupled, A 175
   against 2), so the anchor study's "strength that keeps the incumbent" is a statement about the coupled form. The
   application's anchored seed stays within the (A5) band under both forms, at a sixty times larger distance
   decoupled.
