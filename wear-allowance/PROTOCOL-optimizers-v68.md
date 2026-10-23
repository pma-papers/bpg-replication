# Protocol: the regularizers under three optimizers (roles of the regularizers on PPO; the anchored application), revision 68

Fixed on 2026-10-20 at 18:27 EEST before the campaign. This protocol extends `PROTOCOL-decoupled-v68.md` (fixed at 18:01 the same
day, its 32 runs launched at 18:01 and finished under it) at the author's request of the same afternoon: the aim is to
separate the effect of the regularizer from the effect of the optimizer, so the reruns become a comparison across
optimizers. The 32 runs of the first protocol are the decoupled-Adam arm of rows C2-C5 below and are reused as they
stand. Pilot runs on separate seeds chose the SGD learning rates (Section "Pilots"); their numbers are not used
otherwise. Nothing else was run or inspected before this protocol was fixed.

## Setting

Part A, the roles of the regularizers on PPO: the setting of `PROTOCOL-noise-v60.md` unchanged (wear-allowance
example, mispriced m = 0.03, PPO as in the matched-prior campaign of revision 41, actor started at theta_0 fitted to
the incumbent rule with dispersion 0.1, 8 million environment steps in 32 environments, 256 steps per environment per
rollout, minibatches of 1024, ten epochs, 977 rollouts of 80 optimizer steps, learning rate decayed linearly to zero,
gradient norm clipped at 0.5, 20,000 evaluation episodes, seeds 94001-94008). The pull acts on every actor parameter
(mean network and log dispersion, d = 4418); the noise of variance 2 kappa_1 lr_t is added to every actor parameter
after each optimizer step from a generator seeded with 1,000,000 + seed. One optimizer over actor, critic and
dispersion, as in the learner.

Three arms, differing only in the optimizer and in how the pull enters:

| arm | optimizer | the pull kappa_2 (theta - theta_0) | what it realizes |
|---|---|---|---|
| sgd | plain SGD, no momentum, learning rate lr_sgd (pilot) decayed linearly to zero | added to the clipped actor gradient, so the step on it is exactly lr_t kappa_2 (theta - theta_0) | the Euler step of the paper's particle form; the reference for the regularizer's own effect |
| adam-coupled | Adam, eps 1e-5, learning rate 3e-4 decayed linearly to zero | added to the clipped actor gradient before the Adam step | the paper's recorded runs |
| adam-decoupled | the same Adam | applied to the parameters after the step, theta <- theta - lr_t kappa_2 (theta - theta_0) | the decoupled weight decay of AdamW; the paper's Section 7.1 |

Under SGD the coupled and decoupled forms coincide, so SGD has one arm. The time scales differ: the flow time of a run
is the sum of the learning rates over its 78,160 optimizer steps, 11.7 under Adam and 78,160 lr_sgd / 2 under SGD, and
the relaxation time of the pull is 1 / (lr kappa_2) steps.

Configurations, eight seeds each:

| row | kappa_2 | kappa_1 | purpose |
|---|---|---|---|
| C1 | 10 | 0.009 | the paper's second noise row |
| C2 | 10 | 0.101 | the paper's third noise row |
| C3 | 1 | 0.101 | the kappa_2 dependence |
| C4 | 10 | 0 | anchor only, the strength-to-distance mapping of the anchor study |
| C5 | 1 | 0 | anchor only at the weaker strength |
| C6 | 0 | 0 | no regularizer, the reference of the regularizer-bias comparison |

Learning runs: 3 arms x 6 rows x 8 seeds = 144, of which 72 exist and are reused without rerun: adam-coupled C1, C2 and
C4 from `wear-noise-v60.json` and the matched campaign of revision 41 (`wear-anchor/campaign-matched`, kappa_p = 10),
adam-coupled C6 and adam-decoupled C6 (the unanchored control of that campaign, kappa_p = 0, where no form is
involved), adam-decoupled C2-C5 from `PROTOCOL-decoupled-v68.md`. New: sgd C1-C6 (48), adam-coupled C3 and C5 (16),
adam-decoupled C1 (8): 72 runs, `wear_param_optimizers_v68.py`, records in `wear-optimizers-v68/campaign/`. The reused
coupled records drew their noise from the global generator; the new runs draw from the dedicated one. Nothing else
differs between the reused and the new records of one arm (the arithmetic of the hooks is the same; the decoupled arm
was checked to reproduce `wear_param_noise_decoupled_v68.py` to the digit on the smoke stage).

Zero-objective control: the same three arms and rows C1-C5 with the objective's gradient replaced by zero before the
step, so that pull and noise alone drive the parameters through each optimizer. Eight seeds each, 120 runs, run as a
direct simulation of the optimizer on the actor's parameters (`regularizer_dynamics_v68.py`: the same learner built
from the same seed, the same 78,160 steps and schedule, the same gradient, pull, noise and generator, then the
returned actor evaluated as in every run), which reproduces a PPO run with the gradient zeroed to the digit (checked
on the smoke stage; one such PPO run per arm at row C2, seed 94001, is run in the campaign as the check). The reference
point of this control is the prior policy, the actor at theta_0, whose risk and utility are recorded in every run.

Part B, the anchored application: one run of the anchored configuration of `../application/PROTOCOL-application-v42.md`
(F = 50, kappa_p = 10, residual action map around the incumbent re-planning rule, seed 1, 2.08 million steps,
constant learning rate) per arm, through `../application/train_app_optimizers_v68.py`: adam-coupled is the recorded run
`../application/runs/anchored_F50_k10_s1` (reused), adam-decoupled is the run of the first protocol
(`../application/runs-decoupled-v68/anchored-decoupled_F50_k10_s1`, reused), sgd is new at the constant rate lr_sgd_app
(pilot). The residual control (kappa_p = 0, seed 1) exists for Adam and is run for SGD as its reference.

## Pilots (separate seeds, before the campaign)

SGD needs its own learning rate for PPO to learn at all. Wear allowance: the unregularized configuration C6 on seeds
901 and 902 at lr in {0.001, 0.003, 0.01, 0.03}, full budget; the rule, fixed before the pilot, takes the rate with the
highest mean stopped utility over the two seeds among the rates at which no run diverged (no NaN, |log dispersion|
below 10) and lr kappa_2 < 1 at kappa_2 = 10. Application: the residual control (kappa_p = 0) on seed 990 at lr in
{0.001, 0.01, 0.03}, full budget; the rate with the highest value V. Results:

    lr_sgd = 0.001 (wear allowance; mean stopped utility of the unregularized policy over seeds 901 and 902:
                    5.31 at 0.001, 5.15 at 0.003, 4.56 at 0.01, 3.29 at 0.03; no run diverged, final log dispersion
                    between -2.5 and -4.1; the Adam control of revision 41 reached 4.37)
    lr_sgd_app = 0.01 (application, seed 990, value V of the residual control: -5.946 at 0.001, -5.938 at 0.01,
                    -6.334 at 0.03; the Adam control's seeds 1-6 lie between -5.67 and -5.99)

So lr_sgd kappa_2 is 0.01 at kappa_2 = 10 (relaxation time 100 steps, flow time 39) and lr_sgd_app kappa_p = 0.1
(half-life 7 steps). Under SGD the unregularized wear-allowance policy ends near the unpriced optimum (q_beta 0.58,
U 5.31, against the Adam control's 0.35 and 4.37), so the two optimizers' unregularized references differ; the
comparisons of (E5) are made at the same (kappa_1, kappa_2) and against each optimizer's own reference.

## Derivation (as in the first protocol, extended to the three arms)

For one coordinate x of theta - theta_0 and Adam's or SGD's step a_t on the objective, one step of the decoupled arm
and of the SGD arm is x_{t+1} = (1 - lr_t kappa_2) x_t + a_t + sqrt(2 kappa_1 lr_t) xi_t. Without the objective this
Gaussian autoregression has the stationary variance (kappa_1 / kappa_2) / (1 - lr kappa_2 / 2), independent of the
learning rate up to the factor 1 - lr kappa_2 / 2 (at least 0.9985 under Adam; under SGD at lr_sgd kappa_2, which is
why the control computes the exact value numerically). The stationary spread of the particle form when the noise
carries no pull of its own is therefore

    sigma = sqrt(kappa_1 / kappa_2):   C1 0.0300,   C2 0.1005,   C3 0.318,   C4 and C5 0 (no noise).

Under the coupled Adam arm the pull is normalized by Adam's running root mean square, so its step is about lr_t
whatever the distance, and the stationary law is of Laplace form with spread sqrt(2) kappa_1 = 0.0127 (C1) and 0.143
(C2, C3), independent of kappa_2 and of the learning rate (the paper's Appendix L). Anchor alone (C4, C5) under the
coupled Adam arm oscillates around the anchor with an amplitude of the order of lr = 3e-4 (the recorded pull-alone
spread is 3.2e-4); under SGD and the decoupled arm it stays at the anchor exactly.

The objective adds a_t. Under Adam |a_t| is at most about lr_t, which the decoupled pull balances at |x| <= 1 / kappa_2
(0.1 and 1), an envelope. Under SGD a_t = lr_t g_t with the clipped gradient g_t, balanced at |x| = |g_t| / kappa_2.
The anchor-only rows C4 and C5 measure the objective's displacement s_0 directly in each arm; to the extent that noise
and displacement add in quadrature, spread^2 ~ kappa_1 / kappa_2 + s_0^2 in the sgd and decoupled arms.

## Checks (fixed before the runs)

(E1) Zero-objective control. In the sgd and adam-decoupled arms the spread of C1-C3 lies within 10% of sigma, the
     coordinates are Gaussian (kurtosis between 2.7 and 3.3) and C4, C5 have spread 0. In the adam-coupled arm the
     spread of C2 and C3 lies within 25% of sqrt(2) kappa_1 with the ratio C3/C2 between 0.8 and 1.25, C1 within a
     factor 1.5 of sqrt(2) kappa_1 (its noise step and the pull's step are of the same order there), the coordinates
     heavier-tailed than Gaussian (kurtosis above 4; Laplace is 6), and C4, C5 have spread below 1e-3.
(E2) Learning runs, sgd arm (the regularizer's own effect): the spread of C1-C3 lies within a factor 1.3 of
     sqrt(sigma^2 + s_0^2), s_0 the spread of the matching anchor-only row C4 or C5.
(E3) Learning runs, adam-decoupled arm: the same as (E2) with its own s_0; and the ratio C3/C2 between 2 and 4.
(E4) Learning runs, adam-coupled arm: the spread of C3 within a factor 1.5 of that of C2 (the recorded 0.128), that
     is, independent of kappa_2, and both within a factor 1.5 of sqrt(2) kappa_1. This is the test of the paper's
     explanation of (O16) under the coupled form itself; the first protocol tested it by contrast with the decoupled
     form.
(E5) Position in the risk-utility plane. Reported for every arm and row: breakdown probability q_beta, stopped
     utility U, the interior loss I of the returned policy and U - I, the exact frontier's utility U_ex(q_beta) at the
     attained risk (the failure-only dynamic-programming policies re-evaluated in the simulator, interpolated linearly
     in log q as in the wear-allowance analysis), the gap U_ex(q_beta) - U, the price-risk bound with the idle policy
     as comparison and the measured optimization error, its slack bound - q_beta and factor bound / q_beta, and the
     action relative entropy A to the incumbent. No numerical prediction; the two comparisons to state are
       regularizer bias: the sgd arm at (kappa_1, kappa_2) against the unregularized sgd policy (C6) of the same
       seed, as paired differences in q_beta, U, gap and A;
       optimizer effect: each Adam arm against the sgd arm at the same (kappa_1, kappa_2), as the difference of
       frontier gaps (seed means and paired by seed).
     For the zero-objective control the reference point is the prior policy's q_beta and U.
(E6) Per-coordinate diagnostic, adam-coupled arm. Final |theta_i - theta_0,i| against sqrt(v_i), Adam's running mean of
     the squared gradient of coordinate i at the end of the run (which includes the pull), and against the root mean
     square of the clipped objective gradient of coordinate i over the run. Reported as the rank correlation and the
     median |theta_i - theta_0,i| by decile of sqrt(v_i). The coupled pull's step on coordinate i is lr kappa_2 x_i /
     sqrt(v_i), so it anchors least the coordinates the objective moves most; a positive rank correlation is what that
     reading implies, and the decoupled and sgd arms, whose pull does not depend on v_i, are the comparison. For the
     reused coupled records theta_0 is rebuilt from the seed (the fit is deterministic; its mean squared error is
     checked against the record) and v_i is read from the saved optimizer state.
(E7) The application, sgd arm: anchor distance, miss probability, cost and value against the recorded coupled run
     (distance 0.0070, miss 0.0130, cost 5.287, V -5.891), the decoupled run of the first protocol (0.4185, 0.0170,
     5.063, -5.854), the residual control (2.59, 0.0181, 4.871, -5.709) and the prior policy (0.0128, 5.298, -5.893);
     (A5) of the application protocol as the band (miss within 0.005 of 0.0128, cost within 0.4 of 5.30).

The first protocol's (D1)-(D5) are reported as well, with the same thresholds.

## Budget

All rows at eight seeds and the full budget. Learning runs: 72 new, about 4-7 minutes each on one core (training
250 s unloaded, evaluation 60 s), ten in parallel. Zero-objective control: 120 direct runs of about one to two minutes
each, three in parallel, plus three PPO checks. Application: one sgd run and one sgd residual control, about six
minutes each. Expected wall time of the campaign about one hour after the pilots.

## Reporting

Every run and seed is reported. `analyze_optimizers_v68.py` writes `wear-optimizers-v68.json` (per seed and per arm
and row, the checks and the two comparisons), `wear-optimizers-v68.tex` (the table) and `wear-optimizers-v68.pdf/png`
(spread against kappa_2 and the predictions per arm; the risk-utility plane; the per-coordinate diagnostic);
`coordinate_diagnostic_v68.py` rebuilds the per-coordinate records of the reused coupled runs into
`wear-optimizers-v68/derived-coordinates/`. `REPORT-optimizers-v68.md` holds the summary, the regularizer-bias and
optimizer-effect comparisons, and the application rows.
