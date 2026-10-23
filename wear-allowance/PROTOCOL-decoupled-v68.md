# Protocol: the parameter pull decoupled from Adam (roles of the regularizers on PPO; the anchored application), revision 68

Fixed on 2026-10-20 before any campaign run was started or inspected. Two smoke tests of the runners (seed 990,
a few rollouts, outputs under `wear-noise-decoupled-v68/smoke/` and `../application/_scratch/`) checked that the
scripts run; their numbers are not used.

## Why

Every parameter pull in the paper's runs (the parameter-anchor study, the roles of the regularizers on PPO, the
anchored configuration of the emissions application) adds kappa_2 (theta - theta_0) to the actor gradient after
gradient clipping and before the Adam step (`wear_param_anchor.py`, `wear_param_noise_v60.py`, the toolbox's
`ParameterAnchor` of version 0.1.0). Adam divides each gradient coordinate by its running root mean square, so the
pull's step is about the learning rate whatever the distance to the anchor. The audit of revision 68 (Section 10 of
the paper's revision audit) attributes (O16), the spread proportional to kappa_1, to that coupling, and the paper's
Section 7.1 now recommends the decoupled form, theta <- theta - lr kappa_2 (theta - theta_0) applied to the parameters
after the optimizer's step with the noise added after it. This campaign runs that form.

## The runs

Part A, the roles of the regularizers on PPO. Setting of `PROTOCOL-noise-v60.md` unchanged (wear-allowance example,
mispriced m = 0.03, PPO as in the matched-prior campaign of revision 41: actor started at theta_0 fitted to the
incumbent rule with dispersion 0.1, 8 million steps in 32 environments, 256 steps per environment per rollout,
minibatches of 1024, ten epochs, learning rate 3e-4 decayed linearly to zero, Adam with eps 1e-5, gradient norm
clipped at 0.5, evaluation on 20,000 fresh episodes, seeds 94001-94008). Runner `wear_param_noise_decoupled_v68.py`,
which reuses the revision-41 runner and replaces its optimizer hook. After every optimizer step, in this order, every
actor parameter (mean network and log dispersion, d = 4418) receives the pull theta <- theta - lr_t kappa_2 (theta -
theta_0) and then Gaussian noise of variance 2 kappa_1 lr_t, lr_t the current learning rate clipped at zero. Nothing is
added to the gradient. The noise comes from a dedicated generator seeded with 1,000,000 + seed.

Configurations, eight seeds each (32 runs):

| row | kappa_2 | kappa_1 | role |
|---|---|---|---|
| A1 | 10 | 0.101 | the paper's third noise row (its spread 0.128 coupled), decoupled |
| A2 | 1 | 0.101 | the kappa_2 dependence, which the coupled campaign did not test |
| A3 | 10 | 0 | pull alone, decoupled: the policy gradient's own displacement against the pull |
| A4 | 1 | 0 | the same at the weaker pull |

Rows A3 and A4 are controls added here so that the spread of A1 and A2 can be attributed. Reused without rerun: the
coupled rows of `wear-noise-v60.json` (kappa_2 = 10 with kappa_1 in {0.001, 0.009, 0.101, 0.989}, noise alone) and
the matched pull-alone and control rows of revision 41.

Part B, the anchored application. One run of the anchored configuration of `../application/PROTOCOL-application-v42.md`
(F = 50, kappa_p = 10, residual action map around the incumbent re-planning rule, seed 1, 2.08 million steps,
constant learning rate 3e-4) through `../application/train_app_decoupled_v68.py`, which runs the recorded runner with
the toolbox's anchor replaced by a local decoupled hook (pull on the mean network and the log dispersion after every
optimizer step, no noise). Compared with the coupled record `../application/runs/anchored_F50_k10_s1` and the residual
control `anchored_F50_k0_s1`.

Reported per run, as in revision 60: breakdown probability q_beta and stopped utility U of the stochastic policy, the
mean-action policy's q_beta, the action relative entropy A to the incumbent rule, the parameter distance
||theta - theta_0||, the per-parameter spread ||theta - theta_0|| / sqrt(d), the final log dispersion, the number of
optimizer steps and the learning-rate sum. For the application: anchor distance, miss probability, abatement cost,
value, validation and certificates, against the coupled record.

## Derivation of the expected spread (stated before the runs)

Write x for one coordinate of theta - theta_0 and a_t for Adam's step on the clipped objective gradient in that
coordinate. One decoupled step is

    x_{t+1} = (1 - lr_t kappa_2) x_t + a_t + sqrt(2 kappa_1 lr_t) xi_t,    xi_t ~ N(0, 1).

Without the objective (a_t = 0) and at a fixed learning rate this is a Gaussian autoregression with stationary
variance 2 kappa_1 lr / (1 - (1 - lr kappa_2)^2) = (kappa_1 / kappa_2) / (1 - lr kappa_2 / 2). The learning rate
cancels up to the factor 1 - lr kappa_2 / 2, which is at least 0.9985 here (lr kappa_2 <= 3e-3), so the stationary
spread per coordinate is

    sigma_dec = sqrt(kappa_1 / kappa_2):    0.1005 at kappa_2 = 10,    0.318 at kappa_2 = 1.

This is the spread of the particle form when the noise carries no pull of its own. The particle form's
sqrt(kappa_1 / (kappa_1 + kappa_2)) is the spread when the noise is the relative-entropy term, whose pull -kappa_1 x
toward the prior's center accompanies it; the revision-60 hook injects noise only, so the pull is kappa_2 alone and
sqrt(kappa_1 / kappa_2) applies. The two differ by 0.5% at kappa_2 = 10 (0.1005 against 0.1000) and by 5% at
kappa_2 = 1 (0.318 against 0.303); this campaign cannot tell them apart and does not try to. Since the stationary
variance does not depend on the learning rate, the linear decay of the schedule leaves it unchanged. The relaxation
time 1 / (lr kappa_2) is 333 optimizer steps at kappa_2 = 10 and 3,333 at kappa_2 = 1 at the initial rate, out of
78,080 optimizer steps per run (976 rollouts of 80 steps), so the process reaches its stationary law early and
stays in it; over the last 7,200 (kappa_2 = 10) or 22,800 (kappa_2 = 1) steps the remaining integrated contraction
falls below one and the law is frozen at its stationary value.

The objective adds the term a_t. Adam's step is at most about lr_t per coordinate, attained where the gradient's
direction is persistent, so the pull balances it at |x| = |a_t| / (lr_t kappa_2) <= 1 / kappa_2: 0.1 at kappa_2 = 10
and 1 at kappa_2 = 1, an envelope, not a prediction. Where the gradient's sign fluctuates, as near a stationary point
of the surrogate, a_t is a fraction of lr_t and the displacement is that fraction of 1 / kappa_2. Rows A3 and A4
measure this displacement directly as the pull-alone decoupled spread s_0(kappa_2). To the extent that the noise and
the objective's displacement add in quadrature, the expected spread of the sum is

    spread(A1)^2 ~ kappa_1 / kappa_2 + s_0(10)^2,        spread(A2)^2 ~ kappa_1 / kappa_2 + s_0(1)^2.

Under the coupled form the paper's arithmetic (Appendix L of revision 68) gives a Laplace law of spread sqrt(2) kappa_1
= 0.143 whatever kappa_2 and the learning rate, measured 0.125-0.130 at kappa_2 = 10.

## Checks (fixed before the runs)

(D1) Spread at kappa_2 = 10, kappa_1 = 0.101 (row A1): mean over seeds within a factor 1.3 of sqrt(kappa_1 / kappa_2)
     = 0.1005 once s_0(10) is subtracted in quadrature, that is, sqrt(spread(A1)^2 - s_0(10)^2) in [0.077, 0.131].
(D2) The kappa_2 dependence: the ratio spread(A2) / spread(A1) between 2 and 4 (the particle form gives sqrt(10) =
     3.16; the objective's displacement, larger at the weaker pull, can only raise it). Under the coupled Laplace law
     the ratio would be 1.
(D3) Pull alone decoupled (rows A3 and A4): spread s_0 at or below the envelope 1 / kappa_2 (0.1 and 1), above the
     coupled pull-alone spread 3.2e-4, and larger at kappa_2 = 1 than at kappa_2 = 10. If s_0(10) exceeds 0.077,
     (D1) is undecidable at this kappa_1 and is reported as such.
(D4) Behaviour at kappa_2 = 10, kappa_1 = 0.101: reported, not predicted. In the coupled runs the seeds split between
     the unpriced optimum's risk and policies earning almost nothing, which the paper attributes to the pull's
     gradient at the noised parameters dominating Adam's normalizer, so that the policy gradient's share of a step
     was below one percent. Decoupled, the policy gradient takes its full Adam step. Whether the seeds still split at
     the same spread tells whether that qualification of (O17) was right.
(D5) The application: anchor distance between the coupled record's 0.0070 and the residual control's 2.59, below the
     envelope sqrt(d) / kappa_p (d the number of actor parameters, about 4700, so about 6.9). Miss probability and cost
     reported against (A5) of the application protocol (seed within 0.005 of the prior policy's 0.0128 and cost within
     0.4 of 5.30), the check behind (O27).

What would count as (O16) surviving in the decoupled runs: the spreads of A1 and A2 both within 0.9-1.7 kappa_1
(0.091-0.172), independent of kappa_2 (ratio below 1.5). What would count as (O16) being a property of the coupling:
(D1) and (D2) holding. Anything else is reported as observed.

## Reporting

Every run and seed is reported. `analyze_decoupled_v68.py` writes `wear-noise-decoupled-v68.json` (per seed and per
row, with the checks), `wear-noise-decoupled-v68.tex` (the table, with the coupled rows for comparison) and
`wear-noise-decoupled-v68.pdf/png`; `REPORT-decoupled-v68.md` holds the summary and the application comparison.
Records as in revision 60: `wear-noise-decoupled-v68/campaign/k1_<k1>__k2_<k2>__seed<S>/result.json` with the trained
policy, logs in `noise-decoupled-logs/`; the application run in `../application/runs-decoupled-v68/`.
