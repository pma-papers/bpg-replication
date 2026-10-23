# The regularizers under three optimizers: report (revision 68, second protocol)

Date 2026-10-20. Protocol `PROTOCOL-optimizers-v68.md` (SHA-256 in `PROTOCOL-optimizers-v68.sha256`, fixed at 18:27 EEST
after the pilots and before the campaign). Campaign 18:25-19:10 EEST (zero-objective runs done 18:41, learning runs 19:10): 75 learning runs of `wear_param_optimizers_v68.py`
eight in parallel (`optimizers-campaign-v68.txt`, logs `optimizers-logs/`, records `wear-optimizers-v68/campaign/`),
120 zero-objective runs of `regularizer_dynamics_v68.py` three in parallel (`optimizers-zero-v68.txt`), the
per-coordinate records of the reused runs by `coordinate_diagnostic_v68.py` (`wear-optimizers-v68/derived-coordinates/`),
and two application runs. Runtimes as in `REPORT-decoupled-v68.md`. The machine was shared (load 17-30 on 14 cores).
Analysis `analyze_optimizers_v68.py`: `wear-optimizers-v68.json` (every seed), `wear-optimizers-v68.tex`,
`wear-optimizers-v68.pdf/png`. The decoupled-Adam arm's rows C2-C5 are the runs of the first protocol
(`REPORT-decoupled-v68.md`).

Arms: sgd (plain SGD, lr 0.001 decayed linearly to zero, the pull in the gradient after clipping, so the step on it is
exactly lr kappa_2 (theta - theta_0): the Euler step of the particle form), adam-coupled (Adam, eps 1e-5, lr 3e-4, the
pull in the gradient before the step: the paper's recorded form), adam-decoupled (Adam, the pull on the parameters
after the step). Rows C1 (kappa_2 = 10, kappa_1 = 0.009), C2 (10, 0.101), C3 (1, 0.101), C4 (10, 0), C5 (1, 0),
C6 (0, 0). Eight seeds everywhere, 8 million steps, 78,160 optimizer steps; flow time 39.1 (sgd) and 11.7 (Adam).

## 1. The zero-objective control: each optimizer's stationary law of the regularizer

Pull and noise alone, the objective's gradient zeroed, direct simulation (checked against a PPO run with the gradient
zeroed: one run per arm at C2, seed 94001, with the gradient zeroed inside PPO reproduces the direct simulation's spread and action relative entropy to every printed digit: 0.0985, 0.1296, 0.0990 for sgd, adam-coupled, adam-decoupled). Means over eight seeds (between-seed SD); the predicted spread is sqrt(kappa_1/kappa_2) for sgd
and adam-decoupled and sqrt(2) kappa_1 for adam-coupled; the reference point is the prior policy (q_beta 1.03%,
U 2.73 on these evaluation seeds).

| arm | row | kappa_1 | kappa_2 | predicted | spread | ratio | kurtosis | q_beta (%) | U | A |
|---|---|---|---|---|---|---|---|---|---|---|
| sgd | C1 | 0.009 | 10 | 0.0300 | 0.0299 (0.0003) | 1.00 | 2.98 | 1.2 (2.1) | 2.65 (0.22) | 3.3 |
| sgd | C2 | 0.101 | 10 | 0.1005 | 0.1001 (0.0011) | 1.00 | 2.98 | 4.2 (9.1) | 2.38 (0.68) | 44 |
| sgd | C3 | 0.101 | 1 | 0.318 | 0.3173 (0.0035) | 1.00 | 3.00 | 31.6 (27.5) | 3.57 (2.19) | 2044 |
| sgd | C4, C5 | 0 | 10, 1 | 0 | 0 | -- | -- | 1.0 | 2.72 | 0 |
| adam-decoupled | C1 | 0.009 | 10 | 0.0300 | 0.0298 (0.0003) | 0.99 | 2.98 | 1.4 (1.7) | 2.58 (0.24) | 5.0 |
| adam-decoupled | C2 | 0.101 | 10 | 0.1005 | 0.0998 (0.0010) | 0.99 | 2.98 | 3.5 (7.7) | 2.27 (0.90) | 70 |
| adam-decoupled | C3 | 0.101 | 1 | 0.318 | 0.3181 (0.0030) | 1.00 | 3.02 | 24.8 (28.6) | 3.22 (1.81) | 2476 |
| adam-decoupled | C4, C5 | 0 | 10, 1 | 0 | 0 | -- | -- | 1.0 | 2.72 | 0 |
| adam-coupled | C1 | 0.009 | 10 | 0.0127 | 0.0105 (0.0002) | 0.83 | 4.60 | 1.0 (0.8) | 2.72 (0.10) | 0.5 |
| adam-coupled | C2 | 0.101 | 10 | 0.143 | 0.1281 (0.0019) | 0.90 | 5.94 | 10.3 (14.9) | 2.32 (1.67) | 290 |
| adam-coupled | C3 | 0.101 | 1 | 0.143 | 0.1281 (0.0019) | 0.90 | 5.94 | 10.3 (14.9) | 2.32 (1.67) | 290 |
| adam-coupled | C4, C5 | 0 | 10, 1 | 0 | 0 | -- | -- | 1.0 | 2.72 | 0 |

(E1) holds in all three arms. Under SGD and under the decoupled Adam pull the regularizer alone produces the particle
form's Gaussian law to within 1% at every row, and the pull alone holds the parameters at the anchor exactly. Under
the coupled Adam pull it produces a heavier-tailed law (kurtosis 4.6-5.9; Laplace is 6) of spread 0.90 sqrt(2)
kappa_1 at kappa_1 = 0.101, the same to the fourth digit at kappa_2 = 10 and 1: with no objective, Adam normalizes
the pull's gradient kappa_2 x to a step of size lr sign(x) whatever kappa_2, so the two pulls generate identical
trajectories. The learning run of the paper at C2 gave 0.1277; the regularizer alone gives 0.1281, so the objective
contributed nothing to the recorded spread. At C1 the pull's normalized step (lr) and the noise step (8e-4) are of
the same order and the law is narrower than Laplace (ratio 0.83). The returned policies of this control show what a
spread does to behaviour with no learning: at spread 0.03 (C1) the policy stays near the prior (q_beta 0-6%); at 0.10 (C2) the
seeds split (0-26% under sgd, 0-22% decoupled); at 0.13 (adam-coupled C2) they split more widely (0-35%); at 0.32
(C3) half of the seeds have moved so far from theta_0 that they act at the boundary of the action set, the unpriced
optimum's region (q_beta 54-61%, U 5.0-5.4), which is a property of this example (full-rate operation is the unpriced
optimum), while the other half stay at low risk with a utility below the prior's. The action relative entropy A
grows with the spread from 3 to 2,000.

## 2. Learning runs

Means over eight seeds (between-seed SD); gap = U_ex(q_beta) - U, the exact frontier's utility at the attained
risk less the policy's; slack = price-risk bound (idle comparison, measured delta) minus q_beta, in points; predicted
spread sqrt(kappa_1/kappa_2) for sgd and adam-decoupled, sqrt(2) kappa_1 for adam-coupled. The adam-coupled rows C1,
C2, C4 and C6 and the adam-decoupled rows C2-C6 are the reused records. Full table with every column
`wear-optimizers-v68.tex`, every seed `wear-optimizers-v68.json`.

| arm | row | kappa_1 | kappa_2 | q_beta (%) | U | gap | slack | A | spread | predicted | ratio | sigma_a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sgd | C1 | 0.009 | 10 | 2.1 (2.9) | 2.76 (0.21) | 1.56 | 98 | 3 | 0.0299 | 0.030 | 1.00 | 0.10 |
| sgd | C2 | 0.101 | 10 | 4.3 (9.4) | 2.40 (0.70) | 1.61 | 96 | 44 | 0.1001 | 0.1005 | 1.00 | 0.10 |
| sgd | C3 | 0.101 | 1 | 31.6 (27.6) | 3.57 (2.19) | 1.25 | 68 | 2049 | 0.3173 | 0.318 | 1.00 | 0.11 |
| sgd | C4 | 0 | 10 | 3.3 (0.3) | 2.89 (0.01) | 1.83 | 97 | 2 | 3.2e-4 | 0 | -- | 0.10 |
| sgd | C5 | 0 | 1 | 18.0 (0.7) | 3.61 (0.02) | 1.51 | 82 | 58 | 2.0e-3 | 0 | -- | 0.10 |
| sgd | C6 | 0 | 0 | 58.4 (0.5) | 5.32 (0.03) | 0.02 | 42 | 1101 | 0.0159 | -- | -- | 0.09 |
| adam-coupled | C1 | 0.009 | 10 | 3.2 (2.2) | 2.88 (0.14) | 1.76 | 97 | 3 | 0.0106 | 0.0127 | 0.83 | 0.10 |
| adam-coupled | C2 | 0.101 | 10 | 13.8 (11.5) | 3.08 (0.73) | 1.64 | 86 | 69 | 0.1277 | 0.143 | 0.89 | 0.10 |
| adam-coupled | C3 | 0.101 | 1 | 12.7 (17.0) | 2.67 (1.64) | 1.74 | 87 | 270 | 0.1284 | 0.143 | 0.90 | 0.10 |
| adam-coupled | C4 | 0 | 10 | 3.4 (0.3) | 2.89 (0.02) | 1.84 | 97 | 2 | 3.2e-4 | 0 | -- | 0.10 |
| adam-coupled | C5 | 0 | 1 | 18.7 (0.8) | 3.63 (0.02) | 1.49 | 81 | 62 | 2.1e-3 | 0 | -- | 0.10 |
| adam (either) | C6 | 0 | 0 | 35.0 (3.6) | 4.37 (0.10) | 0.88 | 65 | 276 | 0.0442 | -- | -- | 0.02 |
| adam-decoupled | C1 | 0.009 | 10 | 27.0 (2.6) | 4.07 (0.09) | 1.13 | 73 | 152 | 0.0316 | 0.030 | 1.05 | 0.10 |
| adam-decoupled | C2 | 0.101 | 10 | 12.7 (10.4) | 3.22 (0.67) | 1.73 | 87 | 60 | 0.0999 | 0.1005 | 0.99 | 0.10 |
| adam-decoupled | C3 | 0.101 | 1 | 47.1 (19.6) | 5.03 (0.49) | 0.23 | 53 | 3426 | 0.3182 | 0.318 | 1.00 | 0.17 |
| adam-decoupled | C4 | 0 | 10 | 27.6 (1.2) | 4.16 (0.04) | 1.05 | 72 | 175 | 0.0142 | 0 | -- | 0.10 |
| adam-decoupled | C5 | 0 | 1 | 53.1 (1.2) | 5.12 (0.07) | 0.20 | 47 | 754 | 0.0270 | 0 | -- | 0.08 |

Checks.

* (E2) holds. The sgd arm, the particle form's own Euler step, has spread 0.0299, 0.1001 and 0.3173 at C1-C3 against
  sqrt(kappa_1/kappa_2) = 0.0300, 0.1005, 0.318 (ratios 1.00, 1.00, 1.00). Its objective displacement in the
  anchor-only rows is 3.2e-4 (kappa_2 = 10) and 2.0e-3 (kappa_2 = 1), negligible against the noise spread.
* (E3) holds. The adam-decoupled arm has spread 0.0316, 0.0999, 0.3182 (ratios to sqrt(sigma^2 + s_0^2) 0.95, 0.98,
  1.00; ratio C3/C2 = 3.19). At C1 the objective's displacement s_0 = 0.0142 is of the order of the noise spread 0.03,
  and the measured spread is their quadrature sum.
* (E4) holds. The adam-coupled arm has spread 0.1284 at kappa_2 = 1 against 0.1277 at kappa_2 = 10 (ratio 1.006),
  both 0.89-0.90 of sqrt(2) kappa_1 and equal to the zero-objective coupled spread 0.1281. Under the coupled form the
  pull's strength has no effect on the spread at all, with or without the objective.
* The spread is therefore the regularizer's own property in the sgd and decoupled arms (the particle form's law,
  within 1%) and the optimizer's property in the coupled arm (Adam's Laplace law, independent of kappa_2).

Behaviour. The anchor-only rows separate the arms most sharply. Under SGD and under the coupled Adam pull the rows
C4 and C5 coincide in every column (q_beta 3.3 against 3.4%, U 2.89 against 2.89, A 2 against 2 at kappa_2 = 10;
18.0 against 18.7%, 3.61 against 3.63, 58 against 62 at kappa_2 = 1): at a stationary point of the regularized
objective, g + kappa_2 (theta - theta_0) = 0, Adam's normalization multiplies a zero, so the coupled pull lands where
the particle form's pull lands. The decoupled pull at the same strengths does not: q_beta 27.6% and 53.1%, U 4.16 and
5.12, because its step lr kappa_2 x (3e-4 x at kappa_2 = 10) balances Adam's normalized objective step (about lr)
only at |x| of order 1/kappa_2, far beyond where the SGD pull's step lr_sgd kappa_2 x balances SGD's objective step
lr_sgd g (at |x| = |g|/kappa_2). The seeds split at kappa_1 = 0.101 in every arm (sgd C2: 0.0-27%; coupled C2:
0.0-29%; decoupled C2: 0.9-33%; at kappa_2 = 1 the splits widen to 0-59%, 0-41% and 5-61%), so the splitting is a
property of the noise at that spread, not of the optimizer. With the weak noise C1 the sgd and coupled arms stay at
the anchored values (2.1 and 3.2%); the decoupled arm sits at its anchor-only displacement (27.0%, the C4 value),
since the noise adds little to a pull that already lets the objective move the policy.

The unregularized references differ between the optimizers. SGD at lr 0.001 ends at the unpriced optimum's end of
the exact frontier (q_beta 58.4%, U 5.32, gap 0.02; objective value U - m q_beta/(1 - beta) = 4.97 against the exact
optimum's 5.05 at m = 0.03, an optimization error of 0.08), where the Adam control of revision 41 ends at (35.0%,
4.37), gap 0.88, optimization error 0.88, with its dispersion collapsed to 0.02. In this example the frontier is flat
between q_beta 0.3 and 0.6 (U 5.24 to 5.34), so the priced objective is nearly indifferent there and the SGD learner's
choice of the high-risk end costs it 0.08.

## 3. The regularizer's effect and the optimizer's effect

Regularizer bias, the sgd arm at (kappa_1, kappa_2) against the unregularized sgd policy of the same seed (paired
differences, mean over eight seeds, SD):

| row | kappa_1 | kappa_2 | d q_beta (points) | d U | d gap | d A |
|---|---|---|---|---|---|---|
| C1 | 0.009 | 10 | -56.3 (2.8) | -2.55 (0.19) | +1.54 (0.30) | -1098 (28) |
| C2 | 0.101 | 10 | -54.1 (9.4) | -2.92 (0.70) | +1.59 (0.56) | -1056 (64) |
| C3 | 0.101 | 1 | -26.9 (27.3) | -1.75 (2.18) | +1.22 (1.55) | +948 (1344) |
| C4 | 0 | 10 | -55.2 (0.7) | -2.42 (0.03) | +1.81 (0.01) | -1099 (29) |
| C5 | 0 | 1 | -40.5 (1.1) | -1.71 (0.04) | +1.49 (0.03) | -1043 (28) |

The regularizer moves the policy from the unpriced optimum's end of the frontier to the incumbent's neighbourhood: a
fall in breakdown probability of 40-56 points at the price of 1.7-2.9 in utility, of which 1.5-1.8 is the gap to the
frontier at the attained risk (the incumbent rule is not on the frontier; the prior policy's own gap is 1.69). The
pull sets where the policy is held and the noise how far it spreads, in the paper's terms, and at kappa_2 = 1 with
kappa_1 = 0.101 (C3) the noise is large enough that half of the seeds leave the incumbent altogether.

Optimizer effect, each Adam arm against the sgd arm at the same (kappa_1, kappa_2), as the difference of seed-mean
frontier gaps (standard error of the difference in parentheses; positive means the Adam arm is further from the
frontier):

| row | kappa_1 | kappa_2 | adam-coupled - sgd | adam-decoupled - sgd |
|---|---|---|---|---|
| C1 | 0.009 | 10 | +0.20 (0.12) | -0.43 (0.11) |
| C2 | 0.101 | 10 | +0.03 (0.26) | +0.12 (0.27) |
| C3 | 0.101 | 1 | +0.49 (0.72) | -1.02 (0.57) |
| C4 | 0 | 10 | +0.01 (0.01) | -0.79 (0.01) |
| C5 | 0 | 1 | -0.02 (0.01) | -1.31 (0.02) |
| C6 | 0 | 0 | +0.86 (0.03) | +0.86 (0.03) |

The coupled Adam arm is the sgd arm's equal wherever the pull dominates (C4, C5, and within noise C1, C2): same gap,
same risk, same utility. Its one large difference is the unregularized row C6, where Adam's own optimization error
(0.88) exceeds SGD's (0.08). The decoupled Adam arm differs from sgd by 0.4-1.3 in gap at every pull-dominated row,
in the direction of the frontier, because at these strengths and this learning rate its pull is too weak to hold the
policy at the incumbent, so the policy moves toward the Adam control's region. In the risk-utility plane the
decoupled arm's anchor-only rows sit near the frontier at a higher risk (27.6% and 53.1%), the sgd and coupled arms
near the incumbent at a low risk; the price-risk bound is slack by 42-99 points in every row, since every policy's
risk is far below what the bound with the idle comparison allows.

## 4. Per-coordinate diagnostic (E6)

Rank correlation over the 4418 actor coordinates between the final |theta_i - theta_0,i| and (a) sqrt(v_i), Adam's
second-moment estimate at the end of the run, (b) the root mean square of the clipped objective gradient of coordinate
i over the run (recorded in the new runs only); mean over eight seeds (SD):

| arm | row | (a) vs sqrt(v_i) | (b) vs RMS objective gradient |
|---|---|---|---|
| adam-coupled | C1 (0.009, 10) | 0.86 (0.01) | -- |
| adam-coupled | C2 (0.101, 10) | 0.97 (0.00) | -- |
| adam-coupled | C3 (0.101, 1) | 0.97 (0.00) | 0.00 (0.02) |
| adam-coupled | C4 (0, 10) | 0.29 (0.17) | -- |
| adam-coupled | C5 (0, 1) | 0.45 (0.21) | 0.48 (0.21) |
| adam-decoupled | C1 (0.009, 10) | 0.02 (0.02) | 0.00 (0.02) |
| adam-decoupled | C2 (0.101, 10) | 0.00 (0.02) | -- |
| adam-decoupled | C3 (0.101, 1) | -0.01 (0.02) | -- |
| adam-decoupled | C4 (0, 10) | -0.20 (0.07) | -- |
| adam-decoupled | C5 (0, 1) | 0.23 (0.08) | -- |
| sgd | C1, C2, C3 | -- | 0.00, -0.01, 0.00 (0.02) |
| sgd | C4, C5, C6 | -- | 0.18 (0.10), 0.43 (0.09), 0.56 (0.07) |

Under the coupled pull with noise the final displacement of a coordinate follows Adam's second moment almost
exactly (0.97): the median |theta_i - theta_0,i| rises from 0.009 in the lowest decile of sqrt(v_i) to 0.27 in the
highest, a factor of 29, where the decoupled arm's medians are flat at 0.066-0.069 across the deciles
(`wear-optimizers-v68.pdf`, panel c). This is the coupled pull's mechanism: its step on coordinate i is lr kappa_2 x_i
/ sqrt(v_i), so the coordinates with a large v_i are anchored least. Two qualifications. First, under the coupled form
v_i contains the pull itself, (kappa_2 x_i)^2 plus the objective's part, and at a stationary spread the pull's part
dominates, so (a) is in part the pull measuring its own displacement. Second, against the objective's own gradient
(b) the coupled displacement is uncorrelated in the noise rows (0.00 at C3): with noise, the coordinates the objective
moves most are not the ones that end furthest from the anchor in any arm (sgd and decoupled give 0.00 as well). In the
anchor-only rows, where the policy sits at a stationary point g_i + kappa_2 x_i = 0 and the displacement is
|g_i|/kappa_2 in every form, the correlation with the objective gradient is positive in every arm (0.18-0.48), the
coupled and sgd arms alike. The reading "the coupled pull anchors least the coordinates the objective moves most"
holds for the anchor-only rows in the sense that every pull does, and for the noise rows only with v_i, Adam's own
measure, which the pull dominates.

## 5. The application (F = 50, seed 1)

| arm | kappa_p | lr | contraction per step | anchor distance | miss p_51 | cost | U | V | delta (gap) | validated upper | mean-action p_51 | sigma_a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prior policy | | | | 0 | 0.0128 | 5.298 | -5.253 | -5.893 | | | | 0.135 |
| adam-coupled (record) | 10 | 3e-4 | -- | 0.0070 | 0.0130 | 5.287 | -5.241 | -5.891 | 1.558 (35.9%) | 0.0157 pass | 0.0132 | 0.135 |
| adam-decoupled | 10 | 3e-4 | 0.997 | 0.4185 | 0.0170 | 5.063 | -5.004 | -5.854 | 1.520 (35.1%) | 0.0200 pass | 0.0171 | 0.134 |
| sgd | 10 | 0.01 | 0.9 | 0.0056 | 0.0131 | 5.273 | -5.227 | -5.882 | 1.548 (35.7%) | 0.0158 pass | 0.0133 | 0.135 |
| adam, residual control (record) | 0 | 3e-4 | 1 | 2.592 | 0.0181 | 4.871 | -4.807 | -5.709 | 1.376 (31.7%) | 0.0211 pass | 0.0180 | 0.096 |
| sgd, residual control | 0 | 0.01 | 1 | 0.4703 | 0.0238 | 4.678 | -4.598 | -5.788 | 1.454 (33.6%) | 0.0273 pass | 0.0238 | 0.102 |

(E7) All three anchored arms stay within the (A5) band (miss within 0.005 of 0.0128, cost within 0.4 of 5.30), the
check behind (O27). The SGD arm at lr 0.01 contracts by 0.9 per step (half-life 7 steps) and holds the actor at the
anchor as tightly as the coupled Adam run (distance 0.0056 against 0.0070), while the decoupled Adam arm at lr 3e-4
contracts by 0.997 per step (half-life 231 steps) and settles at 0.42. The regularizer's own effect, the SGD arm
against the SGD control at the same seed: the anchor lowers the miss probability from 0.0238 to 0.0131 and raises the
cost from 4.68 to 5.27, moving the policy from the control's point to the prior's. The optimizer's effect at kappa_p
= 10: the two Adam arms lie within 0.04 of the SGD arm in value and within 0.004 in miss probability; the gap to the
optimum is 35.1-35.9% in all three, since all three sit near the prior, whose gap is 36.0%. The SGD control moves a
fifth as far as the Adam control in parameter distance (0.47 against 2.59) in the same number of steps, the time-scale
difference the protocol notes.

## 6. What this changes in the paper's statements

For the paper session (Section 6.5, Section 7.1, Appendices H, L of revision 68), with the first protocol's list
in `REPORT-decoupled-v68.md`:

1. The particle form's stationary spread holds on the learned controller once the pull is a gradient step. Under
   plain SGD (the particle form's own Euler step) and under the decoupled Adam pull the measured spread is
   sqrt(kappa_1/kappa_2) within 1% at every configuration (0.0299, 0.1001, 0.3173 and 0.0316, 0.0999, 0.3182 against
   0.030, 0.1005, 0.318), and so is the zero-objective control (within 1%, Gaussian coordinates). The sentence "The
   stationary spread of the particle form therefore holds under a gradient pull, applied to the parameters with the
   learning rate as the step" is now supported by runs; "That configuration was not run" is obsolete.
2. (O16), the spread proportional to kappa_1 whatever the pull, is Adam's property. With the objective zeroed the
   coupled pull gives 0.1281 at kappa_2 = 10 and at kappa_2 = 1 (identical trajectories), kurtosis 5.9 (Laplace 6),
   0.90 sqrt(2) kappa_1; the recorded learning run gave 0.1277, so the objective contributed nothing to it, and the
   new coupled run at kappa_2 = 1 gives 0.1284.
3. The splitting of the seeds at kappa_1 = 0.101 ((O17)) happens in all three arms at the same spread, sgd included
   (0-27%, 0-29%, 0.9-33% at kappa_2 = 10), and even in the zero-objective control (0-26%, 0-35%, 0-22%). It is the
   noise's property at a spread comparable to the incumbent's dispersion, not the optimizer's; the paper's
   explanation by the policy gradient's share of Adam's step should be withdrawn.
4. The coupled pull without noise (the anchor study, (O13), (O14), (O27)) lands where the particle form's pull
   lands: the sgd and adam-coupled anchor-only rows coincide at kappa_2 = 10 and 1 (q_beta 3.3 against 3.4%, 18.0
   against 18.7%; U, A and spread alike). The audit's reasoning (the rescaling multiplies a zero at a stationary
   point) is confirmed with numbers. The decoupled pull at the same strengths and Adam's learning rate 3e-4 does not
   hold the policy (27.6% and 53.1%), so the guideline of Section 7.1 needs its calibration sentence: a decoupled
   strength is read through the per-step contraction lr kappa_2, and kappa_2 = 10 at lr 3e-4 (contraction 0.997,
   half-life 231 steps) is a weak pull, while the same kappa_2 under SGD at lr 0.001 (contraction 0.99) or under the
   coupled form holds the incumbent.
5. The optimizer's own contribution is largest with no regularizer: Adam's control sits at gap 0.88 from the frontier
   (optimization error 0.88, dispersion collapsed to 0.02) where SGD's sits at 0.02 (error 0.08). With a pull that
   dominates, coupled Adam and SGD are indistinguishable in the risk-utility plane.
6. Application: all three arms keep the incumbent within the (A5) band at kappa_p = 10 (distances 0.0070 coupled,
   0.4185 decoupled, 0.0056 SGD; miss 0.0130, 0.0170, 0.0131; cost 5.287, 5.063, 5.273).
7. The per-coordinate diagnostic: under the coupled pull with noise the displacement of a coordinate follows Adam's
   second moment (rank correlation 0.97; a factor 29 between the lowest and highest decile), flat under the decoupled
   pull; against the objective's own gradient no arm shows a correlation in the noise rows. Appendix L may state the
   first as the mechanism of the coupled pull with the qualification that v_i contains the pull.

## 7. Amendment of 2026-10-20: the rows of revision 60 that existed only coupled, and the application on six seeds

Date 2026-10-20 (written 22:50 EEST). Protocol `PROTOCOL-optimizers-v68-AMENDMENT.md` (SHA-256 in
`PROTOCOL-optimizers-v68-AMENDMENT.sha256`, fixed at 19:38 after Sections 1-6 above and before any of its runs).
Runs launched 19:38: 48 zero-objective runs of `regularizer_dynamics_v68.py` two in parallel (done 19:47), 22
application runs of `../application/train_app_optimizers_v68.py` three in parallel (done 20:04, launcher
`../application/app_optimizers_run_one_v68.sh`, list `../application/optimizers-amendment-v68.txt`), 48 learning runs of
`wear_param_optimizers_v68.py` six in parallel (list `optimizers-amendment-campaign-v68.txt`, zero-objective list
`optimizers-amendment-zero-v68.txt`, log `optimizers-amendment-v68.log`, run logs `optimizers-logs/`, records
`wear-optimizers-v68/campaign/`). Thirty-six learning runs had finished by 20:05; the last twelve finished between
22:43 and 22:48, although their own training times are 4-9 minutes (234-517 s): the machine was unavailable to them in
between (it is shared; load averages of 20-26 on 14 cores were seen at the end). Every run has a record; no run failed.
Analysis `analyze_optimizers_v68.py`, extended for the new rows and the application (log
`analyze-optimizers-v68-amendment.log`); the entries of Sections 1-6 in `wear-optimizers-v68.json` and
`wear-optimizers-v68.tex` are unchanged to the digit, the new rows are added, and the application table is
`wear-optimizers-v68-application.tex` (key `application` of the JSON).

Rows: C7 (kappa_2 = 10, kappa_1 = 0.001), C8 (10, 0.989), C9 (0, 0.101), the revision-60 rows of nominal sigma 0.01,
0.3 and noise alone. New under sgd (lr 0.001) and adam-decoupled (lr 3e-4), eight seeds each; adam-coupled is the
recorded run of revision 60. Predicted spread: sqrt(kappa_1/kappa_2) = 0.0100 (C7), 0.3145 (C8) for the gradient-step
arms; at C9 there is no pull and the spread is the random-walk value sqrt(2 kappa_1 sum_t lr_t) = 2.808 (sgd, flow
time 39.02) and 1.538 (Adam, flow time 11.71); Laplace sqrt(2) kappa_1 = 0.0014 (C7) and 1.399 (C8) for the coupled
arm.

### 7.1 Zero-objective control (F1)

| arm | row | kappa_1 | kappa_2 | predicted | spread | ratio | kurtosis | q_beta (%) | U | A |
|---|---|---|---|---|---|---|---|---|---|---|
| sgd | C7 | 0.001 | 10 | 0.0100 | 0.0100 (0.0001) | 1.00 | 2.98 | 0.9 (0.7) | 2.70 (0.09) | 0.4 |
| sgd | C8 | 0.989 | 10 | 0.3145 | 0.3131 (0.0035) | 1.00 | 2.98 | 15.6 (22.8) | 2.37 (1.75) | 1392 |
| sgd | C9 | 0.101 | 0 | 2.808 (random walk) | 2.8165 (0.0261) | 1.00 | 3.06 | 19.4 (17.9) | 2.22 (1.71) | 3.3e5 |
| adam-decoupled | C7 | 0.001 | 10 | 0.0100 | 0.0099 (0.0001) | 0.99 | 2.98 | 1.0 (0.8) | 2.67 (0.09) | 0.6 |
| adam-decoupled | C8 | 0.989 | 10 | 0.3145 | 0.3124 (0.0030) | 0.99 | 2.98 | 22.4 (24.7) | 3.15 (2.00) | 1344 |
| adam-decoupled | C9 | 0.101 | 0 | 1.538 (random walk) | 1.5427 (0.0143) | 1.00 | 3.06 | 23.7 (20.1) | 2.49 (1.98) | 4.2e4 |

(F1) holds in both arms: the spread is within 1% of the particle form's value at C7 and C8 and within 0.3% of the
random-walk value at C9, the coordinates are Gaussian (kurtosis 2.98-3.06), and at C9 the per-rollout spread trace
follows sqrt(2 kappa_1 S_k) to within 0.7% from the tenth rollout on (the maximal deviation in both arms). The
recorded coupled C9 spread 1.533 is the same random-walk value (ratio 1.00): with no pull Adam has nothing to
normalize and the noise alone moves the parameters. At spread 0.01 (C7) the regularizer alone leaves the prior policy
where it is (q_beta 0.9-1.0% against the prior's 1.03%); at 0.31 (C8) the seeds split as at C3 (0-59%); at the
random-walk spreads 2.8 and 1.5 (C9) the actor is destroyed (A of 10^4 to 10^5, dispersion 6.4 under sgd and 0.63
under Adam), and its risk is wherever the random actor lands (0-47%).

### 7.2 Learning runs (F2), (F3)

Means over eight seeds (SD); columns as in Section 2; the adam-coupled rows are the reused records.

| arm | row | kappa_1 | kappa_2 | q_beta (%) | range | U | gap | slack | A | spread | predicted | ratio | sigma_a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sgd | C7 | 0.001 | 10 | 2.8 (1.1) | 1.3-4.7 | 2.88 (0.08) | 1.79 | 97 | 2 | 0.0100 | 0.0100 | 1.00 | 0.10 |
| sgd | C8 | 0.989 | 10 | 15.6 (22.8) | 0.0-59.3 | 2.38 (1.75) | 2.03 | 84 | 1392 | 0.3131 | 0.3145 | 1.00 | 0.09 |
| sgd | C9 | 0.101 | 0 | 23.6 (18.9) | 0.0-47.4 | 2.59 (1.67) | 2.31 | 76 | 4.8e5 | 2.8166 | 2.808 | 1.00 | 9.98 |
| adam-coupled | C7 | 0.001 | 10 | 3.2 (0.6) | 2.3-4.1 | 2.89 (0.04) | 1.83 | 97 | 2 | 0.0012 | 0.0014 | 0.87 | 0.10 |
| adam-coupled | C8 | 0.989 | 10 | 16.2 (25.9) | 0.0-59.4 | 2.36 (2.39) | 2.06 | 84 | 40901 | 1.3181 | 1.399 | 0.94 | 0.24 |
| adam-coupled | C9 | 0.101 | 0 | 50.6 (13.6) | 19.7-59.6 | 5.24 (0.14) | 0.06 | 49 | 45592 | 1.5326 | 1.538 | 1.00 | 1.65 |
| adam-decoupled | C7 | 0.001 | 10 | 27.5 (1.3) | 25.4-29.1 | 4.16 (0.04) | 1.05 | 73 | 174 | 0.0170 | 0.0100 | 1.70 | 0.10 |
| adam-decoupled | C8 | 0.989 | 10 | 19.9 (24.6) | 0.0-59.5 | 2.99 (2.05) | 1.68 | 80 | 1640 | 0.3124 | 0.3145 | 0.99 | 0.11 |
| adam-decoupled | C9 | 0.101 | 0 | 51.3 (11.5) | 28.2-60.2 | 5.28 (0.08) | 0.04 | 49 | 84994 | 1.5421 | 1.538 | 1.00 | 3.26 |

* (F2) holds. The sgd arm has spread 0.0100, 0.3131, 2.8166 against sqrt(sigma^2 + s_0^2) = 0.0100, 0.3145, 2.808
  (ratios 1.00, 1.00, 1.00); the objective displaces nothing measurable against the noise in any of the three rows.
* (F3) holds. The adam-decoupled arm has spread 0.0170, 0.3124, 1.5421 (ratios to the target 0.98, 0.99, 1.00). At
  C7 the noise spread 0.010 is below the objective's displacement s_0 = 0.0142 of the anchor-only row C4, and the
  measured spread is their quadrature sum 0.0174 (ratio 0.98), as at C1.
* The coupled records at C7 and C8 are 0.87 and 0.94 of the Laplace value, the range of Section 1 (0.83-0.90);
  at C9 both Adam arms and the coupled record give the random-walk value (1.533, 1.542 against 1.538).

### 7.3 The sequence of revision 60 under each arm (F4)

Seed means at kappa_2 = 10 with kappa_1 rising (rows C4, C7, C1, C2, C8), and noise alone (C9) next to the
unregularized row (C6):

| arm | quantity | kappa_1 = 0 | 0.001 | 0.009 | 0.101 | 0.989 | noise alone | unregularized |
|---|---|---|---|---|---|---|---|---|
| sgd | spread | 0.0003 | 0.0100 | 0.0299 | 0.1001 | 0.3131 | 2.82 | 0.016 |
| sgd | q_beta (%) | 3.3 | 2.8 | 2.1 | 4.3 | 15.6 | 23.6 | 58.4 |
| sgd | range (%) | 2.7-3.6 | 1.3-4.7 | 0.0-7.9 | 0.0-26.6 | 0.0-59.3 | 0.0-47.4 | |
| sgd | U | 2.89 | 2.88 | 2.76 | 2.40 | 2.38 | 2.59 | 5.32 |
| sgd | A | 2 | 2 | 3 | 44 | 1392 | 4.8e5 | 1101 |
| adam-coupled | spread | 0.0003 | 0.0012 | 0.0106 | 0.1277 | 1.3181 | 1.53 | 0.044 |
| adam-coupled | q_beta (%) | 3.4 | 3.2 | 3.2 | 13.8 | 16.2 | 50.6 | 35.0 |
| adam-coupled | range (%) | 2.8-3.8 | 2.3-4.1 | 0.3-6.8 | 0.0-29.3 | 0.0-59.4 | 19.7-59.6 | |
| adam-coupled | U | 2.89 | 2.89 | 2.88 | 3.08 | 2.36 | 5.24 | 4.37 |
| adam-coupled | A | 2 | 2 | 3 | 69 | 40901 | 45592 | 276 |
| adam-decoupled | spread | 0.0142 | 0.0170 | 0.0316 | 0.0999 | 0.3124 | 1.54 | 0.044 |
| adam-decoupled | q_beta (%) | 27.6 | 27.5 | 27.0 | 12.7 | 19.9 | 51.3 | 35.0 |
| adam-decoupled | range (%) | 26.3-30.1 | 25.4-29.1 | 24.1-31.4 | 0.9-33.0 | 0.0-59.5 | 28.2-60.2 | |
| adam-decoupled | U | 4.16 | 4.16 | 4.07 | 3.22 | 2.99 | 5.28 | 4.37 |
| adam-decoupled | A | 175 | 174 | 152 | 60 | 1640 | 84994 | 276 |

Observations. The spread increases with kappa_1 in every arm ((R1) of revision 60 as an observation: in the sgd and
decoupled arms it is the particle form's sqrt(kappa_1/kappa_2) at every row, in the coupled arm Adam's law). The
action relative entropy and the breakdown probability do not rise monotonically in any arm. In the sgd and coupled
arms the three weakest noises (0, 0.001, 0.009) leave the anchored policy where the pull alone holds it (q_beta
2.1-3.4%, U 2.76-2.89, A 2-3), with a slight fall of the seed-mean risk under sgd (3.3, 2.8, 2.1%) as the noise
widens the range of the seeds (2.7-3.6 to 0.0-7.9%) downward; at 0.101 and 0.989 the seeds split (0-27%, 0-59%), and
the seed mean rises because some seeds leave the incumbent. In the decoupled arm the pull alone already lets the
policy go to 27.6%, the weak noises leave it there, and at 0.101 the mean falls to 12.7% because the split sends half
of the seeds to zero risk. The sequence of revision 60, "the breakdown probability rises with kappa_1 from the
anchored value toward the control's" ((R2)), is therefore not what any arm shows: the noise does not move the policy
along the frontier, it splits the seeds between the incumbent's neighbourhood and the boundary of the action set, at
the same spread in every optimizer.

Noise alone ((R3) of revision 60) depends on the optimizer. Under Adam, coupled or decoupled, the random walk of
spread 1.5 ends with a policy at the unpriced optimum's end of the frontier (q_beta 50.6% and 51.3%, U 5.24 and 5.28,
gap 0.06 and 0.04), above the Adam control's risk (35.0%) and on the frontier where the control is not (gap 0.88): the
objective still steers the destroyed actor, since Adam's normalized step is of size lr whatever the gradient's scale.
Under SGD the random walk of spread 2.8 leaves a policy of mean dispersion 10 (final log dispersions from -7.4 to
+4.1 across seeds) at q_beta 23.6% (0-47%) and U 2.59, far below the unregularized SGD policy (58.4%, 5.32, on the
frontier): the clipped gradient step lr g, at most 5e-4 in norm, cannot steer parameters that the noise moves by 2.8
per coordinate. Revision 60's statement that noise alone ends "near the control's" risk and utility was a statement
about Adam.

Regularizer bias (sgd row minus the unregularized sgd policy of the same seed, mean (SD) of the paired differences):
C7 -55.6 (1.1) points of q_beta, U -2.44 (0.06), gap +1.77 (0.06), A -1099 (29), the values of the anchor-only row
C4; C8 -42.8 (23.0), -2.94 (1.77), +2.01 (1.09), A +291 (1820); C9 -34.8 (19.3), -2.73 (1.70), +2.29 (1.26), A
+4.8e5. Optimizer effect (difference of seed-mean frontier gaps, SE): adam-coupled minus sgd +0.04 (0.02) at C7,
+0.03 (0.70) at C8, -2.25 (0.44) at C9; adam-decoupled minus sgd -0.75 (0.02), -0.35 (0.64), -2.27 (0.44). As in
Section 3: coupled Adam and SGD coincide wherever the pull dominates (C7 as C4), the decoupled pull at Adam's rate is
too weak at kappa_2 = 10 (C7 as C4), and the one large optimizer effect besides the unregularized row is noise alone,
where Adam's normalization keeps the objective in control of a destroyed actor and SGD's does not.

Per-coordinate diagnostic (F6) for the new runs: the rank correlation of the final |theta_i - theta_0,i| with the root
mean square objective gradient is 0.00, -0.01, -0.01 (sgd C7-C9) and -0.11, 0.00, -0.01 (adam-decoupled), and with
sqrt(v_i) -0.09, -0.01, -0.01 (adam-decoupled); the medians by decile of sqrt(v_i) are flat (0.009-0.015 at C7,
0.205-0.222 at C8, 0.99-1.06 at C9). As in Section 4, no gradient-step arm shows a coordinate-wise structure with
noise.

### 7.4 The application on six seeds per arm (F7)

F = 50, kappa_p = 10 (anchored) and kappa_p = 0 (residual control), seeds 1-6; mean (SD) over seeds; the (A5) band is
miss within 0.005 of the prior's 0.0128 and cost within 0.4 of the prior's 5.298; "pass" counts the candidates
validated at the family error.

| arm | kappa_p | lr | anchor distance | miss p_51 | cost | U | V | gap (%) | validated upper | pass | (A5) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prior policy | | | 0 | 0.0128 | 5.298 | -5.253 | -5.893 | 36.0 | | | |
| adam-coupled (records) | 10 | 3e-4 | 0.0082 (0.0016) | 0.0131 (0.0004) | 5.291 (0.025) | -5.246 | -5.899 (0.014) | 36.1 | 0.0157 | 6/6 | holds |
| adam-decoupled | 10 | 3e-4 | 0.3162 (0.0708) | 0.0138 (0.0023) | 5.242 (0.147) | -5.194 | -5.882 (0.047) | 35.7 | 0.0165 | 6/6 | holds |
| sgd | 10 | 0.01 | 0.0084 (0.0022) | 0.0129 (0.0005) | 5.302 (0.040) | -5.257 | -5.903 (0.022) | 36.2 | 0.0156 | 6/6 | holds |
| adam, residual control (records) | 0 | 3e-4 | 2.648 (0.079) | 0.0165 (0.0030) | 5.071 (0.187) | -5.012 | -5.835 (0.119) | 34.6 | 0.0194 | 6/6 | |
| adam-decoupled, residual control | 0 | 3e-4 | 2.648 (0.079) | 0.0165 (0.0030) | 5.071 (0.187) | -5.012 | -5.835 (0.119) | 34.6 | 0.0194 | 6/6 | |
| sgd, residual control | 0 | 0.01 | 0.4376 (0.0421) | 0.0171 (0.0057) | 5.123 (0.349) | -5.064 | -5.919 (0.101) | 36.6 | 0.0201 | 6/6 | |

Per seed (anchor distance, miss, cost, V): adam-decoupled kappa_p = 10: s1 0.4185, 0.0170, 5.063, -5.854; s2 0.3705,
0.0118, 5.349, -5.898; s3 0.3095, 0.0126, 5.356, -5.944; s4 0.2138, 0.0130, 5.254, -5.859; s5 0.2921, 0.0118, 5.372,
-5.921; s6 0.2930, 0.0163, 5.058, -5.819. sgd kappa_p = 10: s1 0.0056, 0.0131, 5.273, -5.882; s2 0.0060, 0.0127,
5.323, -5.914; s3 0.0105, 0.0126, 5.353, -5.940; s4 0.0078, 0.0128, 5.292, -5.888; s5 0.0098, 0.0125, 5.328, -5.909;
s6 0.0105, 0.0138, 5.244, -5.886. sgd kappa_p = 0: s1 0.4703, 0.0238, 4.678, -5.788; s2 0.4789, 0.0120, 5.420,
-5.978; s3 0.3987, 0.0118, 5.452, -6.002; s4 0.3780, 0.0195, 5.110, -6.017; s5 0.4307, 0.0123, 5.354, -5.924; s6
0.4690, 0.0231, 4.723, -5.802. The coupled records are in `../application/REPORT.md`; every seed is in the JSON.

(F7) The (A5) band holds for all three anchored arms on six seeds: seed-mean miss 0.0131, 0.0138, 0.0129 (prior
0.0128), cost 5.291, 5.242, 5.302 (prior 5.298), value -5.899, -5.882, -5.903 (prior -5.893); every one of the 36
candidates validates; and in every arm the anchored learner stays closer to the prior in value than its control
(0.006, 0.011, 0.010 against 0.058, 0.058, 0.025). The regularizer's effect (anchored minus control, paired by seed,
mean (SD)): sgd miss -0.0042 (0.0053), cost +0.18 (0.31), V +0.02 (0.09), distance -0.43; adam-coupled -0.0034
(0.0028), +0.22 (0.17), -0.06 (0.12), -2.64; adam-decoupled -0.0027 (0.0024), +0.17 (0.09), -0.05 (0.10), -2.33. The
optimizer's effect at kappa_p = 10 (Adam arm minus sgd, paired): coupled miss +0.0002 (0.0002), cost -0.01 (0.03), V
+0.003 (0.015), distance -0.0002; decoupled +0.0009 (0.0019), -0.06 (0.11), +0.02 (0.03), distance +0.31. The two
controls differ by 2.21 in parameter distance and by 0.08 (0.06) in value (Adam's control the higher), with the same
miss probability. The seed-1 SGD control of Section 5 (miss 0.0238) was the most displaced of its six seeds (0.0118 to
0.0238).

Two reproduction checks. The adam-decoupled seed 1 run again through `train_app_optimizers_v68.py` reproduces the
first protocol's record (`runs-decoupled-v68/anchored-decoupled_F50_k10_s1`) to every digit of distance, miss, cost
and value. The adam-decoupled residual control (kappa_p = 0, where neither form applies a pull) reproduces the
recorded Adam control to every digit on all six seeds, so the pipeline around the learner with toolbox 0.6.1 is the
one that produced the records with toolbox 0.1.0.

### 7.5 What this adds to Section 6

8. The replaced experiment's rows exist now under the three arms at every noise strength of revision 60 (kappa_1 = 0,
   0.001, 0.009, 0.101, 0.989 at kappa_2 = 10, and noise alone). In the sgd and decoupled arms the spread is the
   particle form's at every strength (ratios 0.99-1.00 at C7 and C8; the random walk at C9, ratio 1.00); the
   coupled arm's spread is Adam's at every strength (0.83-0.94 of sqrt(2) kappa_1, the random walk at C9).
9. The monotone-risk statement of revision 60 ((R2)) holds in no arm. The weak noises (sigma <= 0.03) leave the
   anchored policy where the pull holds it, and the stronger ones split the seeds, so the seed mean is not a
   monotone function of kappa_1. The paper's (O17) should state the splitting and not a rise.
10. Noise alone ((R3)) ends on the frontier under Adam (q_beta 51%, gap 0.04-0.06) and in a destroyed policy under
    SGD (24%, gap 2.3, dispersion 10): a statement about the optimizer, not about the regularizer.
11. The application's (O27) holds on six seeds under all three arms (miss 0.0129-0.0138, cost 5.24-5.30, 36 of 36
    validated), with anchor distances 0.008 (coupled), 0.32 (decoupled) and 0.008 (sgd); the SGD control's
    displacement is a sixth of Adam's (0.44 against 2.65) at the same miss probability.
