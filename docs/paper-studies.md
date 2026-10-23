# Worked examples

This document walks through the studies of *The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex
Regularization* as a practitioner would run them with the `barrierpg` toolbox: the wear allowance, the parameter anchor,
the fuel tank, the roles of the regularizers, and the benchmark-suite study of the online appendix. Studies are named as in the paper, not numbered.
Each walk-through states the problem in the paper's vocabulary, the configuration choices and why, the paper's
numbers, and what a practitioner should take from them. The prescriptions behind each choice are in the toolbox's practitioner guide, and its 'under the hood' notebooks run
reduced versions with outputs.

Every number below is from the paper's tables. The campaigns' protocols were fixed before each campaign, and no seed or
checkpoint was selected. For every study below, the simulators, the protocols, the records of every run, the trained
policies and the code that generates every table and figure are in this repository. The studies check the paper's bounds and record what the
learners do; they do not rank barrier policy gradients against other constrained methods.

---

## 1. An asset operated under a wear allowance

### The problem

An asset earns revenue in proportion to its operating rate `a in [0, 1]` and to a market price index
`s in [-1, 1]`; each unit of throughput consumes a random amount of a wear allowance:

```
u(s, a) = (1 + s/2) a / (3/2),    g(a, xi) = 0.1 a xi,    s' = 0.8 s + 0.2 w,
```

with `w` uniform on `[-1, 1]` and `xi` gamma distributed with mean one and shape 1/2. The allowance `b = 1` is not
replenished; exhausting it before the next maintenance access is a breakdown, and every period of downtime costs `m`.
Maintenance access arrives after each period with probability `1 - beta`, `beta = 0.95`, so the geometric risk
`q_beta` is exactly the probability of a breakdown before access, and the price of failure risk is a quantity of the
application. Standing still never fails; every positive rate can fail.

The problem ships with the toolbox as `barrierpg.examples.wear`, with the exact benchmark's frontier:

```python
from barrierpg import Requirement, plan
from barrierpg.examples.wear import WearModel, EXACT_FRONTIER

model = WearModel()
spec = model.spec(ubar=1.0)                                   # b = 1, beta = 0.95, utility in [0, 1]
req = Requirement(T=19, alpha=0.05, zeta=0.05, family=2)
p = plan(spec, req, sup_U=EXACT_FRONTIER[0.0][1], expected_failure_rate=0.01, allowance_error_known=False,
         regularizers=dict(action_reference_trusted=True, price_trusted=False, incumbent_available=True))
print(p.explain())
```

The toolbox's notebooks T1 and T2 (its 'under the hood' track) evaluate rules against the exact frontier and train PPO through the wrapper.

### The paper's campaign

Fifteen configurations, eight seeds each, PPO of Stable-Baselines3 with one setting throughout (32 environments,
rollouts of 256 steps, minibatches of 1024, ten epochs, a learning rate decaying linearly to zero, no entropy bonus,
`8 x 10^6` steps per run), and evaluation of each final stochastic policy on 20,000 fresh episodes. Geometric stopping
(`BudgetWrapper(geometric_stop=True)`) implements the discount with algorithmic discount one. The exact benchmark
solves the stationary problem by dynamic programming on a grid and re-evaluates every exact policy in the continuous
simulator.

| Configuration | `q_beta` (%) | `U` | gap (%) | pass (of 8) | near (%) | at `0.9 b` (%) |
|---|---|---|---|---|---|---|
| unpriced, `m = 0` | 59.90 | 5.35 | -0.2 | 0 | 61.4 | 62.8 |
| fixed price `m = 0.3` | 4.42 | 4.76 | 0.9 | 3 | 29.4 | 45.1 |
| fixed price `m = 1` | 1.90 | 4.50 | 1.7 | 8 | 12.6 | 32.6 |
| fixed price `m = 10` | 0.53 | 4.01 | 4.8 | 8 | 2.0 | 11.0 |
| barrier `theta = 0.1`, `m = 1` | 0.69 | 3.85 | 10.1 | 8 | 1.0 | 1.5 |
| adaptive multiplier, target 0.05 | 5.03 | 4.78 | 1.3 | 0 | 31.5 | 46.2 |
| adaptive multiplier, target 0 | 0.79 | 4.16 | 3.9 | 8 | 3.2 | 15.5 |
| action-regularized, `m = 0.03`, `eta = 0.1` | 1.47 | 2.80 | 37.9 | 8 | 3.2 | 5.7 |

Means over eight seeds. Gap is the relative shortfall from the exact frontier at the attained risk; pass counts the
seeds whose exact binomial upper bound on `q_beta`, simultaneous over the 120 learned policies at confidence 95%, is
at most 0.05. The incumbent rule has `q_beta = 1.08%` and `U = 2.73`. Full table: the wear-allowance campaign in the paper.

### What the practitioner decides, and what came out

1. **Price it.** The unpriced learner runs to failure, as the exact solution does. A price removes most of that risk at
   a small utility cost, because the gamble it forgoes is the last one before exhaustion. Learned fixed-price
   policies lie within a few percent of the exact frontier. The price-risk law (`certificates.price_risk_law`,
   checked by `frontier.check_runs`) has the right dependence on the price and is conservative by a factor of twenty
   to thirty. At `m = 0.1` three of eight seeds stayed at the unpriced optimum: check every seed.
2. **Price or tolerance.** The adaptive multiplier reaches its prescribed level on the same frontier. Its multiplier
   for the target 0.05 converges at the tolerance itself, so no seed passes validation at 0.05: set a prescribed level
   below the tolerance by the width of the confidence bound. For the zero target the multiplier increases throughout
   training; `DualAscent.diagnose` reports that it has not converged.
3. **Margin.** The failure-only price leaves no reserve: at `m = 1`, 32.6% of episodes break down when the true
   allowance is `0.9 b`. The interior barrier at the same price brings this to 1.5% at a utility cost within its
   interior loss. Test the margin with `BudgetWrapper(true_budget_scale=0.9)`; if the size of the error is known,
   plan with the tightened allowance instead.
4. **Action anchor.** At the deliberately low price `m = 0.03` the exact optimum breaks down in about a third of the
   episodes. The action relative entropy to the incumbent rule (`ActionAnchor`) keeps the breakdown probability near
   the incumbent's for `eta >= 0.1`, and the bound `kl(q || q_rho) <= A` holds in every run but is informative only
   when `A` is small. The anchored policies lie far inside the frontier: the anchor protects against a mispriced
   failure risk; it does not replace the price.

---

## 2. The prior on a learned controller

Same problem, price `m = 0.03`. The actor is fitted to the incumbent rule (parameters `theta_0`) and trained with weight
decay toward `theta_0`, the pull part of a Gaussian prior that a finite network carries:

```python
from barrierpg.callbacks import fit_actor_mean, set_log_std, ParameterAnchor
mse = fit_actor_mean(ppo, incumbent_mean, observations)   # theta_0: the actor's mean fitted to the rule
set_log_std(ppo, math.log(0.05))                           # the matched prior: the incumbent's own dispersion
anchor = ParameterAnchor(ppo, kappa_p=10.0).install()      # covers the dispersion by default
```

| Configuration | `q_beta` (%) | `U` | dispersion |
|---|---|---|---|
| prior policy, the learner's default dispersion 0.37 | 6.04 | 2.85 | 0.37 |
| anchor `kappa_p = 100`, default dispersion | 6.43 | 2.86 | 0.37 |
| mean-only anchor `kappa_p = 100` | 34.32 | 4.01 | 7.50 |
| prior policy, matched dispersion (the incumbent rule) | 1.03 | 2.73 | 0.10 |
| unanchored control `kappa_p = 0` | 35.02 | 4.37 | 0.02 |
| matched anchor `kappa_p = 10` | 3.35 | 2.89 | 0.10 |
| matched anchor `kappa_p = 100` | 1.21 | 2.74 | 0.10 |

(Dispersions in the pre-clipping action variable of the paper's learner.) The learner inherits the prior policy, not
the rule the actor was fitted to: evaluate the prior policy as a candidate before training, and anchor the dispersion
too. With the mean network anchored alone, the one free dispersion parameter turned the incumbent rule into a
bang-bang policy. `regularizer_effects` with `ParameterAnchor.effects_inputs()` reports the parameter distance, the
dispersion before and after, inheritance of the prior policy, and a dispersion-escape flag.

---

## 3. A learned controller with a fuel tank

### The problem

LunarLanderContinuous-v3 of Gymnasium with a fuel tank: every frame consumes the fuel its engines burn (0.30 per unit
of main-engine power, 0.03 per unit of side-engine power), running out before touchdown is the failure, and the utility
is the task's own reward, landing bonus and crash penalty included. With `beta = 0.999` and the physical horizon
`H = 1000`:

```python
spec = BudgetSpec(budgets=[22.0], beta=0.999, ubar=100.0, u_minus=100.0, horizon=1000)
stage = Stage(m=0.5, clip_utility=False)     # the crash penalty lies below every price: frozen-multiplier form
env = BudgetWrapper(gym.make("LunarLanderContinuous-v3"), cost=fuel_used, spec=spec, stage=stage)
```

The failing frame receives the tail `m (1 - beta^(H - t))/(1 - beta)` of the price (`barrierpg.spec.failure_tail`). The
toolbox's tests check that the wrapper reproduces the paper's fuel-tank environment step by step.

### The campaign and what came out

Training with the tank from scratch never discovers landing, because random exploration exhausts the tank within
about a hundred frames. Each configuration therefore continues the training of a controller trained without a tank
(`5 x 10^6` steps, then `3 x 10^6` with the tank and the price), which is also how a budget is typically imposed on an
existing controller. Five seeds; every configuration of a seed starts from the same controller.

| Tank 22 units | exhausted (%) | landed (%) | crashed (%) | `U` |
|---|---|---|---|---|
| no tank | 0.0 | 96.2 | 3.7 | 237 |
| `m = 0` | 11.4 | 82.0 | 6.7 | 220 |
| `m = 0.5` | 5.7 | 77.9 | 16.5 | 205 |
| `m = 2` | 10.9 | 40.8 | 48.3 | 123 |
| adaptive, target 0.05, dual step 5 | 5.8 | 72.2 | 22.1 | 193 |
| adaptive, target 0.05, dual step 100 (fixed before the campaign) | 78.3 | 5.0 | 16.7 | 91 |

1. **A moderate price works on a learned controller.** At `m = 0.5` exhaustion falls in nearly every pair of priced and
   unpriced runs with the same seed and tank, while crashes rise and the utility falls.
2. **Do not charge far more for a failure than an episode earns.** At `m = 2` the failing frame costs about ten times
   the landing bonus, several seeds crash in most episodes, and both exhaustion and utility worsen.
   `plan(..., episode_earnings=...)` warns.
3. **Size the dual step.** With the step 100 fixed before the campaign the multiplier increased without bound: the continued
   controller first exhausts the tank in every episode, so the flat penalty carries no gradient. With step 5 the
   multiplier converged near the penalty of the fixed price `m = 0.5`. Take the step from `callbacks.dual_step_suggestion` with the penalty
   of the fixed-price run that attains about the tolerance, and read the trace with `DualAscent.diagnose` (*increasing*
   is the diagnosis of the step-100 runs). This is step-size sensitivity, not a ranking of the two mechanisms.
4. **Small tolerances.** The target 0.02 was missed by both methods: learned policies resolve exhaustion
   probabilities below about five percent only up to their optimization error. Validate the deployed candidate.

---

## 4. The roles of the regularizers

### On the solvable flow

The paper's solvable flow (`barrierpg.examples.plateau`) has one parameter per unit and a feature that is flat on
`[-1, 1]`. From the plateau initial law the value gradient vanishes, so only the regularizers move the units. With the
prior `N(0, 1)`: relative entropy with strength `kappa1` (pull and noise tied), the quadratic potential with strength
`kappa2` (pull only), and their sum.

| Regularizer | `kappa1` (noise) | `kappa2` (pull) | final `V_m` | final spread | OU reference |
|---|---|---|---|---|---|
| relative entropy | 0.1 | 0 | -1.9644 | 2.288 | 1.000 |
| relative entropy | 1 | 0 | -2.2176 | 1.071 | 1.000 |
| potential | 0 | 1 | -2.2533 | 0.000 | 0.000 |
| sum | 0.1 | 1 | -2.2533 | 0.304 | 0.302 |
| sum | 0.1 | 10 | -2.2533 | 0.100 | 0.100 |

The plateau value is -2.2533. The potential alone collapses the units and never improves; relative entropy escapes;
the sums stay on the plateau while the spread is small against the plateau's width. Without the value drift, the sum
is an Ornstein-Uhlenbeck process with stationary spread `sqrt(kappa1/(kappa1+kappa2))` (the OU reference); it gives a
scale for escape from the plateau, not a sharp threshold. In the strongly confined sums the spreads stay close to that
reference; relative entropy alone does not stay at its reference 1, since the value drift changes the spread once the
units leave the plateau.

```python
from barrierpg.examples.plateau import PlateauModel, simulate_particles
run = simulate_particles(PlateauModel(), kappa1=0.1, kappa2=1.0, n=2000, seed=1)
```

The same flow tests the convergence theorem (*Fixed-stage population convergence*): `C_m = 1940`, certified strength `kappa = 2037`
(`convergence.curvature_budget`), every flow converging exponentially far below that threshold, and the minimizer at
the certified strength essentially the prior.

### On a learned controller

In the setting of the parameter-anchor walk-through above with the matched prior, Gaussian noise of variance `2 kappa1 x lr` was added to every
actor parameter after each optimizer step (`ParameterNoise`), at the pull `kappa2 = 10`:

| Configuration | `kappa1` | `kappa2` | reference spread | `q_beta` (%) | `U` | measured spread |
|---|---|---|---|---|---|---|
| pull alone | 0 | 10 | -- | 3.35 | 2.89 | 3.2e-4 |
| pull with noise | 0.001 | 10 | 0.01 | 3.17 | 2.89 | 1.2e-3 |
| pull with noise | 0.009 | 10 | 0.03 | 3.24 | 2.88 | 1.1e-2 |
| pull with noise | 0.101 | 10 | 0.1 | 13.77 | 3.08 | 1.3e-1 |
| pull with noise | 0.989 | 10 | 0.314 | 16.24 | 2.36 | 1.3 |
| noise alone | 0.101 | 0 | -- | 50.62 | 5.24 | 1.5 |
| unanchored control | 0 | 0 | -- | 35.02 | 4.37 | 4.4e-2 |

The reference spread is `sqrt(kappa1/kappa2)`: the injected noise carries no pull of its own, so only the anchor
holds the parameters (`roles.noise_and_anchor_spread`). The anchor's gradient was added to the actor gradient before
the Adam step (the coupled form), and the measured spread grew in proportion to `kappa1` over three decades and crossed
the reference curve. The paper proposes, after the runs, the interaction of the coupled pull with Adam's normalization
of each coordinate as a possible explanation. Risk and utility stayed at the anchored values while the spread was small
against the incumbent's dispersion and moved once it was not, the seeds splitting; noise alone acted as exploration and
abandoned the prior. Vary `kappa1` and measure (`ParameterNoise.spread`, `regularizer_effects(..., noise_kappa1=, anchor_kappa2=, spread=)`).

After revision 68 the same rows were rerun with the pull decoupled from Adam (applied to the parameters after the
step, the toolbox's default from version 0.6.1) and under plain SGD, with a zero-objective control
(`wear-allowance/PROTOCOL-optimizers-v68.md`, `REPORT-optimizers-v68.md`). The formula survives once the pull is a
gradient step: the spread is `sqrt(kappa1 / kappa2)` within 1% under SGD and under the decoupled pull (0.100 and 0.318
at `kappa2 = 10` and `1` with `kappa1 = 0.101`; the noise here carries no pull of its own, so the stationary spread is
`sqrt(kappa1 / kappa2)` rather than `sqrt(kappa1 / (kappa1 + kappa2))`), while the coupled pull gives Adam's Laplace law,
0.128 at both pulls. The coupled pull without noise holds the policy where SGD's does (`kappa2 = 10` keeps the
incumbent in both), but the decoupled pull at the same strength and Adam's learning rate does not: read a decoupled
strength through its per-step contraction `1 - lr kappa2` (`ParameterAnchor.half_life`), not through the strength the
coupled runs used.

---

## 5. A task of a benchmark suite (online appendix)

### Which tasks qualify

A task of a safe-reinforcement-learning suite falls in the paper's class only when its per-step cost is the
consumption of a resource whose exhaustion is the failure, so that a tolerated probability of exhausting the episode's
cost limit is what the task means. In Safety-Gymnasium 1.0.0 only the navigation tasks with obstacles (Goal, Button
and Push at levels 1 and 2, and the vision tasks with hazards) qualify, with the cost limit read as an allowance of
damage. In the other families the cost is zero, or it marks a state that should never be entered (a position outside
the walls, a speed above a limit, a joint outside its range); their requirement holds at every step, which is a
statewise requirement, not a budget. The suite's own formulation of the qualifying tasks limits the expected episode
cost, a different requirement: an expected cost of 25 allows exhausting a budget of 25 in half of the episodes.

### The mapping

SafetyPointGoal1-v0: the budget is the suite's cost limit of 25 per episode, the consumption is the per-step hazard
cost, and the requirement is a tolerated exhaustion probability `alpha` within the episode of 1000 steps.

```python
import safety_gymnasium
from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium     # moves the step cost into info["cost"]

spec = BudgetSpec(budgets=[25.0], beta=0.999, ubar=1.0, horizon=1000)
stage = Stage(m=0.3, clip_utility=False)     # the suite's reward before failure; the tail of the price at the exhausting step
base = SafetyGymnasium2Gymnasium(safety_gymnasium.make("SafetyPointGoal1-v0"))
env = BudgetWrapper(base, cost=lambda obs, action, next_obs, reward, info: [info["cost"]], spec=spec, stage=stage)
```

### What came out

Eight seeds of PPO per configuration, `2 x 10^6` steps from scratch, 200 fresh episodes per final policy, and
validation with the error `0.05` shared among all 72 reported candidates (`0.05/72`).

| Configuration | exhausted (%) | return | mean cost | pass at 0.2 | pass at 0.05 |
|---|---|---|---|---|---|
| unpriced, `m = 0` | 50.2 | 15.6 | 17.2 | 0/8 | 0/8 |
| fixed price `m = 0.03` | 10.3 | 1.7 | 3.7 | 5/8 | 0/8 |
| fixed price `m = 0.1` | 3.1 | -0.5 | 1.1 | 8/8 | 0/8 |
| fixed price `m = 0.3` | 1.2 | -0.8 | 0.7 | 8/8 | 4/8 |
| fixed price `m = 1` | 1.4 | -0.9 | 0.7 | 8/8 | 2/8 |
| adaptive multiplier, tolerance 0.05 | 11.3 | 7.5 | 6.2 | 4/8 | 0/8 |
| adaptive multiplier, tolerance 0.2 | 22.1 | 10.7 | 10.0 | 0/8 | 0/8 |
| expected cost 25, dual step 0.5 | 6.3 | -2.4 | 2.7 | 8/8 | 1/8 |
| expected cost 25, dual step 0.01 (added after the campaign) | 10.6 | 0.7 | 4.1 | 6/8 | 0/8 |

Means over seeds. Full table: the benchmark-suite study in the online appendix.

1. **The price lowers exhaustion, and its cost is visible.** Exhaustion falls with the price from half of the episodes.
   From `m = 0.1` on, the price charges more for a failure than an episode earns, and the return falls to zero or
   below: the condition that `plan(..., episode_earnings=...)` flags.
2. **The value certificate is valid but uninformative here.** The utility bound times the discounted horizon exceeds
   every attained return by two orders of magnitude; the direct binomial test carries the validation.
3. **Size the dual step.** The adaptive multiplier at 0.2 converges to its tolerance, and at 0.05 it rises throughout
   training. Both expected-cost arms met an initial cost far above the limit with a multiplier that rose far above the
   utility of a step; neither realized a policy at the limit, so the comparison between the two requirements at the
   same mean cost remains untested.
4. **Qualify the task before running it.** The requirement comes first; a task whose cost marks forbidden states is
   statewise and calls for another method.

