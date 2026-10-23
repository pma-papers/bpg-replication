# Wear-allowance study (revision 38): protocol

Fixed on 2026-10-02 before any campaign outcome was inspected. Pilot runs (stages
`pilot*`, seeds 91001-91009) chose the training budget, batch size, learning-rate
schedule and the incumbent rule; they are excluded from all reported results.

## Model

Derated operation of an asset under a wear allowance (`model.py`). Price index
s' = 0.8 s + 0.2 w, w ~ Unif[-1,1]; operating rate a in [0,1]; utility
u = (1 + 0.5 s) a / 1.5 in [0,1]; wear g = 0.1 a xi with xi ~ Gamma(0.5, scale 2)
(mean one, coefficient of variation 1.41); allowance b = 1, resource discount one.
A breakdown is a strictly negative remaining allowance. Maintenance access arrives
after each period with probability 1 - beta, beta = 0.95, so the geometric risk
q_beta is the probability of a breakdown before maintenance access. The idle policy
is hard feasible; every positive rate has positive breakdown probability.

## Exact benchmark

`benchmark.py` solves the stationary problem on a grid (21 price nodes, 161 reserve
nodes dense near exhaustion, 81 rates, exact survival probabilities, 16 conditional
shock nodes, 8 price-shock nodes) for failure-only prices m in
{0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30}, for the interior
barrier with scale theta in {0.03, 0.1, 0.3} and m in {0.3, 1, 3, 10}, and for
failure-only plans made with a tightened allowance (0.9 b and 0.8 b). The failure-only
family is also the frozen-multiplier Lagrangian family U - lambda q with
lambda = m / (1 - beta).

## Learning arms (pinned Stable-Baselines3 2.7.1 PPO, runtime numerics-v22/.venv)

Fifteen arms, eight training seeds each (92001-92008), 120 runs:
- fixed price, failure-only: m in {0, 0.1, 0.3, 1, 3, 10} (m = 0 is the unpriced learner);
- fixed price with interior barrier theta = 0.1: m in {1, 3};
- adaptive multiplier (projected dual ascent on the breakdown frequency of completed
  episodes in each rollout): targets alpha = 0.05 with dual steps 5 and 1, alpha = 0.01
  and alpha = 0 with dual step 5; multiplier started at zero;
- incumbent-anchored learning at the under-priced m = 0.03: the stopped action relative
  entropy to a conservative proportional rule (rate z / 3b, pre-clipping standard deviation
  0.1) with weights eta in {0.01, 0.1, 1}.

Every run: 8,000,000 environment steps, 32 environments, 256 steps per rollout,
minibatch 1024, 10 epochs, learning rate 3e-4 decayed linearly to zero, algorithmic
discount one (episodes end at breakdown or maintenance access), GAE 0.95, clip 0.2,
no entropy bonus, two 64-unit layers, initial log standard deviation -1, reward
normalization by the running return scale. No checkpoint or seed selection.

## Evaluation

The final stochastic policy of each run is the candidate. It receives 20,000 fresh
episodes with geometric maintenance access (breakdown indicator, stopped utility,
interior loss, near-exhaustion frequencies), and 20,000 each under an allowance
reduced to 0.9 b and 0.8 b while the policy observes the nominal reserve. The
mean-action policy is evaluated separately as a different candidate. Anchored runs
also estimate the stopped action relative entropy with closed-form Gaussian terms.
A run passes a tolerance alpha when its one-sided exact binomial upper bound, at
error 0.05 divided by the number of candidates in the reported family, is at most alpha.

## Predictions under test (stated before the campaign)

P1. The unpriced learner fails the 5 percent tolerance; attained risk decreases with the price.
P2. Fixed-price and adaptive-multiplier learners lie on the same exact frontier up to
    optimization error; neither is expected to dominate.
P3. A zero-risk target makes the multiplier grow without settling.
P4. The interior barrier lowers near-exhaustion frequencies and the breakdown probability
    under a reduced allowance, at a utility cost equal to its interior loss.
P5. The action-regularization bound kl(q || q_ref) <= A holds and is informative only for
    small A.
All outcomes are reported, including those that contradict a prediction.
