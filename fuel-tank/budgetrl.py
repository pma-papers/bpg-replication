"""budgetrl: barrier policy gradients under resource budgets as drop-in wrappers (revision 39).

The objects of the paper in the form of an environment wrapper and optimizer callbacks, so
that standard policy-gradient software can be used without modification, in the manner of
Saute RL's state-augmentation wrappers.

  BudgetWrapper   gymnasium wrapper: augments the observation with the remaining reserves
                  z_{t+1,k} = (z_{t,k} - g_k) / gamma_k, replaces the reward by the clipped
                  barrier reward r_m = max{-m, u - B(z_{t+1})}, and at failure (some reserve
                  negative) terminates the episode with the discounted tail of the per-step
                  penalty: -m/(1-beta) (absorbing, an infinite physical horizon) or
                  -m (1 - beta^(H-t)) / (1-beta) (physical horizon H). With theta = 0 the
                  barrier is absent and the reward is a failure-only price. The wrapper also
                  supports a prescribed-level multiplier (a flat penalty -lambda at failure,
                  settable during training) for dual-ascent methods.
                  info carries: failure, utility, interior_loss, cost, reserve.
  DualAscent      SB3 callback: lambda <- max(0, lambda + step (observed failure rate - alpha))
                  once per rollout, pushed into every environment through env_method.
  ActionAnchor    SB3 callback: reward shaping -eta [log pi(a|y) - log rho(a|y)] towards a
                  reference action kernel rho (the stopped action relative entropy of the
                  paper, charged on the fly).
  certify         the value-based and direct certificates: geometric-risk bound
                  q <= (ubar - (1-beta) V_m) / (m + ubar), its finite-horizon form, the
                  price--risk law with an idle or given comparator, and exact binomial
                  validation with the resolution floor.

Conventions follow the manuscript: z_0 = b, failure at the first action index tau with
min_k z_{tau+1,k} < 0 (equality is not failure), stopped utility U = E sum_{t<tau} beta^t u_t,
interior loss I_m = E sum_{t<tau} beta^t min{B(z_{t+1}), u_t + m}, and
V_m = U - I_m - m q_beta / (1 - beta).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None


def log_barrier(theta: float, budgets: np.ndarray) -> Callable[[np.ndarray], float]:
    """B(z) = theta sum_k log(1 + b_k / z_k) on the positive orthant, +inf elsewhere."""
    b = np.asarray(budgets, dtype=float)

    def B(z):
        z = np.asarray(z, dtype=float)
        if theta <= 0.0:
            return 0.0
        if np.any(z <= 0.0):
            return math.inf
        return float(theta * np.sum(np.log1p(b / z)))

    return B


class BudgetWrapper(gym.Wrapper if gym is not None else object):
    """See the module docstring.

    cost(obs, action, next_obs, reward, info) -> array of K nonnegative costs for the step.
    utility(obs, action, next_obs, reward, info) -> the one-step utility u; default: reward.
    budgets: K positive numbers; gammas: K cost discounts in (0, 1]; beta: utility discount.
    m: price (clipping level); theta: barrier scale (0 = failure-only price);
    horizon: None for the absorbing implementation, an integer H for the physical-horizon tail.
    penalty: 'price' (the clipped objective) or 'multiplier' (flat -lambda at failure).
    clip_utility: True gives the paper's r_m = max{-m, u - B}; False leaves the pre-failure
    reward u - B unclipped, the frozen-multiplier form U - I - m q/(1-beta) in which only the
    failure is priced. Use False when the task's own utility takes values below -m (for
    example a crash penalty), which clipping would otherwise erase.
    """

    def __init__(self, env, cost, budgets, *, beta, m=0.0, theta=0.0, gammas=None, utility=None,
                 horizon=None, penalty="price", multiplier=0.0, observe="fraction", clip_utility=True):
        super().__init__(env)
        self.clip_utility = bool(clip_utility)
        self.cost_fn, self.utility_fn = cost, utility
        self.budgets = np.asarray(budgets, dtype=float).reshape(-1)
        self.K = len(self.budgets)
        self.gammas = np.ones(self.K) if gammas is None else np.asarray(gammas, dtype=float).reshape(-1)
        self.beta, self.m, self.theta, self.horizon = float(beta), float(m), float(theta), horizon
        self.penalty, self.multiplier, self.observe = penalty, float(multiplier), observe
        self.B = log_barrier(self.theta, self.budgets)
        low = np.append(env.observation_space.low, np.zeros(self.K)).astype(np.float32)
        hi = np.ones(self.K) if observe == "fraction" else self.budgets
        high = np.append(env.observation_space.high, hi).astype(np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    # --- multiplier interface for dual ascent ---------------------------------------------
    def set_multiplier(self, value):
        self.multiplier = float(value)

    def set_price(self, value):
        self.m = float(value)

    # --- gym API ----------------------------------------------------------------------------
    def _obs(self, o):
        z = np.maximum(self.reserve, 0.0)
        extra = z / self.budgets if self.observe == "fraction" else z
        return np.append(o, extra).astype(np.float32)

    def reset(self, **kwargs):
        o, info = self.env.reset(**kwargs)
        self.reserve = self.budgets.copy()
        self.t = 0
        self.used = np.zeros(self.K)
        self.last_obs = o
        return self._obs(o), info

    def failure_reward(self):
        if self.penalty == "multiplier":
            return -self.multiplier
        if self.horizon is None:
            return -self.m / (1.0 - self.beta)
        return -self.m * (1.0 - self.beta ** (self.horizon - self.t)) / (1.0 - self.beta)

    def step(self, action):
        o, r, terminated, truncated, info = self.env.step(action)
        g = np.asarray(self.cost_fn(self.last_obs, action, o, r, info), dtype=float).reshape(-1)
        u = float(r) if self.utility_fn is None else float(self.utility_fn(self.last_obs, action, o, r, info))
        self.reserve = (self.reserve - g) / self.gammas
        self.used += g
        failed = bool(np.any(self.reserve < 0.0))
        if failed:
            reward, utility, interior = self.failure_reward(), 0.0, 0.0
            terminated = True
        else:
            Bz = self.B(self.reserve)
            reward = max(-self.m, u - Bz) if self.clip_utility else u - Bz
            utility, interior = u, (min(Bz, u + self.m) if self.clip_utility else Bz)
        self.t += 1
        self.last_obs = o
        info = dict(info)
        info.update(failure=failed, utility=utility, interior_loss=interior, cost=g.copy(),
                    reserve=self.reserve.copy(), used=self.used.copy())
        if terminated or truncated:
            info["episode_failed"] = failed
        return self._obs(o), float(reward), bool(terminated), bool(truncated), info


# --------------------------------------------------------------------------------------------
# Stable-Baselines3 callbacks
# --------------------------------------------------------------------------------------------
try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover
    BaseCallback = object


class DualAscent(BaseCallback):
    """Prescribed-level multiplier: after each rollout, lambda <- max(0, lambda + step (rate - alpha)),
    where rate is the failure frequency among the episodes that ended in the rollout. The
    environments must be BudgetWrapper(..., penalty='multiplier')."""

    def __init__(self, alpha, step, multiplier0=0.0, verbose=0):
        super().__init__(verbose)
        self.alpha, self.step, self.multiplier = float(alpha), float(step), float(multiplier0)
        self.episodes = self.failures = 0
        self.trace = []

    def _on_training_start(self):
        self.training_env.env_method("set_multiplier", self.multiplier)

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode_failed" in info:
                self.episodes += 1
                self.failures += int(info["episode_failed"])
        return True

    def _on_rollout_end(self):
        rate = self.failures / self.episodes if self.episodes else float("nan")
        if self.episodes:
            self.multiplier = max(0.0, self.multiplier + self.step * (rate - self.alpha))
            self.training_env.env_method("set_multiplier", self.multiplier)
        self.trace.append(dict(steps=int(self.num_timesteps), multiplier=self.multiplier,
                               episodes=self.episodes, failure_rate=rate))
        self.episodes = self.failures = 0


class ActionAnchor(BaseCallback):
    """Charges -eta [log pi(a|y) - log rho(a|y)] on every pre-failure step, where rho is a
    reference kernel given by log_reference(obs_batch, action_batch) -> log densities.
    Requires an on-policy algorithm with a Gaussian policy (PPO); the shaping is applied to the
    rewards before they enter the rollout buffer, so the critic sees the regularized return."""

    def __init__(self, eta, log_reference, verbose=0):
        super().__init__(verbose)
        self.eta, self.log_reference = float(eta), log_reference
        self.total = 0.0; self.count = 0

    def _on_step(self):
        import torch
        obs, act = self.locals["obs_tensor"], self.locals["actions"]
        with torch.no_grad():
            logp = self.model.policy.get_distribution(obs).log_prob(torch.as_tensor(act, device=obs.device)).cpu().numpy()
        ratio = logp - np.asarray(self.log_reference(obs.cpu().numpy(), np.asarray(act)), dtype=float)
        alive = np.array([not info.get("failure", False) for info in self.locals["infos"]])
        self.locals["rewards"][:] -= self.eta * ratio * alive
        self.total += float(np.sum(ratio * alive)); self.count += int(alive.sum())
        return True


# --------------------------------------------------------------------------------------------
# Certificates
# --------------------------------------------------------------------------------------------
def risk_from_value(V_m, m, ubar, beta, T=None):
    """Value-based certificate: q_beta <= (ubar - (1-beta) V_m) / (m + ubar); with T, the
    finite-horizon bound p_T <= min{1, that / beta^T}."""
    q = (ubar - (1.0 - beta) * V_m) / (m + ubar)
    q = min(1.0, max(0.0, q))
    if T is None:
        return q
    return min(1.0, q / beta ** T)


def price_risk_law(m, beta, sup_U, U0=0.0, I0=0.0, q0=0.0, delta=0.0, ubar=None):
    """Price--risk law: an (approximate, error delta) optimizer of the failure-only clipped
    objective has q_beta <= q0 + (1-beta)[sup_U - U0 + I0 + delta]/m, where (U0, I0, q0) belong
    to any comparator; the idle comparator gives q0 = U0 = I0 = 0. With ubar, the variant
    q <= [m q0 + ubar - (1-beta) U0 + (1-beta)(I0 + delta)] / (m + ubar), which needs no
    sup_U, is returned as the second value."""
    base = (1.0 - beta) * (sup_U - U0 + I0 + delta)
    law = min(1.0, q0 + base / m)
    if ubar is None:
        return law
    variant = (m * q0 + ubar - (1.0 - beta) * U0 + (1.0 - beta) * (I0 + delta)) / (m + ubar)
    return law, min(1.0, variant)


def binomial_upper(k, n, err):
    """Exact one-sided (Clopper-Pearson) upper confidence endpoint for a failure probability."""
    from scipy.stats import beta as beta_dist
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - err, k + 1, n - k))


def resolution_floor(n, err):
    """Smallest tolerance that n failure-free validation episodes can certify at error err:
    the exact binomial upper endpoint with zero failures, 1 - err^(1/n)."""
    return 1.0 - err ** (1.0 / n)


def validate(failures, n, alpha, err, n_statements=1):
    """Direct validation of a tolerance alpha with simultaneous error err over n_statements
    policies: returns (pass, upper endpoint, resolution floor)."""
    e = err / n_statements
    up = binomial_upper(failures, n, e)
    return up <= alpha, up, resolution_floor(n, e)


__all__ = ["BudgetWrapper", "DualAscent", "ActionAnchor", "log_barrier", "risk_from_value",
           "price_risk_law", "binomial_upper", "resolution_floor", "validate"]
