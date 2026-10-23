"""Gymnasium environment and policy adapters for the wear-allowance example.

Episodes end at a breakdown or at maintenance access, which arrives after each period with
probability 1 - beta. With algorithmic discount one, the expected episode sum of rewards
therefore equals the beta-discounted objective of the manuscript.
Observation: (price index s, 2 z / b - 1). Action x in [-1, 1] maps to the rate (x + 1) / 2.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from model import Model, barrier, utility, wear


def to_rate(x):
    return (np.clip(np.asarray(x, dtype=float), -1.0, 1.0) + 1.0) / 2.0


def observe(model: Model, s, z):
    return np.column_stack((np.asarray(s), 2.0 * np.maximum(np.asarray(z), 0.0) / model.budget - 1.0)).astype(np.float32)


class WearEnv(gym.Env):
    """objective: 'failure' (price m, m = 0 is the unpriced learner), 'barrier' (price m),
    or 'chance' (failure indicator charged by an externally updated multiplier)."""
    metadata = {"render_modes": []}

    def __init__(self, model: Model, objective="failure", m=1.0, multiplier=0.0):
        super().__init__()
        self.model, self.objective, self.m = model, objective, float(m)
        self.multiplier = float(multiplier)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def set_multiplier(self, value):
        self.multiplier = float(value)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s, self.z = 0.0, self.model.budget
        return observe(self.model, [self.s], [self.z])[0], {}

    def step(self, action):
        M = self.model
        a = float(to_rate(action).reshape(-1)[0])
        xi = self.np_random.gamma(M.wear_shape, 1.0 / M.wear_shape)
        z_next = self.z - float(wear(M, a, xi))
        u = float(utility(M, self.s, a))
        failed = z_next < 0
        if failed:
            lump = self.multiplier if self.objective == "chance" else self.m / (1.0 - M.beta)
            reward = -lump
        elif self.objective == "barrier":
            reward = max(u - float(barrier(M, np.array([z_next]))[0]), -self.m)
        else:
            reward = u
        self.s = M.rho * self.s + (1 - M.rho) * self.np_random.uniform(-1.0, 1.0)
        self.z = z_next
        access = self.np_random.random() > M.beta
        info = dict(failure=bool(failed), utility=0.0 if failed else u)
        if failed or access:
            info["episode_failed"] = bool(failed)
        return observe(M, [self.s], [self.z])[0], float(reward), bool(failed or access), False, info


def model_policy(model: Model, sb3_model, deterministic=False):
    """Adapter from a Stable-Baselines3 model to a rate policy a = policy(s, z)."""
    def policy(s, z):
        x, _ = sb3_model.predict(observe(model, s, z), deterministic=deterministic)
        return to_rate(x).reshape(-1)
    return policy


class Incumbent:
    """Conservative proportional rule used as the reference action kernel: operate at a
    rate proportional to the remaining allowance, a = min(1, z / (span * b)), with Gaussian
    dispersion in the pre-clipping action variable."""

    def __init__(self, model: Model, span=1.5, std=0.2):
        self.model, self.span, self.std = model, span, std

    def mean(self, z):
        rate = np.minimum(1.0, np.maximum(np.asarray(z), 0.0) / (self.span * self.model.budget))
        return 2.0 * rate - 1.0

    def log_density(self, x, z):
        mu = self.mean(z)
        return -0.5 * ((np.asarray(x) - mu) / self.std) ** 2 - np.log(self.std) - 0.5 * np.log(2 * np.pi)

    def policy(self, seed=0):
        rng = np.random.default_rng(seed)

        def policy(s, z):
            return to_rate(self.mean(z) + self.std * rng.standard_normal(len(np.atleast_1d(z))))
        return policy

    def kl_from(self, mean_pi, std_pi, z):
        """KL( N(mean_pi, std_pi^2) || reference ) at reserve z, in the pre-clipping variable."""
        mu = self.mean(z)
        return np.log(self.std / std_pi) + (std_pi ** 2 + (mean_pi - mu) ** 2) / (2 * self.std ** 2) - 0.5
