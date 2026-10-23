"""Derated operation under a wear allowance (revision 38 flagship example).

An asset earns revenue from its operating rate and accumulates wear. A breakdown
(cumulative wear above the allowance) before the next maintenance access stops
production. Maintenance access arrives each period with probability 1 - beta, so the
discount factor is the survival probability of the planning horizon and the geometric
risk q_beta equals the probability of a breakdown before maintenance access.

State: price index s in [-1, 1], remaining wear allowance (reserve) z in [0, b].
Action: operating rate a in [0, 1].
    utility      u(s, a) = (1 + price_swing * s) * a / (1 + price_swing)   in [0, 1]
    wear         g(a, xi) = wear_rate * a * xi,  xi ~ Gamma(wear_shape, 1/wear_shape), mean 1
                 (wear is proportional to throughput, with a random wear per unit)
    reserve      z' = z - g           (resource discount gamma = 1)
    price        s' = rho * s + (1 - rho) * w,  w ~ Unif[-1, 1]   (so |s| <= 1 is invariant)
    failure      z' < 0 (strict); the failing action earns no utility
The idle policy a = 0 is hard feasible with zero utility. Every positive rate has a
positive breakdown probability because the wear shock is unbounded.

Rewards (per period, before discounting by beta):
    utility objective    u                         before failure, 0 at failure
    failure-only price   u                         before failure, -m/(1-beta) at failure
    interior barrier     max(u - B(z'), -m)        before failure, -m/(1-beta) at failure
with B(z') = theta * log(1 + b / z') for z' > 0 and +infinity at z' = 0.
The lump sum -m/(1-beta) is the discounted tail of the per-period downtime penalty m.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Model:
    beta: float = 0.95          # survival probability of the planning horizon per period
    budget: float = 1.0         # wear allowance b
    wear_rate: float = 0.10     # expected wear per period at full rate
    wear_shape: float = 2.0     # shape of the gamma wear shock (unbounded, mean one)
    rho: float = 0.8            # persistence of the price index
    price_swing: float = 0.5    # price = 1 + price_swing * s
    theta: float = 0.1          # barrier scale

    def to_dict(self):
        return asdict(self)


def utility(model: Model, s, a):
    return (1.0 + model.price_swing * s) * a / (1.0 + model.price_swing)


def wear(model: Model, a, xi):
    return model.wear_rate * a * xi


def barrier(model: Model, z_next):
    z = np.asarray(z_next, dtype=float)
    out = np.full(z.shape, np.inf)
    pos = z > 0
    out[pos] = model.theta * np.log1p(model.budget / z[pos])
    return out


def simulate(model: Model, policy, *, n, seed, m=1.0, horizon=None, budget_scale=1.0,
             max_steps=2000):
    """Monte Carlo evaluation of a policy a = policy(s, z_observed).

    horizon=None: geometric maintenance access (each period ends the episode with
    probability 1 - beta); the reported risk is then q_beta = P(breakdown before access)
    and all discounted sums are estimated without discount weights.
    horizon=T: fixed horizon of T+1 actions with discount weights beta**t (no random stop).

    budget_scale < 1 evaluates robustness: the policy observes the nominal reserve
    (allowance b), while breakdown occurs when cumulative wear exceeds budget_scale * b.
    """
    rng = np.random.default_rng(seed)
    b = model.budget
    s = rng.uniform(-1.0, 1.0, n) * 0.0  # start at the mean price
    z_nominal = np.full(n, b)
    true_limit = budget_scale * b
    cum = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    failed = np.zeros(n, dtype=bool)
    tau = np.full(n, -1)
    U = np.zeros(n)            # stopped utility
    I = np.zeros(n)            # interior loss sum_{t<tau} w_t min{B(z'), u + m}
    D = np.zeros(n)            # failure weight (1 or beta**tau)
    min_reserve = np.full(n, b)
    steps = 0
    t = 0
    T = max_steps if horizon is None else horizon + 1
    while alive.any() and t < T:
        ids = np.flatnonzero(alive)
        a = np.clip(policy(s[ids], np.maximum(z_nominal[ids], 0.0)), 0.0, 1.0)
        xi = rng.gamma(model.wear_shape, 1.0 / model.wear_shape, len(ids))
        g = wear(model, a, xi)
        cum_next = cum[ids] + g
        z_next_nominal = z_nominal[ids] - g
        fail = cum_next > true_limit
        w = 1.0 if horizon is None else model.beta ** t
        u = utility(model, s[ids], a)
        ok = ~fail
        U[ids[ok]] += w * u[ok]
        zn = np.maximum(z_next_nominal[ok], 0.0)
        I[ids[ok]] += w * np.minimum(barrier(model, zn), u[ok] + m)
        D[ids[fail]] = w
        failed[ids[fail]] = True
        tau[ids[fail]] = t
        true_reserve = true_limit - cum_next
        min_reserve[ids] = np.minimum(min_reserve[ids], true_reserve)
        cum[ids] = cum_next
        z_nominal[ids] = z_next_nominal
        s[ids] = model.rho * s[ids] + (1 - model.rho) * rng.uniform(-1, 1, len(ids))
        alive[ids[fail]] = False
        steps += len(ids)
        if horizon is None:
            stop = rng.random(len(ids)) > model.beta
            alive[ids[stop & ok]] = False
        t += 1
    risk = float(D.mean())
    k = int(failed.sum())
    return dict(n=n, seed=seed, m=m, horizon=horizon, budget_scale=budget_scale,
                risk=risk, failures=k, U=float(U.mean()), U_se=float(U.std(ddof=1) / np.sqrt(n)),
                I=float(I.mean()), steps=int(steps),
                min_reserve_q=np.quantile(min_reserve, [0.01, 0.05, 0.25, 0.5]).tolist(),
                near={str(e): float(np.mean(min_reserve <= e * b)) for e in (0.02, 0.05, 0.1, 0.2)})
