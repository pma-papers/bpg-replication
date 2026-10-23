"""Shared conventions for the numerical study of the bounded barrier objective.

Conventions follow the manuscript (barrier-policy-gradients-anonymous-v4.tex):
  z_0 = b,  z_{t+1} = (z_t - g(s_t,a_t)) / gamma,  tau = inf{t >= 0 : z_{t+1} < 0},
  r_m = max{-m, u - B(z_{t+1})},  B = theta * log(1 + b/z) on z > 0, +inf otherwise,
  V_m = E sum beta^t r_m,  q_beta = E[beta^tau 1{tau<inf}],  p_T = P(tau <= T),
  U = E sum_{t<tau} beta^t u_t,  I_m = E sum_{t<tau} beta^t min{B(z_{t+1}), u_t + m},
  V_m = U - I_m - m q_beta/(1-beta).
Equality to the budget (z_{t+1} = 0) is not a failure, but the barrier is infinite there.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"


# ----------------------------------------------------------------------------
# Confidence bounds
# ----------------------------------------------------------------------------
def hoeffding_radius(N: int, zeta: float) -> float:
    """One-sided Hoeffding radius for N independent [0,1] variables."""
    return math.sqrt(math.log(1.0 / zeta) / (2.0 * N))


def binomial_upper(k: int, N: int, zeta: float) -> float:
    """Exact one-sided (Clopper-Pearson) upper confidence endpoint for a Bernoulli rate."""
    if k >= N:
        return 1.0
    return float(stats.beta.ppf(1.0 - zeta, k + 1, N - k))


def binomial_lower(k: int, N: int, zeta: float) -> float:
    if k <= 0:
        return 0.0
    return float(stats.beta.ppf(zeta, k, N - k + 1))


# ----------------------------------------------------------------------------
# Barrier and clipped reward
# ----------------------------------------------------------------------------
def log_barrier(z, b, theta=1.0):
    """B(z) = theta * log(1 + b/z) for z > 0, +inf for z <= 0 (elementwise, 1 budget)."""
    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, np.inf)
    pos = z > 0
    out[pos] = theta * np.log1p(b / z[pos])
    return out


def clipped_reward(u, B, m):
    """r_m = max{-m, u - B}; B may be +inf."""
    return np.maximum(-m, u - B)


# ----------------------------------------------------------------------------
# Trajectory bookkeeping (single budget)
# ----------------------------------------------------------------------------
def budget_identity_check(g_seq, b, gamma, tol=1e-9):
    """Check z_{t+1} = gamma^{-(t+1)} (b - C_t) with C_t = sum_{j<=t} gamma^j g_j."""
    g_seq = np.asarray(g_seq, dtype=float)
    z = b
    for t, g in enumerate(g_seq):
        z = (z - g) / gamma
        C = np.sum(gamma ** np.arange(t + 1) * g_seq[: t + 1])
        alt = gamma ** (-(t + 1)) * (b - C)
        if abs(z - alt) > tol * max(1.0, abs(alt)):
            return False
    return True


class TrajectoryStats:
    """Metrics from a batch of simulated trajectories (single budget coordinate).

    Inputs are arrays of shape (H, N): u[t,i], g[t,i], z_next[t,i] = z_{t+1} on
    trajectory i, and a failure time tau[i] in {0,...,H-1} or H (meaning tau >= H,
    i.e. no failure within the simulated horizon).
    All expectations are over the N trajectories; discounted sums are truncated
    at H, and the exact known tail -m beta^tau/(1-beta) is used at failure.
    """

    def __init__(self, u, g, z_next, tau, *, b, beta, gamma, m, theta_B=1.0,
                 gbar=None):
        self.u, self.g, self.z_next = np.asarray(u), np.asarray(g), np.asarray(z_next)
        self.tau = np.asarray(tau)
        self.H, self.N = self.u.shape
        self.b, self.beta, self.gamma, self.m, self.theta_B = b, beta, gamma, m, theta_B
        self.gbar = gbar
        t = np.arange(self.H)[:, None]
        self.alive = t < self.tau[None, :]            # t < tau: pre-failure actions
        self.disc = beta ** np.arange(self.H)[:, None]
        B = log_barrier(self.z_next, b, theta_B)
        self.B = B
        self.r = np.where(self.alive, clipped_reward(self.u, B, m), -m)

    # --- returns -------------------------------------------------------------
    def clipped_return_samples(self, truncated=False):
        """G_i = sum_{t<H} beta^t r_m(t). Unless truncated=True, the exactly known tail
        -m beta^H/(1-beta) is added for trajectories that failed inside the horizon, so that
        the decomposition identity holds exactly; survivors' tails remain omitted."""
        G = np.sum(self.disc * self.r, axis=0)
        if not truncated:
            G = G + np.where(self.tau < self.H, -self.m * self.beta ** self.H / (1 - self.beta), 0.0)
        return G

    def V_m(self):
        return float(np.mean(self.clipped_return_samples()))

    def stopped_utility_samples(self):
        return np.sum(self.disc * np.where(self.alive, self.u, 0.0), axis=0)

    def U(self):
        return float(np.mean(self.stopped_utility_samples()))

    def continuing_utility_samples(self):
        return np.sum(self.disc * self.u, axis=0)

    def interior_loss_samples(self):
        loss = np.minimum(self.B, self.u + self.m)
        return np.sum(self.disc * np.where(self.alive, loss, 0.0), axis=0)

    def I_m(self):
        return float(np.mean(self.interior_loss_samples()))

    # --- risks ---------------------------------------------------------------
    def q_beta_H(self):
        """E[beta^tau 1{tau < H}]; the omitted tail is at most beta^H."""
        failed = self.tau < self.H
        return float(np.mean(np.where(failed, self.beta ** self.tau, 0.0)))

    def p_T(self, T):
        return float(np.mean(self.tau <= T))

    def failure_indicators(self, T):
        return (self.tau <= T).astype(float)

    def decomposition_residual(self):
        """V_m - (U - I_m - m q_beta_H/(1-beta)), which must vanish up to rounding."""
        return self.V_m() - (self.U() - self.I_m() - self.m * self.q_beta_H() / (1 - self.beta))

    # --- reserve margins and eventual risk -----------------------------------
    def min_budget(self, T):
        """min_{t<=T} z_{t+1} over tested actions (failed trajectories give negative values)."""
        return np.min(self.z_next[: T + 1], axis=0)

    def near_exhaustion(self, eps, T):
        """Indicator of {exists t<=T: z_{t+1} <= eps}; includes failures."""
        return (np.min(self.z_next[: T + 1], axis=0) <= eps).astype(float)

    def unresolved(self, T):
        """E_T = {tau > T, z_{T+1} < gbar/(1-gamma)} (requires a cost envelope)."""
        assert self.gbar is not None
        env = self.gbar / (1 - self.gamma)
        return ((self.tau > T) & (self.z_next[T] < env)).astype(float)

    def summary(self, T, zeta=0.05):
        N = self.N
        G = self.clipped_return_samples()
        rad = hoeffding_radius(N, zeta)
        width = (self.m + 1.0) * (1 - self.beta ** self.H) / (1 - self.beta)
        return dict(
            N=N, H=self.H, T=T,
            V_m=self.V_m(), U=self.U(), I_m=self.I_m(),
            V=float(np.mean(self.continuing_utility_samples())),
            q_beta_H=self.q_beta_H(), q_beta_upper=self.q_beta_H() + self.beta ** self.H,
            p_T=self.p_T(T),
            decomposition_residual=self.decomposition_residual(),
            clipped_return_sd=float(np.std(G)),
        )


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def __call__(self):
        return time.perf_counter() - self.t0


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=default))


def load_json(path):
    return json.loads(Path(path).read_text())


def gauss_hermite_normal(n=64):
    """Nodes/weights for E_{N(0,1)} f(X) = sum w_i f(x_i)."""
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    g = rng.uniform(0, 0.1, size=40)
    assert budget_identity_check(g, b=0.6, gamma=0.95)
    assert binomial_upper(0, 20000, 0.05) < 1.5e-4
    assert abs(hoeffding_radius(20000, 0.05) - 0.00865409) < 1e-8
    print("common.py self-checks passed")
