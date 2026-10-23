"""Exact benchmark for the wear-allowance example by dynamic programming on a grid.

The stationary problem has state (s, z). Value iteration with modified policy
iteration solves, for a price m >= 0,
    failure-only:  maximize U - m q / (1 - beta)
    barrier:       maximize U - I_m - m q / (1 - beta)
where U is the stopped utility, q the geometric risk E[beta**tau] (the probability of a
breakdown before maintenance access) and I_m the interior loss. The failure-only family
is also the frozen-multiplier Lagrangian family U - lambda q with lambda = m / (1 - beta).

Grid values are exact for the interpolated grid model; reported comparisons re-evaluate
the interpolated policy table by Monte Carlo in the continuous simulator (model.simulate).
"""
from __future__ import annotations

import numpy as np

from scipy.stats import gamma as gamma_dist

from model import Model, barrier, utility


class Grid:
    def __init__(self, model: Model, ns=21, nz=161, na=81, nxi=16, nw=8):
        self.model = model
        self.s = np.linspace(-1.0, 1.0, ns)
        self.z = model.budget * np.linspace(0.0, 1.0, nz) ** 2      # dense near exhaustion
        self.a = np.linspace(0.0, 1.0, na)
        self.w = -1 + 2 * (np.arange(nw) + 0.5) / nw
        # price transition matrix on the s grid (linear interpolation, averaged over w)
        sn = model.rho * self.s[:, None] + (1 - model.rho) * self.w[None, :]
        pos = (sn + 1.0) / 2.0 * (ns - 1)
        lo = np.clip(np.floor(pos).astype(int), 0, ns - 2)
        fr = pos - lo
        P = np.zeros((ns, ns))
        for j in range(nw):
            np.add.at(P, (np.arange(ns), lo[:, j]), (1 - fr[:, j]) / nw)
            np.add.at(P, (np.arange(ns), lo[:, j] + 1), fr[:, j] / nw)
        self.P = P
        # Wear shock: the survival probability of each (z, a) is exact; the surviving part
        # is integrated with equal-probability nodes of the shock conditional on survival.
        k = model.wear_shape
        dist = gamma_dist(k, scale=1.0 / k)
        scale = model.wear_rate * self.a[None, :]                        # (1, na)
        with np.errstate(divide="ignore"):
            xmax = np.where(scale > 0, self.z[:, None] / np.where(scale > 0, scale, 1.0), np.inf)
        self.survive = dist.cdf(xmax)                                    # (nz, na)
        probs = (np.arange(nxi) + 0.5) / nxi
        xi = dist.ppf(probs[None, None, :] * self.survive[:, :, None])   # (nz, na, nxi)
        zn = self.z[:, None, None] - scale[:, :, None] * xi
        zc = np.clip(zn, 0.0, model.budget)
        hi = np.clip(np.searchsorted(self.z, zc, side="right"), 1, nz - 1)
        self.lo = hi - 1
        self.fr = (zc - self.z[self.lo]) / (self.z[hi] - self.z[self.lo])
        self.zn = zc
        self.u = utility(model, self.s[:, None], self.a[None, :])      # (ns, na)
        self.B = barrier(model, zc)                                      # (nz, na, nxi), inf at 0

    def expect_next(self, V):
        """E[V(s', z')] before masking failures; shape (ns, nz, na, nxi)."""
        Vbar = np.einsum('ij,jz->iz', self.P, V)                        # (ns, nz)
        return Vbar[:, self.lo] * (1 - self.fr) + Vbar[:, self.lo + 1] * self.fr

    def stage(self, objective, m):
        """Pre-failure reward, shape (ns, nz, na, nxi)."""
        u = np.broadcast_to(self.u[:, None, :, None], (len(self.s),) + self.zn.shape)
        if objective == "failure":
            return u
        if objective == "barrier":
            return np.maximum(u - self.B[None], -m)
        raise ValueError(objective)


def solve(grid: Grid, objective: str, m: float, tol=1e-10, eval_sweeps=40, max_iter=400):
    model = grid.model
    beta = model.beta
    r = grid.stage(objective, m)
    lump = -m / (1 - beta)
    surv = grid.survive[None]
    V = np.zeros((len(grid.s), len(grid.z)))
    pol = None
    for it in range(max_iter):
        Q = (r + beta * grid.expect_next(V)).mean(axis=3) * surv + (1 - surv) * lump   # (ns, nz, na)
        new_pol = Q.argmax(axis=2)
        Vn = np.take_along_axis(Q, new_pol[:, :, None], axis=2)[:, :, 0]
        change = np.max(np.abs(Vn - V))
        V = Vn
        if pol is not None and np.array_equal(new_pol, pol) and change < tol:
            break
        pol = new_pol
        V = evaluate_policy(grid, pol, lambda g: (grid.stage(objective, m), lump), V0=V,
                            sweeps=eval_sweeps)
    return dict(policy_index=pol, policy=grid.a[pol], V=V, iterations=it + 1)


def _gather(grid: Grid, pol, arr):
    """Select the policy's action from an (ns, nz, na, nxi) or (nz, na, nxi) array."""
    ns, nz = pol.shape
    zi = np.arange(nz)[None, :].repeat(ns, 0)
    if arr.ndim == 3:
        return arr[zi, pol]                                              # (ns, nz, nxi)
    si = np.arange(ns)[:, None].repeat(nz, 1)
    return arr[si, zi, pol]


def evaluate_policy(grid: Grid, pol, reward_fn, V0=None, sweeps=2000, tol=1e-12):
    """Iterate V = E[ok (r + beta V') + fail * lump] for a fixed action-index table."""
    beta = grid.model.beta
    r_full, lump = reward_fn(grid)
    r = _gather(grid, pol, np.broadcast_to(r_full, (len(grid.s),) + grid.zn.shape))
    surv = _gather(grid, pol, grid.survive[:, :, None])[:, :, 0]
    lo = _gather(grid, pol, grid.lo)
    fr = _gather(grid, pol, grid.fr)
    V = np.zeros(pol.shape) if V0 is None else V0.copy()
    si = np.arange(pol.shape[0])[:, None, None]
    for _ in range(sweeps):
        Vbar = np.einsum('ij,jz->iz', grid.P, V)
        nxt = Vbar[si, lo] * (1 - fr) + Vbar[si, lo + 1] * fr
        Vn = (r + beta * nxt).mean(axis=2) * surv + (1 - surv) * lump
        done = np.max(np.abs(Vn - V)) < tol
        V = Vn
        if done:
            break
    return V


def metrics(grid: Grid, pol, m):
    """Grid-exact stopped utility U, geometric risk q and interior loss I_m at (s=0, z=b)."""
    ns = len(grid.s)
    shape = (ns,) + grid.zn.shape
    u = np.broadcast_to(grid.u[:, None, :, None], shape)
    U = evaluate_policy(grid, pol, lambda g: (u, 0.0))
    q = evaluate_policy(grid, pol, lambda g: (np.zeros(shape), 1.0))
    inter = np.minimum(grid.B[None], u + m)
    I = evaluate_policy(grid, pol, lambda g: (inter, 0.0))
    i0 = ns // 2
    return dict(U=float(U[i0, -1]), q=float(q[i0, -1]), I=float(I[i0, -1]))


def policy_function(grid: Grid, policy_table):
    """Bilinear interpolation of an (ns, nz) action table for use in model.simulate."""
    s_grid, z_grid = grid.s, grid.z
    ns, nz = policy_table.shape

    def policy(s, z):
        ps = (np.asarray(s) + 1.0) / 2.0 * (ns - 1)
        ls = np.clip(np.floor(ps).astype(int), 0, ns - 2)
        fs = ps - ls
        zc = np.clip(np.asarray(z), 0.0, z_grid[-1])
        hz = np.clip(np.searchsorted(z_grid, zc, side="right"), 1, nz - 1)
        lz = hz - 1
        fz = (zc - z_grid[lz]) / (z_grid[hz] - z_grid[lz])
        return ((1 - fs) * (1 - fz) * policy_table[ls, lz] + (1 - fs) * fz * policy_table[ls, hz]
                + fs * (1 - fz) * policy_table[ls + 1, lz] + fs * fz * policy_table[ls + 1, hz])

    return policy
