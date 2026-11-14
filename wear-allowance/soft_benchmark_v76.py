"""Revision 76: benchmark for the action-regularized configurations of the wear-allowance example.

The action-regularized objective at the price m and the strength eta is
    J_eta(pi) = U(pi) - m q_beta(pi) / (1 - beta) - eta A(pi),
with A the stopped action relative entropy to the incumbent rule (a Gaussian in the pre-clipping action variable,
mean 2 min(1, z / (span b)) - 1, dispersion 0.1). Over all action kernels its maximizer solves the soft Bellman equation
    V(s, z) = eta log int rho(dx | z) exp(Q(s, z, clip x) / eta),
    Q(s, z, a) = P(survive | z, a) E[u(s, a) + beta V(s', z') | survive] - P(fail | z, a) m / (1 - beta),
and is pi(dx | s, z) proportional to rho(dx | z) exp(Q(s, z, clip x) / eta). The script solves this equation on the grid of
dp.py with a finer action grid, the clipped tails of the reference being the atoms at the rates 0 and 1, and then
evaluates the resulting kernel by Monte Carlo in the continuous simulator, as the learned policies are evaluated.
The Monte Carlo value of this kernel is a lower bound on the supremum of J_eta, so J_eta of a learned policy below it
is an optimization error of at least the difference. The learner's policies are Gaussian with one dispersion for all
states, a smaller class, so part of a difference can be the class and not the optimization.

Runtime ~/.venvs/constrained-rl/bin/python. Output soft-benchmark-v76.json.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.special import log_ndtr, logsumexp
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dp import Grid  # noqa: E402
from model import Model, utility  # noqa: E402

M_PRICE, SPAN, STD_X = 0.03, 3.0, 0.1
ETAS = [3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]


def log_reference(model, z, a):
    """log of the reference mass of each cell of the rate grid a (cells bounded by midpoints, tails in the end cells)."""
    mu = np.minimum(1.0, np.maximum(z, 0.0) / (SPAN * model.budget))[..., None]      # mean rate; rate = (x + 1) / 2
    sd = STD_X / 2.0
    edges = (a[1:] + a[:-1]) / 2.0
    lo = (edges - mu) / sd                                                           # (..., na - 1) standardized edges
    out = np.empty(mu.shape[:-1] + (len(a),))
    out[..., 0] = log_ndtr(lo[..., 0])
    out[..., -1] = log_ndtr(-lo[..., -1])
    # interior cells: log(Phi(b) - Phi(a)) computed from the side on which both tails are small
    a_, b_ = lo[..., :-1], lo[..., 1:]
    left = b_ <= 0
    la = np.where(left, log_ndtr(a_), log_ndtr(-b_)); lb = np.where(left, log_ndtr(b_), log_ndtr(-a_))
    out[..., 1:-1] = lb + np.log1p(-np.exp(np.minimum(la - lb, -1e-300)))
    return out


def solve_soft(grid, eta, tol=1e-9, max_iter=3000):
    model = grid.model; beta = model.beta; lump = -M_PRICE / (1 - beta)
    surv = grid.survive[None]
    logw = log_reference(model, grid.z, grid.a)[None]                                # (1, nz, na)
    u = grid.u[:, None, :, None]
    V = np.zeros((len(grid.s), len(grid.z)))
    for it in range(max_iter):
        Q = (u + beta * grid.expect_next(V)).mean(axis=3) * surv + (1 - surv) * lump
        Vn = eta * logsumexp(logw + Q / eta, axis=2)
        change = np.max(np.abs(Vn - V)); V = Vn
        if change < tol:
            break
    Q = (u + beta * grid.expect_next(V)).mean(axis=3) * surv + (1 - surv) * lump
    return dict(Q=Q, V=V, iterations=it + 1)


def simulate_soft(model, grid, Q, eta, n, seed, max_steps=2000):
    """Monte Carlo evaluation of the kernel pi proportional to rho exp(Q / eta) in the continuous simulator."""
    rng = np.random.default_rng(seed)
    ns, nz = len(grid.s), len(grid.z)
    s = np.zeros(n); z = np.full(n, model.budget); alive = np.ones(n, dtype=bool)
    U = np.zeros(n); A = np.zeros(n); failed = np.zeros(n, dtype=bool)
    t = 0
    while alive.any() and t < max_steps:
        ids = np.flatnonzero(alive)
        ps = (s[ids] + 1.0) / 2.0 * (ns - 1); ls = np.clip(np.floor(ps).astype(int), 0, ns - 2); fs = (ps - ls)[:, None]
        zc = np.clip(z[ids], 0.0, grid.z[-1]); hz = np.clip(np.searchsorted(grid.z, zc, side="right"), 1, nz - 1); lz = hz - 1
        fz = ((zc - grid.z[lz]) / (grid.z[hz] - grid.z[lz]))[:, None]
        q = ((1 - fs) * (1 - fz) * Q[ls, lz] + (1 - fs) * fz * Q[ls, hz] + fs * (1 - fz) * Q[ls + 1, lz] + fs * fz * Q[ls + 1, hz])
        logw = log_reference(model, z[ids], grid.a)
        logp = logw + q / eta; logp -= logsumexp(logp, axis=1, keepdims=True)
        p = np.exp(logp)
        A[ids] += np.sum(p * (logp - logw), axis=1, where=p > 0)
        j = (p.cumsum(axis=1) > rng.random(len(ids))[:, None]).argmax(axis=1)
        a = grid.a[j]
        xi = rng.gamma(model.wear_shape, 1.0 / model.wear_shape, len(ids))
        z_next = z[ids] - model.wear_rate * a * xi
        fail = z_next < 0
        U[ids[~fail]] += utility(model, s[ids], a)[~fail]
        failed[ids[fail]] = True
        s[ids] = model.rho * s[ids] + (1 - model.rho) * rng.uniform(-1, 1, len(ids))
        z[ids] = z_next
        stop = fail | (rng.random(len(ids)) > model.beta)
        alive[ids[stop]] = False
        t += 1
    se = lambda x: float(x.std(ddof=1) / np.sqrt(n))
    Jn = U - M_PRICE / (1 - model.beta) * failed - eta * A
    return dict(n=n, risk=float(failed.mean()), U=float(U.mean()), U_se=se(U), A=float(A.mean()), A_se=se(A), J=float(Jn.mean()), J_se=se(Jn))


if __name__ == "__main__":
    model = Model(wear_shape=0.5)
    grid = Grid(model, na=241)
    rows = []
    for eta in ETAS:
        t0 = time.time()
        sol = solve_soft(grid, eta)
        mc = simulate_soft(model, grid, sol["Q"], eta, n=100_000, seed=760_000 + int(1000 * eta))
        rows.append(dict(eta=eta, grid_V=float(sol["V"][len(grid.s) // 2, -1]), iterations=sol["iterations"], **mc))
        print(f"eta={eta:g}: grid V={rows[-1]['grid_V']:.3f} | simulated J={mc['J']:.3f} ({mc['J_se']:.3f}) U={mc['U']:.3f} q={100 * mc['risk']:.2f}% A={mc['A']:.3g}"
              f"  [{sol['iterations']} sweeps, {time.time() - t0:.0f}s]", flush=True)
    (HERE / "soft-benchmark-v76.json").write_text(json.dumps(dict(m=M_PRICE, incumbent=dict(span=SPAN, std=STD_X), grid=dict(ns=21, nz=161, na=241), rows=rows), indent=1) + "\n")
