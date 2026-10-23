"""The role of the prior in the solvable flow (revision 39).

Same model and scheme as regularizer_flow.py (plateau example, KL alone, omega = 0), with the
prior Gamma = N(x0, sigma^2) in place of N(0, 1). Relative entropy to a Gaussian prior is
    KL(mu || Gamma) = -h(mu) + (1 / 2 sigma^2) int |x - x0|^2 dmu + (1/2) log(2 pi sigma^2),
so kappa KL is an entropy term with weight kappa plus quadratic weight decay toward x0 with
weight kappa / sigma^2; the curvature is lambda_W = 1 / sigma^2 and the flow's diffusion is
    d_l p = -d_theta(p G) + kappa d_theta( d_theta p + p (theta - x0) / sigma^2 ).
Runs: width sweep sigma in {1/2, 1, 2} at x0 = 0 and location sweep x0 in {0, 1, 2} at
sigma = 1, each at kappa in {10, 1}; plus the pairs (kappa, sigma) with equal kappa / sigma^2
to show that the rate depends on the pair only through that ratio while the minimizer does
not. Reported: observed energy and W_2 rates, the induced prior policy's tilt c_Gamma =
int chi dGamma and its V_m and q_beta, and the minimizer's V_m, q_beta and KL.
"""
from __future__ import annotations

import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
sys.path.insert(0, str(ROOT))
from density_flow import FirstActionModel, chi, chi_prime  # noqa: E402
from regularizer_flow import PARAMS, bernoulli, fit_rate, w2  # noqa: E402


def gauss(theta, x0, sigma):
    return np.exp(-0.5 * ((theta - x0) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


class PriorFlow:
    def __init__(self, model, kappa, x0, sigma, *, L=10.0, N=1000):
        self.model, self.kappa, self.x0, self.sigma = model, kappa, x0, sigma
        self.L, self.N = L, N
        self.h = 2 * L / N
        self.theta = -L + (np.arange(N) + 0.5) * self.h
        self.faces = -L + np.arange(N + 1) * self.h
        self.phi = gauss(self.theta, x0, sigma)
        self.feat_cells = chi(self.theta)
        self.feat_prime_faces = chi_prime(self.faces)

    def c_of(self, p):
        return float(np.sum(p * self.feat_cells) * self.h)

    def kl(self, p):
        f = p / self.phi
        return float(np.sum(np.where(p > 0, p * np.log(np.where(p > 0, f, 1.0)), 0.0)) * self.h)

    def energy(self, p):
        c = self.c_of(p)
        st = self.model.expectations(c)
        H = self.kl(p)
        return -st["V_m"] + self.kappa * H, c, st, H

    def velocity_G(self, p):
        return self.model.dV(self.c_of(p)) * self.feat_prime_faces

    def step_implicit(self, p, dt):
        G = self.velocity_G(p)
        D = np.full(self.N + 1, self.kappa)
        v = G - D * (self.faces - self.x0) / self.sigma ** 2
        Pe = v * self.h / D
        Bm, Bp = bernoulli(-Pe), bernoulli(Pe)
        cj = (D / self.h)[1:-1] * dt / self.h
        main = np.ones(self.N)
        main[:-1] += cj * Bm[1:-1]
        main[1:] += cj * Bp[1:-1]
        ab = np.zeros((3, self.N))
        ab[0, 1:] = -cj * Bp[1:-1]
        ab[1, :] = main
        ab[2, :-1] = -cj * Bm[1:-1]
        return solve_banded((1, 1), ab, p)

    def run(self, p0, l_max, *, n_record=160, max_steps=4_000_000):
        p = p0.copy()
        rec, snaps = [], []
        record_times = np.unique(np.concatenate([[0.0], np.geomspace(l_max * 1e-6, l_max, n_record)]))
        l = 0.0; k = 0; next_idx = 0; increases = 0; J_prev = None
        t0 = time.monotonic()
        while True:
            if next_idx < len(record_times) and l >= record_times[next_idx] - 1e-15:
                J, c, st, H = self.energy(p)
                rec.append(dict(l=l, J=J, c=c, H=H, mass=float(np.sum(p) * self.h), **{k_: st[k_] for k_ in ("V_m", "U", "I_m", "q")}))
                snaps.append(p.copy())
                if J_prev is not None and J > J_prev + 1e-10 * max(1.0, abs(J)):
                    increases += 1
                J_prev = J
                next_idx += 1
            if l >= l_max or k >= max_steps:
                break
            G = self.velocity_G(p)
            rate = self.kappa * (1 + (self.L + abs(self.x0)) ** 2 / self.sigma ** 2)
            dt = min(0.05 / rate, 0.2 * self.h / (np.max(np.abs(G)) + 1e-30), l_max / 200)
            if next_idx < len(record_times) and record_times[next_idx] > l:
                dt = min(dt, record_times[next_idx] - l)
            p = self.step_implicit(p, dt)
            l += dt; k += 1
        return dict(records=rec, snaps=snaps, p_final=p, steps=k, energy_increases=increases, wall=time.monotonic() - t0)


def initial_plateau(theta, h):
    p = np.where(np.abs(theta) <= 0.5, gauss(theta, 0.0, 1.0), 0.0)
    return p / (np.sum(p) * h)


def one_run(args):
    name, kappa, x0, sigma, P = args
    model = FirstActionModel("plateau", P["eps"], P["m"], P["beta"], P["gamma"])
    flow = PriorFlow(model, kappa, x0, sigma)
    p0 = initial_plateau(flow.theta, flow.h)
    lam_W = 1.0 / sigma ** 2
    l_max = 40.0 / (kappa * lam_W)
    res = flow.run(p0, l_max)
    rec = res["records"]
    Jf = rec[-1]["J"]
    gap = np.array([x["J"] - Jf for x in rec])
    dist = np.array([w2(s, res["p_final"], flow.faces, flow.h) for s in res["snaps"]])
    ls = np.array([x["l"] for x in rec])
    th = np.linspace(-40, 40, 400001)
    c_G = float(np.trapezoid(chi(th) * gauss(th, x0, sigma), th))
    stG = model.expectations(c_G)
    out = dict(name=name, kappa=kappa, x0=x0, sigma=sigma, lambda_W=lam_W, kappa_lambda=kappa * lam_W, l_max=l_max,
               steps=res["steps"], energy_increases=res["energy_increases"], wall=res["wall"], E0=float(gap[0]), W2_0=float(dist[0]),
               rate_energy=fit_rate(ls, gap), rate_W2=fit_rate(ls, dist, lo=1e-4, hi=1e-1),
               prior=dict(c=c_G, V_m=stG["V_m"], q=stG["q"], U=stG["U"]),
               final={k: rec[-1][k] for k in ("J", "V_m", "U", "I_m", "q", "H", "c", "mass")},
               records=[dict(l=x["l"], gap=float(g), W2=float(d), V_m=x["V_m"], q=x["q"], H=x["H"]) for x, g, d in zip(rec, gap, dist)],
               density=dict(theta=flow.theta[::5].tolist(), p_final=res["p_final"][::5].tolist(), phi=flow.phi[::5].tolist()))
    print(f"{name:14s} kappa={kappa:5g} x0={x0:4g} sigma={sigma:4g} kappa/sigma^2={kappa * lam_W:6.3g} steps={res['steps']:7d} "
          f"rateE={out['rate_energy']:8.4g} rateW2={out['rate_W2']:8.4g} prior: c={c_G:.3f} V_m={stG['V_m']:.4f} q={stG['q']:.4f} | "
          f"min: V_m={rec[-1]['V_m']:.4f} q={rec[-1]['q']:.4f} KL={rec[-1]['H']:.3g} c={rec[-1]['c']:.3f} inc={res['energy_increases']}", flush=True)
    return out


def main(workers=4):
    P = PARAMS
    jobs = []
    for kappa in (10.0, 1.0):
        for sigma in (0.5, 1.0, 2.0):
            jobs.append((f"w{sigma:g}/k{kappa:g}", kappa, 0.0, sigma, P))
        for x0 in (1.0, 2.0):
            jobs.append((f"x{x0:g}/k{kappa:g}", kappa, x0, 1.0, P))
    # equal kappa / sigma^2 = 1 with different splits
    for kappa, sigma in ((4.0, 2.0), (0.25, 0.5)):
        jobs.append((f"ratio1/k{kappa:g}s{sigma:g}", kappa, 0.0, sigma, P))
    model = FirstActionModel("plateau", P["eps"], P["m"], P["beta"], P["gamma"])
    out = dict(params=P, stats_mu0=model.expectations(0.0), runs={})
    with Pool(workers) as pool:
        for res in pool.imap_unordered(one_run, jobs):
            out["runs"][res["name"]] = res
    (ROOT / "prior-flow.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__" and not (len(sys.argv) > 2 and sys.argv[2] == "path"):
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)


def path_main(workers=4):
    """Regularization path: minimizers over kappa for four priors, relative entropy alone."""
    P = PARAMS
    priors = [(0.0, 0.5), (0.0, 1.0), (0.0, 2.0), (2.0, 1.0)]
    kappas = (30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03)
    jobs = [(f"path/x{x0:g}s{sigma:g}/k{kappa:g}", kappa, x0, sigma, P) for x0, sigma in priors for kappa in kappas]
    model = FirstActionModel("plateau", P["eps"], P["m"], P["beta"], P["gamma"])
    cs = np.linspace(0.0, 1.0, 2001)
    curve = [model.expectations(c) for c in cs]
    out = dict(params=P, stats_mu0=model.expectations(0.0), priors=priors, kappas=kappas,
               curve=dict(c=cs.tolist(), **{k: [x[k] for x in curve] for k in ("V_m", "U", "q", "I_m")}), runs={})
    with Pool(workers) as pool:
        for res in pool.imap_unordered(one_run, jobs):
            res.pop("records", None); res.pop("density", None)
            out["runs"][res["name"]] = res
    (ROOT / "prior-path.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "path":
    path_main(int(sys.argv[1]))
