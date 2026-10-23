"""Numerical test of Theorem thm:flow for general convex regularizers (revision 38).

Setting: Example ex:regularization-advantage (the plateau first-action model) with the
parameters of the earlier study (numerics/density_flow.py): beta = 0.9, gamma = 1/2,
eps = 0.2, m = 1, b = 1, B(z) = log(1 + 1/z), uniform action reference, feature
psi = a chi(theta), prior Gamma = N(0,1), initial law mu_0 = Gamma(. | [-1/2, 1/2]).

Regularizers H_{omega,G} = KL(mu || Gamma) + omega [int G(f) dGamma - G(1)] with the power
generators G(v) = v^r / (r - 1) of Proposition prop:generators (r = 2 is KL + omega chi^2),
and KL alone (omega = 0). The Wasserstein gradient flow of J = -V_m + kappa H is
    d_l p = -d_theta(p G) + kappa d_theta( p d_theta[ log f + omega G'(f) ] ),   f = p / phi,
and since d_theta[omega G'(f)] = omega G''(f) f d_theta log f = omega r f^(r-1) d_theta log f,
the diffusion coefficient is kappa (1 + omega r f^(r-1)) (kappa (1 + 2 omega f) for r = 2).
The scheme is the semi-implicit Scharfetter-Gummel finite-volume scheme of the earlier
study on [-L, L] with no-flux boundaries, with the generator factor frozen at the old f.

Reported per run: the energy gap J(mu_l) - J*, the W_2 distance to the final law (1-D
quantile formula), the theorem's envelopes E_0 exp(-2 Lambda_m l) and
W_2(mu_0, mu*) exp(-Lambda_m l) when Lambda_m > 0, the observed exponential rate, the
first time at which V_m(mu_l) >= V_m(mu_0) + Delta/2 against the example's prediction, and
the minimizer's V_m, U, q_beta, I_m and H (the regularization bias).
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
from density_flow import FirstActionModel, chi, chi_prime, feature_constants, phi_gauss  # noqa: E402

PARAMS = dict(beta=0.9, gamma=0.5, eps=0.2, m=1.0, omega=1.0, L=8.0, N=800, ubar=1.0, lambda_W=1.0)
REGULARIZERS = [("KL", 0.0, 2.0), ("r1.5", 1.0, 1.5), ("r2", 1.0, 2.0), ("r3", 1.0, 3.0)]


def bernoulli(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-8
    out[small] = 1.0 - 0.5 * x[small]
    xs = x[~small]
    out[~small] = xs / np.expm1(xs)
    return out


def certified(m, beta, ubar, lambda_W, margin=1.05):
    L0, L1, L2 = feature_constants("plateau")
    c_psi = L2 / (1 - beta) ** 2 + L1 ** 2 / (1 - beta) ** 3
    C_m = (m + ubar) * c_psi
    return margin * C_m / lambda_W, dict(L0=L0, L1=L1, L2=L2, c_psi=c_psi, C_m=C_m)


class GeneratorFlow:
    """1-D flow of J = -V_m + kappa H_{omega,G_r} for the plateau model, prior N(0,1)."""

    def __init__(self, model, kappa, omega, r, *, L=8.0, N=800):
        self.model, self.kappa, self.omega, self.r = model, kappa, omega, r
        self.L, self.N = L, N
        self.h = 2 * L / N
        self.theta = -L + (np.arange(N) + 0.5) * self.h
        self.faces = -L + np.arange(N + 1) * self.h
        self.phi = phi_gauss(self.theta)
        self.feat_cells = chi(self.theta)
        self.feat_prime_faces = chi_prime(self.faces)

    def factor(self, f):
        """1 + omega G''(f) f = 1 + omega r f^(r-1): the generator's diffusion factor."""
        return 1.0 + self.omega * self.r * np.power(np.maximum(f, 0.0), self.r - 1)

    def c_of(self, p):
        return float(np.sum(p * self.feat_cells) * self.h)

    def regularizer(self, p):
        f = p / self.phi
        kl = float(np.sum(np.where(p > 0, p * np.log(np.where(p > 0, f, 1.0)), 0.0)) * self.h)
        gen = float((np.sum(self.phi * np.power(np.maximum(f, 0.0), self.r)) * self.h - 1.0) / (self.r - 1))
        return kl + self.omega * gen, kl, gen

    def energy(self, p):
        c = self.c_of(p)
        st = self.model.expectations(c)
        H, kl, gen = self.regularizer(p)
        return -st["V_m"] + self.kappa * H, c, st, H, kl, gen

    def velocity_G(self, p):
        return self.model.dV(self.c_of(p)) * self.feat_prime_faces

    def step_implicit(self, p, dt):
        G = self.velocity_G(p)
        f_cells = p / self.phi
        f_faces = np.empty(self.N + 1)
        f_faces[1:-1] = 0.5 * (f_cells[:-1] + f_cells[1:])
        f_faces[0], f_faces[-1] = f_cells[0], f_cells[-1]
        D = self.kappa * self.factor(f_faces)
        v = G - D * self.faces
        Pe = v * self.h / D
        Bm, Bp = bernoulli(-Pe), bernoulli(Pe)
        coef = D / self.h
        cj = coef[1:-1] * dt / self.h
        main = np.ones(self.N)
        main[:-1] += cj * Bm[1:-1]
        main[1:] += cj * Bp[1:-1]
        upper = -cj * Bp[1:-1]
        lower = -cj * Bm[1:-1]
        ab = np.zeros((3, self.N))
        ab[0, 1:] = upper
        ab[1, :] = main
        ab[2, :-1] = lower
        return solve_banded((1, 1), ab, p)

    def run(self, p0, l_max, *, n_record=160, max_steps=6_000_000):
        p = p0.copy()
        rec, snaps = [], []
        record_times = np.unique(np.concatenate([[0.0], np.geomspace(l_max * 1e-6, l_max, n_record)]))
        l = 0.0; k = 0; next_idx = 0; increases = 0; J_prev = None
        t0 = time.monotonic()
        while True:
            if next_idx < len(record_times) and l >= record_times[next_idx] - 1e-15:
                J, c, st, H, kl, gen = self.energy(p)
                rec.append(dict(l=l, J=J, c=c, H=H, KL=kl, gen=gen, mass=float(np.sum(p) * self.h),
                                pmin=float(np.min(p)), **{k_: st[k_] for k_ in ("V_m", "U", "V", "I_m", "q", "lam")}))
                snaps.append(p.copy())
                if J_prev is not None and J > J_prev + 1e-10 * max(1.0, abs(J)):
                    increases += 1
                J_prev = J
                next_idx += 1
            if l >= l_max or k >= max_steps:
                break
            G = self.velocity_G(p)
            fmax = float(np.max(p / self.phi))
            rate = self.kappa * float(self.factor(fmax)) * (1 + self.L ** 2)
            dt = min(0.05 / rate, 0.2 * self.h / (np.max(np.abs(G)) + 1e-30), l_max / 200)
            if next_idx < len(record_times) and record_times[next_idx] > l:
                dt = min(dt, record_times[next_idx] - l)
            p = self.step_implicit(p, dt)
            l += dt; k += 1
        return dict(records=rec, snaps=snaps, p_final=p, steps=k, energy_increases=increases,
                    wall=time.monotonic() - t0)


def initial_plateau(theta, h):
    p = np.where(np.abs(theta) <= 0.5, phi_gauss(theta), 0.0)
    return p / (np.sum(p) * h)


def quantiles(p, faces, h, u):
    F = np.concatenate([[0.0], np.cumsum(p) * h])
    F = np.maximum.accumulate(F) / F[-1]
    return np.interp(u, F, faces)


def w2(p, q, faces, h, n_u=4000):
    u = (np.arange(n_u) + 0.5) / n_u
    return float(np.sqrt(np.mean((quantiles(p, faces, h, u) - quantiles(q, faces, h, u)) ** 2)))


def fit_rate(l, y, lo=1e-8, hi=1e-2):
    """Slope of log y over the window y/y[0] in [lo, hi]; nan if fewer than four points."""
    l = np.asarray(l); y = np.asarray(y)
    rel = y / y[0]
    sel = (rel <= hi) & (rel >= lo) & (y > 0)
    if sel.sum() < 4:
        return float("nan")
    a = np.polyfit(l[sel], np.log(y[sel]), 1)
    return float(-a[0])


def one_run(args):
    name, omega, r, kappa, l_max, P = args
    model = FirstActionModel("plateau", P["eps"], P["m"], P["beta"], P["gamma"])
    flow = GeneratorFlow(model, kappa, omega, r, L=P["L"], N=P["N"])
    p0 = initial_plateau(flow.theta, flow.h)
    res = flow.run(p0, l_max)
    rec = res["records"]
    Jf = rec[-1]["J"]
    gap = np.array([x["J"] - Jf for x in rec])
    dist = np.array([w2(s, res["p_final"], flow.faces, flow.h) for s in res["snaps"]])
    ls = np.array([x["l"] for x in rec])
    for x, g, d in zip(rec, gap, dist):
        x["gap"] = float(g); x["W2"] = float(d)
    out = dict(name=name, omega=omega, r=r, kappa=kappa, l_max=l_max, steps=res["steps"],
               energy_increases=res["energy_increases"], wall=res["wall"],
               E0=float(gap[0]), W2_0=float(dist[0]),
               rate_energy=fit_rate(ls, gap), rate_W2=fit_rate(ls, dist, lo=1e-4, hi=1e-1),
               final={k: rec[-1][k] for k in ("J", "V_m", "U", "I_m", "q", "H", "KL", "gen", "c", "mass", "pmin")},
               records=[{k: v for k, v in x.items()} for x in rec],
               density=dict(theta=flow.theta[::4].tolist(), p_final=res["p_final"][::4].tolist()))
    print(f"{name:5s} kappa={kappa:9.4g} r={r} omega={omega} steps={res['steps']:8d} wall={res['wall']:6.1f}s "
          f"E0={gap[0]:.4g} rate_E={out['rate_energy']:.4g} rate_W2={out['rate_W2']:.4g} "
          f"V_m {rec[0]['V_m']:.4f}->{rec[-1]['V_m']:.4f} q {rec[0]['q']:.4f}->{rec[-1]['q']:.4f} "
          f"H->{rec[-1]['H']:.3g} inc={res['energy_increases']}", flush=True)
    return out


def main(workers=4):
    P = PARAMS
    kappa_cert, consts = certified(P["m"], P["beta"], P["ubar"], P["lambda_W"])
    model = FirstActionModel("plateau", P["eps"], P["m"], P["beta"], P["gamma"])
    th = np.linspace(-40, 40, 400001)
    c_Gamma = float(np.trapezoid(chi(th) * phi_gauss(th), th))
    st0, stG = model.expectations(0.0), model.expectations(c_Gamma)
    cs = np.linspace(0.0, 1.0, 2001)
    curve = [model.expectations(c) for c in cs]
    out = dict(params=P, constants=consts, kappa_cert=kappa_cert, c_Gamma=c_Gamma,
               stats_mu0=st0, stats_Gamma=stG, Delta=stG["V_m"] - st0["V_m"],
               sup_V_m=max(x["V_m"] for x in curve),
               curve=dict(c=cs.tolist(), **{k: [x[k] for x in curve] for k in ("V_m", "U", "q", "I_m")}),
               runs={})
    kappas = [("cert", kappa_cert), ("k10", 10.0), ("k1", 1.0), ("k0.1", 0.1)]
    jobs = []
    for kname, kappa in kappas:
        Lam = kappa * P["lambda_W"] - consts["C_m"]
        l_max = 12.0 / Lam if Lam > 0 else 60.0 / kappa
        for name, omega, r in REGULARIZERS:
            jobs.append((f"{kname}/{name}", omega, r, kappa, l_max, P))
    # the certified runs are the longest; start them first
    with Pool(workers) as pool:
        for res in pool.imap_unordered(one_run, jobs):
            res["Lambda"] = res["kappa"] * P["lambda_W"] - consts["C_m"]
            out["runs"][res["name"]] = res
    # example prediction: l >= log(2 E0 / Delta) / (2 Lambda) implies V_m(mu_l) >= V_m(mu_0) + Delta/2
    for res in out["runs"].values():
        if res["Lambda"] > 0:
            res["l_predicted"] = math.log(2 * res["E0"] / out["Delta"]) / (2 * res["Lambda"])
            target = st0["V_m"] + out["Delta"] / 2
            hit = [x["l"] for x in res["records"] if x["V_m"] >= target]
            res["l_observed_half_Delta"] = hit[0] if hit else None
    (ROOT / "regularizer-flow.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
