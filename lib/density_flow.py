"""Tier B: the analytical regularization examples and a 1-D parameter-density flow.

Two first-action models from the manuscript, both with S = {0,1} x [0,1], A = [0,1],
x ~ Unif[0,1] redrawn every step, zero cost and utility after the first action:
  plateau    (Example ex:regularization-advantage): g = 1 - a + eps x, u = a,
              psi = a chi(theta), chi flat on [-1,1]; failure iff a < eps x.
  compatible (Example ex:compatible): g = a + eps x, u = a,
              psi = c tanh(theta)(a - 1/2); failure iff a + eps x > 1.
Budget b = 1, cost discount gamma, utility discount beta, barrier B(z) = log(1+1/z),
clipping level m. The uniform action reference makes the action density proportional to
exp(lambda a) with lambda = int chi dmu (plateau) or lambda = c int tanh dmu (compatible).

Complete clipped returns are computed by quadrature (no simulation). The regularized flow
    d_l p = -d_theta(p G) + kappa d_theta(p [d_theta log(p/phi) + 2 omega d_theta(p/phi)])
is solved by a semi-implicit Scharfetter-Gummel finite-volume scheme on [-L, L] with
no-flux boundaries (mass-conserving, positivity-preserving). Unregularized transport
(kappa = 0) uses explicit upwind advection. The truncated solver approximates the
whole-space problem; refinement checks are reported, not a convergence theorem.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import solve_banded
from scipy import stats
from common import RESULTS, dump_json, Timer

SQ2PI = math.sqrt(2 * math.pi)


def phi_gauss(theta):
    return np.exp(-0.5 * theta ** 2) / SQ2PI


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def chi(theta):
    theta = np.asarray(theta, dtype=float)
    s = np.abs(theta) - 1.0
    out = np.zeros_like(theta)
    pos = s > 0
    out[pos] = np.exp(-1.0 / s[pos] ** 2)
    return out


def chi_prime(theta):
    theta = np.asarray(theta, dtype=float)
    s = np.abs(theta) - 1.0
    out = np.zeros_like(theta)
    pos = s > 0
    out[pos] = np.sign(theta[pos]) * 2.0 * s[pos] ** -3 * np.exp(-1.0 / s[pos] ** 2)
    return out


def chi_second(theta):
    theta = np.asarray(theta, dtype=float)
    s = np.abs(theta) - 1.0
    out = np.zeros_like(theta)
    pos = s > 0
    sp = s[pos]
    out[pos] = (4.0 * sp ** -6 - 6.0 * sp ** -4) * np.exp(-1.0 / sp ** 2)
    return out


def feature_constants(kind, c_feat=2.0):
    """(L0, L1, L2) sup bounds of psi and its first two theta-derivatives."""
    th = np.linspace(-12, 12, 480001)
    if kind == "plateau":
        return 1.0, float(np.max(np.abs(chi_prime(th)))), float(np.max(np.abs(chi_second(th))))
    sech2 = 1 / np.cosh(th) ** 2
    return c_feat / 2, c_feat / 2 * float(np.max(sech2)), c_feat / 2 * float(np.max(np.abs(-2 * sech2 * np.tanh(th))))


# ----------------------------------------------------------------------------
# First-action return models by quadrature
# ----------------------------------------------------------------------------
class FirstActionModel:
    """Quadrature representation of complete returns as functions of the first action.

    Nodes (a_k, x_k, w_k) cover the unit square with the failure boundary resolved
    exactly and a square-root substitution that removes the logarithmic barrier
    singularity at the boundary. Expectations under the tilt exp(lambda a) are
    ratios of weighted sums over the same nodes.
    """

    def __init__(self, kind, eps, m, beta, gamma, n_x=160, n_s=160, t_max=400):
        self.kind, self.eps, self.m, self.beta, self.gamma = kind, eps, m, beta, gamma
        xg, xw = np.polynomial.legendre.leggauss(n_x)
        x = 0.5 * (xg + 1); wx = 0.5 * xw
        sg, sw = np.polynomial.legendre.leggauss(n_s)
        s = 0.5 * (sg + 1); ws = 0.5 * sw
        # survivor region parametrised by s in (0,1); failure region by plain Gauss-Legendre
        A, X, W, FAIL = [], [], [], []
        for xi, wxi in zip(x, wx):
            if kind == "plateau":
                a0 = eps * xi                    # failure iff a < a0
                # survivor: a = a0 + (1-a0) s^2
                a_s = a0 + (1 - a0) * s ** 2
                w_s = wxi * ws * 2 * (1 - a0) * s
                # failure region [0, a0]
                a_f = a0 * 0.5 * (sg + 1)
                w_f = wxi * 0.5 * sw * a0
            else:
                a0 = 1 - eps * xi                # failure iff a > a0
                a_s = a0 - a0 * s ** 2
                w_s = wxi * ws * 2 * a0 * s
                a_f = a0 + (1 - a0) * 0.5 * (sg + 1)
                w_f = wxi * 0.5 * sw * (1 - a0)
            A += [a_s, a_f]; X += [np.full_like(a_s, xi), np.full_like(a_f, xi)]
            W += [w_s, w_f]; FAIL += [np.zeros_like(a_s, dtype=bool), np.ones_like(a_f, dtype=bool)]
        self.a = np.concatenate(A); self.x = np.concatenate(X)
        self.w = np.concatenate(W); self.fail = np.concatenate(FAIL)
        # budgets and returns
        if kind == "plateau":
            g0 = 1 - self.a + eps * self.x
        else:
            g0 = self.a + eps * self.x
        z1 = (1 - g0) / gamma
        self.z1 = z1
        surv = ~self.fail
        # first reward
        with np.errstate(divide="ignore", invalid="ignore"):
            B1 = np.where(z1 > 0, np.log1p(1.0 / np.where(z1 > 0, z1, 1.0)), np.inf)
        r0 = np.where(surv, np.maximum(-m, self.a - B1), -m)
        # later rewards: u = 0, z_{t+1} = z1 / gamma^t, t >= 1
        t = np.arange(1, t_max + 1)
        disc = beta ** t
        R = r0.copy(); I = np.where(surv, np.minimum(B1, self.a + m), 0.0)
        zs = z1[surv][:, None] / gamma ** t[None, :]
        Bt = np.log1p(1.0 / zs)
        R[surv] += np.sum(disc[None, :] * np.maximum(-m, -Bt), axis=1)
        I[surv] += np.sum(disc[None, :] * np.minimum(Bt, m), axis=1)
        # exact geometric tail for failed nodes and the remainder beyond t_max
        R[self.fail] = -m / (1 - beta)
        assert beta ** t_max * m < 1e-10 * max(1.0, m)
        self.R = R; self.I = I
        self.U = np.where(surv, self.a, 0.0)          # stopped utility (u = a at t=0)
        self.Vcont = self.a                            # continuing utility

    def expectations(self, lam):
        wt = self.w * np.exp(lam * (self.a - 0.5))
        Z = np.sum(wt)
        E = lambda f: float(np.sum(wt * f) / Z)
        q = E(self.fail.astype(float))
        out = dict(lam=float(lam), V_m=E(self.R), U=E(self.U), V=E(self.Vcont), I_m=E(self.I), q=q,
                   Ea=E(self.a))
        out["dV_dlam"] = E(self.a * self.R) - out["Ea"] * out["V_m"]
        out["decomposition_residual"] = out["V_m"] - (out["U"] - out["I_m"] - self.m * q / (1 - self.beta))
        return out

    def dV(self, lam):
        wt = self.w * np.exp(lam * (self.a - 0.5))
        Z = np.sum(wt)
        Ea = np.sum(wt * self.a) / Z
        return float(np.sum(wt * self.a * self.R) / Z - Ea * np.sum(wt * self.R) / Z)

    def C_B(self, lam):
        """(1-beta) times the pre-failure comparator cost sum_{t<tau} beta^t [ubar - u + B]."""
        # ubar = 1; pre-failure: t=0 term (1 - a + B(z1)) + sum_{t>=1} beta^t (1 + B(z1/gamma^t))
        wt = self.w * np.exp(lam * (self.a - 0.5)); Z = np.sum(wt)
        surv = ~self.fail
        t = np.arange(1, 401); disc = self.beta ** t
        zs = self.z1[surv][:, None] / self.gamma ** t[None, :]
        B1 = np.log1p(1.0 / self.z1[surv])
        tail = np.sum(disc[None, :] * (1 + np.log1p(1.0 / zs)), axis=1)
        val = np.zeros_like(self.a)
        val[surv] = (1 - self.a[surv] + B1) + tail
        return float((1 - self.beta) * np.sum(wt * val) / Z)


# ----------------------------------------------------------------------------
# 1-D density flow solver
# ----------------------------------------------------------------------------
def bernoulli(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-8
    out[small] = 1.0 - 0.5 * x[small]
    xs = x[~small]
    out[~small] = xs / np.expm1(xs)
    return out


class DensityFlow:
    def __init__(self, model, kind, kappa, omega, *, L=8.0, N=800, c_feat=2.0):
        self.model, self.kind, self.kappa, self.omega = model, kind, kappa, omega
        self.L, self.N, self.c_feat = L, N, c_feat
        self.h = 2 * L / N
        self.theta = -L + (np.arange(N) + 0.5) * self.h
        self.faces = -L + np.arange(N + 1) * self.h
        self.phi = phi_gauss(self.theta)
        self.phi_faces = phi_gauss(self.faces)
        if kind == "plateau":
            self.feat, self.feat_prime = chi, chi_prime
            self.lam_of_c = lambda c: c
            self.dlam_dc = 1.0
        else:
            self.feat, self.feat_prime = np.tanh, lambda th: 1 / np.cosh(th) ** 2
            self.lam_of_c = lambda c: c_feat * c
            self.dlam_dc = c_feat
        self.feat_cells = self.feat(self.theta)
        self.feat_prime_faces = self.feat_prime(self.faces)

    # --- functionals -----------------------------------------------------------
    def c_of(self, p):
        return float(np.sum(p * self.feat_cells) * self.h)

    def regularizer(self, p):
        f = p / self.phi
        kl = np.sum(np.where(p > 0, p * np.log(np.where(p > 0, f, 1.0)), 0.0)) * self.h
        chi2 = np.sum(self.phi * (f - 1) ** 2) * self.h
        return kl + self.omega * chi2, kl, chi2

    def energy(self, p):
        c = self.c_of(p)
        st = self.model.expectations(self.lam_of_c(c))
        H, kl, chi2 = self.regularizer(p)
        J = -st["V_m"] + self.kappa * H
        return J, c, st, H, kl, chi2

    def velocity_G(self, p):
        c = self.c_of(p)
        return self.model.dV(self.lam_of_c(c)) * self.dlam_dc * self.feat_prime_faces

    # --- steps -----------------------------------------------------------------
    def step_implicit(self, p, dt):
        G = self.velocity_G(p)
        f_cells = p / self.phi
        f_faces = np.empty(self.N + 1)
        f_faces[1:-1] = 0.5 * (f_cells[:-1] + f_cells[1:])
        f_faces[0], f_faces[-1] = f_cells[0], f_cells[-1]
        D = self.kappa * (1 + 2 * self.omega * f_faces)
        v = G - D * self.faces
        Pe = v * self.h / D
        Bm, Bp = bernoulli(-Pe), bernoulli(Pe)
        coef = D / self.h                    # flux F_{i+1/2} = coef[i+1](Bm p_i - Bp p_{i+1}) for interior faces
        # interior faces index j = 1..N-1 connect cells j-1 and j
        cj = coef[1:-1] * dt / self.h
        main = np.ones(self.N)
        main[:-1] += cj * Bm[1:-1]
        main[1:] += cj * Bp[1:-1]
        upper = -cj * Bp[1:-1]               # coefficient of p_{i+1} in row i
        lower = -cj * Bm[1:-1]               # coefficient of p_{i-1} in row i
        ab = np.zeros((3, self.N))
        ab[0, 1:] = upper
        ab[1, :] = main
        ab[2, :-1] = lower
        return solve_banded((1, 1), ab, p)

    def step_transport(self, p, dt):
        v = self.velocity_G(p)
        F = np.zeros(self.N + 1)
        vin = v[1:-1]
        F[1:-1] = np.where(vin > 0, vin * p[:-1], vin * p[1:])
        return p - dt / self.h * (F[1:] - F[:-1])

    def run(self, p0, l_max, *, dt=None, n_record=200, tol=1e-12, min_steps=50, max_steps=2_000_000):
        p = p0.copy()
        rec = []
        record_times = np.unique(np.concatenate([[0.0], np.geomspace(l_max * 1e-6, l_max, n_record)]))
        l = 0.0; k = 0; next_idx = 0; increases = 0; J_prev = None
        timer = Timer()
        while True:
            if next_idx < len(record_times) and l >= record_times[next_idx] - 1e-15:
                J, c, st, H, kl, chi2 = self.energy(p)
                rec.append(dict(l=l, J=J, c=c, H=H, KL=kl, chi2=chi2, mass=float(np.sum(p) * self.h),
                                pmin=float(np.min(p)), **{k_: st[k_] for k_ in ("V_m", "U", "V", "I_m", "q", "lam")}))
                if J_prev is not None and J > J_prev + 1e-10 * max(1.0, abs(J)):
                    increases += 1
                J_prev = J
                next_idx += 1
            if l >= l_max or k >= max_steps:
                break
            if self.kappa > 0:
                G = self.velocity_G(p)
                fmax = float(np.max(p / self.phi))
                rate = self.kappa * (1 + 2 * self.omega * fmax) * (1 + self.L ** 2)
                dt_k = dt if dt is not None else min(0.05 / rate, 0.2 * self.h / (np.max(np.abs(G)) + 1e-30), l_max / 200)
                if next_idx < len(record_times):
                    dt_k = min(dt_k, record_times[next_idx] - l) if record_times[next_idx] > l else dt_k
                p = self.step_implicit(p, dt_k)
            else:
                G = self.velocity_G(p)
                dt_k = dt if dt is not None else min(0.5 * self.h / (np.max(np.abs(G)) + 1e-30), l_max / 200)
                if next_idx < len(record_times) and record_times[next_idx] > l:
                    dt_k = min(dt_k, record_times[next_idx] - l)
                p = self.step_transport(p, dt_k)
            l += dt_k; k += 1
        return dict(records=rec, p_final=p, steps=k, energy_increases=increases, wall=timer(),
                    theta=self.theta)


def initial_plateau(theta, h):
    p = np.where(np.abs(theta) <= 0.5, phi_gauss(theta), 0.0)
    return p / (np.sum(p) * h)


def initial_gauss(theta, h):
    p = phi_gauss(theta)
    return p / (np.sum(p) * h)


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def certified_kappa(kind, m, beta, ubar=1.0, c_feat=2.0, lambda_W=1.0, margin=1.05):
    L0, L1, L2 = feature_constants(kind, c_feat)
    c_psi = L2 / (1 - beta) ** 2 + L1 ** 2 / (1 - beta) ** 3
    C_m = (m + ubar) * c_psi
    return margin * C_m / lambda_W, dict(L0=L0, L1=L1, L2=L2, c_psi=c_psi, C_m=C_m)


def run_plateau(beta=0.9, gamma=0.5, eps=0.2, m=1.0, omega=1.0, L=8.0, N=800):
    model = FirstActionModel("plateau", eps, m, beta, gamma)
    kappa_cert, consts = certified_kappa("plateau", m, beta)
    out = dict(params=dict(beta=beta, gamma=gamma, eps=eps, m=m, omega=omega, L=L, N=N),
               constants=consts, kappa_cert=kappa_cert, runs={})
    # quadrature sanity: prior tilt zero, Gamma tilt c_Gamma
    th = np.linspace(-40, 40, 400001)
    c_Gamma = float(np.trapezoid(chi(th) * phi_gauss(th), th))
    out["c_Gamma"] = c_Gamma
    out["stats_mu0"] = model.expectations(0.0)
    out["stats_Gamma"] = model.expectations(c_Gamma)
    lam_grid = np.linspace(-2, 8, 201)
    out["curve"] = dict(lam=lam_grid.tolist(), **{k: [model.expectations(l)[k] for l in lam_grid] for k in ("V_m", "U", "V", "q", "I_m")})
    Delta = out["stats_Gamma"]["V_m"] - out["stats_mu0"]["V_m"]
    out["Delta"] = Delta
    out["c_lower_bound"] = Delta * (1 - beta) ** 2 / (m + 1)      # manuscript's explicit bound
    settings = [("transport", 0.0, 0.0), ("cert", kappa_cert, omega), ("cert_KL", kappa_cert, 0.0),
                ("k1", 1.0, omega), ("k1_KL", 1.0, 0.0), ("k0.1", 0.1, omega), ("k0.1_KL", 0.1, 0.0), ("k0.01", 0.01, omega)]
    for name, kappa, om in settings:
        flow = DensityFlow(model, "plateau", kappa, om, L=L, N=N)
        p0 = initial_plateau(flow.theta, flow.h)
        Lam = kappa * 1.0 - consts["C_m"]
        l_max = 10.0 if kappa == 0 else (12.0 / Lam if Lam > 0 else 60.0 / max(kappa, 0.05))
        res = flow.run(p0, l_max, n_record=120)
        J0 = res["records"][0]["J"]; Jf = res["records"][-1]["J"]
        out["runs"][name] = dict(kappa=kappa, omega=om, Lambda=Lam, l_max=l_max, steps=res["steps"],
                                 energy_increases=res["energy_increases"], wall=res["wall"],
                                 records=res["records"], E0_est=J0 - Jf,
                                 density=dict(theta=flow.theta[::4].tolist(), p_final=res["p_final"][::4].tolist()))
        # snapshots at a few times
        r = res["records"]
        print(f"plateau {name:9s} kappa={kappa:.4g} Lambda={Lam:.3g} steps={res['steps']} "
              f"c: {r[0]['c']:.4f}->{r[-1]['c']:.4f}  V_m: {r[0]['V_m']:.4f}->{r[-1]['V_m']:.4f} "
              f"U: {r[0]['U']:.4f}->{r[-1]['U']:.4f} q: {r[0]['q']:.4f}->{r[-1]['q']:.4f} "
              f"mass={r[-1]['mass']:.6f} inc={res['energy_increases']} wall={res['wall']:.1f}s")
    # density snapshots along the kappa=1 flow for the figure
    flow = DensityFlow(model, "plateau", 1.0, omega, L=L, N=N)
    p0 = initial_plateau(flow.theta, flow.h)
    snaps = {}
    p = p0.copy(); l = 0.0
    for target in (0.0, 0.05, 0.3, 2.0, 10.0, 60.0):
        while l < target - 1e-12:
            G = flow.velocity_G(p)
            fmax = float(np.max(p / flow.phi))
            dt = min(0.05 / (flow.kappa * (1 + 2 * omega * fmax) * (1 + L ** 2)), 0.2 * flow.h / (np.max(np.abs(G)) + 1e-30), target - l)
            p = flow.step_implicit(p, dt); l += dt
        snaps[str(target)] = p[::2].tolist()
    out["snapshots"] = dict(theta=flow.theta[::2].tolist(), kappa=1.0, p=snaps,
                            G_over_Vprime=chi_prime(flow.theta[::2]).tolist())
    dump_json(out, RESULTS / "plateau_flow.json")
    return out


def run_compatible(beta=0.9, gamma=0.5, eps=0.02, omega=1.0, c_feat=2.0, L=8.0, N=800):
    out = dict(params=dict(beta=beta, gamma=gamma, eps=eps, omega=omega, c_feat=c_feat, L=L, N=N), cases={})
    for m in (10.0, 300.0):
        model = FirstActionModel("compatible", eps, m, beta, gamma)
        kappa_cert, consts = certified_kappa("compatible", m, beta, c_feat=c_feat)
        st0 = model.expectations(0.0)
        cb = model.C_B(0.0)
        bound_cb = 1 + (1 + gamma) * math.log(1 + gamma) - gamma * math.log(gamma)
        case = dict(m=m, kappa_cert=kappa_cert, constants=consts, stats_Gamma=st0,
                    one_minus_beta_CB=cb, one_minus_beta_CB_bound=bound_cb,
                    certificate_delta0=eps / 2 + bound_cb / (m + 1), runs={})
        lam_grid = np.linspace(-6, 3, 181)
        case["curve"] = dict(lam=lam_grid.tolist(), **{k: [model.expectations(l)[k] for l in lam_grid] for k in ("V_m", "U", "V", "q", "I_m")})
        R_Gamma = 1.0 / (1 - beta) - st0["V_m"]
        case["R_Gamma"] = R_Gamma
        settings = [("cert", kappa_cert), ("k100", 100.0), ("k10", 10.0), ("k1", 1.0)]
        for name, kappa in settings:
            flow = DensityFlow(model, "compatible", kappa, omega, L=L, N=N, c_feat=c_feat)
            p0 = initial_gauss(flow.theta, flow.h)
            Lam = kappa - consts["C_m"]
            l_max = 12.0 / Lam if Lam > 0 else 40.0 / kappa
            res = flow.run(p0, l_max, n_record=100)
            recs = res["records"]
            for r_ in recs:   # theorem quantities along the flow
                r_["allowance_q"] = 0.5 * consts["L0"] * math.sqrt(2 * R_Gamma / (kappa * (1 - beta)))
                r_["allowance_q_actual_H"] = 0.5 * consts["L0"] * math.sqrt(2 * r_["H"] / (1 - beta))
                r_["improvement_lower"] = kappa * r_["H"]
            case["runs"][name] = dict(kappa=kappa, Lambda=Lam, l_max=l_max, steps=res["steps"],
                                      energy_increases=res["energy_increases"], wall=res["wall"], records=recs,
                                      density=dict(theta=flow.theta[::4].tolist(), p_final=res["p_final"][::4].tolist()))
            r0, rf = recs[0], recs[-1]
            print(f"compatible m={m:g} {name:5s} kappa={kappa:.4g} Lambda={Lam:.3g} steps={res['steps']} "
                  f"lam: {r0['lam']:.3f}->{rf['lam']:.3f} V_m: {r0['V_m']:.3f}->{rf['V_m']:.3f} "
                  f"U: {r0['U']:.4f}->{rf['U']:.4f} q: {r0['q']:.5f}->{rf['q']:.5f} H={rf['H']:.3g} "
                  f"gain={rf['V_m']-r0['V_m']:.4g} >= kH={kappa*rf['H']:.4g}  inc={res['energy_increases']} wall={res['wall']:.1f}s")
        out["cases"][str(m)] = case
    dump_json(out, RESULTS / "compatible_flow.json")
    return out


def run_audit(beta=0.9, gamma=0.5, eps=0.2, m=1.0, omega=1.0, kappa=1.0):
    """Independent refinements of mesh, domain, time step and quadrature for the plateau flow."""
    out = dict(params=dict(beta=beta, gamma=gamma, eps=eps, m=m, omega=omega, kappa=kappa), rows=[])
    l_max = 20.0
    base_model = FirstActionModel("plateau", eps, m, beta, gamma)
    ref = None
    for label, N, L, dtf, nq in (("N200", 200, 8.0, None, 160), ("N400", 400, 8.0, None, 160),
                                  ("N800", 800, 8.0, None, 160), ("N1600", 1600, 8.0, None, 160),
                                  ("L6", 800, 6.0, None, 160), ("L10", 800, 10.0, None, 160),
                                  ("dt/4", 800, 8.0, 0.25, 160), ("quad80", 800, 8.0, None, 80),
                                  ("quad320", 800, 8.0, None, 320)):
        model = base_model if nq == 160 else FirstActionModel("plateau", eps, m, beta, gamma, n_x=nq, n_s=nq)
        flow = DensityFlow(model, "plateau", kappa, omega, L=L, N=N)
        p0 = initial_plateau(flow.theta, flow.h)
        if dtf is None:
            res = flow.run(p0, l_max, n_record=40)
        else:
            # quarter of the automatic step: emulate by a fixed small dt
            G = flow.velocity_G(p0)
            fmax = float(np.max(p0 / flow.phi))
            dt0 = min(0.05 / (kappa * (1 + 2 * omega * fmax) * (1 + L ** 2)), 0.2 * flow.h / (np.max(np.abs(G)) + 1e-30), l_max / 200)
            res = flow.run(p0, l_max, dt=dtf * dt0, n_record=40)
        recs = res["records"]
        row = dict(label=label, N=N, L=L, steps=res["steps"], mass=recs[-1]["mass"], pmin=min(r["pmin"] for r in recs[1:]),
                   energy_increases=res["energy_increases"], c_final=recs[-1]["c"], J_final=recs[-1]["J"],
                   c_at_1=[r["c"] for r in recs if abs(r["l"] - 1.0) < 1e-9][0] if any(abs(r["l"] - 1.0) < 1e-9 for r in recs) else None,
                   wall=res["wall"])
        out["rows"].append(row)
        print(f"audit {label:8s} steps={res['steps']:7d} mass={row['mass']:.8f} pmin={row['pmin']:.2e} inc={row['energy_increases']} "
              f"c_final={row['c_final']:.6f} J_final={row['J_final']:.6f} wall={res['wall']:.1f}s")
    dump_json(out, RESULTS / "flow_audit.json")
    return out


def quadrature_checks():
    """Implementation checks: decomposition identity, monotonicity, prior comparator cost bound."""
    for kind, eps, m in (("plateau", 0.2, 1.0), ("compatible", 0.02, 300.0)):
        model = FirstActionModel(kind, eps, m, 0.9, 0.5)
        for lam in (-3.0, 0.0, 2.0):
            st = model.expectations(lam)
            assert abs(st["decomposition_residual"]) < 1e-9, st
            # derivative check against finite differences
            fd = (model.expectations(lam + 1e-5)["V_m"] - model.expectations(lam - 1e-5)["V_m"]) / 2e-5
            assert abs(fd - st["dV_dlam"]) < 1e-6 * max(1.0, abs(fd)), (fd, st["dV_dlam"])
        # zero tilt failure probability equals eps/2 exactly
        assert abs(model.expectations(0.0)["q"] - eps / 2) < 1e-10
    # compatible-example comparator bound: (1-beta) C_B(Gamma) <= 1 + (1+g)log(1+g) - g log g
    model = FirstActionModel("compatible", 0.02, 300.0, 0.9, 0.5)
    cb = model.C_B(0.0)
    bound = 1 + 1.5 * math.log(1.5) - 0.5 * math.log(0.5)
    assert cb <= bound + 1e-9, (cb, bound)
    # certificate number from the manuscript
    assert 0.01 + (1 + 2 * math.log(2)) / 301 < 0.018
    # Bernoulli function and equilibrium of the solver: Gaussian is stationary for G=0
    flow = DensityFlow(model, "compatible", 5.0, 1.0, L=8.0, N=400)
    flow.model.dV = lambda lam: 0.0
    p = initial_gauss(flow.theta, flow.h)
    q = flow.step_implicit(p, 0.01)
    assert np.max(np.abs(q - p)) < 1e-6 * np.max(p), np.max(np.abs(q - p))
    print("quadrature/solver checks passed: (1-beta)C_B(Gamma) =", round(cb, 6), "<=", round(bound, 6))


if __name__ == "__main__":
    import sys
    quadrature_checks()
    which = sys.argv[1:] or ["plateau", "compatible", "audit"]
    if "plateau" in which:
        run_plateau()
    if "compatible" in which:
        run_compatible()
    if "audit" in which:
        run_audit()
