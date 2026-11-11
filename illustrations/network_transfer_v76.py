"""Revision 76: the transfer of a population law to one sampled network (Theorem 13 of the paper) in a model where
everything is exact (for Appendix G.1 of the Online Resource; the visuals of the verification companion, redrawn).

The two-state model: calm and stressed, switching with probability 0.2 in each period, starting calm; in each period
before failure the action is safe or exhausting, and the action reference takes the exhausting action with
probability 0.05 when calm and 0.2 when stressed; beta = 0.95. A unit with parameter x has the score
L0 sin(x + c_s) on the exhausting action in the state s, with c = (0, pi/2) and L0 = 1, and the unit law is
N(0.3, 1). A network of n units is n independent draws; 4000 networks per width, error probability zeta = 0.1.
The utilities are 0.4 for the safe action and 1 for the exhausting one, with a failure-only price m = 1.
(a) The 90 percent quantile of the deviation of the geometric risk against the width, with eps_n of Theorem 13.
(b) The same for the clipped value, the continuing value and the stopped utility, with their bounds.
(c) The deviation of the risk through the horizon T = 10 with the direct horizon bound.
(d) The relative entropy through T between the two policies: its mean and its 90 percent quantile, with the bounds
    from Hoeffding's lemma and Markov's inequality.
Output: network-transfer-v76 (pdf/png) and network-transfer-v76.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BETA, L0, ZETA, T = 0.95, 1.0, 0.1, 10
PS = np.array([[0.8, 0.2], [0.2, 0.8]]); RHO = np.array([0.05, 0.2]); CPH = np.array([0.0, np.pi / 2])
MX, SX = 0.3, 1.0
US, UE, M, UBAR, UMINUS = 0.4, 1.0, 1.0, 1.0, 0.0
NS, REPS, SEED = np.array([10, 30, 100, 300, 1000, 3000, 10000]), 4000, 10

f_mu = L0 * np.exp(-SX ** 2 / 2) * np.sin(MX + CPH)                       # the population score in each state
policy = lambda f: RHO * np.exp(f) / (RHO * np.exp(f) + 1 - RHO)          # probability of the exhausting action
w = np.linalg.solve((np.eye(2) - BETA * PS).T, np.array([1.0, 0.0]))      # sum_t beta^t P(s_t = s)


def kl2(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def occupation(p):
    """sum_t beta^t P(tau >= t, s_t = s) from the calm state, for exhausting probabilities p of shape (reps, 2)."""
    Mx = np.eye(2)[None] - BETA * (1 - p)[:, :, None] * PS[None]
    return np.linalg.solve(np.transpose(Mx, (0, 2, 1)), np.tile([1.0, 0.0], (len(p), 1))[:, :, None])[:, :, 0]


def values(p):
    p = np.atleast_2d(p); oc = occupation(p)
    U = (oc * (1 - p) * US).sum(1); q = (oc * p).sum(1)
    V = (w[None, :] * ((1 - p) * US + p * UE)).sum(1)
    return U, V, U - M * q / (1 - BETA), q


def risk_through(p, T_):
    p = np.atleast_2d(p); a = np.tile([1.0, 0.0], (len(p), 1)); cum = np.zeros(len(p))
    for _ in range(T_ + 1):
        cum += (a * p).sum(1); a = (a * (1 - p)) @ PS
    return cum


p_mu = policy(f_mu)
U_mu, V_mu, Vm_mu, q_mu = [float(v[0]) for v in values(p_mu)]
pT_mu = float(risk_through(p_mu, T)[0])
st = np.array([1.0, 0.0]); wT = np.zeros(2)
for _ in range(T + 1):
    wT += st; st = st @ PS                                                # sum_{t <= T} P(s_t = s)
rng = np.random.default_rng(SEED)
quant = lambda x: float(np.quantile(x, 1 - ZETA))
rows = dict(dq=[], dVm=[], dV=[], dU=[], dpT=[], EDT=[], QDT=[])
for n in NS:
    X = rng.normal(MX, SX, (REPS, n))
    pn = policy(L0 * np.sin(X[:, :, None] + CPH[None, None, :]).mean(1))  # the network's policy in each state
    Kn = kl2(p_mu[None, :], pn); Dn = Kn @ w; DT = Kn @ wT
    Un, Vn, Vmn, qn = values(pn)
    assert (np.abs(qn - q_mu) <= np.sqrt(Dn / 2) + 1e-12).all()
    dp = np.abs(risk_through(pn, T) - pT_mu)
    assert (dp <= np.sqrt(DT / 2) + 1e-12).all()
    for k, v in (("dq", np.abs(qn - q_mu)), ("dVm", np.abs(Vmn - Vm_mu)), ("dV", np.abs(Vn - V_mu)), ("dU", np.abs(Un - U_mu)), ("dpT", dp), ("QDT", DT)):
        rows[k].append(quant(v))
    rows["EDT"].append(float(DT.mean()))
rows = {k: np.array(v) for k, v in rows.items()}
eps_n = np.minimum(1, L0 / (2 * np.sqrt(NS * (1 - BETA) * ZETA)))
b_h = np.minimum(1, L0 * np.sqrt(T + 1) / (2 * np.sqrt(NS * ZETA)))
assert (rows["dq"] <= eps_n).all() and (rows["dVm"] <= (M + UBAR) / (1 - BETA) * eps_n).all() and (rows["dpT"] <= b_h).all()
assert (rows["EDT"] <= (T + 1) * L0 ** 2 / (2 * NS)).all()

fig, axes = plt.subplots(2, 2, figsize=(TEXTW, 4.6), gridspec_kw=dict(wspace=0.45, hspace=0.95))
LG = dict(fontsize=FS - 1.5, handlelength=1.8, labelspacing=0.3, loc="upper center", bbox_to_anchor=(0.5, -0.30))
ax = axes[0, 0]
ax.loglog(NS, rows["dq"], "o-", ms=3.5, color=PAL[0], lw=1.2, label="$|q_\\beta(\\pi_n)-q_\\beta(\\pi_\\mu)|$")
ax.loglog(NS, eps_n, color=PAL[7], lw=1.1, ls="--", label="$\\varepsilon_n$")
ax.set_xlabel("number of units $n$"); ax.set_ylabel("deviation"); ax.legend(**LG); panel_label(ax, "a")
ax = axes[0, 1]
for k, c, mk, lab in (("dVm", PAL[7], "o", "$|V_m(\\pi_n)-V_m(\\mu)|$"), ("dV", PAL[2], "s", "$|V(\\pi_n)-V(\\pi_\\mu)|$"), ("dU", PAL[0], "^", "$|U(\\pi_n)-U(\\pi_\\mu)|$")):
    ax.loglog(NS, rows[k], marker=mk, ms=3.5, color=c, lw=1.2, label=lab)
ax.loglog(NS, (M + UBAR) / (1 - BETA) * eps_n, color=PAL[7], lw=1.1, ls="--", label="$(m+\\bar u)\\varepsilon_n/(1-\\beta)$")
ax.loglog(NS, (UMINUS + UBAR) / (1 - BETA) * eps_n, color="k", lw=1.1, ls="-.", label="$(u_-+\\bar u)\\varepsilon_n/(1-\\beta)$")
ax.set_xlabel("number of units $n$"); ax.set_ylabel("deviation"); ax.legend(ncol=1, **LG); panel_label(ax, "b")
ax = axes[1, 0]
ax.loglog(NS, rows["dpT"], "o-", ms=3.5, color=PAL[0], lw=1.2, label=f"$|p_T(\\pi_n)-p_T(\\pi_\\mu)|$, $T={T}$")
ax.loglog(NS, b_h, color=PAL[7], lw=1.1, ls="--", label="$L_0\\sqrt{T+1}/(2\\sqrt{n\\zeta})$")
ax.set_xlabel("number of units $n$"); ax.set_ylabel("deviation"); ax.legend(**LG); panel_label(ax, "c")
ax = axes[1, 1]
ax.loglog(NS, rows["EDT"], "o-", ms=3.5, color=PAL[0], lw=1.2, label="mean")
ax.loglog(NS, (T + 1) * L0 ** 2 / (2 * NS), color=PAL[0], lw=1.0, ls="--", label="$(T+1)L_0^2/(2n)$")
ax.loglog(NS, rows["QDT"], "s-", ms=3.2, color=PAL[3], lw=1.2, label="90% quantile")
ax.loglog(NS, (T + 1) * L0 ** 2 / (2 * NS * ZETA), color=PAL[3], lw=1.0, ls="-.", label="$(T+1)L_0^2/(2n\\zeta)$")
ax.set_xlabel("number of units $n$"); ax.set_ylabel("relative entropy through $T$"); ax.legend(ncol=2, **LG); panel_label(ax, "d")
out = dict(widths=NS.tolist(), networks_per_width=REPS, seed=SEED, zeta=ZETA, population=dict(q_beta=q_mu, U=U_mu, V=V_mu, V_m=Vm_mu, p_T=pT_mu),
           eps_n=eps_n.tolist(), quantiles={k: v.tolist() for k, v in rows.items()}, direct_horizon_bound=b_h.tolist())
(ROOT / "network-transfer-v76.json").write_text(json.dumps(out, indent=1) + "\n")
save(fig, "network-transfer-v76", ROOT)
print("population", out["population"]); print("dq", rows["dq"].round(5), "eps_n", eps_n.round(4))
