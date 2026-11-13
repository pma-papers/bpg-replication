"""Revision 76: the validation bounds of Sect. 5.2 of the paper on the wear-allowance example (for Appendix G.1 of
the Online Resource).

The candidate is the incumbent rule of the wear-allowance study with its dispersion: the rate
min{1, z/(3b)} + 0.05 x (standard normal), clipped to [0, 1]. The error probability is zeta = 0.05, and 100,000
validation episodes are drawn. In this example maintenance access arrives at a geometric time, so the geometric risk
q_beta is the probability of a breakdown before access, the requirement of the example itself. The empirical
certificate bounds q_beta by
    [ubar - (1-beta) V_hat_H] / (m + ubar) + (1 - beta^H) sqrt(log(1/zeta) / (2N)) + m beta^H / (m + ubar),
and the bound on p_T is beta^(-T) times this bound, with beta^(H-T) in the last term. The bound charges the whole
range ubar = 1 of the utility as if it were a failure cost, so it is informative when the price is large against
ubar; the figure covers the prices from 1 to 300 (the study's prices reach 30).
(a) The three terms of the bound on q_beta at m = 30 and H = 200 against the number of episodes, with q_beta.
(b) The bound on q_beta from all episodes against the price for H = 50, 75 and 200, with q_beta and with the left
    sides of the necessary condition at T = 0.
(c) The bound on p_T against the horizon T at three prices (H = 200), with p_T of the candidate.
(d) A task that ends after H actions: the bound on p_10 against H at three prices, with p_10 of the candidate.
2026-11-13: redrawn. The first version showed the price 1 and constant rates, where every bound is close to one.
Output: validation-bounds-v76 (pdf/png) and validation-bounds-v76.json. About a minute.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BETA, THETA_B, UBAR, ZETA = 0.95, 0.1, 1.0, 0.05
N, HMAX, M_A, T_D = 100_000, 200, 30.0, 10
HS, MS_C, MS_D = (50, 75, 200), (30.0, 100.0, 300.0), (10.0, 30.0, 100.0)
PRICES = np.sort(np.r_[np.geomspace(1.0, 300, 26), 30.0])
# The legends print equation numbers of the paper (M-) and of its Online Resource (O-). They are recorded in the
# JSON output under "cited", and the manuscript's build compares them with the numbers of the typeset documents.
CITED = {"M-eq:validation-resolution-floor": "37"}
N_F = CITED["M-eq:validation-resolution-floor"]


def paths(n, H, seed):
    """Failure times and the pre-failure combined reward u - B (nan from failure on) of n paths of the candidate."""
    rng = np.random.default_rng(seed)
    z = np.ones(n); s = np.zeros(n); alive = np.ones(n, bool); tau = np.full(n, H + 1); pre = np.full((n, H), np.nan)
    for t in range(H):
        a = np.clip(np.minimum(1.0, z / 3.0) + 0.05 * rng.standard_normal(n), 0.0, 1.0)
        u = (1 + s / 2) * a / 1.5
        xi = rng.gamma(0.5, 2.0, n); w = rng.uniform(-1, 1, n)
        zn = z - 0.1 * a * xi
        fail = alive & (zn < 0); surv = alive & ~fail
        with np.errstate(divide="ignore"):
            B = np.where(zn > 0, THETA_B * np.log1p(1 / np.where(zn > 0, zn, 1.0)), np.inf)
        pre[surv, t] = (u - B)[surv]
        tau[fail] = t; alive = surv; z = np.where(alive, zn, z); s = 0.8 * s + 0.2 * w
    return tau, pre


disc = BETA ** np.arange(HMAX)
tau, pre = paths(N, HMAX, 31)                                  # u - B does not depend on the price: one sample serves every price
q_beta = float(np.where(tau < HMAX, BETA ** tau, 0.0).mean())  # E[beta^tau; tau < infinity], up to beta^200
p = lambda T: float((tau <= T).mean())


def clipped(m, H, n=N):
    """The clipped rewards of the first n episodes for H actions at the price m."""
    return np.where(np.isnan(pre[:n, :H]), -m, np.maximum(-m, pre[:n, :H]))


def parts(m, H, n=N, T=0):
    """The plug-in estimate, the sampling error and the truncation error of the bound on p_T (on q_beta for T = 0)."""
    lam = np.sqrt(np.log(1 / ZETA) / (2 * n))
    V = float((clipped(m, H, n) * disc[None, :H]).sum(1).mean())
    return (UBAR - (1 - BETA) * V) / ((m + UBAR) * BETA ** T), (1 - BETA ** H) / BETA ** T * lam, m / (m + UBAR) * BETA ** (H - T)


left_side = lambda H, n, T=0: BETA ** (H - T) + (1 - BETA ** H) / BETA ** T * np.sqrt(np.log(1 / ZETA) / (2 * n))
Ngrid = np.unique(np.geomspace(100, N, 30).astype(int))
P = np.array([parts(M_A, HMAX, int(n)) for n in Ngrid])
cert = {H: np.array([sum(parts(m, H)) for m in PRICES]) for H in HS}
assert all((cert[H] >= q_beta).all() for H in HS), "a bound lies below the sample value of q_beta"

fig, axes = plt.subplots(2, 2, figsize=(TEXTW, 4.6), gridspec_kw=dict(wspace=0.45, hspace=0.95))
LG = dict(fontsize=FS - 1.5, handlelength=1.8, labelspacing=0.3, loc="upper center", bbox_to_anchor=(0.5, -0.30))
ax = axes[0, 0]
ax.stackplot(Ngrid, P[:, 2], P[:, 1], P[:, 0], colors=[MUTED, PAL[3], PAL[0]], alpha=0.55, labels=["truncation error", "sampling error", "plug-in estimate"])
ax.axhline(q_beta, color="k", lw=0.9, label="$q_\\beta$ of the candidate")
ax.set_xscale("log"); ax.set_ylim(0, 0.2); ax.set_xlabel("validation episodes $N$"); ax.set_ylabel("bound on $q_\\beta$")
ax.legend(ncol=2, **LG); panel_label(ax, "a")
ax = axes[0, 1]
for H, c, ls in zip(HS, (PAL[2], PAL[0], PAL[3]), ("-", "--", "-.")):
    ax.plot(PRICES, np.minimum(1, cert[H]), color=c, lw=1.2, ls=ls, label=f"$H={H}$")
    ax.axhline(left_side(H, N), color=c, lw=0.7, ls=":")
ax.axhline(q_beta, color="k", lw=0.9, label="$q_\\beta$ of the candidate")
ax.plot([], [], color=TEXT2, lw=0.7, ls=":", label=f"left side of ({N_F}), $T=0$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3e-3, 1.5); ax.set_xlabel("price $m$"); ax.set_ylabel("bound on $q_\\beta$"); ax.legend(ncol=2, **LG); panel_label(ax, "b")
# the bound on p_T at every horizon T, from the same sample
TG = np.arange(0, 41)
ax = axes[1, 0]
byT = {}
for m, c, ls in zip(MS_C, (PAL[2], PAL[0], PAL[7]), ("-", "--", "-.")):
    byT[m] = np.array([min(1.0, sum(parts(m, HMAX, N, int(T_)))) for T_ in TG])
    ax.plot(TG, byT[m], color=c, lw=1.2, ls=ls, label=f"$m={m:g}$")
pTG = np.array([p(int(T_)) for T_ in TG])
assert all((byT[m] >= pTG).all() for m in MS_C)
ax.axhline(0, color=MUTED, lw=0.5)
ax.plot(TG, pTG, color="k", lw=1.2, label="$p_T$ of the candidate")
ax.set_ylim(-0.03, 0.4);          # the axis starts below zero, so that the candidate's line stands clear of it
ax.spines["bottom"].set_position(("data", -0.03)); ax.set_xlabel("horizon $T$"); ax.set_ylabel("bound on $p_T$"); ax.legend(ncol=2, **LG); panel_label(ax, "c")
# a task that ends after H actions: the price is charged from failure to action H - 1
HG = np.arange(T_D + 5, HMAX + 1)
bound = {}
for m in MS_D:
    VH = np.cumsum(clipped(m, HMAX) * disc[None, :], axis=1).mean(0)       # the value of the task of H actions, H = 1, ..., 200
    bound[m] = np.array([min(1.0, (UBAR * (1 - BETA ** H) / (1 - BETA) - VH[H - 1]) / ((m + UBAR) * BETA ** T_D * (1 - BETA ** (H - T_D)) / (1 - BETA))) for H in HG])
pTD = p(T_D)
assert all((bound[m] >= pTD).all() for m in MS_D)
ax = axes[1, 1]
for m, c, ls in zip(MS_D, (PAL[3], PAL[2], PAL[0]), (":", "-", "--")):
    ax.plot(HG, bound[m], color=c, lw=1.2, ls=ls, label=f"$m={m:g}$")
ax.axhline(0, color=MUTED, lw=0.5)
ax.axhline(pTD, color="k", lw=1.2, label=f"$p_{{{T_D}}}$ of the candidate")
ax.set_ylim(-0.03, 0.4); ax.set_xlabel("task length $H$"); ax.set_ylabel(f"bound on $p_{{{T_D}}}$"); ax.legend(ncol=2, **LG); panel_label(ax, "d")
out = dict(cited=CITED, candidate="incumbent rule min{1, z/(3b)} with dispersion 0.05 of the rate", episodes=N, zeta=ZETA, q_beta=q_beta,
           p_T={str(T_): p(T_) for T_ in (5, 10, 20, 40)},
           bound_on_q_beta_at_m30_H200_by_episodes={str(int(n)): float(s) for n, s in zip(Ngrid[[0, len(Ngrid) // 2, -1]], P.sum(1)[[0, len(Ngrid) // 2, -1]])},
           terms_at_m30_H200_all_episodes=dict(zip(("plug_in", "sampling", "truncation"), (float(x) for x in P[-1]))),
           bound_on_q_beta_by_price={f"{m:g}": {str(H): float(sum(parts(m, H))) for H in HS} for m in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)},
           left_side_of_the_condition_T0={str(H): float(left_side(H, N)) for H in HS},
           bound_on_p_T={f"{m:g}": {str(T_): float(byT[m][T_]) for T_ in (0, 10, 20, 40)} for m in MS_C},
           task_of_H_actions=dict(T=T_D, p_T=pTD, bound_at_H50={f"{m:g}": float(bound[m][50 - HG[0]]) for m in MS_D},
                                  bound_at_H200={f"{m:g}": float(bound[m][-1]) for m in MS_D}, smallest_bound={f"{m:g}": float(bound[m].min()) for m in MS_D}))
(ROOT / "validation-bounds-v76.json").write_text(json.dumps(out, indent=1) + "\n")
save(fig, "validation-bounds-v76", ROOT)
print(json.dumps(out, indent=1))
