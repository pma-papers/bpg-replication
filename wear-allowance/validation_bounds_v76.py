"""Revision 76: the validation bounds of Sect. 5.2 of the paper on the wear-allowance example (for Appendix G.1 of
the Online Resource; the visuals of the verification companion, redrawn).

The policy runs at the constant rate 0.15; the requirement concerns the failure probability through T = 10; the
error probability is zeta = 0.05; 100,000 validation episodes, truncated after H actions.
(a) The value-based bound on p_T at the price m = 1 and H = 100 against the number of episodes, as the sum of its
    three terms (the policy term, the sampling error and the truncation tail), with the floor that no policy can
    pass, which is the sampling error and the truncation tail at their smallest.
(b) The value-based bound from 10,000 episodes against the price for H = 50, 100 and 200, with the floors and with
    the bound that the geometric risk itself gives, beta^(-T) q_beta. No episode fails by T in the sample.
(c) A task that ends after H = 50 actions, under the rate 0.8 and the price m = 10: the discounted shortfall of
    the clipped return from its upper bound on each path. Every path that fails by T lies beyond the threshold.
(d) The bound for a task of H actions against H at three prices, with the sample value of p_T.
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

BETA, THETA_B, UBAR, T, ZETA = 0.95, 0.1, 1.0, 10, 0.05
A, N, HS = 0.15, 100_000, (50, 100, 200)
PRICES = np.sort(np.r_[np.geomspace(0.03, 30, 24), 1.0])
# The legends print equation numbers of the paper (M-) and of its Online Resource (O-). They are recorded in the
# JSON output under "cited", and the manuscript's build compares them with the numbers of the typeset documents.
CITED = {"M-eq:validation-resolution-floor": "39"}
N_F = CITED["M-eq:validation-resolution-floor"]


def rewards(a, n, H, seed, m):
    """Failure times and per-period clipped barrier rewards of n paths at the constant rate a, for H actions."""
    rng = np.random.default_rng(seed)
    z = np.ones(n); s = np.zeros(n); alive = np.ones(n, bool); tau = np.full(n, H + 1); r = np.full((n, H), -float(m))
    for t in range(H):
        u = (1 + s / 2) * a / 1.5
        xi = rng.gamma(0.5, 2.0, n); w = rng.uniform(-1, 1, n)
        zn = z - 0.1 * a * xi
        fail = alive & (zn < 0); surv = alive & ~fail
        with np.errstate(divide="ignore"):
            B = np.where(zn > 0, THETA_B * np.log1p(1 / np.where(zn > 0, zn, 1.0)), np.inf)
        r[surv, t] = np.maximum(-m, u - B)[surv]
        tau[fail] = t; alive = surv; z = np.where(alive, zn, z); s = 0.8 * s + 0.2 * w
    return tau, r


disc = BETA ** np.arange(max(HS))
# the pre-failure combined reward u - B does not depend on the price, so one simulation serves every price
tau, r0 = rewards(A, N, max(HS), 31, 1e9)                    # with a price this large the clip does not bind before failure
pre = np.where(r0 > -1e8, r0, np.nan)                         # u - B before failure, nan from failure on; clipped at each price below


def mean_return(m, H, n):
    """Mean clipped return of the first n episodes truncated after H actions, at the price m."""
    x = np.where(np.isnan(pre[:n, :H]), -m, np.maximum(-m, pre[:n, :H]))
    return float((x * disc[None, :H]).sum(1).mean())


def parts(m, H, n):
    lam = np.sqrt(np.log(1 / ZETA) / (2 * n))
    return (UBAR - (1 - BETA) * mean_return(m, H, n)) / ((m + UBAR) * BETA ** T), (1 - BETA ** H) / BETA ** T * lam, m / (m + UBAR) * BETA ** (H - T)


floor = lambda H, n: BETA ** (H - T) + (1 - BETA ** H) / BETA ** T * np.sqrt(np.log(1 / ZETA) / (2 * n))
Ngrid = np.unique(np.geomspace(100, N, 30).astype(int))
P = np.array([parts(1.0, 100, int(n)) for n in Ngrid])
assert (P.sum(1) >= floor(100, Ngrid) - 1e-12).all()
NB = 10_000
cert = {H: np.array([sum(parts(m, H, NB)) for m in PRICES]) for H in HS}
pT = float((tau <= T).mean()); q = float(np.where(tau <= max(HS), BETA ** np.minimum(tau, max(HS)), 0.0).mean())

fig, axes = plt.subplots(2, 2, figsize=(TEXTW, 4.6), gridspec_kw=dict(wspace=0.45, hspace=0.95))
LG = dict(fontsize=FS - 1.5, handlelength=1.8, labelspacing=0.3, loc="upper center", bbox_to_anchor=(0.5, -0.30))
ax = axes[0, 0]
ax.stackplot(Ngrid, P[:, 2], P[:, 1], P[:, 0], colors=[MUTED, PAL[3], PAL[0]], alpha=0.55, labels=["truncation tail", "sampling error", "policy term"])
ax.plot(Ngrid, floor(100, Ngrid), color="k", lw=1.1, ls="--", label=f"left side of ({N_F})")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 3); ax.set_xlabel("validation episodes $N$"); ax.set_ylabel(f"bound on $p_{{{T}}}$")
ax.legend(ncol=2, **LG); panel_label(ax, "a")
ax = axes[0, 1]
for H, c, ls in zip(HS, (PAL[2], PAL[0], PAL[3]), ("-", "--", "-.")):
    ax.plot(PRICES, np.minimum(1, cert[H]), color=c, lw=1.2, ls=ls, label=f"$H={H}$")
    ax.axhline(floor(H, NB), color=c, lw=0.7, ls=":")
ax.axhline(q / BETA ** T, color="k", lw=0.9, label="$\\beta^{-T}q_\\beta$ of the policy")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("price $m$"); ax.set_ylabel(f"bound on $p_{{{T}}}$"); ax.legend(ncol=2, **LG); panel_label(ax, "b")
# a task that ends after H actions, at the riskier rate 0.8
MS, HG = (3.0, 10.0, 30.0), np.arange(T + 1, 201)
bound, keep = {}, None
for m in MS:
    tau2, r2 = rewards(0.8, 20_000, 200, 33, m)
    VH = np.cumsum(r2 * disc[None, :200], axis=1).mean(0)                 # the truncated clipped value for H = 1, ..., 200
    bound[m] = np.array([min(1.0, (UBAR * (1 - BETA ** H) / (1 - BETA) - VH[H - 1]) / ((m + UBAR) * BETA ** T * (1 - BETA ** (H - T)) / (1 - BETA))) for H in HG])
    if m == 10.0:
        keep = (tau2, r2)
tau2, r2 = keep
pT2 = float((tau2 <= T).mean())
assert all((bound[m] >= pT2 - 0.01).all() for m in MS)
HSHOW, MSHOW = 50, 10.0
D = UBAR * (1 - BETA ** HSHOW) / (1 - BETA) - r2[:, :HSHOW] @ disc[:HSHOW]
thr = (MSHOW + UBAR) * BETA ** T * (1 - BETA ** (HSHOW - T)) / (1 - BETA)
assert (D[tau2 <= T] >= thr - 1e-9).all()
ax = axes[1, 0]
bins = np.linspace(0, D.max() * 1.02, 60)
ax.hist(D[tau2 > T], bins=bins, color=PAL[2], alpha=0.6, lw=0, label="paths that survive $T$")
ax.hist(D[tau2 <= T], bins=bins, color=PAL[7], alpha=0.6, lw=0, label="paths that fail by $T$")
ax.axvline(thr, color="k", lw=1.1, ls="--", label="$(m+\\bar u)\\beta^T(1-\\beta^{H-T})/(1-\\beta)$")
ax.set_xlabel("$\\sum_{t<H}\\beta^t(\\bar u-r_m)$"); ax.set_ylabel("paths"); ax.legend(ncol=1, **{**LG, "bbox_to_anchor": (0.5, -0.38)}); panel_label(ax, "c")
ax = axes[1, 1]
for m, c, ls in zip(MS, (PAL[2], PAL[0], PAL[7]), ("-", "--", "-.")):
    ax.plot(HG, bound[m], color=c, lw=1.2, ls=ls, label=f"$m={m:g}$")
ax.axhline(pT2, color="k", lw=0.9, label=f"$p_{{{T}}}$ of the policy")
ax.set_ylim(0, 1.02); ax.set_xlabel("task length $H$"); ax.set_ylabel(f"bound on $p_{{{T}}}$"); ax.legend(ncol=2, **LG); panel_label(ax, "d")
out = dict(cited=CITED, rate=A, episodes=N, T=T, zeta=ZETA, p_T=pT, q_beta=q, bound_at_m1_H100={str(int(n)): float(s) for n, s in zip(Ngrid[[0, len(Ngrid) // 2, -1]], P.sum(1)[[0, len(Ngrid) // 2, -1]])},
           floor_at_10000_episodes={str(H): float(floor(H, NB)) for H in HS}, smallest_bound_over_prices={str(H): float(cert[H].min()) for H in HS},
           task_of_H_actions=dict(rate=0.8, p_T=pT2, threshold=float(thr), bound_at_H200={f"{m:g}": float(bound[m][-1]) for m in MS}, smallest_bound={f"{m:g}": float(bound[m].min()) for m in MS}))
(ROOT / "validation-bounds-v76.json").write_text(json.dumps(out, indent=1) + "\n")
save(fig, "validation-bounds-v76", ROOT)
print(json.dumps(out, indent=1))
