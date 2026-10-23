"""Regularization paths and bounds versus behavior (reads prior-path.json and the wear-anchor campaign)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

D = json.loads((ROOT / "prior-path.json").read_text())
P = D["params"]
PRIORS = [tuple(p) for p in D["priors"]]
KAPPAS = list(D["kappas"])
COL = {(0.0, 0.5): PAL[0], (0.0, 1.0): PAL[1], (0.0, 2.0): PAL[2], (2.0, 1.0): PAL[3]}
LAB = {(0.0, 0.5): "$N(0,1/4)$", (0.0, 1.0): "$N(0,1)$", (0.0, 2.0): "$N(0,4)$", (2.0, 1.0): "$N(2,1)$"}


def runs_of(prior):
    x0, sigma = prior
    return [D["runs"][f"path/x{x0:g}s{sigma:g}/k{k:g}"] for k in KAPPAS]


def allowance(kappa, R_Gamma, L0=1.0, cH=2.0, T=0):
    return 0.5 * L0 * math.sqrt((T + 1) * cH * R_Gamma / kappa)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.3), gridspec_kw=dict(wspace=0.45))
    # (a) minimizer's tilt against kappa, with the prior policies' tilts
    ax = axes[0]
    for prior in PRIORS:
        rs = runs_of(prior)
        ax.plot(KAPPAS, [r["final"]["c"] for r in rs], marker="o", ms=3.5, mec="white", mew=0.5, color=COL[prior], label=LAB[prior])
        ax.axhline(rs[0]["prior"]["c"], color=COL[prior], lw=0.7, ls=":")
    c_sup = D["curve"]["c"][int(np.argmax(D["curve"]["V_m"]))]
    ax.axhline(c_sup, color=TEXT2, lw=0.8, ls="--")
    ax.annotate("unregularized optimum", (KAPPAS[-1], c_sup), textcoords="offset points", xytext=(0, -9), fontsize=6.5, color=TEXT2, ha="left")
    ax.set_xscale("log"); ax.set_xlabel("$\\kappa$"); ax.set_ylabel("tilt $\\int\\chi\\,d\\mu^*$ of the minimizer")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=6.0, borderaxespad=0.2, labelspacing=0.3, title="prior (dotted: its tilt)", title_fontsize=6.0); panel_label(ax, "a")
    # (b) the (q, U) plane: the class curve, prior policies and the paths of minimizers
    ax = axes[1]
    q = np.array(D["curve"]["q"]); U = np.array(D["curve"]["U"])
    ax.plot(100 * q, U, color=TEXT2, lw=1.0, label="represented class")
    for prior in PRIORS:
        rs = runs_of(prior)
        ax.plot([100 * r["final"]["q"] for r in rs], [r["final"]["U"] for r in rs], marker="o", ms=3, mec="white", mew=0.4, color=COL[prior], lw=0.9)
        ax.plot([100 * rs[0]["prior"]["q"]], [rs[0]["prior"]["U"]], marker="*", ms=8, mec="white", mew=0.5, color=COL[prior], ls="none")
    i = int(np.argmax(U))
    ax.plot([100 * q[i]], [U[i]], marker="s", ms=4, color=TEXT2, ls="none")
    ax.annotate("unregularized optimum", (100 * q[i], U[i]), textcoords="offset points", xytext=(6, -3), fontsize=6.5, color=TEXT2, va="top")
    ax.annotate("$\\mu_0$ (plateau)", (100 * D["stats_mu0"]["q"], D["stats_mu0"]["U"]), textcoords="offset points", xytext=(-4, 4), fontsize=6.5, color=TEXT2, ha="right")
    handles = [Line2D([], [], marker="*", ls="none", color=TEXT2, ms=7, label="prior policy $\\pi_\\Gamma$"),
               Line2D([], [], marker="o", ls="-", color=TEXT2, ms=3, label="minimizers, $\\kappa$ decreasing")]
    ax.legend(handles=handles, loc="center right", fontsize=6.0, borderaxespad=0.2, labelspacing=0.3)
    ax.set_xlabel("geometric risk $q_\\beta$ (%)"); ax.set_ylabel("stopped utility $U$"); panel_label(ax, "b")
    # (c) bounds versus behavior: deviation of the minimizer's risk from the prior policy's, and the allowance
    ax = axes[2]
    for prior in PRIORS:
        rs = runs_of(prior)
        dev = [abs(r["final"]["q"] - r["prior"]["q"]) for r in rs]
        ax.plot(KAPPAS, np.maximum(dev, 1e-6), marker="o", ms=3.5, mec="white", mew=0.5, color=COL[prior])
    R = P["ubar"] / (1 - P["beta"]) - runs_of((0.0, 1.0))[0]["prior"]["V_m"]
    ax.plot(KAPPAS, [allowance(k, R) for k in KAPPAS], color=TEXT2, ls="--", label="risk slack bound")
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=":")
    ax.annotate("risk is at most one", (KAPPAS[-1], 1.0), textcoords="offset points", xytext=(0, 3), fontsize=6.5, color=TEXT2, ha="left")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-6, 30)
    ax.set_xlabel("$\\kappa$"); ax.set_ylabel("$|q_\\beta(\\mu^*)-q_\\beta(\\pi_\\Gamma)|$")
    ax.legend(loc="lower left", fontsize=6.0, borderaxespad=0.2); panel_label(ax, "c")
    save(fig, "prior-path", ROOT)
    for prior in PRIORS:
        for r in runs_of(prior):
            print(f"{LAB[prior]:10s} kappa={r['kappa']:5g} c*={r['final']['c']:.3f} (prior {r['prior']['c']:.3f}) q*={100*r['final']['q']:.2f} (prior {100*r['prior']['q']:.2f}) U*={r['final']['U']:.4f} (prior {r['prior']['U']:.4f}) rate={r['rate_energy']:.3g} KL={r['final']['H']:.3g}")
    print("sup: c=%.3f q=%.2f U=%.4f; mu0: q=%.2f U=%.4f" % (c_sup, 100 * q[i], U[i], 100 * D["stats_mu0"]["q"], D["stats_mu0"]["U"]))


if __name__ == "__main__":
    main()
