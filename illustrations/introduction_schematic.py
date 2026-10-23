"""Revision 55 introduction schematic, restored from revision 48.
Its frontier is schematic, not an empirical fit."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, LS, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT   # the figures are written beside this script (logic.pdf, curvature.pdf)

# ---------------------------------------------------------------- toy constants
B_BUDGET, THETA, U_ACT, M_PRICE, BETA = 1.0, 0.15, 0.6, 1.0, 0.9


def barrier(z):
    z = np.asarray(z, dtype=float)
    out = np.full_like(z, np.inf)
    pos = z > 0
    out[pos] = THETA * np.log1p(B_BUDGET / z[pos])
    return out


# ---------------------------------------------------------------- Figure 1
def logic():
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.6), gridspec_kw=dict(wspace=0.3, width_ratios=[1.0, 1.15]))
    # (a) one-step reward against the next reserve
    ax = axes[0]
    z = np.linspace(-0.25, 1.0, 2001)
    zp = z[z > 1e-4]
    inf_b = U_ACT - barrier(zp)
    ax.axvspan(-0.25, 0.0, color=MUTED, alpha=0.15, lw=0)
    ax.text(-0.125, 0.85, "failure", ha="center", va="center", fontsize=9, color=TEXT2)
    ax.plot(zp, np.maximum(inf_b, -3.5), color=MUTED, lw=1.0, ls="--", label="infinite barrier $u-B(z')$")
    ax.annotate("$\\to-\\infty$", xy=(0.004, -1.45), xytext=(0.03, -1.55), fontsize=9, color=TEXT2, va="center")
    bounded = np.where(z > 0, np.maximum(U_ACT - barrier(np.maximum(z, 1e-9)), -M_PRICE), -M_PRICE)
    ax.plot(z, bounded, color=PAL[0], lw=1.6, label="bounded barrier $\\max\\{-m,\\,u-B(z')\\}$")
    fo = np.where(z > 0, U_ACT, -M_PRICE)
    ax.plot(z, fo, color=PAL[1], lw=1.1, ls=(0, (4, 2)), label="failure-only price")
    ax.fill_between(z[z > 0], bounded[z > 0], U_ACT, color=PAL[0], alpha=0.12, lw=0)
    ax.text(0.33, 0.28, "interior loss", fontsize=9, color=PAL[0])
    ax.axhline(-M_PRICE, color=TEXT2, lw=0.6, ls=":")
    ax.text(0.03, -M_PRICE - 0.08, "$-m$ from failure onward", fontsize=9, color=TEXT2, va="top")
    ax.set_xlim(-0.25, 1.0); ax.set_ylim(-2.2, 1.05)
    ax.set_xlabel("next reserve $z'$ (budget $b=1$)"); ax.set_ylabel("one-step reward")
    panel_label(ax, "(a)")

    # (b) risk-utility plane of a represented class (schematic frontier)
    ax = axes[1]
    qf, qmax = 0.08, 0.55
    q = np.linspace(qf, qmax, 400)
    s = 0.09
    U = 1.0 - np.exp(-(q - qf) / s) * 0.75          # concave, increasing frontier of the class
    dU = 0.75 / s * np.exp(-(q - qf) / s)
    ax.axvspan(0.0, qf, color=MUTED, alpha=0.18, lw=0)
    ax.plot(q, U, color=TEXT2, lw=1.4)
    ax.text(0.43, 1.04, "frontier of the class", fontsize=9, color=TEXT2, ha="center", va="bottom")
    ax.annotate("below the floor: no policy,\nno multiplier converges",
                xy=(qf * 0.5, 0.12), xytext=(0.13, 0.05), fontsize=9, color=TEXT2, va="bottom",
                arrowprops=dict(arrowstyle="->", color=TEXT2, lw=0.6))
    # a price: tangent with slope m/(1-beta) (scaled for the schematic)
    slope = 3.2
    i = int(np.argmin(np.abs(dU - slope)))
    qm, Um = q[i], U[i]
    xs = np.linspace(qm - 0.05, qm + 0.12, 10)
    ax.plot(xs, Um + slope * (xs - qm), color=PAL[0], lw=1.0)
    ax.plot([qm], [Um], marker="o", color=PAL[0], ms=5, ls="none")
    ax.annotate("fixed-price optimum,\nslope $m/(1-\\beta)$", xy=(qm, Um), xytext=(qm + 0.03, Um - 0.44),
                fontsize=9, color=PAL[0], arrowprops=dict(arrowstyle="->", color=PAL[0], lw=0.6))
    # a tolerance and the prescribed-level target
    alpha = 0.30
    Ua = 1.0 - np.exp(-(alpha - qf) / s) * 0.75
    ax.axvline(alpha, color=PAL[1], lw=0.8, ls=":")
    ax.plot([alpha], [Ua], marker="s", color=PAL[1], ms=4.5, ls="none")
    ax.annotate("tolerance $\\alpha$", xy=(alpha, Ua), xytext=(alpha + 0.05, Ua - 0.14),
                fontsize=9, color=PAL[1], arrowprops=dict(arrowstyle="->", color=PAL[1], lw=0.6))
    ax.annotate("", xy=(alpha, Um - 0.06), xytext=(qm, Um - 0.06), arrowprops=dict(arrowstyle="<->", color=TEXT2, lw=0.6))
    ax.text((qm + alpha) / 2, Um - 0.1, "unused risk", fontsize=9, color=TEXT2, ha="center", va="top")
    ax.set_xlim(0.0, qmax); ax.set_ylim(0.0, 1.14)
    ax.set_xlabel("geometric risk $q_\\beta$"); ax.set_ylabel("utility net of interior loss")
    ax.set_xticks([0, qf, alpha]); ax.set_xticklabels(["0", "floor", "$\\alpha$"]); ax.set_yticks([])
    panel_label(ax, "(b)")
    legend_below(fig, *axes[0].get_legend_handles_labels(), ncol=1, axes=[axes[0]])
    save(fig, "logic", OUT)


# ---------------------------------------------------------------- Section 3 figure

if __name__ == "__main__":
    logic()
