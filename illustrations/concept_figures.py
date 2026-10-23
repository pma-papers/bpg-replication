"""Illustration figures of revision 46 (theory part), computed from explicit toy models.

  logic.pdf      (revision 46; superseded by introduction_schematic.py) Figure 1: (a) one-step reward against the next reserve for an infinite barrier, the
                 bounded barrier and a failure-only price; (b) the risk-utility plane of a represented
                 class (schematic frontier) with its risk floor, a price, a tolerance and the methods.
  identity.pdf   (revision 46; superseded by running_example_figures.py) Section 3: the identity V_m = U - I_m - m beta^tau/(1-beta) on one trajectory.
  curvature.pdf  Section 4: the curvature budget along a generalized geodesic (toy one-dimensional
                 objective) and the minimizer's position as the regularizer's strength grows.

Run: ~/.venvs/constrained-rl/bin/python concept_figures.py  (writes into ../figures/).
Every number printed in a panel is computed here; the toy parameters are the constants below.
"""
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
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.55), gridspec_kw=dict(wspace=0.32, width_ratios=[1.0, 1.15]))
    # (a) one-step reward against the next reserve
    ax = axes[0]
    z = np.linspace(-0.25, 1.0, 2001)
    zp = z[z > 1e-4]
    inf_b = U_ACT - barrier(zp)
    ax.axvspan(-0.25, 0.0, color=MUTED, alpha=0.15, lw=0)
    ax.text(-0.125, 0.85, "failure", ha="center", va="center", fontsize=9, color=TEXT2)
    ax.plot(zp, np.maximum(inf_b, -3.5), color=MUTED, lw=1.0, ls="--", label="infinite barrier $u-B(z')$")
    ax.annotate("infinite barrier $\\to-\\infty$", xy=(0.004, -1.45), xytext=(0.03, -1.42), fontsize=9, color=TEXT2, va="center")
    bounded = np.where(z > 0, np.maximum(U_ACT - barrier(np.maximum(z, 1e-9)), -M_PRICE), -M_PRICE)
    ax.plot(z, bounded, color=PAL[0], lw=1.6, label="bounded barrier $\\max\\{-m,\\,u-B(z')\\}$")
    fo = np.where(z > 0, U_ACT, -M_PRICE)
    ax.plot(z, fo, color=PAL[1], lw=1.1, ls=(0, (4, 2)), label="failure-only price")
    ax.fill_between(z[z > 0], bounded[z > 0], U_ACT, color=PAL[0], alpha=0.12, lw=0)
    ax.text(0.33, 0.28, "interior loss", fontsize=9, color=PAL[0])
    ax.axhline(-M_PRICE, color=TEXT2, lw=0.6, ls=":")
    ax.text(0.03, -M_PRICE - 0.08, "$-m$: charged every period from failure onward", fontsize=9, color=TEXT2, va="top")
    ax.set_xlim(-0.25, 1.0); ax.set_ylim(-2.2, 1.05)
    ax.set_xlabel("next reserve $z'$ (budget $b=1$)"); ax.set_ylabel("one-step reward")
    ax.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.0, 0.0))
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
    ax.text(0.40, 0.985, "frontier of the class", fontsize=9, color=TEXT2, ha="center", va="bottom")
    ax.annotate("below the risk floor of a full-support class:\nno policy, and no multiplier settles; an\nalmost-sure requirement ($q_\\beta=0$) needs\na method that restricts the actions",
                xy=(qf * 0.5, 0.12), xytext=(0.15, 0.06), fontsize=9, color=TEXT2, va="bottom",
                arrowprops=dict(arrowstyle="->", color=TEXT2, lw=0.6))
    # a price: tangent with slope m/(1-beta) (scaled for the schematic)
    slope = 3.2
    i = int(np.argmin(np.abs(dU - slope)))
    qm, Um = q[i], U[i]
    xs = np.linspace(qm - 0.05, qm + 0.12, 10)
    ax.plot(xs, Um + slope * (xs - qm), color=PAL[0], lw=1.0)
    ax.plot([qm], [Um], marker="o", color=PAL[0], ms=5, ls="none")
    ax.annotate("fixed-price optimum:\nslope $m/(1-\\beta)$", xy=(qm, Um), xytext=(qm + 0.045, Um - 0.36),
                fontsize=9, color=PAL[0], arrowprops=dict(arrowstyle="->", color=PAL[0], lw=0.6))
    # a tolerance and the prescribed-level target
    alpha = 0.30
    Ua = 1.0 - np.exp(-(alpha - qf) / s) * 0.75
    ax.axvline(alpha, color=PAL[1], lw=0.8, ls=":")
    ax.plot([alpha], [Ua], marker="s", color=PAL[1], ms=4.5, ls="none")
    ax.annotate("tolerance $\\alpha$:\nprescribed-level target", xy=(alpha, Ua), xytext=(alpha + 0.03, Ua - 0.33),
                fontsize=9, color=PAL[1], arrowprops=dict(arrowstyle="->", color=PAL[1], lw=0.6))
    ax.annotate("", xy=(alpha, Um - 0.06), xytext=(qm, Um - 0.06), arrowprops=dict(arrowstyle="<->", color=TEXT2, lw=0.6))
    ax.text((qm + alpha) / 2, Um - 0.1, "unused risk", fontsize=9, color=TEXT2, ha="center", va="top")
    ax.set_xlim(0.0, qmax); ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("geometric risk $q_\\beta$"); ax.set_ylabel("utility net of interior loss, $U-\\mathcal{I}_m$")
    ax.set_xticks([0, qf, alpha]); ax.set_xticklabels(["0", "floor", "$\\alpha$"]); ax.set_yticks([])
    panel_label(ax, "(b)")
    save(fig, "logic", OUT)


# ---------------------------------------------------------------- Section 3 figure
def identity():
    g, H = 0.09, 40
    t = np.arange(H)
    z_next = B_BUDGET - g * (t + 1)                    # reserve after each action
    tau = int(np.argmax(z_next < 0))                    # first action after which the reserve is negative
    disc = BETA ** t
    u = np.full(H, U_ACT)
    Bz = barrier(z_next)
    loss = np.where(t < tau, np.minimum(Bz, u + M_PRICE), 0.0)
    U = float(np.sum(disc[:tau] * u[:tau]))
    I = float(np.sum(disc[:tau] * loss[:tau]))
    tail = M_PRICE * BETA ** tau / (1 - BETA)
    Vm = U - I - tail
    r = np.where(t < tau, np.maximum(-M_PRICE, u - Bz), -M_PRICE)
    Vm_direct = float(np.sum(disc * r) + (-M_PRICE) * BETA ** H / (1 - BETA))
    assert abs(Vm - Vm_direct) < 1e-9, (Vm, Vm_direct)
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.3), gridspec_kw=dict(wspace=0.3, width_ratios=[0.8, 1.2]))
    ax = axes[0]
    zz = np.concatenate([[B_BUDGET], z_next])
    ax.step(np.arange(H + 1)[: tau + 4], zz[: tau + 4], where="post", color=PAL[0])
    ax.axhline(0, color=TEXT2, lw=0.6)
    ax.axvline(tau, color=PAL[1], lw=0.8, ls=":")
    ax.text(tau - 0.3, 0.75, f"failing\naction\n$\\tau={tau}$", fontsize=9, color=PAL[1], ha="right")
    ax.set_xticks(range(0, tau + 4, 3))
    ax.set_xlabel("period $t$"); ax.set_ylabel("reserve $z_t$"); ax.set_xlim(0, tau + 4)
    panel_label(ax, "(a)")
    ax = axes[1]
    T = 26
    ax.bar(t[:tau], disc[:tau] * u[:tau], color=PAL[2], width=0.8, label="utility before failure, sums to $U$")
    ax.bar(t[:tau], -disc[:tau] * loss[:tau], color=PAL[0], width=0.8, label="interior loss, sums to $\\mathcal{I}_m$")
    ax.bar(t[tau:T], -disc[tau:T] * M_PRICE, color=PAL[1], width=0.8,
           label="price from failure onward, sums to $m\\beta^\\tau/(1-\\beta)$")
    ax.axhline(0, color=TEXT2, lw=0.6)
    ax.text(T - 0.5, 0.55, f"$U={U:.2f}$, $\\mathcal{{I}}_m={I:.2f}$,\n$m\\beta^\\tau/(1-\\beta)={tail:.2f}$,\n$V_m={Vm:.2f}$",
            fontsize=9, ha="right", va="top", color=TEXT2)
    ax.set_xlabel("period $t$ (discounted terms; the tail continues)"); ax.set_ylabel("$\\beta^t\\times$ term")
    ax.set_xlim(-0.8, T); ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(-1.15, 0.72)
    panel_label(ax, "(b)")
    save(fig, "identity", OUT)
    return dict(tau=tau, U=U, I=I, tail=tail, Vm=Vm)


# ---------------------------------------------------------------- Section 4 figure
def curvature():
    th = np.linspace(0.0, 1.0, 4001)
    amp, freq = 0.025, 3.0
    f = 0.35 * (th - 0.85) ** 2 + amp * np.cos(2 * np.pi * freq * th)       # -V_m along the curve (toy)
    f2 = np.gradient(np.gradient(f, th), th)
    C = float(-f2.min())                                                    # curvature deficit of the toy
    th_prior = 0.15

    def total(lam):
        return f + 0.5 * lam * (th - th_prior) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.3), gridspec_kw=dict(wspace=0.32))
    ax = axes[0]
    ax.plot(th, f - f.min(), color=TEXT2, lw=1.2, label="$-V_m$, curvature $\\geq -C_m$")
    for lam, col, ls, lab in ((0.5 * C, PAL[1], "--", "$-V_m+\\kappa\\mathcal{H}$ at $\\kappa\\lambda_{\\mathcal{H}}=C_m/2$, not convex"),
                              (1.5 * C, PAL[0], "-.", "$-V_m+\\kappa\\mathcal{H}$ at $\\kappa\\lambda_{\\mathcal{H}}=1.5\\,C_m$, convex")):
        y = total(lam)
        ax.plot(th, y - y.min(), color=col, lw=1.2, ls=ls, label=lab)
        k = np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1
        ax.plot(th[k], (y - y.min())[k], marker="o", color=col, ms=3.5, ls="none")
    ax.axvline(th_prior, color=MUTED, lw=0.7, ls=":")
    ax.text(th_prior + 0.01, ax.get_ylim()[1] * 0.92 if False else 0.62, "prior", fontsize=9, color=TEXT2)
    ax.set_ylim(0, 0.68)
    ax.set_xlabel("position along a generalized geodesic"); ax.set_ylabel("objective (shifted)")
    panel_label(ax, "(a)")
    ax = axes[1]
    lams = np.geomspace(1e-3 * C, 30 * C, 300)
    mins = []
    for lam in lams:
        y = total(lam)
        mins.append(th[int(np.argmin(y))])
    ax.plot(lams / C, mins, color=PAL[0])
    ax.axvline(1.0, color=PAL[1], lw=0.8, ls=":")
    ax.text(1.15, 0.50, "threshold\n$\\kappa\\lambda_{\\mathcal{H}}=C_m$", fontsize=9, color=PAL[1])
    ax.axhline(th_prior, color=MUTED, lw=0.7, ls=":"); ax.text(1.2e-3, th_prior + 0.02, "prior", fontsize=9, color=TEXT2)
    ax.axhline(float(th[np.argmin(f)]), color=MUTED, lw=0.7, ls="--")
    ax.text(1.2e-3, float(th[np.argmin(f)]) - 0.09, "unregularized\nminimizer", fontsize=9, color=TEXT2, va="top")
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    ax.set_xlabel("regularization strength $\\kappa\\lambda_{\\mathcal{H}}/C_m$"); ax.set_ylabel("global minimizer")
    panel_label(ax, "(b)")
    legend_below(fig, *axes[0].get_legend_handles_labels(), ncol=1)
    save(fig, "curvature", OUT)
    return dict(C=C)


if __name__ == "__main__":
    # Only the curvature figure is still drawn here. The paper's logic.pdf comes from introduction_schematic.py and
    # identity.pdf from running_example_figures.py; logic() and identity() are the revision-46 versions, kept as records
    # and not called, so that running this script cannot overwrite the current figures.
    print("curvature", curvature())
