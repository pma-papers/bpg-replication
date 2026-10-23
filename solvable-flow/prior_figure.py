"""Figure and table of the prior sweep (reads prior-flow.json)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, SEQ, TEXT2, MUTED, TEXTW, FS, MK, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from regularizer_figure import sci  # noqa: E402

D = json.loads((ROOT / "prior-flow.json").read_text())
R = D["runs"]


def label(r):
    return f"$\\sigma={r['sigma']:g}$, $x_0={r['x0']:g}$, $\\kappa={r['kappa']:g}$"


def main():
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.2), gridspec_kw=dict(wspace=0.55))
    # (a) observed energy rate against kappa / sigma^2
    ax = axes[0]
    xs = np.array([r["kappa_lambda"] for r in R.values()])
    grid = np.geomspace(xs.min() / 2, xs.max() * 2, 50)
    ax.plot(grid, 4 * grid, color=MUTED, ls=":", lw=0.9, label="$4\\kappa\\lambda_W$")
    ax.plot(grid, 2 * grid, color=MUTED, ls="--", lw=0.9, label="$2\\kappa\\lambda_W$")
    for r in R.values():
        shifted = r["x0"] != 0
        ax.plot([r["kappa_lambda"]], [r["rate_energy"]], marker="s" if shifted else "o", ms=4.5, mec="white", mew=0.6,
                color=PAL[1] if shifted else PAL[0], ls="none")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$\\kappa\\lambda_W=\\kappa/\\sigma^2$"); ax.set_ylabel("observed rate of the objective gap")
    handles = [Line2D([], [], marker="o", ls="none", color=PAL[0], ms=4.5, label="centered prior, $x_0=0$"),
               Line2D([], [], marker="s", ls="none", color=PAL[1], ms=4.5, label="shifted prior, $x_0>0$"),
               Line2D([], [], color=MUTED, ls=":", label="$4\\kappa\\lambda_W$"), Line2D([], [], color=MUTED, ls="--", label="$2\\kappa\\lambda_W$")]
    legend_below(fig, handles, ncol=1, axes=[ax]); panel_label(ax, "a")
    # (b) minimizer's clipped value against the prior policy's
    ax = axes[1]
    lo = min(min(r["prior"]["V_m"], r["final"]["V_m"]) for r in R.values()) - 0.02
    hi = max(max(r["prior"]["V_m"], r["final"]["V_m"]) for r in R.values()) + 0.02
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls=":")
    KMK = {10.0: "o", 1.0: "s", 4.0: "^", 0.25: "D"}   # markers beside the strength colours; open markers: shifted priors
    for r in R.values():
        col = {10.0: SEQ[5], 1.0: SEQ[3], 4.0: PAL[2], 0.25: PAL[3]}.get(r["kappa"], PAL[4])
        shifted = r["x0"] != 0
        ax.plot([r["prior"]["V_m"]], [r["final"]["V_m"]], marker=KMK.get(r["kappa"], "v"), ms=4.5, mec=(col if shifted else "white"), mfc=("white" if shifted else col), mew=(1.0 if shifted else 0.6), color=col, ls="none")
    ax.axhline(D["stats_mu0"]["V_m"], color=TEXT2, lw=0.7, ls="--")
    ax.annotate("plateau", (hi - 0.01, D["stats_mu0"]["V_m"]), textcoords="offset points", xytext=(0, 3), fontsize=FS, color=TEXT2, ha="right")
    handles = [Line2D([], [], marker="o", ls="none", color=SEQ[5], ms=4.5, label="$\\kappa=10$"),
               Line2D([], [], marker="s", ls="none", color=SEQ[3], ms=4.5, label="$\\kappa=1$"),
               Line2D([], [], marker="^", ls="none", color=PAL[2], ms=4.5, label="$\\kappa=4$, $\\sigma=2$"),
               Line2D([], [], marker="D", ls="none", color=PAL[3], ms=4.5, label="$\\kappa=1/4$, $\\sigma=1/2$"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec=TEXT2, mew=1.0, ms=4.5, label="open: shifted prior")]
    ax.set_xlabel("$V_m$ of the prior policy $\\pi_\\Gamma$"); ax.set_ylabel("$V_m$ of the minimizer"); panel_label(ax, "b")
    legend_below(fig, handles, ncol=1, axes=[ax])
    # (c) prior and minimizer densities for the width sweep at kappa = 10
    ax = axes[2]
    for sigma, col, mk in ((0.5, SEQ[2], "o"), (1.0, SEQ[4], "s"), (2.0, SEQ[6], "^")):
        r = R[f"w{sigma:g}/k10"]
        th = np.array(r["density"]["theta"])
        ax.plot(th, r["density"]["phi"], color=col, ls=":", lw=0.9)
        ax.plot(th, r["density"]["p_final"], color=col, marker=mk, markevery=(len(th) // 20, len(th) // 10), ms=3.5, mec="white", mew=0.4, label=f"$\\sigma={sigma:g}$")
    ax.axvspan(-1, 1, color=MUTED, alpha=0.12, lw=0)
    ax.annotate("$\\chi=0$", (0, 0.72), fontsize=FS, color=TEXT2, ha="center")
    ax.set_xlim(-5, 5); ax.set_xlabel("$\\theta$"); ax.set_ylabel("density")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax]); panel_label(ax, "c")
    save(fig, "prior-flow", ROOT)
    # table
    rows = []
    order = sorted(R.values(), key=lambda r: (r["x0"], r["sigma"], -r["kappa"]))
    for r in order:
        rows.append(f"${r['kappa']:g}$ & ${r['x0']:g}$ & ${r['sigma']:g}$ & ${r['kappa_lambda']:g}$ & ${r['rate_energy']:.3g}$ & ${r['rate_W2']:.3g}$ & "
                    f"${r['prior']['c']:.3f}$ & ${r['prior']['V_m']:.4f}$ & ${100 * r['prior']['q']:.2f}$ & "
                    f"${r['final']['V_m']:.4f}$ & ${100 * r['final']['q']:.2f}$ & {sci(r['final']['H'])} \\\\")
    tex = ("\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{rrrrrrrrrrrr}\n\\toprule\n"
           " & & & & \\multicolumn{2}{c}{observed rates} & \\multicolumn{3}{c}{prior policy $\\pi_\\Gamma$} & \\multicolumn{3}{c}{minimizer} \\\\\n"
           "\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}\n"
           "$\\kappa$ & $x_0$ & $\\sigma$ & $\\kappa\\lambda_W$ & gap & $W_2$ & $c_\\Gamma$ & $V_m$ & $q_\\beta$ (\\%) & $V_m$ & $q_\\beta$ (\\%) & $\\KL$ \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (ROOT / "prior-table.tex").write_text(tex)
    for r in order:
        print(f"k={r['kappa']:5g} x0={r['x0']:3g} s={r['sigma']:4g} kl={r['kappa_lambda']:5g} rateE={r['rate_energy']:8.4g} prior V={r['prior']['V_m']:.4f} q={100*r['prior']['q']:.2f} | min V={r['final']['V_m']:.4f} q={100*r['final']['q']:.2f} KL={r['final']['H']:.2g}")
    print("mu0", D["stats_mu0"]["V_m"], 100 * D["stats_mu0"]["q"])


if __name__ == "__main__":
    main()
