"""Revision 76: the figure of the gradient flow for the integrands, in two panels, and the panel below the threshold.

The figure of revisions 41 to 75 had three panels. In revision 76 the paper keeps the objective gap above the
threshold (a) and the improvement of the minimizer over the prior against the strength (b); the objective gap at
the strength 1, below the threshold, is a figure of Online Resource 1. The data and the drawing of each panel are
those of regularizer_figure_v41.py (imported), with the labels of revision 46 and the axis label "learning time".
Outputs: regularizer-flow-v76 (pdf/png), regularizer-flow-below-v76 (pdf/png). The digests of the plotted data are
compared with regularizer-digests-v41.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))
import regularizer_figure_v41 as m  # noqa: E402
from plotstyle import TEXT2, MUTED, TEXTW, FS, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from digest_v41 import digest  # noqa: E402

m.LABEL = {"KL": "relative entropy", "r1.5": "rel. entropy $+\\,\\omega G$, $r=3/2$",
           "r2": "rel. entropy $+\\,\\omega\\chi^2$ ($r=2$)", "r3": "rel. entropy $+\\,\\omega G$, $r=3$"}
D, REF = m.D, json.loads((HERE / "regularizer-digests-v41.json").read_text())


def main():
    C_sharp = m.sharp_constant()["C_sharp"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.2), gridspec_kw=dict(wspace=0.42))
    ax = axes[0]
    arrs = m.panel_decay(ax, "cert", "", 0.004, C_sharp=C_sharp)
    assert digest(*arrs) == REF["regularizer/a flow curves"]["after"]
    ax.set_xlabel("learning time $\\ell$"); ax.set_ylabel("objective gap, relative to $E_0$"); panel_label(ax, "a")
    handles, labels = ax.get_legend_handles_labels()
    ax = axes[1]
    ks = [D["runs"][f"{k}/KL"]["kappa"] for k in m.KAPPAS]
    VG = D["stats_Gamma"]["V_m"]
    arrs = [ks]
    for reg in m.ORDER:
        y = np.array([D["runs"][f"{k}/{reg}"]["final"]["V_m"] - VG for k in m.KAPPAS]); arrs.append(y)
        ax.plot(ks, np.maximum(y, 1e-7), marker=m.MARK[reg], ms=3.5, mec="white", mew=0.5, color=m.COL[reg], label=m.LABEL[reg])
    assert digest(*arrs) == REF["regularizer/c improvement over the prior"]["after"]
    ax.axhline(D["sup_V_m"] - VG, color=TEXT2, ls="-", lw=0.8)
    ax.annotate("$\\sup V_m-V_m(\\Gamma)$", (ks[1], D["sup_V_m"] - VG), textcoords="offset points", xytext=(0, 3), fontsize=FS, color=TEXT2, ha="center")
    ax.axvline(D["kappa_cert"], color=MUTED, lw=0.7, ls=":")
    ax.annotate("threshold", (D["kappa_cert"], 3e-7), textcoords="offset points", xytext=(-3, 0), fontsize=FS, color=TEXT2, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-7, 2)
    ax.set_xlabel("strength $\\kappa$"); ax.set_ylabel("$V_m(\\mu^*)-V_m(\\Gamma)$"); panel_label(ax, "b")
    legend_below(fig, handles, labels, ncol=3)
    save(fig, "regularizer-flow-v76", HERE)

    fig, ax = plt.subplots(figsize=(0.5 * TEXTW, 2.2))
    arrs = m.panel_decay(ax, "k1", "", 8.0)
    assert digest(*arrs) == REF["regularizer/b flow curves"]["after"]
    ax.set_xlabel("learning time $\\ell$"); ax.set_ylabel("objective gap, relative to $E_0$")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=1)
    save(fig, "regularizer-flow-below-v76", HERE)


if __name__ == "__main__":
    main()
