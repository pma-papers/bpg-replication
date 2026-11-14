"""Revision 76: the prior figure of Online Resource 1 in two panels.

Panel (b) of the revision-47 figure, the clipped value of the minimizer against the strength, is panel (a) of the
figure of the paper drawn by prior_strength_figure_v76.py. This script draws the two panels that stay in Online
Resource 1 with the code of prior_figure_v47.py: (a) the observed rates against the weight-decay strength and (b) the
deviation of the minimizer's risk from its prior policy's against the strength, with the bound. Their data digests are
checked against prior-digests-v41.json. Output: prior-v76.pdf/png."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V41 = HERE.parent / "solvable-flow"
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(V41))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import prior_combined_figure_v41 as m  # noqa: E402
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, MK, panel_label, legend_below  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from digest_v41 import digest  # noqa: E402
from scipy.stats import norm  # noqa: E402


def main():
    m.apply_rerun()
    assert not m.STATUS["not_converged"], m.STATUS
    D, S, P = m.D, m.S, m.P
    digests = {}
    fig, axes = plt.subplots(1, 2, figsize=(0.72 * TEXTW, 2.2), gridspec_kw=dict(wspace=0.62))
    PMK = {prior: MK[i] for i, prior in enumerate(m.PRIORS)}   # markers beside the prior colours
    # (a) rate collapse: the revision-41 drawing
    ax = axes[0]
    xs = np.array([r["kappa_lambda"] for r in S.values()]); ys = np.array([r["rate_energy"] for r in S.values()])
    digests["prior/a rate collapse"] = digest(xs, ys)
    grid = np.geomspace(xs.min() / 2, xs.max() * 2, 50)
    ax.plot(grid, 4 * grid, color=MUTED, ls=":", lw=0.9)
    ax.plot(grid, 2 * grid, color=MUTED, ls="--", lw=0.9)
    for r in S.values():
        shifted = r["x0"] != 0
        ax.plot([r["kappa_lambda"]], [r["rate_energy"]], marker="s" if shifted else "o", ms=4.5, mec="white", mew=0.6,
                color=PAL[1] if shifted else PAL[0], ls="none")
    handles = [Line2D([], [], marker="o", ls="none", color=PAL[0], ms=4.5, label="centered prior"),
               Line2D([], [], marker="s", ls="none", color=PAL[1], ms=4.5, label="shifted prior"),
               Line2D([], [], color=MUTED, ls=":", label="$4\\kappa\\lambda_W$"), Line2D([], [], color=MUTED, ls="--", label="$2\\kappa\\lambda_W$")]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$\\kappa\\lambda_W=\\kappa/\\sigma^2$")
    ax.set_ylabel("observed rate of the objective gap"); panel_label(ax, "a")
    legend_below(fig, handles, ncol=1, axes=[ax])

    # (c) bounds versus behaviour: the revision-41 drawing
    ax = axes[1]
    arrs = []
    for prior in m.PRIORS:
        rs = m.runs_of(prior)
        dev = np.array([abs(r["final"]["q"] - r["prior"]["q"]) for r in rs]); arrs.append(dev)
        ax.plot(m.KAPPAS, np.maximum(dev, 1e-6), marker=PMK[prior], ms=3.5, mec="white", mew=0.5, color=m.COL[prior])
    digests["prior/c deviations"] = digest(m.KAPPAS, *arrs)
    R = P["ubar"] / (1 - P["beta"]) - m.runs_of((0.0, 1.0))[0]["prior"]["V_m"]
    ax.plot(m.KAPPAS, [m.slack(k, R) for k in m.KAPPAS], color=TEXT2, ls="--", label="bound")
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-6, 30)
    ax.set_xlabel("$\\kappa$"); ax.set_ylabel("$|q_\\beta(\\mu^*)-q_\\beta(\\pi_\\Gamma)|$")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax]); panel_label(ax, "b")

    fig.savefig(HERE / "prior-v76.pdf", bbox_inches="tight"); fig.savefig(HERE / "prior-v76.png", dpi=200, bbox_inches="tight")
    ref = json.loads((V41 / "prior-digests-v41.json").read_text())
    check = {"a": digests["prior/a rate collapse"] == ref["prior/a rate collapse"]["after"],
             "c": digests["prior/c deviations"] == ref["prior/c deviations"]["after"]}
    print("digests equal to v41:", check)
    assert all(check.values())


if __name__ == "__main__":
    main()
