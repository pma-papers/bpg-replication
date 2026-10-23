"""Revision 47: the prior figure of the solvable flow with panel (b) redrawn.

Panel (b) of revision 41 showed the regularization paths in the plane of geometric risk and stopped
utility, measured against the unregularized optimum only. Read that way, regularization seemed only to
raise risk and lower utility. The stopped utility also leaves out the price charged from failure
onward. Panel (b) now plots the clipped value V_m = U - I_m - m q/(1-beta) of each minimizer against the
strength kappa. Two reference lines mark the plateau policy, where unregularized transport from the
initial law mu_0 stays, and the unregularized optimum of the class, which transport from mu_0 does not
reach without regularization. Panels (a) and (c) are drawn by the revision-41 code unchanged; their data
digests are checked against prior-digests-v41.json. The data are those of revision 41 (solvable-flow path
and flow files, with the revision-41 rerun substituted for the one run that had not converged).
Outputs: prior-v47.pdf/png, prior-digests-v47.json."""
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


def main():
    m.apply_rerun()
    assert not m.STATUS["not_converged"], m.STATUS
    D, S, P = m.D, m.S, m.P
    digests = {}
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.2), gridspec_kw=dict(wspace=0.72))
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

    # (b) clipped value of the minimizer against kappa
    ax = axes[1]
    K = np.array(m.KAPPAS, dtype=float)
    v_plateau = D["stats_mu0"]["V_m"]
    v_opt = float(np.max(D["curve"]["V_m"]))
    ax.axhline(v_opt, color=TEXT2, lw=0.9)
    ax.axhline(v_plateau, color=TEXT2, lw=0.9, ls="--")
    arrs = []
    x_star = K.max() * 2.2
    for prior in m.PRIORS:
        rs = m.runs_of(prior)
        v = np.array([r["final"]["V_m"] for r in rs]); arrs.append(v)
        ax.plot(K, v, marker=PMK[prior], ms=3.5, mec="white", mew=0.4, color=m.COL[prior], lw=0.9, label=m.LAB[prior])
        ax.plot([x_star], [rs[0]["prior"]["V_m"]], marker="*", ms=8, mec="white", mew=0.5, color=m.COL[prior], ls="none")
    digests["prior/b clipped value of the minimizers"] = digest(K, *arrs, [v_plateau, v_opt])
    ax.text(K.min(), v_opt + 0.012, "unregularized optimum", fontsize=FS, color=TEXT2, va="bottom")
    ax.text(K.min(), v_plateau - 0.03, "plateau", fontsize=FS, color=TEXT2, va="top")
    ax.set_xscale("log"); ax.set_xlim(K.min() / 1.6, x_star * 1.6)
    ax.set_ylim(v_plateau - 0.12, v_opt + 0.10)
    ax.set_xticks([0.03, 0.1, 1, 10, x_star]); ax.set_xticklabels(["0.03", "0.1", "1", "10", "$\\star$"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("$\\kappa$ ($\\star$: the prior policy)"); ax.set_ylabel("clipped value $V_m$ of the minimizer")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=2, axes=[ax], title="prior", title_fontsize=FS)
    panel_label(ax, "b")

    # (c) bounds versus behaviour: the revision-41 drawing
    ax = axes[2]
    arrs = []
    for prior in m.PRIORS:
        rs = m.runs_of(prior)
        dev = np.array([abs(r["final"]["q"] - r["prior"]["q"]) for r in rs]); arrs.append(dev)
        ax.plot(m.KAPPAS, np.maximum(dev, 1e-6), marker=PMK[prior], ms=3.5, mec="white", mew=0.5, color=m.COL[prior])
    digests["prior/c deviations"] = digest(m.KAPPAS, *arrs)
    R = P["ubar"] / (1 - P["beta"]) - m.runs_of((0.0, 1.0))[0]["prior"]["V_m"]
    ax.plot(m.KAPPAS, [m.slack(k, R) for k in m.KAPPAS], color=TEXT2, ls="--", label="risk slack bound")
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-6, 30)
    ax.set_xlabel("$\\kappa$"); ax.set_ylabel("$|q_\\beta(\\mu^*)-q_\\beta(\\pi_\\Gamma)|$")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax]); panel_label(ax, "c")

    fig.savefig(HERE / "prior-v47.pdf", bbox_inches="tight"); fig.savefig(HERE / "prior-v47.png", dpi=200, bbox_inches="tight")
    ref = json.loads((V41 / "prior-digests-v41.json").read_text())
    check = {"a": digests["prior/a rate collapse"] == ref["prior/a rate collapse"]["after"],
             "c": digests["prior/c deviations"] == ref["prior/c deviations"]["after"]}
    (HERE / "prior-digests-v47.json").write_text(json.dumps(dict(digests=digests, equal_to_v41=check,
                                                                 plateau_V_m=v_plateau, optimum_V_m=v_opt), indent=1) + "\n")
    print("digests equal to v41:", check)
    assert all(check.values())
    for prior in m.PRIORS:
        rs = m.runs_of(prior)
        print(m.LAB[prior], "prior V_m %.4f" % rs[0]["prior"]["V_m"], "minimizers", ["%.4f" % r["final"]["V_m"] for r in rs])
    print("plateau V_m %.4f, optimum V_m %.4f" % (v_plateau, v_opt))


if __name__ == "__main__":
    main()
