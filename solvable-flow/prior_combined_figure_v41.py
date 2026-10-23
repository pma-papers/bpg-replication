"""Revision 41: the combined prior figure (rate collapse from solvable-flow/prior-flow.json; regularization
paths and bounds versus behaviour from solvable-flow/prior-path.json).

Change against solvable-flow/prior_combined_figure.py (review finding): the run path/x0s0.5/k0.03 did not converge
within its horizon (rate_energy NaN). If prior-path-rerun.json (a rerun with a longer horizon, written by
prior_path_rerun.py) exists, its final and rate values replace that run's; otherwise the run's points are drawn
with an open marker and the legend says "not converged within the horizon". The status is written to
prior-path-status-v41.json. Outputs: prior-combined-v41 (pdf/png), prior-digests-v41.json."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
V39 = ROOT.parent / "solvable-flow"
sys.path.insert(0, str(ROOT.parent / "lib"))
sys.path.insert(0, str(ROOT))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from digest_v41 import digest  # noqa: E402

S = json.loads((V39 / "prior-flow.json").read_text())["runs"]
D = json.loads((V39 / "prior-path.json").read_text())
P = D["params"]
PRIORS = [tuple(p) for p in D["priors"]]
KAPPAS = list(D["kappas"])
COL = {(0.0, 0.5): PAL[0], (0.0, 1.0): PAL[1], (0.0, 2.0): PAL[2], (2.0, 1.0): PAL[3]}
LAB = {(0.0, 0.5): "$N(0,1/4)$", (0.0, 1.0): "$N(0,1)$", (0.0, 2.0): "$N(0,4)$", (2.0, 1.0): "$N(2,1)$"}
RERUN = ROOT / "prior-path-rerun.json"
STATUS = dict(not_converged=[], substituted={})


def apply_rerun():
    def bad(r):
        v = r.get("rate_energy")
        return v is None or (isinstance(v, float) and math.isnan(v))
    rerun = None
    if RERUN.exists():
        try:
            rerun = json.loads(RERUN.read_text())
        except json.JSONDecodeError:
            rerun = None  # still being written
    if rerun is not None and rerun.get("name") in D["runs"] and not bad(rerun):
        run = D["runs"][rerun["name"]]
        before = dict(final=run["final"], rate_energy=run["rate_energy"], rate_W2=run["rate_W2"], l_max=run["l_max"], steps=run["steps"])
        for k in ("final", "rate_energy", "rate_W2", "l_max", "steps", "horizon_factor", "energy_increases", "wall"):
            if k in rerun:
                run[k] = rerun[k]
        STATUS["substituted"][rerun["name"]] = dict(before=before, after={k: run[k] for k in ("final", "rate_energy", "rate_W2", "l_max", "steps", "horizon_factor")})
    for name, run in D["runs"].items():
        if bad(run):
            STATUS["not_converged"].append(name)
    STATUS["rerun_file_present"] = RERUN.exists()


def runs_of(prior):
    x0, sigma = prior
    return [D["runs"][f"path/x{x0:g}s{sigma:g}/k{k:g}"] for k in KAPPAS]


def slack(kappa, R_Gamma, L0=1.0, cH=2.0, T=0):
    return 0.5 * L0 * math.sqrt((T + 1) * cH * R_Gamma / kappa)


def main():
    apply_rerun()
    notconv = set(STATUS["not_converged"])
    digests = {}
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.3), gridspec_kw=dict(wspace=0.45))
    # (a) rate collapse (prior-flow.json; unchanged)
    ax = axes[0]
    xs = np.array([r["kappa_lambda"] for r in S.values()]); ys = np.array([r["rate_energy"] for r in S.values()])
    d = digest(xs, ys); digests["prior/a rate collapse"] = dict(before=d, after=d, changed=False)
    grid = np.geomspace(xs.min() / 2, xs.max() * 2, 50)
    ax.plot(grid, 4 * grid, color=MUTED, ls=":", lw=0.9)
    ax.plot(grid, 2 * grid, color=MUTED, ls="--", lw=0.9)
    for r in S.values():
        shifted = r["x0"] != 0
        ax.plot([r["kappa_lambda"]], [r["rate_energy"]], marker="s" if shifted else "o", ms=4.5, mec="white", mew=0.6, color=PAL[1] if shifted else PAL[0], ls="none")
    handles = [Line2D([], [], marker="o", ls="none", color=PAL[0], ms=4.5, label="centered prior"),
               Line2D([], [], marker="s", ls="none", color=PAL[1], ms=4.5, label="shifted prior"),
               Line2D([], [], color=MUTED, ls=":", label="$4\\kappa\\lambda_W$"), Line2D([], [], color=MUTED, ls="--", label="$2\\kappa\\lambda_W$")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.0, borderaxespad=0.2, labelspacing=0.3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$\\kappa\\lambda_W=\\kappa/\\sigma^2$"); ax.set_ylabel("observed rate of the objective gap"); panel_label(ax, "a")
    # (b) regularization paths
    ax = axes[1]
    q = np.array(D["curve"]["q"]); U = np.array(D["curve"]["U"])
    ax.plot(100 * q, U, color=TEXT2, lw=1.0)
    arrs = []
    for prior in PRIORS:
        rs = runs_of(prior)
        pq = [100 * r["final"]["q"] for r in rs]; pU = [r["final"]["U"] for r in rs]; arrs += [pq, pU]
        ax.plot(pq, pU, marker="o", ms=3, mec="white", mew=0.4, color=COL[prior], lw=0.9, label=LAB[prior])
        for r, x, y in zip(rs, pq, pU):
            if r["name"] in notconv:
                ax.plot([x], [y], marker="o", ms=3.6, mfc="white", mec=COL[prior], mew=0.8, ls="none", zorder=4)
        ax.plot([100 * rs[0]["prior"]["q"]], [rs[0]["prior"]["U"]], marker="*", ms=8, mec="white", mew=0.5, color=COL[prior], ls="none")
    digests["prior/b path end points"] = dict(after=digest(*arrs), changed=bool(STATUS["substituted"]))
    i = int(np.argmax(U))
    ax.plot([100 * q[i]], [U[i]], marker="s", ms=4, color=TEXT2, ls="none")
    ax.annotate("unregularized optimum", (100 * q[i], U[i]), textcoords="offset points", xytext=(6, -3), fontsize=6.5, color=TEXT2, va="top")
    ax.annotate("$\\mu_0$", (100 * D["stats_mu0"]["q"], D["stats_mu0"]["U"]), textcoords="offset points", xytext=(-4, 4), fontsize=6.5, color=TEXT2, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    if notconv:
        handles.append(Line2D([], [], marker="o", ms=3.6, mfc="white", mec=TEXT2, mew=0.8, ls="none")); labels.append("not converged within the horizon")
    ax.legend(handles, labels, loc="center right", fontsize=6.0, borderaxespad=0.2, labelspacing=0.3, title="prior ($\\star$: its policy)", title_fontsize=6.0)
    ax.set_xlabel("geometric risk $q_\\beta$ (%)"); ax.set_ylabel("stopped utility $U$"); panel_label(ax, "b")
    # (c) bounds versus behaviour
    ax = axes[2]
    arrs = []
    for prior in PRIORS:
        rs = runs_of(prior)
        dev = np.array([abs(r["final"]["q"] - r["prior"]["q"]) for r in rs]); arrs.append(dev)
        ax.plot(KAPPAS, np.maximum(dev, 1e-6), marker="o", ms=3.5, mec="white", mew=0.5, color=COL[prior])
        for r, k, y in zip(rs, KAPPAS, dev):
            if r["name"] in notconv:
                ax.plot([k], [max(y, 1e-6)], marker="o", ms=4.2, mfc="white", mec=COL[prior], mew=0.8, ls="none", zorder=4)
    digests["prior/c deviations"] = dict(after=digest(KAPPAS, *arrs), changed=bool(STATUS["substituted"]))
    R = P["ubar"] / (1 - P["beta"]) - runs_of((0.0, 1.0))[0]["prior"]["V_m"]
    ax.plot(KAPPAS, [slack(k, R) for k in KAPPAS], color=TEXT2, ls="--", label="risk slack bound")
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=":")
    handles, labels = ax.get_legend_handles_labels()
    if notconv:
        handles.append(Line2D([], [], marker="o", ms=4.2, mfc="white", mec=TEXT2, mew=0.8, ls="none")); labels.append("not converged within the horizon")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-6, 30)
    ax.set_xlabel("$\\kappa$"); ax.set_ylabel("$|q_\\beta(\\mu^*)-q_\\beta(\\pi_\\Gamma)|$")
    ax.legend(handles, labels, loc="lower left", fontsize=6.0, borderaxespad=0.2, labelspacing=0.3); panel_label(ax, "c")
    save(fig, "prior-combined-v41", ROOT)
    (ROOT / "prior-path-status-v41.json").write_text(json.dumps(STATUS, indent=1) + "\n")
    (ROOT / "prior-digests-v41.json").write_text(json.dumps(digests, indent=1) + "\n")
    print("rerun file present:", STATUS["rerun_file_present"]); print("substituted:", json.dumps(STATUS["substituted"], indent=1)); print("not converged (open markers):", STATUS["not_converged"])


if __name__ == "__main__":
    main()
