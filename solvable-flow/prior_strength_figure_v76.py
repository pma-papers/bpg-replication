"""Revision 76: the figure of the paper on the prior, the strength and a strength lowered in steps (analytical example).

(a) The clipped value of the minimizer under relative entropy against the strength, for the four priors of the
    regularization paths (prior-path.json, with the rerun of the one run that had not converged). The stars at the
    right end are the prior policies. The open markers at the left end are the limits of unregularized transport
    started from each prior: the clipped value increases in the tilt and the feature increases in |x| outside the
    plateau, so every unit there moves outward and its feature tends to one, the units on the plateau do not move, and
    the limiting tilt is the prior's mass outside the plateau.
(b), (c) The clipped value against the learning time when the strength is lowered in steps, under the priors N(0,1) and
    N(0,4), for three of the configurations of decreasing_strength_v76.py (means over its seeds).
The hues of the priors in (a) are not used for the configurations in (b) and (c).
Output: prior-strength-v76 (pdf/png).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, MK, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

p = json.loads((ROOT / "prior-path.json").read_text())
rerun = json.loads((ROOT / "prior-path-rerun.json").read_text())
p["runs"][rerun["name"]] = rerun
d = json.loads((ROOT / "decreasing-strength-v76.json").read_text())
c, V = np.array(p["curve"]["c"]), np.array(p["curve"]["V_m"])
v_initial, v_sup = p["stats_mu0"]["V_m"], float(V.max())

fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.35), sharey=True, gridspec_kw=dict(wspace=0.12, width_ratios=[1.15, 1, 1]))
ax = axes[0]
K = np.array(sorted(p["kappas"]))
x_star, x_zero = K.max() * 2.4, K.min() / 3.0
for i, ((x0, s), lab) in enumerate(zip(p["priors"], ("$N(0,1/4)$", "$N(0,1)$", "$N(0,4)$", "$N(2,1)$"))):
    rs = sorted([r for r in p["runs"].values() if r["x0"] == x0 and r["sigma"] == s], key=lambda r: r["kappa"])
    ax.plot(K, [r["final"]["V_m"] for r in rs], marker=MK[i], ms=3.5, mec="white", mew=0.4, color=PAL[i], lw=0.9, label=lab)
    ax.plot([x_star], [rs[0]["prior"]["V_m"]], marker="*", ms=8, mec="white", mew=0.5, color=PAL[i], ls="none")
    mass = norm.cdf(-1, x0, s) + norm.sf(1, x0, s)
    ax.plot([x_zero], [np.interp(mass, c, V)], marker=MK[i], ms=4, mfc="white", mec=PAL[i], mew=1.0, ls="none")
ax.axvline(np.sqrt(x_zero * K.min()), color=MUTED, lw=0.6, ls=":")
ax.set_xscale("log"); ax.set_xlim(x_zero / 1.7, x_star * 1.7)
ax.set_xticks([x_zero, 0.1, 1, 10, x_star]); ax.set_xticklabels(["$0$", "0.1", "1", "10", "$\\star$"]); ax.xaxis.set_minor_locator(plt.NullLocator())
ax.set_xlabel("strength $\\kappa$"); ax.set_ylabel("clipped value $V_m$"); panel_label(ax, "a")
handles_a, labels_a = ax.get_legend_handles_labels()

T = np.array(d["times"])
CONFIGS = (("fixed", "fixed $\\kappa=0.03$", TEXT2, "-"), ("equal", "stages of length $2$", PAL[7], "--"), ("rule", "rule", PAL[6], ":"))
for ax, key, name, lab in ((axes[1], "sigma = 1", "$N(0,1)$", "b"), (axes[2], "sigma = 2", "$N(0,4)$", "c")):
    block = d["priors"][key]
    for k, label, col, ls in CONFIGS:
        ax.plot(T, block["configurations"][k]["mean"], color=col, ls=ls, lw=1.3, label=label)
    ax.axhline(block["minimizer_values"]["0.03"], color=MUTED, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("learning time $\\ell$"); panel_label(ax, lab)
    ax.text(0.05, 0.88, "prior " + name, transform=ax.transAxes, fontsize=FS, color=TEXT2)
for ax in axes:
    ax.axhline(v_sup, color=MUTED, lw=0.8); ax.axhline(v_initial, color=MUTED, lw=0.8, ls="--")
axes[0].set_ylim(v_initial - 0.06, v_sup + 0.05)
legend_below(fig, handles_a, labels_a, ncol=2, axes=[axes[0]], title="prior", title_fontsize=FS)
legend_below(fig, *axes[1].get_legend_handles_labels(), ncol=2, axes=[axes[1], axes[2]])
save(fig, "prior-strength-v76", ROOT)
