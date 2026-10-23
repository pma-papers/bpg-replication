"""Revision 60: the bound of Corollary thm:combined (eq. combinedrisk) on the solvable flow.

Comparison law: the prior Gamma = N(0,1), for which the regularization cost R(Gamma) vanishes. The bound reads
  q_beta(mu_l) <= q_beta(Gamma) + (1-beta)/m [sup_C U - U(pi_Gamma) + I_m(pi_Gamma) + E_0 exp(-2 Lambda_m l)],
capped at one. sup_C U is the largest stopped utility over the represented class (the curve of the example).
Panel (a): the bound against l at the certified strength (margin 1.05) for relative entropy and the three
power integrands, from the stored runs of solvable-flow/regularizer-flow.json, with the attained q_beta(mu_l)
and the learning form of Proposition prop:fixed-price(iv) evaluated with the observed objective gap in place
of the envelope. Panel (b): relative entropy at margins 1.05, 1.5, 2, 4 (corollary-flow-v60.json).
Outputs: corollary-bound-v60.pdf/png and corollary-bound-v60.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, MK, panel_label, legend_below, save  # noqa: E402

REG = json.loads((HERE.parent / "solvable-flow" / "regularizer-flow.json").read_text())
COR = json.loads((HERE / "corollary-flow-v60.json").read_text())
P = REG["params"]; G = REG["stats_Gamma"]
supU = max(REG["curve"]["U"])
FIXED = G["q"] + (1 - P["beta"]) / P["m"] * (supU - G["U"] + G["I_m"])
LABELS = {"KL": "relative entropy", "r1.5": "$r=3/2$", "r2": "$r=2$", "r3": "$r=3$"}


def bound(run, use_observed=False):
    rec = run["records"]; l = np.array([x["l"] for x in rec]); q = np.array([x["q"] for x in rec])
    gap = np.array([x["gap"] for x in rec]) if use_observed else run["E0"] * np.exp(-2 * run["Lambda"] * l)
    return l, q, np.minimum(1.0, FIXED + (1 - P["beta"]) / P["m"] * gap)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.3), gridspec_kw=dict(wspace=0.3))
    out = dict(supU=supU, fixed_part=FIXED, q_Gamma=G["q"], U_Gamma=G["U"], I_Gamma=G["I_m"], crossing={})
    ax = axes[0]
    for i, key in enumerate(["KL", "r1.5", "r2", "r3"]):
        run = REG["runs"][f"cert/{key}"]
        l, q, b = bound(run); _, _, bo = bound(run, use_observed=True)
        lp = np.maximum(l, 1e-5)
        ax.plot(lp, b, color=PAL[i], lw=1.0, marker=MK[i], markevery=(0.04 * i, 0.16), ms=3.5, mec="white", mew=0.4, label=LABELS[key])
        ax.plot(lp, bo, color=PAL[i], lw=0.8, ls="--")
        ax.plot(lp, q, color=PAL[i], lw=0.8, ls=":")
        out["crossing"][key] = float(np.log(run["E0"] * (1 - P["beta"]) / P["m"] / FIXED) / (2 * run["Lambda"]))
    ax.axhline(FIXED, color=MUTED, lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-5, REG["runs"]["cert/KL"]["l_max"]); ax.set_ylim(0.06, 1.3)
    ax.set_xlabel("learning time $\\ell$"); ax.set_ylabel("breakdown probability $q_\\beta$")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=2, axes=[ax]); panel_label(ax, "a")
    ax = axes[1]
    runs = [("1.05", REG["runs"]["cert/KL"])] + [(f"{m:g}", COR["runs"][f"m{m:g}/KL"]) for m in COR["margins"]]
    for i, (mg, run) in enumerate(runs):
        l, q, b = bound(run); lp = np.maximum(l, 1e-5)
        ax.plot(lp, b, color=PAL[i], lw=1.0, marker=MK[i], markevery=(0.04 * i, 0.16), ms=3.5, mec="white", mew=0.4, label=f"$\\kappa={mg}\\,C_m/\\lambda_W$")
        ax.plot(lp, q, color=PAL[i], lw=0.8, ls=":")
        out["crossing"][f"KL-margin{mg}"] = float(np.log(run["E0"] * (1 - P["beta"]) / P["m"] / FIXED) / (2 * run["Lambda"]))
    ax.axhline(FIXED, color=MUTED, lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-5, REG["runs"]["cert/KL"]["l_max"]); ax.set_ylim(0.06, 1.3)
    ax.set_xlabel("learning time $\\ell$"); ax.set_ylabel("breakdown probability $q_\\beta$")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=2, axes=[ax]); panel_label(ax, "b")
    save(fig, "corollary-bound-v60", HERE)
    (HERE / "corollary-bound-v60.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
