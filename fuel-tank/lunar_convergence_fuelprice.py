"""Convergence of the runs of the fuel-tank study with valuable fuel (2026-11-16).

A flat end of a training trace under a learning rate decayed to zero does not show convergence, since the decay
stops a run in any case. This script compares, at the prices of the check stages, the stage of the study with the
same design under a constant learning rate and under twice the training.

Usage:  python lunar_convergence_fuelprice.py STAGE CONSTANT-STAGE [LONGER-STAGE]
Outputs (this directory): lunar-STAGE-convergence.json (per price and stage: failure probability, median fuel and
stopped utility of the final policies, per seed and as means), lunar-STAGE-convergence-table.tex, and
lunar-STAGE-convergence (pdf/png): the failure rate of training against the training steps, one panel per price,
mean over the seeds (line) and their range (band), each point a mean over about 2 percent of the training.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

stages = sys.argv[1:]
NAMES = ["learning rate decayed to zero", "constant learning rate", "decayed, twice the training"]
STYLE = [(PAL[0], "-"), (PAL[1], "--"), (PAL[2], "-.")]


def load(stage):
    runs = {}
    for f in sorted((ROOT / "lunar" / stage).glob("*/result.json")):
        r = json.loads(f.read_text())
        if r["arm"]["kind"] == "fixed":
            runs.setdefault(r["arm"]["m"], {})[r["seed"]] = r
    return runs


R = [load(s) for s in stages]
prices = sorted(set(R[1]) & set(R[0]))
seeds = sorted(R[0][prices[0]])
out = dict(stages=dict(zip(stages, NAMES)), prices=prices, seeds=seeds, configurations={})
rows = []
for m in prices:
    out["configurations"][f"{m:g}"] = {}
    for stage, runs, name in zip(stages, R, NAMES):
        if m not in runs or sorted(runs[m]) != seeds:
            continue
        S = [runs[m][s]["stochastic"] for s in seeds]
        c = dict(steps=runs[m][seeds[0]]["steps"], risk=[x["risk"] for x in S], fuel_median=[x["fuel_quantiles"][2] for x in S], U=[x["U"] for x in S])
        c.update(risk_mean=float(np.mean(c["risk"])), fuel_mean=float(np.mean(c["fuel_median"])), U_mean=float(np.mean(c["U"])))
        tr = [np.array([x["failure_rate"] for x in runs[m][s]["trace"]]) for s in seeds]
        c["training_failure_rate_last_tenths"] = [float(np.mean([np.nanmean(t[len(t) * i // 10:len(t) * (i + 1) // 10]) for t in tr])) for i in (7, 8, 9)]
        out["configurations"][f"{m:g}"][stage] = c
        rows.append(f"${m:g}$ & {name} & ${c['steps'] / 1e6:.0f}$ & " + ", ".join(f"${100 * x:.1f}$" for x in c["risk"]) + f" & ${100 * c['risk_mean']:.1f}$ & ${c['fuel_mean']:.1f}$ & ${c['U_mean']:.0f}$ \\\\")
(ROOT / f"lunar-{stages[0]}-convergence.json").write_text(json.dumps(out, indent=1) + "\n")
(ROOT / f"lunar-{stages[0]}-convergence-table.tex").write_text(
    "\\begin{tabular}{lllrrrr}\n\\toprule\nPrice $m$ & Training & steps ($10^6$) & failed (\\%), the five seeds & mean & fuel & $U$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

fig, axes = plt.subplots(1, len(prices), figsize=(TEXTW, 2.2), sharey=True, gridspec_kw=dict(wspace=0.12))
for ax, m, tag in zip(np.atleast_1d(axes), prices, "abcdef"):
    for stage, runs, name, (col, ls) in zip(stages, R, NAMES, STYLE):
        if m not in runs or sorted(runs[m]) != seeds:
            continue
        tr = [runs[m][s]["trace"] for s in seeds]
        n = min(len(t) for t in tr); k = max(1, n // 50)
        steps = np.array([t["steps"] for t in tr[0][:n]])
        F = np.array([[x["failure_rate"] for x in t[:n]] for t in tr])
        nb = n // k
        Fb = np.nanmean(F[:, :nb * k].reshape(len(seeds), nb, k), axis=2); sb = steps[:nb * k].reshape(nb, k).mean(1) / 1e6
        ax.plot(sb, np.maximum(Fb.mean(0), 1e-3), color=col, ls=ls, lw=1.1, label=name)
        ax.fill_between(sb, np.maximum(Fb.min(0), 1e-3), np.maximum(Fb.max(0), 1e-3), color=col, alpha=0.15, lw=0)
    ax.set_yscale("log"); ax.set_ylim(0.004, 1.2); ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0]); ax.set_yticklabels(["1%", "3%", "10%", "30%", "100%"])
    ax.set_xlabel("training steps ($10^6$)")
    ax.text(0.5, 1.03, "no price" if m == 0 else f"price $m={m:g}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=FS)
    panel_label(ax, tag)
np.atleast_1d(axes)[0].set_ylabel("failure rate in training")
legend_below(fig, *np.atleast_1d(axes)[0].get_legend_handles_labels(), ncol=3)
save(fig, f"lunar-{stages[0]}-convergence", ROOT)
for m in prices:
    for stage, c in out["configurations"][f"{m:g}"].items():
        print(f"m={m:<4g} {out['stages'][stage]:32s} failed {100 * c['risk_mean']:5.1f} {[round(100 * x, 1) for x in c['risk']]}  fuel {c['fuel_mean']:5.1f}  U {c['U_mean']:6.1f}  training rate in the last tenths {[round(x, 3) for x in c['training_failure_rate_last_tenths']]}")
