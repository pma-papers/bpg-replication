"""Convergence of the runs of the final design of the lander study (2026-11-16).

The design trains at a constant learning rate and ends with a phase in which the rate is decayed to zero. A level
end of a training trace under a decaying rate does not show convergence, since the decay stops a run in any case.
This script therefore sets the design next to the same runs continued at the constant rate to twice the training,
and next to the earlier runs whose rate was decayed from the start.

Usage:  python lunar_convergence_design.py CONSTANT-STAGE FINAL-PHASE-STAGE LONGER-STAGE DECAYED-STAGE
        (for the final design: fuelprice-20-const fuelprice-20-anneal fuelprice-20-const-30m fuelprice-20-long)
The longer stage repeats the constant stage with the same seeds and goes on, so its traces contain those of the
constant stage; the script checks this.

Outputs (this directory): lunar-FINAL-PHASE-STAGE-convergence.json (per price of the longer stage and per schedule:
failure probability, median fuel and stopped utility of the final policies, per seed and as means),
lunar-FINAL-PHASE-STAGE-convergence-table.tex, and lunar-FINAL-PHASE-STAGE-convergence (pdf/png): the failure rate
of training against the training steps, one panel per price and one line per seed; solid lines are the runs at the
constant rate, dashed lines the final phase of the design, which leaves them at the end of the constant phase.
Each point is a mean over about 2 percent of the longer training. A price that the decayed stage does not hold
is reported without that schedule.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

const, final, longer, decayed = sys.argv[1:5]
SCHEDULES = [(decayed, "decayed from the start"), (const, "constant"), (final, "constant, then decayed (the design)"), (longer, "constant, twice the training")]


def load(stage):
    runs = {}
    for f in sorted((ROOT / "lunar" / stage).glob("*/result.json")):
        r = json.loads(f.read_text())
        if r["arm"]["kind"] == "fixed":
            runs.setdefault(r["arm"]["m"], {})[r["seed"]] = r
    return runs


R = {s: load(s) for s, _ in SCHEDULES}
prices = sorted(R[longer])
seeds = sorted(R[longer][prices[0]])
steps_const = R[const][prices[0]][seeds[0]]["steps"]
total = {decayed: R[decayed][prices[0]][seeds[0]]["steps"], const: steps_const, final: steps_const + R[final][prices[0]][seeds[0]]["steps"], longer: R[longer][prices[0]][seeds[0]]["steps"]}

out = dict(schedules={s: n for s, n in SCHEDULES}, training_steps=total, prices=prices, seeds=seeds, configurations={})
rows = []
for m in prices:
    out["configurations"][f"{m:g}"] = {}
    first = next(stage for stage, _ in SCHEDULES if m in R[stage])
    for stage, name in SCHEDULES:
        if m not in R[stage]:
            continue
        S = [R[stage][m][s]["stochastic"] for s in seeds]
        c = dict(risk=[x["risk"] for x in S], fuel_median=[x["fuel_quantiles"][2] for x in S], U=[x["U"] for x in S])
        c.update(risk_mean=float(np.mean(c["risk"])), fuel_mean=float(np.mean(c["fuel_median"])), U_mean=float(np.mean(c["U"])))
        out["configurations"][f"{m:g}"][stage] = c
        rows.append((f"${m:g}$" if stage == first else "") + f" & {name} & ${total[stage] / 1e6:.0f}$ & " + ", ".join(f"${100 * x:.1f}$" for x in c["risk"])
                    + f" & ${100 * c['risk_mean']:.1f}$ & ${c['fuel_mean']:.1f}$ & ${c['U_mean']:.0f}$ \\\\" + ("\n\\addlinespace" if stage == longer and m != prices[-1] else ""))
    # the longer stage contains the constant stage: same seeds, same settings, hence the same trace over its length
    for s in seeds:
        a = [x["failure_rate"] for x in R[const][m][s]["trace"]]
        b = [x["failure_rate"] for x in R[longer][m][s]["trace"]][:len(a)]
        assert np.allclose(np.nan_to_num(a), np.nan_to_num(b)), (m, s)
out["longer_stage_contains_constant_stage"] = True
(ROOT / f"lunar-{final}-convergence.json").write_text(json.dumps(out, indent=1) + "\n")
(ROOT / f"lunar-{final}-convergence-table.tex").write_text(
    "\\begin{tabular}{llrlrrr}\n\\toprule\nPrice $m$ & Learning rate & Steps ($10^6$) & Failed (\\%), the five seeds & Mean & Fuel & $U_H$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def binned(trace, k):
    f = np.array([x["failure_rate"] for x in trace], dtype=float); st = np.array([x["steps"] for x in trace], dtype=float)
    nb = len(f) // k
    return st[:nb * k].reshape(nb, k).mean(1), np.nanmean(f[:nb * k].reshape(nb, k), axis=1)


fig, axes = plt.subplots(1, len(prices), figsize=(TEXTW, 2.45 if len(prices) <= 2 else 2.1), gridspec_kw=dict(wspace=0.3 if len(prices) <= 2 else 0.16))
k = max(1, len(R[longer][prices[0]][seeds[0]]["trace"]) // 50)
for ax, m, tag in zip(np.atleast_1d(axes), prices, "abcdef"):
    for i, s in enumerate(seeds):
        x, y = binned(R[longer][m][s]["trace"], k)
        ax.plot(x / 1e6, y, color=PAL[i], lw=1.0)
        xa, ya = binned(R[final][m][s]["trace"], k)
        j = int(np.searchsorted(x, steps_const)) - 1                                  # the last point of the constant phase
        ax.plot(np.r_[x[j], steps_const + xa] / 1e6, np.r_[y[j], ya], color=PAL[i], lw=1.0, ls=(0, (3, 1.5)))
        ax.plot([total[final] / 1e6], [R[final][m][s]["stochastic"]["risk"]], marker="o", ms=3.2, color=PAL[i], mec="white", mew=0.4, zorder=5)
    ax.axvline(steps_const / 1e6, color=MUTED, lw=0.6, ls=":")
    if m == 0:
        ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    else:
        ax.set_yscale("log"); ax.set_ylim(0.008, 1.2); ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0]); ax.set_yticklabels(["1%", "3%", "10%", "30%", "100%"])
        ax.minorticks_off()
        if len(prices) > 2:                                                         # the priced panels share one axis, labelled at the right of the last of them
            if m == prices[-1]:
                ax.yaxis.tick_right(); ax.tick_params(axis="y", right=True, left=False)
            else:
                ax.set_yticklabels([])
    ax.set_xlim(0, total[longer] / 1e6); ax.set_xticks(np.arange(0, total[longer] / 1e6 + 1, 5 if len(prices) <= 2 else 10))
    ax.set_xlabel("training steps ($10^6$)" if len(prices) <= 2 else "steps ($10^6$)")
    ax.text(0.5, 1.03, "no price" if m == 0 else f"price $m={m:g}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=FS)
    panel_label(ax, tag)
np.atleast_1d(axes)[0].set_ylabel("failure rate in training")
handles = [Line2D([], [], color=TEXT2, lw=1.0), Line2D([], [], color=TEXT2, lw=1.0, ls=(0, (3, 1.5))), Line2D([], [], color=TEXT2, marker="o", ms=3.2, ls="none")]
legend_below(fig, handles, ["constant learning rate", "final phase with the rate decayed to zero", "final policy of the design, $1000$ episodes"], ncol=2)
save(fig, f"lunar-{final}-convergence", ROOT)
for m in prices:
    for stage, name in SCHEDULES:
        if stage not in out["configurations"][f"{m:g}"]:
            continue
        c = out["configurations"][f"{m:g}"][stage]
        print(f"m={m:<4g} {name:38s} {total[stage] / 1e6:4.0f}M  failed {100 * c['risk_mean']:5.1f} {[round(100 * x, 1) for x in c['risk']]}  fuel {c['fuel_mean']:5.1f}  U {c['U_mean']:6.1f}")
