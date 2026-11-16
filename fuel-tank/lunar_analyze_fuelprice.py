"""Summary, table and figure of a stage of the fuel-tank study with valuable fuel (2026-11-14).

The stages of this design have one tank, a list of prices and five seeds. The failure is the loss of the lander or
an empty tank before touchdown, and the utility charges the stage's value per unit of fuel (lunar.py, options
crash_fails, tank_until_touchdown and fuel_price; lunar-protocol.json).

Usage:  python lunar_analyze_fuelprice.py STAGE
Outputs (this directory): lunar-STAGE.json (per configuration: means, standard deviations and the values of the
seeds; the fall of the failure probability against the run without a price, seed by seed; where the files
priced-risk.json of lunar_reeval.py --stage STAGE exist, the priced risk q_{beta,H}, the frames of the failures and
the value U - m' q_{beta,H}/(1-beta) of every trained policy at every price m'), lunar-STAGE-table.tex,
lunar-STAGE-values.tex (the values, one row per training price), and lunar-STAGE (pdf/png) with three panels:
(a) the failure probability against the price, (b) the median fuel use against the price, (c) the stopped utility
against the failure probability.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

stage = sys.argv[1]
protocol = json.loads((ROOT / "lunar-protocol.json").read_text())
runs = {}
for f in sorted((ROOT / "lunar" / stage).glob("*/result.json")):
    r = json.loads(f.read_text())
    runs.setdefault(r["arm"]["m"], {})[r["seed"]] = r
seeds = sorted(runs[0.0])
incomplete = [m for m in sorted(runs) if sorted(runs[m]) != seeds]
prices = [m for m in sorted(runs) if m not in incomplete]            # a price enters when all its seeds have finished
if incomplete:
    print("prices left out, not all seeds finished:", incomplete)
FIELDS = ("risk", "exhausted", "lost", "landed", "U")
summary = dict(stage=stage, options=protocol["stages"][stage]["arm_overrides"], tank=runs[0.0][seeds[0]]["arm"]["budget"], steps=runs[0.0][seeds[0]]["steps"], seeds=seeds, prices=prices, configurations={})
for m in prices:
    S = [runs[m][s]["stochastic"] for s in seeds]
    c = {k: dict(mean=float(np.mean([x[k] for x in S])), sd=float(np.std([x[k] for x in S], ddof=1)), median=float(np.median([x[k] for x in S])), per_seed=[x[k] for x in S]) for k in FIELDS}
    fuel = [x["fuel_quantiles"][2] for x in S]
    c["fuel_median"] = dict(mean=float(np.mean(fuel)), sd=float(np.std(fuel, ddof=1)), per_seed=fuel)
    c["mean_action_risk"] = float(np.mean([runs[m][s]["mean_action"]["risk"] for s in seeds]))
    c["charge_at_frame_120"] = m * (1 - 0.999 ** (1000 - 120)) / (1 - 0.999)
    pr = [ROOT / "lunar" / stage / f"lunar-b{runs[m][s]['arm']['budget']:g}-m{m:g}__seed{s}" / "priced-risk.json" for s in seeds]
    if all(f.exists() for f in pr):
        Q = [json.loads(f.read_text()) for f in pr]
        assert all(abs(q["risk"] - x["risk"]) < 1e-12 and abs(q["U"] - x["U"]) < 1e-6 for q, x in zip(Q, S))    # the same thousand episodes
        frames = np.concatenate([q["exhaustion_frames"] for q in Q])
        c["priced_risk"] = dict(mean=float(np.mean([q["priced_risk"] for q in Q])), per_seed=[q["priced_risk"] for q in Q])
        if len(frames):
            c["failing_frame"] = dict(median=float(np.median(frames)), q10=float(np.quantile(frames, 0.1)), q90=float(np.quantile(frames, 0.9)))
            c["charge_at_median_failing_frame"] = m * (1 - 0.999 ** (1000 - float(np.median(frames)))) / (1 - 0.999)
    if m > 0:
        fall = [runs[0.0][s]["stochastic"]["risk"] - runs[m][s]["stochastic"]["risk"] for s in seeds]
        c["fall_of_failure_against_no_price"] = dict(per_seed=fall, mean=float(np.mean(fall)), seeds_with_a_fall=int(sum(x > 0 for x in fall)))
    summary["configurations"][f"{m:g}"] = c
C = summary["configurations"]
if all("priced_risk" in C[f"{m:g}"] for m in prices):
    # value at the price m' of the policy trained at the price m, seed by seed, and its mean; for each m' the training price with the largest mean
    V = {f"{m:g}": {f"{mp:g}": [u - mp * q / (1 - 0.999) for u, q in zip(C[f"{m:g}"]["U"]["per_seed"], C[f"{m:g}"]["priced_risk"]["per_seed"])] for mp in prices} for m in prices}
    best = {f"{mp:g}": max(prices, key=lambda m: np.mean(V[f"{m:g}"][f"{mp:g}"])) for mp in prices}
    summary["value_at_price_of_policy_trained_at"] = dict(note="rows: training price m; columns: price m' at which the value U - m' q_{beta,H}/(1-beta) is taken; means over the seeds",
        mean={m: {mp: float(np.mean(v)) for mp, v in row.items()} for m, row in V.items()}, per_seed=V, training_price_with_the_largest_mean_value=best,
        shortfall_of_own_price={f"{m:g}": float(np.mean(V[f"{best[f'{m:g}']:g}"][f"{m:g}"]) - np.mean(V[f"{m:g}"][f"{m:g}"])) for m in prices})
    rows = []
    for m in prices:
        cells = []
        for mp in prices:
            x = f"{np.mean(V[f'{m:g}'][f'{mp:g}']):.0f}"
            cells.append(f"$\\mathbf{{{x}}}$" if best[f"{mp:g}"] == m else f"${x}$")
        rows.append(f"${m:g}$ & " + " & ".join(cells) + " \\\\")
    (ROOT / f"lunar-{stage}-values.tex").write_text("\\begin{tabular}{l" + "r" * len(prices) + "}\n\\toprule\n & \\multicolumn{" + str(len(prices)) + "}{c}{Value at the price $m'$} \\\\\n\\cmidrule(lr){2-" + str(len(prices) + 1) + "}\nTrained at $m$ & "
        + " & ".join(f"${mp:g}$" for mp in prices) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
(ROOT / f"lunar-{stage}.json").write_text(json.dumps(summary, indent=1) + "\n")

rows = []
for m in prices:
    c = C[f"{m:g}"]
    rows.append(f"${m:g}$ & ${100 * c['risk']['mean']:.1f}\\pm{100 * c['risk']['sd']:.1f}$ & ${100 * c['lost']['mean']:.1f}$ & ${100 * c['exhausted']['mean']:.1f}$ & ${100 * c['landed']['mean']:.1f}$ & "
                + (f"${c['priced_risk']['mean']:.4f}$ & " if "priced_risk" in c else "-- & ") +
                f"${c['fuel_median']['mean']:.1f}$ & ${c['U']['mean']:.0f}\\pm{c['U']['sd']:.0f}$ & " + (f"${c['fall_of_failure_against_no_price']['seeds_with_a_fall']}/{len(seeds)}$" if m > 0 else "--") + " \\\\")
(ROOT / f"lunar-{stage}-table.tex").write_text(
    "\\begin{tabular}{lrrrrrrrr}\n\\toprule\nPrice $m$ & failed (\\%) & lander lost (\\%) & tank empty (\\%) & landed (\\%) & $q_{\\beta,H}$ & fuel & $U$ & fell \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.35), gridspec_kw=dict(wspace=0.62))
xs = np.arange(len(prices))
for ax, key, lab, scale in ((axes[0], "risk", "failure probability", 1.0), (axes[1], "fuel_median", "median fuel use", 1.0)):
    for i, m in enumerate(prices):
        c = C[f"{m:g}"][key]
        ax.plot([i] * len(seeds), c["per_seed"], marker="o", ms=2.6, color=MUTED, ls="none", zorder=2)
    ax.plot(xs, [C[f"{m:g}"][key]["mean"] for m in prices], marker="D", ms=4.5, mec="white", mew=0.6, color=PAL[0], lw=1.0, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels([f"{m:g}" for m in prices], fontsize=FS - 2, rotation=45)
    ax.set_xlabel("price $m$"); ax.set_ylabel(lab)
axes[0].set_yscale("log"); axes[0].set_yticks([0.01, 0.03, 0.1, 0.3, 1.0]); axes[0].set_yticklabels(["1%", "3%", "10%", "30%", "100%"]); axes[0].set_ylim(0.004, 1.0)
panel_label(axes[0], "a"); panel_label(axes[1], "b")
ax = axes[2]
for m in prices:
    c = C[f"{m:g}"]
    ax.plot(c["risk"]["per_seed"], c["U"]["per_seed"], marker="o", ms=2.6, color=MUTED, ls="none", zorder=2)
ax.plot([C[f"{m:g}"]["risk"]["mean"] for m in prices], [C[f"{m:g}"]["U"]["mean"] for m in prices], marker="D", ms=4.5, mec="white", mew=0.6, color=PAL[0], lw=0.8, zorder=3)
for m in prices:
    c = C[f"{m:g}"]
    ax.annotate(f"{m:g}", (c["risk"]["mean"], c["U"]["mean"]), textcoords="offset points", xytext=(-5, -2) if 0 < m else (5, 3), ha="right" if 0 < m else "left", va="center", fontsize=FS - 2, color=TEXT2)
ax.set_xscale("log"); ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0]); ax.set_xticklabels(["1%", "3%", "10%", "30%", "100%"], fontsize=FS - 2); ax.set_xlim(0.004, 1.0)
ax.set_xlabel("failure probability"); ax.set_ylabel("stopped utility $U$"); panel_label(ax, "c")
save(fig, f"lunar-{stage}", ROOT)
print(f"{stage}: tank {summary['tank']:g}, {summary['steps']} steps, options {summary['options']}")
for m in prices:
    c = C[f"{m:g}"]
    print(f"  m={m:<5g} failed {100 * c['risk']['mean']:5.1f} +- {100 * c['risk']['sd']:4.1f} (median {100 * c['risk']['median']:5.1f})  lost {100 * c['lost']['mean']:5.1f}  empty {100 * c['exhausted']['mean']:4.1f}  landed {100 * c['landed']['mean']:5.1f}  "
          f"fuel {c['fuel_median']['mean']:5.1f}  U {c['U']['mean']:7.1f} +- {c['U']['sd']:5.1f}" + (f"  fell in {c['fall_of_failure_against_no_price']['seeds_with_a_fall']}/{len(seeds)}" if m > 0 else ""))
if "value_at_price_of_policy_trained_at" in summary:
    W = summary["value_at_price_of_policy_trained_at"]
    print("  value at the price m' (columns) of the policy trained at m (rows), means over seeds:")
    for m in prices:
        print(f"    {m:<5g} " + " ".join(f"{W['mean'][f'{m:g}'][f'{mp:g}']:8.1f}" for mp in prices))
    print("  best training price for each m':", W["training_price_with_the_largest_mean_value"], " shortfall of the own price:", {k: round(v, 1) for k, v in W["shortfall_of_own_price"].items()})
