"""The lander problem in the form chosen on 2026-11-18: the ledger with revenue 36, fuel at 1 per unit, a tank of 24 units.

Reads the runs of the stages ledger-const (constant rate) and ledger-anneal (rate decayed to zero) and gives, for every
worth of the vehicle k (the price charges 36 k for a failure at frame 100):
    the table of the evaluated policies (failed, of which lander lost and tank empty before touchdown, completed on the
    pad, fuel used, stopped utility), with means, standard deviations and medians over the seeds;
    the comparison by seed of every price with no price;
    the training traces over both phases (failure rate, fuel used and utility of the training episodes), with the
    means over windows at the end of the constant phase, as a check that the runs have settled.

Usage:  python lunar_ledger_main.py [CONST ANNEAL [PREFIX [TAG]]]      (default ledger-const ledger-anneal ledger-R36-b24 main; runtime ~/.venvs/constrained-rl)
Outputs: lunar-ledger-TAG.json, lunar-ledger-TAG-table.tex, lunar-ledger-TAG (pdf/png), lunar-ledger-TAG-traces (pdf/png).
The sets of runs: TAG main is the first set (stages ledger-const, ledger-anneal; started from the controller trained without a tank),
TAG staged the second (ledger-staged-const, ledger-staged-anneal; started from the controller trained without a price), TAG staged2 its
continuation (ledger-staged-const2, ledger-staged-anneal2). The priced objective is approximated by U - m TAIL (failure share), the charge
of a failure at frame 100.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, MUTED, TEXT2, FS, TEXTW, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CONST, ANNEAL = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("ledger-const", "ledger-anneal")
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "ledger-R36-b24"
TAG = sys.argv[4] if len(sys.argv) > 4 else "main"
REVENUE = 36.0
TAIL = (1 - 0.999 ** 900) / 0.001                      # the charge of a failure at frame 100 is m TAIL

final, traces = defaultdict(dict), defaultdict(dict)
for f in sorted((ROOT / "lunar" / ANNEAL).glob(f"{PREFIX}-k*__seed*/result.json")):
    r = json.loads(f.read_text()); k = float(r["arm"]["vehicle_worth_in_landings"])
    final[k][r["seed"]] = r
    tr = []
    for stage in (CONST, ANNEAL):
        p = ROOT / "lunar" / stage / f.parent.name / "progress.jsonl"
        q = ROOT / "lunar" / stage / f.parent.name / "result.json"
        rec = [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else json.loads(q.read_text()).get("trace", []) if q.exists() else []
        off = tr[-1]["steps"] if tr else 0
        tr += [{**x, "steps": x["steps"] + off, "phase": stage} for x in rec]
    traces[k][r["seed"]] = tr
KS = sorted(final)
if not KS:
    sys.exit("no runs found")
SEEDS = sorted(set.intersection(*[set(final[k]) for k in KS]))


def col(k, name, kind="stochastic"):
    return np.array([final[k][s][kind][name] for s in sorted(final[k])], float)


out = dict(stages=[CONST, ANNEAL], revenue=REVENUE, worth=KS, seeds_common=SEEDS, cells={}, against_no_price={}, settled={})
rows = []
print(f"{'k':>5s} {'m':>7s} {'n':>2s} {'failed % mean sd median':>24s} {'lost':>5s} {'empty':>5s} {'compl':>6s} {'fuel med':>8s} {'U mean sd med':>17s} {'priced med':>10s}   per seed failed %")
for k in KS:
    q, U, fuel = col(k, "risk"), col(k, "U"), col(k, "fuel_mean")
    c = dict(m=final[k][next(iter(final[k]))]["arm"]["m"], n=len(q), seeds=sorted(final[k]), risk=q.tolist(), lost=col(k, "lost").tolist(), exhausted=col(k, "exhausted").tolist(),
             completed=col(k, "completed").tolist(), fuel=fuel.tolist(), U=U.tolist(), risk_mean_action=col(k, "risk", "mean_action").tolist())
    c["priced_objective"] = (U - c["m"] * TAIL * q).tolist()
    out["cells"][f"{k:g}"] = c
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    print(f"{k:5g} {c['m']:7.4f} {c['n']:2d} {100 * q.mean():8.2f} {100 * sd(q):6.2f} {100 * np.median(q):7.2f} {100 * np.mean(c['lost']):5.1f} {100 * np.mean(c['exhausted']):5.1f} {100 * np.mean(c['completed']):6.1f} {np.median(fuel):8.2f} {U.mean():7.2f} {sd(U):4.2f} {np.median(U):5.2f} {np.median(c['priced_objective']):10.2f}   {np.round(100 * q, 1).tolist()}")
    rows.append(f"${k:g}$ & ${c['m']:.3f}$ & ${100 * q.mean():.1f}\\pm{100 * sd(q):.1f}$ & ${100 * np.median(q):.1f}$ & ${100 * np.mean(c['exhausted']):.1f}$ & ${100 * np.mean(c['completed']):.1f}$"
                f" & ${fuel.mean():.1f}\\pm{sd(fuel):.1f}$ & ${U.mean():.1f}\\pm{sd(U):.1f}$ \\\\")
(ROOT / f"lunar-ledger-{TAG}-table.tex").write_text(
    "\\begin{tabular}{@{}rrrrrrrr@{}}\n\\toprule\nWorth $k$ & Price $m$ & Failed (\\%) & Median (\\%) & Tank empty (\\%) & Completed (\\%) & Fuel used & Utility $U$ \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

print("\nagainst no price, by seed (differences no price minus price; t with n - 1 degrees of freedom)")
for k in KS[1:]:
    S = sorted(set(final[k]) & set(final[KS[0]]))
    d = {name: np.array([final[KS[0]][s]["stochastic"][name] - final[k][s]["stochastic"][name] for s in S]) for name in ("risk", "fuel_mean", "U")}
    t = {name: float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 1 and v.std() > 0 else float("nan") for name, v in d.items()}
    out["against_no_price"][f"{k:g}"] = dict(n=len(S), mean_difference={n: float(v.mean()) for n, v in d.items()}, t=t, failures_fell_in_seeds=int((d["risk"] > 0).sum()))
    print(f"   k={k:5g} n={len(S):2d}  failed {100 * d['risk'].mean():+5.2f} points (t {t['risk']:+5.2f}, fell in {int((d['risk'] > 0).sum())} seeds)   fuel {d['fuel_mean'].mean():+5.2f} (t {t['fuel_mean']:+5.2f})   U {d['U'].mean():+5.2f} (t {t['U']:+5.2f})")


def window(tr, lo, hi, key):
    x = [(t["episodes"], t[key]) for t in tr if lo < t["steps"] / 1e6 <= hi and key in t and t["episodes"]]
    return float(sum(e * v for e, v in x) / sum(e for e, _ in x)) if x else float("nan")


end_const = max(t["steps"] for k in KS for tr in traces[k].values() for t in tr if t["phase"] == CONST) / 1e6
total = max(t["steps"] for k in KS for tr in traces[k].values() for t in tr) / 1e6
W = [(end_const * a, end_const * b) for a, b in ((0.4, 0.6), (0.6, 0.8), (0.8, 1.0))] + [(total - (total - end_const) * 0.2, total)]
print(f"\ntraining episodes, means over the seeds in windows of the constant phase ({', '.join(f'{a:.0f}-{b:.0f}' for a, b in W[:3])} million steps) and at the end of the decayed phase")
for k in KS:
    line = {}
    for key in ("failure_rate", "fuel", "U"):
        line[key] = [float(np.nanmean([window(tr, a, b, key) for tr in traces[k].values()])) for a, b in W]
    out["settled"][f"{k:g}"] = dict(windows_million_steps=W, **line)
    print(f"   k={k:5g}  failed % " + " ".join(f"{100 * v:5.2f}" for v in line["failure_rate"]) + "   fuel " + " ".join(f"{v:5.2f}" for v in line["fuel"]) + "   U " + " ".join(f"{v:6.2f}" for v in line["U"]))
(ROOT / f"lunar-ledger-{TAG}.json").write_text(json.dumps(out, indent=1) + "\n")

# figure of the evaluated policies: failed, fuel, utility against the worth of the vehicle (seeds as points, medians as line)
fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.3), gridspec_kw=dict(wspace=0.6))
x = np.arange(len(KS))
TOP = 6.0                                              # failure shares above TOP percent are drawn as triangles at the top of panel a
for ax, (name, scale, lab) in zip(axes, (("risk", 100.0, "failed (%)"), ("fuel_mean", 1.0, "fuel used"), ("U", 1.0, "utility $U$"))):
    for xi, k in zip(x, KS):
        v = scale * col(k, name)
        if name == "risk":
            hi = v > TOP; ax.plot([xi] * int(hi.sum()), [TOP * 0.97] * int(hi.sum()), ls="", marker="^", ms=3.5, color=PAL[1]); v = v[~hi]
        ax.plot([xi] * len(v), v, ls="", marker=".", ms=3, color=PAL[0], alpha=0.45)
    ax.plot(x, [scale * np.median(col(k, name)) for k in KS], color=PAL[0], marker="o", ms=3, lw=1.1)
    ax.set_xticks(x); ax.set_xticklabels([f"{k:g}" for k in KS], fontsize=FS - 3, rotation=45); ax.set_xlabel("worth of the vehicle $k$"); ax.set_ylabel(lab)
axes[0].set_ylim(0, TOP)
axes[2].set_ylim(min(np.percentile(col(k, "U"), 15) for k in KS) - 1.5, None)        # collapsed runs lie below the panel
for ax, letter in zip(axes, "abc"):
    panel_label(ax, letter)
save(fig, f"lunar-ledger-{TAG}", ROOT)

# traces over both phases
fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.3), gridspec_kw=dict(wspace=0.6))
grid = np.linspace(0.5, total, 160)
for i, k in enumerate(KS):
    for ax, key, scale in zip(axes, ("failure_rate", "fuel", "U"), (100.0, 1.0, 1.0)):
        curves = []
        for tr in traces[k].values():
            if not tr or key not in tr[-1]:
                continue
            sx = np.array([t["steps"] for t in tr]) / 1e6; sv = scale * np.array([t.get(key, np.nan) for t in tr], float)
            kern = np.ones(15); sv = np.nan_to_num(sv, nan=np.nanmean(sv))
            curves.append(np.interp(grid, sx, np.convolve(sv, kern, mode="same") / np.convolve(np.ones(len(sv)), kern, mode="same")))
        if curves:
            ax.plot(grid, np.median(curves, 0), color=plt.cm.viridis(i / max(len(KS) - 1, 1)), lw=0.9, label=f"${k:g}$")      # median over the seeds
for ax, lab, letter in zip(axes, ("failed in training (%)", "fuel used", "utility $U$"), "abc"):
    ax.axvline(end_const, color=MUTED, lw=0.6, ls=":"); ax.set_xlabel("steps (millions)"); ax.set_ylabel(lab); panel_label(ax, letter)
axes[0].set_yscale("log"); axes[0].set_ylim(0.5, 100); axes[0].set_yticks([1, 3, 10, 30, 100]); axes[0].set_yticklabels(["1", "3", "10", "30", "100"]); axes[0].minorticks_off()
axes[1].set_ylim(16.5, 24.5)
axes[0].legend(frameon=False, fontsize=FS - 3, ncol=2, title="worth $k$", title_fontsize=FS - 3, loc="upper right", handlelength=1.2)
save(fig, f"lunar-ledger-{TAG}-traces", ROOT)
