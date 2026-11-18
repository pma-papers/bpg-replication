"""Figures and tables of the landing problem for the paper (draft of 18 November 2026, revision 77).

The landing problem is built on the simulator of LunarLanderContinuous-v3: a revenue of 36 is paid when the lander
comes to rest on the pad, fuel costs 1 per unit, the tank holds 24 units, and the lander fails when it is lost (crash
or out of the frame) or when the tank is empty before touchdown (lunar.py, the ledger arms ledger-R36-b24-k*).
The price m is such that a failure at frame 100 is charged k landings, m = 36 k (1 - beta) / (1 - beta^900).

Three sets of runs, ten seeds each, at k = 0, 0.25, 0.5, 1, 2, 4, 8, 16:
    design   started from the controller of the same seed trained on the landing problem without a price (3e7 steps),
             5e7 steps at a constant learning rate and 1e7 with the rate decayed to zero (stages ledger-staged-const,
             ledger-staged-const2, ledger-staged-anneal2)
    shorter  the same start, 2e7 steps at the constant rate and 1e7 decayed (ledger-staged-const, ledger-staged-anneal)
    other start  started from the controller of the simulator's own task (no tank, its own reward, 5e6 steps), 3e7 steps
             at the constant rate and 1e7 decayed (ledger-const, ledger-anneal)
Every final policy was evaluated on 2000 episodes on the machine that trained it; priced-risk.json (lunar_reeval.py,
run there) holds the failing frames of the same episodes. The fuel before touchdown is from 500 further episodes per
policy (lunar_touchdown_split.py), and the flights from the first episode of each of the 32 evaluation environments
(lunar_problem_eval_v76.py, run there).

Usage:  python lunar_landing_paper.py OUTDIR       (runtime ~/.venvs/constrained-rl)
Outputs in OUTDIR: figures landing-flights, landing-summary, landing-descent-seeds, landing-convergence (pdf/png);
tables landing.tex, landing-values.tex, landing-convergence.tex; landing-numbers.json with every number the text uses.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, MUTED, TEXT2, FS, TEXTW, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
BETA, H, R = 0.999, 1000, 36.0
TAIL = (1 - BETA ** 900) / (1 - BETA)
PREFIX = "ledger-R36-b24"
SETS = {"design": "ledger-staged-anneal2", "shorter": "ledger-staged-anneal", "other start": "ledger-anneal"}


def load(stage):
    res = defaultdict(dict)
    for f in sorted((ROOT / "lunar" / stage).glob(f"{PREFIX}-k*__seed*/result.json")):
        r = json.loads(f.read_text()); k = float(r["arm"]["vehicle_worth_in_landings"])
        pr = f.parent / "priced-risk.json"
        r["priced"] = json.loads(pr.read_text()) if pr.exists() else None
        res[k][r["seed"]] = r
    split = json.loads((ROOT / f"lunar-touchdown-split-{stage}.json").read_text())
    for k in res:
        for s, r in res[k].items():
            r["split"] = split[f"{PREFIX}-k{k:g}__seed{s}"]
    return res


D = {name: load(stage) for name, stage in SETS.items()}
KS = sorted(D["design"])
SEEDS = sorted(D["design"][0.0])
M = {k: D["design"][k][SEEDS[0]]["arm"]["m"] for k in KS}
for k in KS:
    assert abs(M[k] - R * k / TAIL) < 1e-9


def arr(name, k, f):
    return np.array([f(D[name][k][s]) for s in SEEDS], float)


risk = lambda r: r["stochastic"]["risk"]
U = lambda r: r["stochastic"]["U"]
fuel = lambda r: r["stochastic"]["fuel_mean"]
before = lambda r: r["split"]["fuel_before_touchdown"]
q = lambda r: r["priced"]["priced_risk"] if r["priced"] else np.nan
burn_out = lambda r: r["split"]["ends_with_tank_empty"] > 0.5
collapsed = lambda r: r["stochastic"]["risk"] > 0.5
value = lambda r, m: U(r) - m * q(r) / (1 - BETA)
numbers = dict(revenue=R, beta=BETA, horizon=H, charge_factor_at_frame_100=TAIL, worth=KS, price={f"{k:g}": M[k] for k in KS}, seeds=SEEDS)

# ---------------------------------------------------------------- tables
rows = []
cells = {}
for k in KS:
    rk, uk = arr("design", k, risk), arr("design", k, U)
    fell = int(np.sum(rk < arr("design", 0.0, risk))) if k else None
    c = dict(failed_mean=float(rk.mean()), failed_sd=float(rk.std(ddof=1)), failed_median=float(np.median(rk)),
             lost=float(arr("design", k, lambda r: r["stochastic"]["lost"]).mean()), empty=float(arr("design", k, lambda r: r["stochastic"]["exhausted"]).mean()),
             completed=float(arr("design", k, lambda r: r["stochastic"]["completed"]).mean()), q=float(np.nanmean(arr("design", k, q))),
             fuel_before=float(arr("design", k, before).mean()), fuel=float(arr("design", k, fuel).mean()),
             frames_before=float(arr("design", k, lambda r: r["split"]["frames_before_touchdown"]).mean()),
             U_mean=float(uk.mean()), U_sd=float(uk.std(ddof=1)), fell=fell)
    frames = [t for s in SEEDS if D["design"][k][s]["priced"] for t in D["design"][k][s]["priced"]["exhaustion_frames"]]
    c["median_failing_frame"] = float(np.median(frames)) if frames else None
    cells[f"{k:g}"] = c
    rows.append(f"${k:g}$ & ${M[k]:.3f}$ & ${100 * c['failed_mean']:.2f}\\pm{100 * c['failed_sd']:.2f}$ & ${100 * c['lost']:.2f}$ & ${100 * c['empty']:.2f}$ & ${100 * c['completed']:.1f}$"
                f" & ${c['q']:.4f}$ & ${c['fuel_before']:.1f}$ & ${c['fuel']:.1f}$ & ${c['U_mean']:.2f}\\pm{c['U_sd']:.2f}$ & {'--' if fell is None else f'${fell}/{len(SEEDS)}$'} \\\\")
numbers["design"] = cells
(OUT / "landing.tex").write_text(
    "\\begin{tabular}{rrrrrrrrrrr}\n\\toprule\n$k$ & $m$ & Failed (\\%) & Lost (\\%) & Empty (\\%) & Pad (\\%) & $q_{\\beta,H}$ & Fuel before & Fuel & $U_H$ & Fell \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# the value of every policy at every price (means over the seeds)
V = np.array([[float(np.mean([value(D["design"][k][s], M[kp]) for s in SEEDS])) for kp in KS] for k in KS])
best = V.max(axis=0)
short = [float(best[i] - V[i, i]) for i in range(len(KS))]
numbers["values"] = dict(matrix=V.tolist(), shortfall_at_own_price=short)
lines = []
for i, k in enumerate(KS):
    lines.append(f"${k:g}$ & " + " & ".join((f"$\\mathbf{{{v:.2f}}}$" if abs(v - best[j]) < 1e-9 else f"${v:.2f}$") for j, v in enumerate(V[i])) + " \\\\")
(OUT / "landing-values.tex").write_text(
    "\\begin{tabular}{r" + "r" * len(KS) + "}\n\\toprule\n & \\multicolumn{" + str(len(KS)) + "}{c}{Value at the charge $k'$} \\\\\n\\cmidrule(lr){2-" + str(len(KS) + 1) + "}\n"
    "Trained at $k$ & " + " & ".join(f"${k:g}$" for k in KS) + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# convergence and the starting point: the three sets side by side
lines, conv = [], {}
for k in KS:
    row = [f"${k:g}$"]
    conv[f"{k:g}"] = {}
    for key, f, fmt in (("failed", lambda r: 100 * risk(r), "{:.2f}"), ("fuel", fuel, "{:.1f}"), ("value", lambda r, k=k: value(r, M[k]), "{:.2f}")):
        for name in ("other start", "shorter", "design"):
            v = arr(name, k, f)
            conv[f"{k:g}"][f"{key} {name}"] = dict(mean=float(np.nanmean(v)), median=float(np.nanmedian(v)))
            row.append("$" + fmt.format(np.nanmedian(v)) + "$")
    lines.append(" & ".join(row) + " \\\\")
numbers["convergence_medians"] = conv
numbers["other_start"] = dict(collapsed=[f"k={k:g} seed {s}" for k in KS for s in SEEDS if collapsed(D["other start"][k][s])],
                              burn_out=[f"k={k:g} seed {s}" for k in KS for s in SEEDS if burn_out(D["other start"][k][s])])
numbers["design_burn_out_or_collapsed"] = sum(burn_out(D[n][k][s]) or collapsed(D[n][k][s]) for n in ("design", "shorter") for k in KS for s in SEEDS)
(OUT / "landing-convergence.tex").write_text(
    "\\begin{tabular}{r*{9}{r}}\n\\toprule\n & \\multicolumn{3}{c}{Failed (\\%)} & \\multicolumn{3}{c}{Fuel} & \\multicolumn{3}{c}{Value} \\\\\n"
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n$k$ & A & B & C & A & B & C & A & B & C \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# paired comparison with no price (design)
pair = {}
for k in KS[1:]:
    d = arr("design", 0.0, risk) - arr("design", k, risk)
    pair[f"{k:g}"] = dict(fell_in=int((d > 0).sum()), mean_fall_points=float(100 * d.mean()), t=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))))
numbers["paired_with_no_price"] = pair

# ---------------------------------------------------------------- figures
xs = np.arange(len(KS))
xt = [f"{k:g}" for k in KS]


def per_price_panel(ax, name, f, scale=1.0, color=PAL[0], line=True, marker="o"):
    for xi, k in zip(xs, KS):
        v = scale * arr(name, k, f)
        ax.plot(np.full(len(v), xi) + np.linspace(-0.12, 0.12, len(v)), v, ls="", marker=".", ms=3, color=color, alpha=0.45)
    if line:
        ax.plot(xs, [scale * np.median(arr(name, k, f)) for k in KS], color=color, marker=marker, ms=3.2, lw=1.1)


# summary: failed, fuel before touchdown and in all, utility
fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.15), gridspec_kw=dict(wspace=0.55))
per_price_panel(axes[0], "design", risk, 100.0)
axes[0].set_ylim(0, None); axes[0].set_ylabel("failed (%)")
per_price_panel(axes[1], "design", fuel, color=PAL[0])
per_price_panel(axes[1], "design", before, color=PAL[1], marker="s")
axes[1].set_ylabel("fuel used")
axes[1].legend(handles=[Line2D([], [], color=PAL[0], marker="o", ms=3, lw=1.1, label="in all"), Line2D([], [], color=PAL[1], marker="s", ms=3, lw=1.1, label="before touchdown")],
               frameon=False, fontsize=FS - 2, loc="upper left", handlelength=1.4)
per_price_panel(axes[2], "design", U)
axes[2].set_ylabel("utility $U_H$")
for ax, letter in zip(axes, "abc"):
    ax.set_xticks(xs); ax.set_xticklabels(xt, fontsize=FS - 2.5, rotation=45); ax.set_xlabel("charge $k$"); panel_label(ax, letter)
save(fig, "landing-summary", OUT)

# flights of the median seed without a price and at the largest charge, and the descent profile of every charge
P = json.loads((ROOT / f"lunar-paths-{SETS['design']}.json").read_text())
END_MK = {"landed": "o", "lost": "v", "empty": "x"}
END_COL = {"landed": PAL[2], "lost": PAL[3], "empty": PAL[7]}
XLIM, YLIM = (-0.5, 0.42), (-0.1, 0.45)
TMAX = 250
fsum = {s: sum(D["design"][k][s]["stochastic"]["risk"] for k in (0.0, KS[-1])) for s in SEEDS}
MED = sorted(SEEDS, key=lambda s: fsum[s])[len(SEEDS) // 2]
numbers["flights_seed"] = MED
HB = np.linspace(0.05, 1.4, 28)          # from the height 0.05 up, below which the legs touch the ground and the lander rests


def profile(paths):
    """median descent speed (minus the vertical velocity) in height bins, over all frames of the flights"""
    hs, vs = [], []
    for p in paths:
        hs.append(np.array(p["y"])); vs.append(-np.array(p["vy"]))
    h, v = np.concatenate(hs), np.concatenate(vs)
    idx = np.digitize(h, HB)
    mid = 0.5 * (HB[1:] + HB[:-1])
    med = np.array([np.median(v[idx == i + 1]) if np.sum(idx == i + 1) > 20 else np.nan for i in range(len(mid))])
    return mid, med


fig, (axf, axp) = plt.subplots(1, 2, figsize=(TEXTW, 2.25), gridspec_kw=dict(wspace=0.32))
COLK = {0.0: plt.cm.viridis(0.0), KS[-1]: plt.cm.viridis(1.0)}
for k in (0.0, KS[-1]):
    v = P[f"{PREFIX}-k{k:g}__seed{MED}"]
    for p in v["paths"]:
        t = np.arange(len(p["y"]))
        axf.plot(t, p["y"], color=COLK[k], lw=0.6, alpha=0.6 if k == 0 else 0.8, zorder=2 if k == 0 else 3)
        if p["end"] != "landed":
            axf.plot(t[-1], p["y"][-1], marker=END_MK[p["end"]], color="k", ms=3.5, mew=0.9, mfc="white", ls="none", zorder=5)
axf.set_xlim(0, TMAX); axf.set_ylim(-0.05, 1.5); axf.set_xlabel("frame"); axf.set_ylabel("height")
axf.legend(handles=[Line2D([], [], color=COLK[0.0], lw=1.0, label="no price"), Line2D([], [], color=COLK[KS[-1]], lw=1.0, label=f"charge $k={KS[-1]:g}$")],
           frameon=False, fontsize=FS - 2, loc="upper right", handlelength=1.4)
panel_label(axf, "a")
prof = {}
for i, k in enumerate(KS):
    paths = [p for s in SEEDS for p in P[f"{PREFIX}-k{k:g}__seed{s}"]["paths"]]
    mid, med = profile(paths)
    prof[f"{k:g}"] = med.tolist()
    axp.plot(mid, med, color=plt.cm.viridis(i / (len(KS) - 1)), lw=1.0, label=f"${k:g}$")
numbers["descent_profile"] = dict(height=(0.5 * (HB[1:] + HB[:-1])).tolist(), median_descent_speed=prof)
numbers["flights_ends"] = {f"{k:g}": [p["end"] for p in P[f"{PREFIX}-k{k:g}__seed{MED}"]["paths"]] for k in (0.0, KS[-1])}
axp.set_xlim(0, 1.0); axp.set_ylim(0, None); axp.set_xlabel("height"); axp.set_ylabel("descent speed")
axp.legend(frameon=False, fontsize=FS - 2.5, ncol=2, title="charge $k$", title_fontsize=FS - 2.5, loc="lower right", handlelength=1.2, columnspacing=0.8)
panel_label(axp, "b")
save(fig, "landing-flights", OUT)

# descent profile per seed (Online Resource)
fig, axes = plt.subplots(2, 5, figsize=(TEXTW, 3.3), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.12, hspace=0.35))
for ax, s, j in zip(axes.ravel(), SEEDS, range(len(SEEDS))):
    for i, k in enumerate(KS):
        mid, med = profile(P[f"{PREFIX}-k{k:g}__seed{s}"]["paths"])
        ax.plot(mid, med, color=plt.cm.viridis(i / (len(KS) - 1)), lw=0.8, label=f"${k:g}$")
    ax.set_title(f"seed {j + 1}", fontsize=FS - 1); ax.set_xlim(0, 1.0); ax.set_xticks([0.25, 0.75]); ax.tick_params(labelsize=FS - 2.5)
for ax in axes[1]:
    ax.set_xlabel("height", fontsize=FS - 1)
for ax in axes[:, 0]:
    ax.set_ylabel("descent speed", fontsize=FS - 1)
axes[0, 0].set_ylim(0, None)
hl = axes[0, 0].get_legend_handles_labels()
fig.legend(hl[0], [f"$k={x[1:-1]}$" for x in hl[1]], loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=8, frameon=False, fontsize=FS - 2.5, handlelength=1.2, columnspacing=0.9)
save(fig, "landing-descent-seeds", OUT)

# training traces of the design: 2e7 + 3e7 steps at the constant rate, then 1e7 decayed
STAGES = (("ledger-staged-const", 0), ("ledger-staged-const2", 2e7), ("ledger-staged-anneal2", 5e7))
tr = defaultdict(lambda: defaultdict(list))
for st, off in STAGES:
    for p in sorted((ROOT / "lunar" / st).glob(f"{PREFIX}-k*__seed*/progress.jsonl")):
        m = re.match(rf"{PREFIX}-k([\d.]+)__seed(\d+)", p.parent.name)
        tr[float(m[1])][int(m[2])] += [{**json.loads(x), "steps": json.loads(x)["steps"] + off} for x in p.read_text().splitlines() if x.strip()]
grid = np.linspace(0.3e6, 6e7, 200)
fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.25), gridspec_kw=dict(wspace=0.55))
settle = {}
for i, k in enumerate(KS):
    for ax, key, sc in zip(axes, ("failure_rate", "fuel", "U"), (100.0, 1.0, 1.0)):
        curves = []
        for s in SEEDS:
            t = tr[k][s]
            x = np.array([r["steps"] for r in t], float); y = sc * np.array([r[key] for r in t], float)
            kern = np.ones(31)
            ys = np.convolve(y, kern, mode="same") / np.convolve(np.ones(len(y)), kern, mode="same")
            curves.append(np.interp(grid, x, ys))
        ax.plot(grid / 1e6, np.median(curves, 0), color=plt.cm.viridis(i / (len(KS) - 1)), lw=0.9, label=f"${k:g}$")
    w = lambda key, lo, hi: float(np.mean([np.sum([r["episodes"] * r[key] for r in tr[k][s] if lo < r["steps"] <= hi]) / np.sum([r["episodes"] for r in tr[k][s] if lo < r["steps"] <= hi]) for s in SEEDS]))
    settle[f"{k:g}"] = {key: [w(key, a * 1e6, b * 1e6) for a, b in ((20, 30), (30, 40), (40, 50))] for key in ("failure_rate", "fuel", "U")}
numbers["training_windows_20_30_40_50"] = settle
for ax, lab, letter in zip(axes, ("failed in training (%)", "fuel used", "utility $U_H$"), "abc"):
    for xv in (20, 50):
        ax.axvline(xv, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("steps (millions)"); ax.set_ylabel(lab); panel_label(ax, letter)
axes[0].set_ylim(0, 4)
hl = axes[0].get_legend_handles_labels()
fig.legend(hl[0], [f"$k={x[1:-1]}$" for x in hl[1]], loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=8, frameon=False, fontsize=FS - 2.5, handlelength=1.2, columnspacing=0.9)
save(fig, "landing-convergence", OUT)

(OUT / "landing-numbers.json").write_text(json.dumps(numbers, indent=1) + "\n")
print(json.dumps({k: numbers[k] for k in ("flights_seed", "paired_with_no_price", "other_start", "design_burn_out_or_collapsed")}, indent=1))
print("shortfall at the own price:", np.round(short, 3).tolist())
print("q per charge:", [round(cells[f'{k:g}']['q'], 5) for k in KS])
print("median failing frame:", [cells[f'{k:g}']['median_failing_frame'] for k in KS])
