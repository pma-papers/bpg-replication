"""Flight paths near the landing pad for the policies of one stage, in the form of the revised figure of the
fuel-tank problem (2026-11-14): the last part of the flight in 32 episodes of one seed, one panel per configuration,
a path in the colour of its end, a marker where it ended, and above each panel the frequencies of the ends over the
1000 evaluation episodes of the policy, with its stopped utility and median fuel use.

Usage:  python lunar_paths_figure.py STAGE OUTPUT-NAME ARM [ARM ...] [--seed SEED | --seed all]
Without --seed the seed is the one whose failure probability, summed over the configurations shown, is the median
of the five seeds. With --seed all the panels show the 32 episodes of every seed (160 flights), and the numbers
above a panel are means over the seeds. With --seed each and one ARM the panels are the seeds of that configuration. With --seed grid the rows are the
configurations and the columns the seeds; above a panel are then the percentages of the three ends alone, in the
colours of the ends. Reads lunar-paths-STAGE.json (lunar_problem_eval_v76.py --stage STAGE); writes OUTPUT-NAME (pdf/png) and OUTPUT-NAME-numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

args = sys.argv[1:]
seed = None
if "--seed" in args:
    i = args.index("--seed"); seed = args[i + 1] if args[i + 1] in ("all", "each", "grid") else int(args[i + 1]); args = args[:i] + args[i + 2:]
stage, name, arms = args[0], args[1], args[2:]
ENDS = ("landed", "lost", "empty")
END_MK = {"landed": "o", "lost": "v", "empty": "x"}
END_COL = {"landed": PAL[2], "lost": PAL[3], "empty": PAL[7]}
END_LAB = {"landed": "landed", "lost": "lander lost", "empty": "tank empty before touchdown"}
END_SHORT = {"landed": "landed", "lost": "lost", "empty": "empty"}
XLIM, YLIM = (-0.5, 0.42), (-0.1, 0.45)
d = json.loads((ROOT / f"lunar-paths-{stage}.json").read_text())
seeds = sorted({v["seed"] for k, v in d.items() if v["arm"] in arms})
fail = {s: sum(1 - d[f"{a}__seed{s}"]["frequencies"]["landed"] for a in arms) for s in seeds}
if seed is None:
    seed = sorted(seeds, key=lambda s: fail[s])[len(seeds) // 2]


ONE_TANK = len({d[f"{a}__seed{seeds[0]}"]["budget"] for a in arms}) == 1      # with one tank in the figure the panels are named by the price alone


def label(v):
    if v["budget"] > 1e5:
        return "no fuel limit"
    price = "no price" if v["m"] == 0 else f"price $m={v['m']:g}$"
    return price if ONE_TANK else f"tank ${v['budget']:g}$, {price}"


if seed == "grid":                                    # rows the configurations, columns the seeds
    nrow, ncol = len(arms), len(seeds)
    fig, axes = plt.subplots(nrow, ncol, figsize=(TEXTW, 0.78 * nrow + 0.75), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.08, hspace=0.42), squeeze=False)
    out = dict(stage=stage, seed="grid")
    for i, arm in enumerate(arms):
        for j, sd in enumerate(seeds):
            ax, v = axes[i, j], d[f"{arm}__seed{sd}"]
            for e in ENDS:
                for q in v["paths"]:
                    if q["end"] != e:
                        continue
                    ax.plot(q["x"], q["y"], color=END_COL[e], lw=0.5, alpha=0.55 if e == "landed" else 0.85, zorder=2 if e == "landed" else 3)
                    ax.plot(np.clip(q["x"][-1], *XLIM), np.clip(q["y"][-1], *YLIM), marker=END_MK[e], color=END_COL[e], ms=2.6, mew=0.7, mfc="white", ls="none", zorder=5, clip_on=False)
            ax.plot([-0.2, 0.2], [0, 0], color="k", lw=1.8, solid_capstyle="butt", zorder=4)
            ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
            ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.grid(False); ax.set_xticks([-0.3, 0, 0.3]); ax.set_yticks([0, 0.2, 0.4]); ax.tick_params(labelsize=FS - 2.5)
            f = v["frequencies"]
            for e, xpos, ha in zip(ENDS, (0.0, 0.5, 1.0), ("left", "center", "right")):
                ax.text(xpos, 1.03, f"{100 * f[e]:.0f}%", transform=ax.transAxes, ha=ha, va="bottom", fontsize=FS - 2.8, color=END_COL[e])
            if i == 0:
                ax.text(0.5, 1.24, f"seed {j + 1}", transform=ax.transAxes, ha="center", va="bottom", fontsize=FS - 1)
            if j == 0:
                ax.set_ylabel("no price" if v["m"] == 0 else f"$m={v['m']:g}$", fontsize=FS - 1)
            out[f"{arm} seed {sd}"] = dict(frequencies=f, U=v["U"], fuel_median=v["fuel_median"], paths={e: sum(q["end"] == e for q in v["paths"]) for e in ENDS})
    axes[-1, ncol // 2].set_xlabel("horizontal position", fontsize=FS - 1)
    handles = [Line2D([], [], color="k", lw=2.0, label="landing pad")] + [Line2D([], [], color=END_COL[e], marker=END_MK[e], mfc="white", mew=0.9, lw=0.8, ms=3.5, label=END_LAB[e]) for e in ENDS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=FS - 1.5, columnspacing=1.0, handletextpad=0.5)
    fig.subplots_adjust(bottom=0.135, top=0.94)
    (ROOT / f"{name}-numbers.json").write_text(json.dumps(out, indent=1) + "\n")
    save(fig, name, ROOT)
    sys.exit(0)
EACH = seed == "each"
if EACH:
    assert len(arms) == 1
    panels = [(arms[0], s) for s in seeds]
else:
    panels = [(a, seed) for a in arms]
ncol = 3 if EACH else (2 if len(arms) > 3 else len(arms))
nrow = int(np.ceil(len(panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(TEXTW, (1.95 if ncol == 2 else 1.75) * nrow if nrow > 1 else 2.2), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.75), squeeze=False)
out = dict(stage=stage, seed=seed, failure_summed_over_the_panels_per_seed={str(s): round(fail[s], 3) for s in seeds})
for ax, (arm, sd), tag in zip(axes.ravel(), panels, "abcdefgh"):
    vs = [d[f"{arm}__seed{s}"] for s in (seeds if sd == "all" else [sd])]
    v = dict(vs[0], paths=[p for x in vs for p in x["paths"]], U=float(np.mean([x["U"] for x in vs])), fuel_median=float(np.mean([x["fuel_median"] for x in vs])),
             frequencies={k: float(np.mean([x["frequencies"][k] for x in vs])) for k in vs[0]["frequencies"]})
    for e in ENDS:
        for p in v["paths"]:
            if p["end"] != e:
                continue
            ax.plot(p["x"], p["y"], color=END_COL[e], lw=0.7 if seed != "all" or e != "landed" else 0.5, alpha=(0.55 if seed != "all" else 0.3) if e == "landed" else 0.85, zorder=2 if e == "landed" else 3)
            ax.plot(np.clip(p["x"][-1], *XLIM), np.clip(p["y"][-1], *YLIM), marker=END_MK[e], color=END_COL[e], ms=4.5, mew=1.1, mfc="white", ls="none", zorder=5, clip_on=False)
    ax.plot([-0.2, 0.2], [0, 0], color="k", lw=2.5, solid_capstyle="butt", zorder=4)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.grid(False)
    f = v["frequencies"]
    ax.text(0.5, 1.33, f"seed {seeds.index(sd) + 1}" if EACH else label(v), transform=ax.transAxes, ha="center", va="bottom", fontsize=FS)
    ax.text(0.5, 1.17, f"utility ${v['U']:.0f}$, fuel ${v['fuel_median']:.1f}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=FS - 2.3, color=TEXT2)
    for e, xpos, ha in zip(ENDS, (0.0, 0.5, 1.0), ("left", "center", "right")):
        ax.text(xpos, 1.03, f"{END_SHORT[e]} {100 * f[e]:.{1 if seed == 'all' else 0}f}%", transform=ax.transAxes, ha=ha, va="bottom", fontsize=FS - (3.6 if ncol == 3 else 2.3), color=END_COL[e])
    panel_label(ax, tag, y=1.33)
    out[f"{arm} seed {sd}" if EACH else arm] = dict(frequencies=f, U=v["U"], fuel_median=v["fuel_median"], paths={e: sum(p["end"] == e for p in v["paths"]) for e in ENDS})
for ax in axes[-1]:
    ax.set_xlabel("horizontal position")
for ax in axes[:, 0]:
    ax.set_ylabel("height")
for ax in axes.ravel()[len(panels):]:
    ax.axis("off")
for i, ax in enumerate(axes.ravel()[:len(panels)]):                       # a panel above an empty place carries the axis of its column
    if i + ncol >= len(panels) and i // ncol < nrow - 1:
        ax.tick_params(labelbottom=True); ax.set_xlabel("horizontal position")
handles = [Line2D([], [], color="k", lw=2.5, label="landing pad")] + [Line2D([], [], color=END_COL[e], marker=END_MK[e], mfc="white", mew=1.1, lw=0.9, ms=4.5, label=END_LAB[e]) for e in ENDS]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, (-0.03 if EACH else 0.02) if nrow > 1 else -0.08), ncol=4, frameon=False, fontsize=FS - 1, columnspacing=1.0, handletextpad=0.5)
assert name != f"lunar-paths-{stage}", "the output name must differ from the file of the recorded flights"
(ROOT / f"{name}-numbers.json").write_text(json.dumps(out, indent=1) + "\n")
save(fig, name, ROOT)
print(json.dumps(out))
