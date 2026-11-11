"""Revision 76: the benchmark task as a picture (for the Online Resource).

(a) One layout of the task seen from above: the hazards, the goals given in turn and the path of the robot under
    the policy trained without a price, in an episode that exhausts the cost budget.
(b) The same layout under the policy trained at the price 0.03.
(c) The cumulative cost against the step in six episodes of each policy, with the budget of 25.
Reads safety-problem-v76.json (safety_problem_eval_v76.py). Output: safety-problem-v76 (pdf/png).
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
from matplotlib.patches import Circle, Patch  # noqa: E402

d = json.loads((ROOT / "safety-problem-v76.json").read_text())
A0, A1 = "fixed_m0_b25", "fixed_m0.03_b25"
C0, C1, CH, CG = TEXT2, PAL[6], PAL[7], PAL[2]
# the layout shown: the first reset seed in which the episode without a price exhausts the budget
rs = next((e["reset_seed"] for e in d[A0]["episodes"] if e["end"] == "exhausted"), d[A0]["episodes"][0]["reset_seed"])
ep = {a: next(e for e in d[a]["episodes"] if e["reset_seed"] == rs) for a in (A0, A1)}
fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.95), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1, 1.15]))
for ax, a, col, tag in ((axes[0], A0, C0, "a"), (axes[1], A1, C1, "b")):
    e = ep[a]; L = e["layout"]
    for h in L["hazards"]:
        ax.add_patch(Circle(h, L["hazard_radius"], color=CH, alpha=0.30, lw=0))
    for k, (t, gx, gy) in enumerate(e["goals"]):
        ax.add_patch(Circle((gx, gy), L["goal_radius"], facecolor=CG, alpha=0.22, edgecolor=CG, lw=0.6))
        ax.text(gx, gy, str(k + 1), ha="center", va="center", fontsize=FS - 2, color=CG)
    p = np.array(e["path"])
    ax.plot(p[:, 0], p[:, 1], color=col, lw=0.8)
    ax.plot(p[0, 0], p[0, 1], marker="s", color=col, ms=4, ls="none")
    ax.plot(p[-1, 0], p[-1, 1], marker="x" if e["end"] == "exhausted" else "o", color=col, ms=5, mew=1.2, mfc="white", ls="none", zorder=5)
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("no price" if a == A0 else f"price $m={d[a]['m']:g}$"); panel_label(ax, tag)
ax = axes[2]
for a, col in ((A0, C0), (A1, C1)):
    for e in d[a]["episodes"]:
        c = np.array(e["cost"]); ax.plot(np.arange(len(c)), c, color=col, lw=0.9, alpha=0.9)
        if e["end"] == "exhausted":
            ax.plot(len(c) - 1, c[-1], marker="x", color=col, ms=5, mew=1.2, ls="none", zorder=5)
ax.axhline(d[A0]["budget"], color=CH, lw=1.2)
ax.set_xlabel("step"); ax.set_ylabel("cumulative cost"); ax.set_ylim(-0.8, 29); ax.set_xlim(0, 1000); panel_label(ax, "c")
handles = [Patch(color=CH, alpha=0.30, label="hazard"), Patch(facecolor=CG, alpha=0.3, edgecolor=CG, label="goals, in turn"),
           Line2D([], [], color=C0, lw=1.0, label="no price"), Line2D([], [], color=C1, lw=1.0, label=f"price $m={d[A1]['m']:g}$"),
           Line2D([], [], color=TEXT2, marker="s", ls="none", ms=4, label="start"),
           Line2D([], [], color=TEXT2, marker="x", ls="none", ms=5, mew=1.2, label="budget exhausted"),
           Line2D([], [], color=CH, lw=1.2, label="budget $25$")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=FS, columnspacing=1.2, handletextpad=0.5)
save(fig, "safety-problem-v76", ROOT)
print("layout of reset seed", rs, {a: (ep[a]["end"], ep[a]["steps"], ep[a]["cost"][-1], len(ep[a]["goals"]) - 1) for a in ep})
print({a: dict(goals=[len(e["goals"]) - 1 for e in d[a]["episodes"]], cost=[e["cost"][-1] for e in d[a]["episodes"]], ends=[e["end"] for e in d[a]["episodes"]]) for a in d})
