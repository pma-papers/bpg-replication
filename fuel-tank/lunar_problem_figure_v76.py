"""Revision 76: the fuel-tank problem as flight paths (for the Online Resource).

Ten episodes of one seed in the plane of the lander's position (the first two coordinates of the observation;
the landing pad is at height zero between -0.2 and 0.2): (a) the controller trained without a fuel limit,
(b) the policy trained at the tank of 20 units without a price, (c) the policy trained at that tank at the
price 0.5. The marker shows how each episode ended; an episode that exhausts the tank ends in the air.
Reads lunar-setup-v76.json (lunar_setup_eval.py). Output: lunar-problem-v76 (pdf/png).
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

SEED = 72001
END_MK = {"landed": "o", "crashed": "v", "exhausted": "x"}
END_COL = {"landed": PAL[2], "crashed": PAL[3], "exhausted": PAL[7]}
d = json.loads((ROOT / "lunar-setup-v76.json").read_text())
fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.95), sharey=True, gridspec_kw=dict(wspace=0.12))
counts = {}
for ax, (arm, lab), tag in zip(axes, (("lunar-unlimited", "no fuel limit"), ("lunar-b20-m0", "tank $20$, no price"), ("lunar-b20-m0.5", "tank $20$, price $m=0.5$")), "abc"):
    v = d[f"{arm}__seed{SEED}"]
    for p in v["paths"]:
        ax.plot(p["x"], p["y"], color=TEXT2, lw=0.7, alpha=0.75)
        ax.plot(p["x"][-1], p["y"][-1], marker=END_MK[p["end"]], color=END_COL[p["end"]], ms=5, mew=1.2, mfc="white", ls="none", zorder=4)
    counts[arm] = {e: sum(p["end"] == e for p in v["paths"]) for e in END_MK}
    ax.plot([-0.2, 0.2], [0, 0], color="k", lw=2.5, solid_capstyle="butt", zorder=3)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.08, 1.5); ax.set_xlabel(f"horizontal position\n{lab}")
    ax.grid(False); panel_label(ax, tag)
axes[0].set_ylabel("height")
handles = [Line2D([], [], color="k", lw=2.5, label="landing pad")] + [Line2D([], [], color=END_COL[e], marker=END_MK[e], mfc="white", mew=1.2, ls="none", ms=5, label=lab)
                                                                       for e, lab in (("landed", "landed"), ("crashed", "crashed"), ("exhausted", "tank exhausted"))]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=FS, columnspacing=1.2, handletextpad=0.5)
save(fig, "lunar-problem-v76", ROOT)
print(counts)
