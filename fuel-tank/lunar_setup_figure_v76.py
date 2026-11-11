"""Revision 76: the setup of the fuel-tank study in one figure, at the smallest tank (20 units).

(a) Fuel used per episode by the controller trained without a fuel limit (five seeds pooled), with the tanks of
    the study marked: what a tank asks of the controller.
(b) Fuel used against the frame in ten episodes of one seed at the tank of 20 units, for the policy trained
    without a price and for the policy trained at the price 0.5, with the end of each episode marked; the axes show
    the last eight units of the tank, where the episodes differ.
(c) Fuel used per episode at the tank of 20 units (five seeds pooled) without a price and at the price 0.5; an
    episode that exhausts the tank ends at 20.
Reads lunar-setup-v76.json (lunar_setup_eval.py). Output: lunar-setup-v76 (pdf/png), lunar-setup-numbers-v76.json.
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

TANK, PRICE, SEED, TANKS = 20, 0.5, 72001, (24, 23, 22, 21, 20)
C0, C1, CT = MUTED, PAL[6], PAL[7]            # no price, the price 0.5 (as in the results figure), the tank
END_MK = {"landed": "o", "crashed": "v", "exhausted": "x"}
d = json.loads((ROOT / "lunar-setup-v76.json").read_text())
arm = lambda name: [v for v in d.values() if v["arm"] == name]
pool = lambda name: np.concatenate([np.array(v["used"]) for v in arm(name)])
share = lambda name, end: float(np.mean([e == end for v in arm(name) for e in v["end"]]))
A0, A1 = f"lunar-b{TANK}-m0", f"lunar-b{TANK}-m{PRICE:g}"
numbers = {}

fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.0), gridspec_kw=dict(wspace=0.62))
# (a) the controller without a fuel limit, and the tanks
ax = axes[0]
u = pool("lunar-unlimited")
ax.hist(u, bins=np.arange(14, 34.01, 0.5), density=True, color=TEXT2, alpha=0.45, lw=0)
for b in TANKS:
    ax.axvline(b, color=CT, lw=1.3 if b == TANK else 0.7, ls="-" if b == TANK else "--")
numbers["share_of_episodes_without_fuel_limit_using_more_than"] = {str(b): float(np.mean(u > b)) for b in TANKS}
numbers["median_fuel_without_fuel_limit"] = float(np.median(u))
ax.set_xlim(14, 34); ax.set_xlabel("fuel used in an episode,\nno fuel limit"); ax.set_ylabel("density"); panel_label(ax, "a")
# (b) ten episodes of one seed at the tank
ax = axes[1]
for name, col in ((A0, C0), (A1, C1)):
    v = next(x for x in arm(name) if x["seed"] == SEED)
    for p in v["paths"]:
        y = np.array(p["used"]); t = np.arange(1, len(y) + 1)
        ax.plot(t, y, color=col, lw=0.9, alpha=0.9)
        ax.plot(t[-1], y[-1], marker=END_MK.get(p["end"], "."), color=col, ms=4.5, mew=1.1, mfc="white", ls="none", zorder=4)
    numbers[f"paths_{name}_seed{SEED}"] = [p["end"] for p in v["paths"]]
ax.axhline(TANK, color=CT, lw=1.3)
ax.set_ylim(TANK - 8, TANK + 0.8); ax.set_xlim(60, 215); ax.set_xlabel("frame"); ax.set_ylabel(f"fuel used, tank ${TANK}$"); panel_label(ax, "b")
# (c) fuel used per episode at the tank
ax = axes[2]
bins = np.arange(10, TANK + 0.51, 0.5)
x0, x1 = np.minimum(pool(A0), TANK + 0.25), np.minimum(pool(A1), TANK + 0.25)      # an exhausted episode ends at the tank: last bin
ax.hist(x0, bins=bins, density=True, color=C0, alpha=0.6, lw=0)
ax.hist(x1, bins=bins, density=True, histtype="step", color=C1, lw=1.2)
ax.axvline(TANK, color=CT, lw=1.3)
for name in ("lunar-unlimited", A0, A1):
    numbers[name] = dict(exhausted=share(name, "exhausted"), landed=share(name, "landed"), crashed=share(name, "crashed"),
                         median_fuel=float(np.median(pool(name))), median_fuel_of_landings=float(np.median(np.concatenate(
                             [np.array(v["used"])[np.array(v["end"]) == "landed"] for v in arm(name)]))))
ax.set_xlim(10, TANK + 1.2); ax.set_xlabel(f"fuel used in an episode,\ntank ${TANK}$"); ax.set_ylabel("density"); panel_label(ax, "c")
handles = [Line2D([], [], color=C0, lw=4, alpha=0.6, label="no price"), Line2D([], [], color=C1, lw=1.2, label=f"price $m={PRICE:g}$"),
           Line2D([], [], color=CT, lw=1.3, label=f"tank ${TANK}$"), Line2D([], [], color=CT, lw=0.7, ls="--", label="tanks $21$ to $24$"),
           Line2D([], [], color=TEXT2, marker="o", mfc="white", ls="none", ms=4.5, label="landed"),
           Line2D([], [], color=TEXT2, marker="v", mfc="white", ls="none", ms=4.5, label="crashed"),
           Line2D([], [], color=TEXT2, marker="x", ls="none", ms=4.5, mew=1.1, label="tank exhausted")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=4, frameon=False, fontsize=FS, columnspacing=1.2, handletextpad=0.5)
(ROOT / "lunar-setup-numbers-v76.json").write_text(json.dumps(numbers, indent=1) + "\n")
save(fig, "lunar-setup-v76", ROOT)
print(json.dumps(numbers, indent=1))
