"""Revision 76: one figure of the noise and the anchor on the learned controller under three optimizers.

Redraws the records of the campaigns of revisions 41, 60 and 68 (wear-anchor/campaign-matched, wear-noise/campaign,
wear-noise-decoupled-v68/campaign, wear-optimizers-v68/campaign); no new runs. Every marker is one seed.
  (a) spread of the actor parameters around x_0 against the noise strength, anchor strength 10 and 1
  (b) breakdown probability of every seed against its spread, all configurations with anchor and noise
  (c) breakdown probability under the anchor alone against the anchor strength
Runtime ~/.venvs/constrained-rl/bin/python. Output wear-three-optimizers-v76.pdf/png.
"""
from __future__ import annotations
import glob, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from plotstyle import PAL, MK, TEXT2, MUTED, FS, TEXTW, panel_label, plt  # noqa: E402

D = 4418
ARMS = [("sgd", "plain gradient steps", PAL[2], MK[0]), ("adam-coupled", "Adam, anchor in the gradient (the runs of the paper)", PAL[1], MK[1]), ("adam-decoupled", "Adam, anchor applied after the step", PAL[0], MK[3])]
data = defaultdict(list)       # (arm, k1, k2) -> list of (spread, risk %, U)


def take(pattern, arm, k1=None, k2=None):
    for f in sorted(glob.glob(str(HERE / pattern))):
        r = json.load(open(f)); a = r["arm"]
        key = (arm, float(a.get("kappa_1", 0.0) if k1 is None else k1), float(a["kappa_p"] if k2 is None else k2))
        data[key].append((r["distance"] / math.sqrt(D), 100 * r["stochastic"]["risk"], r["stochastic"]["U"]))


for name in glob.glob(str(HERE / "wear-optimizers-v68/campaign/*__seed94001")):
    base = Path(name).name.rsplit("__seed", 1)[0]
    if "zerograd" not in base:
        take(f"wear-optimizers-v68/campaign/{base}__seed*/result.json", base.split("__")[0])
take("wear-noise/campaign/*/result.json", "adam-coupled")
take("wear-anchor/campaign-matched/kappa*/result.json", "adam-coupled", k1=0.0)
take("wear-noise-decoupled-v68/campaign/*/result.json", "adam-decoupled")
for k in (0.0,):                                    # without the anchor the two Adam forms are the same runs
    data[("adam-decoupled", 0.0, 0.0)] = data[("adam-coupled", 0.0, 0.0)]

fig, axes = plt.subplots(1, 3, figsize=(TEXTW * 1.45, TEXTW * 0.52))
ax = axes[0]
k = np.geomspace(6e-4, 1.6, 50)
ax.plot(k, np.sqrt(k / 10), color=TEXT2, lw=0.9, ls=":"); ax.plot(k, np.sqrt(k / 1), color=TEXT2, lw=0.9, ls=":"); ax.plot(k, math.sqrt(2) * k, color=TEXT2, lw=0.9, ls="--")
ax.annotate(r"$\sqrt{\kappa_1/10}$", (1.2e-3, math.sqrt(1.2e-4)), xytext=(2, -11), textcoords="offset points", fontsize=FS - 1, color=TEXT2)
ax.annotate(r"$\sqrt{\kappa_1}$", (1.2e-3, math.sqrt(1.2e-3)), xytext=(2, 4), textcoords="offset points", fontsize=FS - 1, color=TEXT2)
ax.annotate(r"$\sqrt{2}\,\kappa_1$", (0.25, 0.35), xytext=(4, -12), textcoords="offset points", fontsize=FS - 1, color=TEXT2)
for arm, lab, c, mk in ARMS:
    for (a, k1, k2), v in data.items():
        if a == arm and k1 > 0 and k2 > 0:
            s = np.mean([x[0] for x in v])
            ax.plot(k1, s, ls="", marker=mk, ms=5.5, color=c, mfc=(c if k2 == 10 else "none"), mew=1.1, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"noise strength $\kappa_1$"); ax.set_ylabel(r"spread $\|x-x_0\|/\sqrt{d}$")

ax = axes[1]
for arm, lab, c, mk in ARMS:
    first = True
    for (a, k1, k2), v in sorted(data.items()):
        if a == arm and k1 > 0 and k2 > 0:
            ax.plot([x[0] for x in v], [x[1] for x in v], ls="", marker=mk, ms=4, color=c, mfc=(c if k2 == 10 else "none"), mew=0.9, alpha=0.85, label=lab if first else None); first = False
ax.axvline(0.1, color=MUTED, lw=0.8, ls=":"); ax.annotate("incumbent's\ndispersion", (0.1, 42), xytext=(-3, 0), textcoords="offset points", ha="right", fontsize=FS - 1.5, color=TEXT2)
ax.axhline(1.08, color=TEXT2, lw=0.8, ls=":"); ax.axhline(59.6, color=TEXT2, lw=0.8, ls="--")
ax.annotate("optimum without a price", (0.0, 59.6), xycoords=("axes fraction", "data"), xytext=(2, 2), textcoords="offset points", ha="left", fontsize=FS - 1.5, color=TEXT2)
ax.set_xscale("log"); ax.set_xlabel(r"spread $\|x-x_0\|/\sqrt{d}$"); ax.set_ylabel("breakdown probability (%)"); ax.set_ylim(-2, 68)

ax = axes[2]
pos = {0.0: 0, 1.0: 1, 10.0: 2}
for i, (arm, lab, c, mk) in enumerate(ARMS):
    for k2, x0 in pos.items():
        v = data.get((arm, 0.0, k2), [])
        if v:
            ax.plot(x0 + 0.22 * (i - 1) + np.zeros(len(v)), [x[1] for x in v], ls="", marker=mk, ms=4, color=c, alpha=0.85)
ax.axhline(1.08, color=TEXT2, lw=0.8, ls=":"); ax.axhline(31.6, color=TEXT2, lw=0.8, ls="--")
ax.annotate("incumbent", (1.0, 1.08), xycoords=("axes fraction", "data"), xytext=(-2, 2), textcoords="offset points", ha="right", fontsize=FS - 1.5, color=TEXT2)
ax.annotate("exact optimum", (0.0, 31.6), xycoords=("axes fraction", "data"), xytext=(2, -9), textcoords="offset points", ha="left", fontsize=FS - 1.5, color=TEXT2)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["0", "1", "10"]); ax.set_xlabel(r"anchor strength $\kappa_2$, no noise"); ax.set_ylabel("breakdown probability (%)"); ax.set_ylim(-2, 68)
for a_, s in zip(axes, "abc"):
    panel_label(a_, s, x=-0.16)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(w_pad=1.0)
fig.savefig(HERE / "wear-three-optimizers-v76.pdf"); fig.savefig(HERE / "wear-three-optimizers-v76.png")
for key in sorted(data):
    v = np.array(data[key]); print(key, len(v), "spread %.4f" % v[:, 0].mean(), "q %.1f [%.1f, %.1f]" % (v[:, 1].mean(), v[:, 1].min(), v[:, 1].max()), "U %.2f" % v[:, 2].mean())
