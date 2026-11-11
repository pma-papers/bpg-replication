"""Revision 76: the wear-allowance problem in one figure (for the Online Resource; no learned policy, no new runs).

(a) Twelve episodes under two exact policies with the same wear draws, price paths and access times: the exact
    optimum without a price and the exact optimum at the price m = 1 (failure-only). An episode ends with a
    breakdown, when the allowance is exhausted, or with maintenance access, which arrives with probability
    1 - beta in each period.
(b) The exact optimum at m = 1 as a map: operating rate against the price index and the remaining allowance
    (darker is a higher rate; contours at 0.25, 0.5 and 0.75).
(c) The wear of one period at full rate: the probability that it exceeds a given part of the allowance.
Reads benchmark.json (the exact dynamic program of the study). Output: wear-problem-v76 (pdf/png) and
wear-problem-v76.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gamma as gamma_dist

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
sys.path.insert(0, str(ROOT))
from plotstyle import PAL, SEQ, TEXT2, MUTED, TEXTW, FS, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from model import Model, wear  # noqa: E402

bench = json.loads((ROOT / "benchmark.json").read_text())
model = Model(**bench["model"])
s_grid, z_grid = np.array(bench["s_grid"]), np.array(bench["z_grid"])
table = np.array(bench["policy_tables"]["failure-m1"])            # (ns, nz): rate at (price index, allowance)
table0 = np.array(bench["policy_tables"]["failure-m0"])           # the exact optimum without a price
N_EP, SEED, T_MAX = 12, 20261111, 80


def rate(s, z, tab):
    i = np.interp(s, s_grid, np.arange(len(s_grid))); lo = int(min(np.floor(i), len(s_grid) - 2)); f = i - lo
    return float((1 - f) * np.interp(z, z_grid, tab[lo]) + f * np.interp(z, z_grid, tab[lo + 1]))


rng = np.random.default_rng(SEED)
xi = rng.gamma(model.wear_shape, 1.0 / model.wear_shape, (N_EP, T_MAX))
w = rng.uniform(-1, 1, (N_EP, T_MAX))
access = rng.geometric(1 - model.beta, N_EP)                      # the period after which access arrives
episodes = {"no price": [], "price": []}
for name in episodes:
    for e in range(N_EP):
        s, z, path, end = 0.0, model.budget, [model.budget], "access"
        for t in range(min(int(access[e]), T_MAX)):
            a = rate(s, z, table0 if name == "no price" else table)
            z = z - wear(model, a, xi[e, t])
            if z < 0:
                path.append(0.0); end = "breakdown"; break
            path.append(z); s = model.rho * s + (1 - model.rho) * w[e, t]
        episodes[name].append(dict(path=path, end=end))

fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.05), gridspec_kw=dict(wspace=0.72, width_ratios=[1.1, 1.0, 0.95]))
C0, C1 = MUTED, PAL[0]
# (a) paired episodes
ax = axes[0]
for name, col, z_ in (("no price", C0, 2), ("price", C1, 3)):
    for ep in episodes[name]:
        y = np.array(ep["path"]); t = np.arange(len(y))
        ax.plot(t, y, color=col, lw=0.8, alpha=0.9, zorder=z_)
        ax.plot(t[-1], y[-1], marker="x" if ep["end"] == "breakdown" else "o", color=col, ms=4.2, mew=1.0, mfc="white", ls="none", zorder=z_ + 2, clip_on=False)
ax.set_xlim(0, 30); ax.set_ylim(0, 1.02); ax.set_xlabel("period"); ax.set_ylabel("remaining allowance $z$"); panel_label(ax, "a")
# (b) the exact policy at m = 1
ax = axes[1]
im = ax.pcolormesh(s_grid, z_grid, table.T, cmap="Blues", vmin=0, vmax=1, shading="auto", rasterized=True)
cs = ax.contour(s_grid, z_grid, table.T, levels=[0.25, 0.5, 0.75], colors="k", linewidths=0.6)
ax.clabel(cs, fmt="%.2f", fontsize=FS - 2)
ax.set_xlabel("price index $s$"); ax.set_ylabel("remaining allowance $z$"); ax.grid(False); panel_label(ax, "b")
# (c) the wear of one period at full rate
ax = axes[2]
z = np.linspace(0, 1, 401)
dist = gamma_dist(model.wear_shape, scale=1.0 / model.wear_shape)
ax.plot(z, dist.sf(z / model.wear_rate), color=TEXT2, lw=1.2)
ax.set_yscale("log"); ax.set_ylim(1e-3, 1.2); ax.set_xlim(0, 1)
ax.set_xlabel("part $z$ of the allowance"); ax.set_ylabel("probability of wear above $z$"); panel_label(ax, "c")
handles = [Line2D([], [], color=C0, lw=1.2, label="exact optimum, no price"), Line2D([], [], color=C1, lw=1.2, label="exact optimum, $m=1$"),
           Line2D([], [], color=TEXT2, marker="x", ls="none", ms=4.5, mew=1.0, label="breakdown"),
           Line2D([], [], color=TEXT2, marker="o", mfc="white", ls="none", ms=4.5, label="maintenance access")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False, fontsize=FS, columnspacing=1.2, handletextpad=0.5)
out = dict(seed=SEED, episodes=N_EP, model=bench["model"],
           ends={k: {e: sum(ep["end"] == e for ep in v) for e in ("breakdown", "access")} for k, v in episodes.items()},
           wear_exceeds={f"{q:g}": float(dist.sf(q / model.wear_rate)) for q in (0.1, 0.25, 0.5, 1.0)},
           mean_wear_per_period_at_full_rate=model.wear_rate,
           exact={r["kind"] + "-m" + f"{r['m']:g}": r["grid"] for r in bench["rows"] if r["kind"] == "failure" and r["m"] in (0, 1)})
(ROOT / "wear-problem-v76.json").write_text(json.dumps(out, indent=1) + "\n")
save(fig, "wear-problem-v76", ROOT)
print(json.dumps({k: out[k] for k in ("ends", "wear_exceeds", "exact")}, indent=1))
