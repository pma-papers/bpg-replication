"""Revision 60: the particle evolution under the three regularizer types, as pictures.

Same particle system as roles_flow_v59.py (solvable plateau flow, prior N(0,1), plateau initial
law, Euler-Maruyama, 2000 units, one seed): relative entropy (K, kappa_1 = 0.1), the quadratic potential
(P, kappa_2 = 1) and their sum (K+P, kappa_1 = kappa_2 = 1). Stored: the unit positions at the snapshot
times l (kappa_1 + kappa_2) in {0, 1, 4, 40} and the paths of eight units. Outputs:
particle-evolution-v60.json (summaries only) and particle-evolution-v60.pdf/png.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE.parent / "solvable-flow"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from density_flow import FirstActionModel, chi, chi_prime, phi_gauss  # noqa: E402
from plotstyle import PAL, TEXT2, MUTED, TEXTW, save  # noqa: E402

P = json.loads((HERE.parent / "solvable-flow" / "prior-flow.json").read_text())["params"]
model = FirstActionModel("plateau", P["eps"], P["m"], P["beta"], P["gamma"])
N, SEED, DT_FACTOR = 2000, 1, 0.02
CONFIGS = [("relative entropy, $\\kappa_1=0.1$", 0.1, 0.0), ("potential, $\\kappa_2=1$", 0.0, 1.0),
           ("sum, $\\kappa_1=\\kappa_2=1$", 1.0, 1.0)]
SNAP = [0.0, 1.0, 4.0, 40.0]      # in units of 1/(kappa_1+kappa_2)
N_PATHS = 8


def sample_plateau(n, rng):
    out = np.empty(0)
    while out.size < n:
        x = rng.standard_normal(4 * n); out = np.concatenate([out, x[np.abs(x) <= 0.5]])
    return out[:n]


def simulate(k1, k2, n_path_rec=400):
    rate = k1 + k2; l_max = SNAP[-1] / rate; dt = DT_FACTOR / rate
    snap_times = [s / rate for s in SNAP]
    path_times = np.linspace(0.0, l_max, n_path_rec)
    rng = np.random.default_rng(100 * SEED + 7)
    x = sample_plateau(N, rng); l = 0.0
    snaps, paths, stats = [], [], []
    i_s = i_p = 0
    while True:
        while i_s < len(snap_times) and l >= snap_times[i_s] - 1e-12:
            c = float(np.mean(chi(x))); st = model.expectations(c)
            snaps.append(x.copy()); stats.append(dict(l=l, V_m=st["V_m"], q=st["q"], spread=float(np.std(x)))); i_s += 1
        while i_p < len(path_times) and l >= path_times[i_p] - 1e-12:
            paths.append(x[:N_PATHS].copy()); i_p += 1
        if i_s >= len(snap_times): break
        h = min(dt, snap_times[i_s] - l, (path_times[i_p] - l) if i_p < len(path_times) else dt)
        h = max(h, 1e-12)
        c = float(np.mean(chi(x)))
        x = x + (model.dV(c) * chi_prime(x) - rate * x) * h + math.sqrt(2 * k1 * h) * rng.standard_normal(N)
        l += h
    return snaps, stats, np.array(paths), path_times[: len(paths)]


def main():
    fig, axes = plt.subplots(len(CONFIGS), len(SNAP) + 1, figsize=(TEXTW, 4.6),
                             gridspec_kw=dict(wspace=0.12, hspace=0.45, width_ratios=[1, 1, 1, 1, 1.6]))
    grid = np.linspace(-3.5, 3.5, 400); out = {}
    for row, (lab, k1, k2) in enumerate(CONFIGS):
        snaps, stats, paths, pt = simulate(k1, k2)
        out[lab] = dict(kappa1=k1, kappa2=k2, snapshots=stats)
        rate = k1 + k2
        for col, (x, st) in enumerate(zip(snaps, stats)):
            ax = axes[row, col]
            ax.axvspan(-1, 1, color=MUTED, alpha=0.12, lw=0)
            ax.plot(grid, phi_gauss(grid), color=MUTED, lw=0.7, ls=":")
            ax.hist(x, bins=np.linspace(-3.5, 3.5, 71), density=True, color=PAL[row], alpha=0.85, lw=0)
            ax.set_xlim(-3.5, 3.5); ax.set_ylim(0, 2.2); ax.set_yticks([])
            ax.set_title(f"$\\ell(\\kappa_1+\\kappa_2)={SNAP[col]:g}$", fontsize=7, pad=2)
            ax.text(0.03, 0.95, f"$V_m={st['V_m']:.3f}$\nspread ${st['spread']:.2f}$", transform=ax.transAxes,
                    fontsize=5.5, va="top", color=TEXT2)
            if col == 0:
                ax.set_ylabel(lab, fontsize=7)
            if row == len(CONFIGS) - 1:
                ax.set_xlabel("unit parameter $x$", fontsize=7)
            ax.tick_params(labelsize=6)
        ax = axes[row, -1]
        ax.axhspan(-1, 1, color=MUTED, alpha=0.12, lw=0)
        for j in range(N_PATHS):
            ax.plot(pt * rate, paths[:, j], color=PAL[row], lw=0.6, alpha=0.8)
        ax.set_ylim(-3.5, 3.5); ax.set_xlim(0, SNAP[-1]); ax.yaxis.tick_right()
        ax.set_title("paths of eight units", fontsize=7, pad=2); ax.tick_params(labelsize=6)
        if row == len(CONFIGS) - 1:
            ax.set_xlabel("$\\ell(\\kappa_1+\\kappa_2)$", fontsize=7)
    save(fig, "particle-evolution-v60", HERE)
    (HERE / "particle-evolution-v60.json").write_text(json.dumps(out, indent=1))
    for lab, v in out.items():
        print(lab, [(round(s["l"], 2), round(s["V_m"], 4), round(s["spread"], 3)) for s in v["snapshots"]])


if __name__ == "__main__":
    main()
