"""Revision 59: the roles of regularizers on the solvable flow, by the particle form.

Three regularizers of the unit law, all admissible, on the solvable flow (plateau example, prior N(0,1)):
  (K) relative entropy to the prior with strength kappa_1: the case of Leahy et al. (2022);
      particle form: pull -kappa_1 x plus noise of variance 2 kappa_1 (noise and pull tied by kappa_1);
  (P) the quadratic potential int x^2/2 dmu with strength kappa_2: deterministic pull -kappa_2 x, no noise;
  (K+P) their sum: pull -(kappa_1 + kappa_2) x and noise 2 kappa_1, chosen separately.
From the plateau initial law (units in [-1/2, 1/2], where the feature chi and its gradient vanish), the
value gradient is zero, so a regularizer without noise cannot move the law off the plateau: (P) collapses
the units to the prior's center and the clipped value never improves, whereas any noise lets units reach
|x| > 1 where the feature acts, and the value rises toward the minimizer of the regularized objective.
Runs: (K) kappa_1 in {0.1, 1}; (P) kappa_2 in {1, 10}; (K+P) kappa_1 = 0.1 with kappa_2 in {1, 10}; and a
sweep of the noise strength kappa_1 in {0.01, 0.03, 0.1, 0.3, 1} at the fixed pull kappa_2 = 1.
n = 2000 units, five seeds, Euler-Maruyama with dt = 0.02 / (kappa_1 + kappa_2), horizon l_max = 40 / (kappa_1 + kappa_2)
(the horizon rule of the finite-volume runs). Reported: clipped value V_m and the spread (standard deviation)
of the units over time, and their final values. Outputs: roles-flow-v59.json, roles-flow-v59.pdf/png.
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
from density_flow import FirstActionModel, chi, chi_prime  # noqa: E402
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, legend_below, save  # noqa: E402

P = json.loads((HERE.parent / "solvable-flow" / "prior-flow.json").read_text())["params"]
model = FirstActionModel("plateau", P["eps"], P["m"], P["beta"], P["gamma"])
N, SEEDS, DT_FACTOR = 2000, [1, 2, 3, 4, 5], 0.02
CONFIGS = [("K", 0.1, 0.0), ("K", 1.0, 0.0), ("P", 0.0, 1.0), ("P", 0.0, 10.0), ("K+P", 0.1, 1.0), ("K+P", 0.1, 10.0)]
SWEEP = [0.01, 0.03, 0.1, 0.3, 1.0]        # noise strengths at the fixed pull kappa_2 = 1


def sample_plateau(n, rng):
    out = np.empty(0)
    while out.size < n:
        x = rng.standard_normal(4 * n); out = np.concatenate([out, x[np.abs(x) <= 0.5]])
    return out[:n]


def simulate(k1, k2, seed, n=N, n_rec=120):
    rate = k1 + k2; l_max = 40.0 / rate; dt = DT_FACTOR / rate
    rec_times = np.unique(np.concatenate([[0.0], np.geomspace(l_max * 1e-4, l_max, n_rec)]))
    rng = np.random.default_rng(100 * seed + 7)
    x = sample_plateau(n, rng); l = 0.0; i = 0; out = []
    while True:
        while i < len(rec_times) and l >= rec_times[i] - 1e-12:
            c = float(np.mean(chi(x))); st = model.expectations(c)
            out.append(dict(l=l, V_m=st["V_m"], q=st["q"], spread=float(np.std(x)), c=c)); i += 1
        if i >= len(rec_times): break
        h = min(dt, rec_times[i] - l)
        c = float(np.mean(chi(x)))
        x = x + (model.dV(c) * chi_prime(x) - rate * x) * h + math.sqrt(2 * k1 * h) * rng.standard_normal(n)
        l += h
    return out


def stack(runs, key):
    return np.array([[r[key] for r in run] for run in runs])


def main():
    results = {}
    for name, k1, k2 in CONFIGS:
        runs = [simulate(k1, k2, s) for s in SEEDS]
        results[(name, k1, k2)] = runs
        V = stack(runs, "V_m"); S_ = stack(runs, "spread")
        print(f"{name:4s} kappa1={k1:5g} kappa2={k2:5g}: final V_m {V[:, -1].mean():.4f}+-{V[:, -1].std(ddof=1):.4f}  spread {S_[:, -1].mean():.4f}  q {100*stack(runs,'q')[:, -1].mean():.3f}%")
    sweep = {}
    for k1 in SWEEP:
        runs = [simulate(k1, 1.0, s) for s in SEEDS]
        sweep[k1] = dict(V=stack(runs, "V_m")[:, -1], spread=stack(runs, "spread")[:, -1])
        print(f"sweep kappa1={k1:5g} kappa2=1: final V_m {sweep[k1]['V'].mean():.4f}+-{sweep[k1]['V'].std(ddof=1):.4f} spread {sweep[k1]['spread'].mean():.4f}")
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.2), gridspec_kw=dict(wspace=0.75))
    style = {("K", 0.1, 0.0): (PAL[0], "-", "relative entropy, $\\kappa_1=0.1$"), ("K", 1.0, 0.0): (PAL[0], "--", "relative entropy, $\\kappa_1=1$"),
             ("P", 0.0, 1.0): (PAL[1], "-", "potential, $\\kappa_2=1$"), ("P", 0.0, 10.0): (PAL[1], "--", "potential, $\\kappa_2=10$"),
             ("K+P", 0.1, 1.0): (PAL[2], "-", "sum, $\\kappa_1=0.1$, $\\kappa_2=1$"), ("K+P", 0.1, 10.0): (PAL[2], "--", "sum, $\\kappa_1=0.1$, $\\kappa_2=10$")}
    for key, runs in results.items():
        col, ls, lab = style[key]; l = np.array([r["l"] for r in runs[0]])
        axes[0].plot(l, stack(runs, "V_m").mean(axis=0), color=col, ls=ls, lw=1.0, label=lab)
        axes[1].plot(l, np.maximum(stack(runs, "spread").mean(axis=0), 1e-3), color=col, ls=ls, lw=1.0)
    axes[0].set_xscale("log"); axes[0].set_xlabel("learning time $\\ell$"); axes[0].set_ylabel("clipped value $V_m$"); panel_label(axes[0], "a")
    legend_below(fig, *axes[0].get_legend_handles_labels(), ncol=2, axes=axes[:2])
    axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_ylim(8e-4, 4.0); axes[1].set_xlabel("learning time $\\ell$"); axes[1].set_ylabel("spread of the units"); panel_label(axes[1], "b")
    ks = np.array(SWEEP); Vm = np.array([sweep[k]["V"].mean() for k in SWEEP]); Vs = np.array([sweep[k]["V"].std(ddof=1) for k in SWEEP])
    sp = np.array([sweep[k]["spread"].mean() for k in SWEEP])
    axes[2].errorbar(ks, Vm, yerr=Vs, color=PAL[2], marker="o", ms=3.5, lw=1.0, capsize=0, label="final $V_m$")
    axes[2].set_xscale("log"); axes[2].set_xlabel("noise strength $\\kappa_1$"); axes[2].set_ylabel("final clipped value $V_m$")
    ax2 = axes[2].twinx(); ax2.plot(ks, sp, color=MUTED, marker="s", ms=3, lw=0.8, ls=":", label="final spread"); ax2.set_ylabel("final spread", color=MUTED)
    ax2.tick_params(axis="y", colors=MUTED)
    panel_label(axes[2], "c")
    h1, l1 = axes[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    legend_below(fig, h1 + h2, l1 + l2, ncol=1, axes=[axes[2]])
    save(fig, "roles-flow-v59", HERE)
    out = dict(params=P, n=N, seeds=SEEDS, dt_factor=DT_FACTOR,
               configs=[dict(name=k[0], kappa1=k[1], kappa2=k[2], final_V_m_mean=float(stack(r, "V_m")[:, -1].mean()), final_V_m_sd=float(stack(r, "V_m")[:, -1].std(ddof=1)),
                             final_spread_mean=float(stack(r, "spread")[:, -1].mean()), final_q_mean=float(stack(r, "q")[:, -1].mean()), l_max=float(runs[0][-1]["l"]) if False else 40.0 / (k[1] + k[2]))
                        for k, r in results.items()],
               sweep=[dict(kappa1=k, kappa2=1.0, final_V_m_mean=float(sweep[k]["V"].mean()), final_V_m_sd=float(sweep[k]["V"].std(ddof=1)), final_spread_mean=float(sweep[k]["spread"].mean())) for k in SWEEP],
               plateau_V_m=float(model.expectations(0.0)["V_m"]))
    (HERE / "roles-flow-v59.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
