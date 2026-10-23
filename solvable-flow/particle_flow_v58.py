"""Revision 58: the particle (finite-network) form of the population flow on the solvable flow.

The population flow of the paper, for relative entropy to the Gaussian prior N(x0, sigma^2), has the
Lagrangian (particle) form, for n units x_1..x_n with empirical law mu_n and tilt c_n = mean chi(x_i),
    dx_i = [ V_m'(c_n) chi'(x_i) - kappa (x_i - x0) / sigma^2 ] dl + sqrt(2 kappa) dB_i ,
because in this example the clipped value depends on the law only through c = int chi dmu, so the
parameter gradient of its first variation is V_m'(c) chi'(x). The empirical law of this mean-field
Langevin system should follow the finite-volume solution of the flow (solvable-flow/prior-flow.json)
as n grows. This script simulates the system by Euler-Maruyama from the plateau initial law for
n in N_LIST and SEEDS seeds, and compares V_m(c_n(l)), c_n(l) and q_beta(c_n(l)) with the stored
finite-volume records of the same runs. Outputs: particle-flow-v58.json, particle-flow-v58.pdf/png.
Runtime: ~/.venvs/constrained-rl/bin/python particle_flow_v58.py
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
from matplotlib.lines import Line2D  # noqa: E402
from density_flow import FirstActionModel, chi, chi_prime  # noqa: E402
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, LS, panel_label, legend_below, save  # noqa: E402

F = json.loads((HERE.parent / "solvable-flow" / "prior-flow.json").read_text())
P = F["params"]
RUNS = ["w1/k10", "w1/k1"]          # prior N(0,1), kappa = 10 and 1
N_LIST = [100, 1000, 10000]
SEEDS = [1, 2, 3, 4, 5]
DT_FACTOR = 0.02                    # dt = DT_FACTOR / (kappa lambda_W), the stiffest rate
model = FirstActionModel("plateau", P["eps"], P["m"], P["beta"], P["gamma"])


def sample_plateau(n, rng):
    """The plateau initial law: N(0,1) conditioned on [-1/2, 1/2]."""
    out = np.empty(0)
    while out.size < n:
        x = rng.standard_normal(4 * n)
        out = np.concatenate([out, x[np.abs(x) <= 0.5]])
    return out[:n]


def simulate(run, n, seed):
    r = F["runs"][run]
    kappa, x0, sigma, l_max = r["kappa"], r["x0"], r["sigma"], r["l_max"]
    rec_times = np.array([q["l"] for q in r["records"]])
    rng = np.random.default_rng(1000 * seed + n)
    x = sample_plateau(n, rng)
    dt = DT_FACTOR / (kappa / sigma ** 2)
    out = []; l = 0.0; next_i = 0
    while True:
        while next_i < len(rec_times) and l >= rec_times[next_i] - 1e-12:
            c = float(np.mean(chi(x))); st = model.expectations(c)
            out.append(dict(l=l, c=c, V_m=st["V_m"], q=st["q"], U=st["U"]))
            next_i += 1
        if l >= l_max - 1e-12 or next_i >= len(rec_times):
            break
        h = min(dt, rec_times[next_i] - l)
        c = float(np.mean(chi(x)))
        drift = model.dV(c) * chi_prime(x) - kappa * (x - x0) / sigma ** 2
        x = x + drift * h + math.sqrt(2 * kappa * h) * rng.standard_normal(n)
        l += h
    return out


def main():
    results = {}
    for run in RUNS:
        r = F["runs"][run]
        pde = {k: np.array([q[k] for q in r["records"]]) for k in ("l", "V_m", "q")}
        results[run] = dict(kappa=r["kappa"], x0=r["x0"], sigma=r["sigma"], l_max=r["l_max"], pde=pde, particles={})
        for n in N_LIST:
            series = [simulate(run, n, s) for s in SEEDS]
            L = series[0]
            arr = {k: np.array([[q[k] for q in sr] for sr in series]) for k in ("V_m", "c", "q")}
            results[run]["particles"][n] = dict(l=np.array([q["l"] for q in L]), **arr)
            err = np.abs(arr["V_m"].mean(axis=0) - pde["V_m"][: arr["V_m"].shape[1]])
            print(f"{run:8s} n={n:6d}: final V_m particles {arr['V_m'][:, -1].mean():.4f} +- {arr['V_m'][:, -1].std(ddof=1):.4f}"
                  f"  PDE {pde['V_m'][-1]:.4f}; max |mean - PDE| over time {err.max():.4f}; final q particles {100*arr['q'][:, -1].mean():.3f}% PDE {100*pde['q'][-1]:.3f}%")
    # figure: V_m against learning time, particles (mean over seeds, band = seed range) against the finite-volume solution
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.3), gridspec_kw=dict(wspace=0.4))
    for ax, run, lab in zip(axes, RUNS, ("a", "b")):
        R = results[run]; pde = R["pde"]
        ax.plot(pde["l"], pde["V_m"], color=TEXT2, lw=1.4, label="population flow (finite volume)", zorder=5)
        for k, n in enumerate(N_LIST):
            Pn = R["particles"][n]
            m = Pn["V_m"].mean(axis=0); lo = Pn["V_m"].min(axis=0); hi = Pn["V_m"].max(axis=0)
            ax.plot(Pn["l"], m, color=PAL[k], lw=1.0, ls=LS[k + 1], label=f"$n={n}$ units")
            ax.fill_between(Pn["l"], lo, hi, color=PAL[k], alpha=0.18, lw=0)
        ax.set_xscale("log"); ax.set_xlabel("learning time $\\ell$"); ax.set_ylabel("clipped value $V_m$")
        panel_label(ax, lab)
    legend_below(fig, *axes[0].get_legend_handles_labels(), ncol=4)
    save(fig, "particle-flow-v58", HERE)
    summary = {run: dict(kappa=R["kappa"], sigma=R["sigma"], x0=R["x0"], l_max=R["l_max"], dt=DT_FACTOR / (R["kappa"] / R["sigma"] ** 2),
                         pde_final=dict(V_m=float(R["pde"]["V_m"][-1]), q=float(R["pde"]["q"][-1])),
                         particles={str(n): dict(final_V_m_mean=float(R["particles"][n]["V_m"][:, -1].mean()), final_V_m_sd=float(R["particles"][n]["V_m"][:, -1].std(ddof=1)),
                                                 final_q_mean=float(R["particles"][n]["q"][:, -1].mean()),
                                                 max_abs_dev_V_m=float(np.abs(R["particles"][n]["V_m"].mean(axis=0) - R["pde"]["V_m"][: R["particles"][n]["V_m"].shape[1]]).max()),
                                                 max_abs_dev_q=float(np.abs(R["particles"][n]["q"].mean(axis=0) - R["pde"]["q"][: R["particles"][n]["q"].shape[1]]).max()))
                                    for n in N_LIST})
               for run, R in results.items()}
    (HERE / "particle-flow-v58.json").write_text(json.dumps(dict(params=P, seeds=SEEDS, n_list=N_LIST, dt_factor=DT_FACTOR, runs=summary), indent=1))


if __name__ == "__main__":
    main()
