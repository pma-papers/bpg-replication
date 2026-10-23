"""Revision 60: the value-based validation certificate (eqs. lowerV and risk-horizon-bound) in the wear example.

Panel (a): the certified upper bound on p_T against the horizon T for the fixed-price PPO policies of the
wear-allowance campaign (wear-allowance/campaign/fixed-m{1,3,10}, eight seeds, stochastic evaluation), with
the measured clipped value V_m = U - m q_beta/(1-beta) (no barrier, so the interior loss is zero) in place
of the mean truncated return, at N = 20,000 validation episodes, H = 200, zeta = 0.05 (solid), and in the
population limit N -> infinity (dotted); the dashed lines are the re-evaluated q_beta of the exact dynamic-program policy at
the same prices, which p_T cannot exceed. Panel (b): the resolution floor (eq. validation-resolution-floor),
the smallest tolerance the value-based test can certify, against T for N in {1e3, 1e4, 1e5} with
H = T + 100, and the zero-failure endpoint 1 - zeta^(1/N) of the direct binomial test (horizontal).
Outputs: validation-bound-v60.pdf/png and validation-bound-v60.json.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plotstyle import PAL, TEXT2, MUTED, TEXTW, panel_label, save  # noqa: E402

BETA, UBAR, ZETA, H, N_VAL = 0.95, 1.0, 0.05, 200, 20000
PRICES = [1, 3, 10]
BENCH = json.loads((HERE.parent / "wear-allowance" / "benchmark.json").read_text())


def bench_q(m):
    for r in BENCH["rows"]:
        if abs(r["m"] - m) < 1e-12:
            return r["nominal"]["risk"]   # re-evaluated breakdown probability of the exact policy
    raise KeyError(m)


def certificate(V_hat, m, T, N, H=H, zeta=ZETA):
    hoeff = 0.0 if N is None else (m + UBAR) * (1 - BETA ** H) / (1 - BETA) * np.sqrt(np.log(1 / zeta) / (2 * N))
    V_low = V_hat - hoeff - m * BETA ** H / (1 - BETA)
    return np.minimum(1.0, (UBAR - (1 - BETA) * V_low) / ((m + UBAR) * BETA ** T))


def floor(T, N, H, zeta=ZETA):
    return BETA ** (H - T) + (1 - BETA ** H) / BETA ** T * np.sqrt(np.log(1 / zeta) / (2 * N))


def main():
    T = np.arange(0, 81)
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.5), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]; out = {}
    for i, m in enumerate(PRICES):
        rows = [json.load(open(f)) for f in sorted(glob.glob(str(HERE.parent / "wear-allowance" / "campaign" / f"fixed-m{m}__seed*/result.json")))]
        V = np.array([r["stochastic"]["U"] - m * r["stochastic"]["risk"] / (1 - BETA) for r in rows])
        q = np.array([r["stochastic"]["risk"] for r in rows])
        out[f"m{m}"] = dict(n=len(rows), V_m=float(V.mean()), q_beta=float(q.mean()), exact_q=bench_q(m),
                            bound_T0_N=float(certificate(V.mean(), m, 0, N_VAL)), bound_T0_pop=float(certificate(V.mean(), m, 0, None)))
        ax.plot(T, certificate(V.mean(), m, T, N_VAL), color=PAL[i], lw=1.0, label=f"$m={m}$")
        ax.plot(T, certificate(V.mean(), m, T, None), color=PAL[i], lw=0.8, ls=":")
        ax.axhline(bench_q(m), color=PAL[i], lw=0.7, ls="--")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.2); ax.set_xlim(0, 80)
    ax.set_xlabel("validation horizon $T$ (periods)"); ax.set_ylabel("certified bound on $p_T$")
    ax.legend(frameon=False, fontsize=6, loc="lower right"); panel_label(ax, "a")
    ax = axes[1]
    for i, N in enumerate([1e3, 1e4, 1e5]):
        ax.plot(T, floor(T, N, T + 100), color=PAL[i], lw=1.0, label=f"$N=10^{int(np.log10(N))}$")
        ax.axhline(1 - ZETA ** (1 / N), color=PAL[i], lw=0.7, ls="--")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1.2); ax.set_xlim(0, 80)
    ax.set_xlabel("validation horizon $T$ (periods)"); ax.set_ylabel("smallest certifiable tolerance")
    ax.legend(frameon=False, fontsize=6, loc="lower right"); panel_label(ax, "b")
    save(fig, "validation-bound-v60", HERE)
    (HERE / "validation-bound-v60.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
