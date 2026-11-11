"""Revision 76: the geometric risk and the lower tail of the failure time (Remark 2 of the paper), drawn on the
wear-allowance example for constant operating rates (for Appendix D.2 of the Online Resource; the visuals of the
verification companion, redrawn). 200,000 simulated paths per rate.

T_eps is the mean failure time over the earliest eps-fraction of the paths. Remark 2: T_eps >= log(eps/q_beta)/theta
with theta = log(1/beta).
(a) The law of the failure time at the rate 0.15, with T_0.1 and its bound.
(b) T_eps against eps for three rates, with the bounds.
(c) T_0.05 against the geometric risk for nineteen rates, with the bound from the geometric risk and the bound
    that the failure probability by the horizon T = 10 gives, which never exceeds T + 1.
Output: failure-time-tail-v76 (pdf/png) and failure-time-tail-v76.json. A few minutes.
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

BETA, N, TMAX = 0.95, 200_000, 3000
THETA = np.log(1 / BETA)
RATES, EPS_C, T_C = (0.1, 0.15, 0.2), 0.05, 10
# The legends print equation numbers of the paper (M-) and of its Online Resource (O-). They are recorded in the
# JSON output under "cited", and the manuscript's build compares them with the numbers of the typeset documents.
CITED = {"O-eq:failure-time-bound": "E.1", "O-eq:failure-time-horizon": "E.2"}
N_B, N_H = (CITED[k] for k in CITED)


def failure_times(a, seed):
    """Failure times of N paths of the wear-allowance example at the constant rate a (TMAX + 1 when none)."""
    rng = np.random.default_rng(seed)
    z = np.ones(N); alive = np.ones(N, bool); tau = np.full(N, TMAX + 1)
    for t in range(TMAX):
        xi = rng.gamma(0.5, 2.0, N); rng.uniform(-1, 1, N)          # the price index draw keeps the stream of the companion
        zn = z - 0.1 * a * xi
        fail = alive & (zn < 0)
        tau[fail] = t; alive &= ~fail
        z = np.where(alive, zn, z)
        if not alive.any():
            break
    return tau


def q_beta(tau):
    return float(np.where(tau <= TMAX, BETA ** np.minimum(tau, TMAX), 0.0).mean())


def tail_mean(tau, eps):
    """Mean of the earliest eps-fraction of the failure times."""
    ts = np.sort(tau.astype(float)); n = len(ts); k = eps * n; j = int(np.floor(k))
    return float((ts[:j].sum() + (k - j) * (ts[j] if j < n else ts[-1])) / k)


tau = {a: failure_times(a, 2026) for a in RATES}
fam = np.round(np.geomspace(0.04, 1.0, 19), 3)
tfam = {a: failure_times(a, 2027) for a in fam}
epsg = np.geomspace(1e-3, 1, 40)
Teps = {a: np.array([tail_mean(tau[a], e) for e in epsg]) for a in RATES}
q = {a: q_beta(tau[a]) for a in RATES}
assert all((Teps[a] >= np.log(epsg / q[a]) / THETA - 1e-9).all() for a in RATES)
Tc = np.array([tail_mean(tfam[a], EPS_C) for a in fam]); pT = np.array([(tfam[a] <= T_C).mean() for a in fam]); qf = np.array([q_beta(tfam[a]) for a in fam])
from_pT = (T_C + 1) * np.clip(1 - pT / EPS_C, 0, None)

C3 = (PAL[2], PAL[0], PAL[3]); LS3 = ("-", "--", "-.")
fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.05), gridspec_kw=dict(wspace=0.62))
ax = axes[0]
a0 = 0.15
pm = np.bincount(np.minimum(tau[a0], 260), minlength=261)[:260] / N
cut = np.quantile(tau[a0], 0.1); ts = np.arange(260)
ax.bar(ts, pm, width=1.0, color=np.where(ts < cut, PAL[7], MUTED), lw=0)
t01 = tail_mean(tau[a0], 0.1); b01 = np.log(0.1 / q[a0]) / THETA
ax.axvline(t01, color=PAL[7], lw=1.2); ax.axvline(b01, color="k", lw=1.0, ls="--")
ax.set_xlim(0, 200); ax.set_xlabel("failure time $\\tau$"); ax.set_ylabel("probability"); panel_label(ax, "a")
ax = axes[1]
for a, c, ls in zip(RATES, C3, LS3):
    ax.plot(epsg, Teps[a], color=c, lw=1.3, ls=ls)
    bd = np.log(epsg / q[a]) / THETA
    ax.plot(epsg[bd > 0], bd[bd > 0], color=c, lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xlabel("fraction $\\varepsilon$"); ax.set_ylabel("$T_\\varepsilon$ (periods)"); panel_label(ax, "b")
ax = axes[2]
qs = np.geomspace(qf.min() * 0.7, 1, 200)
ax.plot(qs, np.clip(np.log(EPS_C / qs) / THETA, 0, None), color="k", lw=1.0, ls="--")
ax.plot(qf, Tc, "o", ms=3.2, color=PAL[0]); ax.plot(qf, from_pT, "s", ms=3, mfc="none", color=PAL[3])
ax.set_xscale("log"); ax.set_xlabel("geometric risk $q_\\beta$"); ax.set_ylabel(f"$T_{{{EPS_C}}}$ (periods)"); panel_label(ax, "c")
handles = [Line2D([], [], color=PAL[7], lw=1.2, label="$T_{0.1}$, mean of the earliest tenth"), Line2D([], [], color="k", lw=1.0, ls="--", label=f"right side of ({N_B})")]
handles += [Line2D([], [], color=c, lw=1.3, ls=ls, label=f"rate $a={a:g}$") for a, c, ls in zip(RATES, C3, LS3)]
handles += [Line2D([], [], color=PAL[0], marker="o", ms=3.2, ls="none", label=f"$T_{{{EPS_C}}}$ of a constant rate"),
            Line2D([], [], color=PAL[3], marker="s", mfc="none", ms=3, ls="none", label=f"right side of ({N_H}), $T={T_C}$")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False, fontsize=FS, columnspacing=1.2, handletextpad=0.5)
out = dict(cited=CITED, paths=N, seeds=[2026, 2027], beta=BETA, q_beta={f"{a:g}": q[a] for a in RATES}, T_0_1={f"{a:g}": tail_mean(tau[a], 0.1) for a in RATES},
           bound_0_1={f"{a:g}": float(np.log(0.1 / q[a]) / THETA) for a in RATES}, fraction_of_panel_c=EPS_C, horizon_of_panel_c=T_C,
           family={f"{a:g}": dict(q_beta=float(x), T=float(t), p_T=float(p_), bound_from_p_T=float(g)) for a, x, t, p_, g in zip(fam, qf, Tc, pT, from_pT)})
(ROOT / "failure-time-tail-v76.json").write_text(json.dumps(out, indent=1) + "\n")
save(fig, "failure-time-tail-v76", ROOT)
print(json.dumps({k: out[k] for k in ("q_beta", "T_0_1", "bound_0_1")}, indent=1))
