"""Revision 76: the price, the time after failure, the overshoot and the CVaR of the consumption in one figure
(for Appendix D.2 of the Online Resource; the visuals of the verification companion, redrawn).

A task of H = 20 actions without discounting; the consumption of an action is uniform on [c_-, c_+] = [0.01, 0.09]
and the budget is b = 1; 100,000 simulated paths.
(a) One path that fails: the reserve after each action, the N_H periods charged after failure and the overshoot
    O_H at the horizon.
(b) The overshoot against the time after failure on 3,000 failed paths, between c_-(N_H - 1) and c_+ N_H.
(c) The total consumption: the budget is its (1 - p)-quantile for the failure probability p, and the mean of the
    tail beyond it is the average value-at-risk at the level p.
(d) The function r + E(c - r)^+ / p, whose minimum, at r = b, is that average value-at-risk.
Output: cvar-consumption-v76 (pdf/png) and cvar-consumption-v76.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CM, CP, H, N, B, SEED = 0.01, 0.09, 20, 100_000, 1.0, 9
rng = np.random.default_rng(SEED)
cons = rng.uniform(CM, CP, (N, H))
cum = np.cumsum(cons, axis=1)                                 # consumption through action t
total = cum[:, -1]
failed = total > B
tau = np.where(failed, np.argmax(cum > B, axis=1), H)
NH = np.where(failed, H - tau, 0)                              # periods from failure to the horizon
OH = np.maximum(total - B, 0)                                  # overshoot at the horizon
assert (OH[failed] <= CP * NH[failed] + 1e-12).all() and (OH[failed] >= CM * (NH[failed] - 1) - 1e-12).all()
p = float(failed.mean())
avar = B + OH.mean() / p
rgrid = np.linspace(0.75, 1.25, 201)
phi = np.array([r + np.maximum(total - r, 0).mean() / p for r in rgrid])
assert abs(rgrid[np.argmin(phi)] - B) < 0.01
i = int(np.argmax(failed & (NH == 4)))

C_F, C_O = PAL[7], PAL[3]
fig, axes = plt.subplots(2, 2, figsize=(TEXTW, 3.9), gridspec_kw=dict(wspace=0.42, hspace=0.55))
ax = axes[0, 0]
z = B - cum[i]
ax.plot(np.arange(H), z, "o-", ms=3, color=PAL[0], lw=1.0)
ax.axhline(0, color=TEXT2, lw=0.7)
ax.axvspan(tau[i] - 0.5, H - 0.5, color=C_F, alpha=0.10, lw=0)
ax.text(tau[i] + 1.5, 0.5, f"$N_H={NH[i]}$ periods\ncharged", ha="center", color=C_F, fontsize=FS - 1)
ax.annotate("", xy=(H - 0.6, 0), xytext=(H - 0.6, z[-1]), arrowprops=dict(arrowstyle="<->", color=C_O))
ax.text(H - 4.6, z[-1] / 2 - 0.03, f"$O_H={OH[i]:.3f}$", ha="right", va="center", color=C_O, fontsize=FS - 1)
ax.set_xlabel("action index $t$"); ax.set_ylabel("reserve $z_{t+1}$"); panel_label(ax, "a")
ax = axes[0, 1]
jit = rng.uniform(-0.18, 0.18, N)
sel = np.flatnonzero(failed)[:3000]
ax.plot(NH[sel] + jit[sel], OH[sel], ".", ms=1.6, color=PAL[0], alpha=0.5, rasterized=True)
nn = np.arange(0, NH.max() + 2)
ax.plot(nn, CP * nn, color=C_F, lw=1.1, label="$c_+N_H$")
ax.plot(nn, CM * np.maximum(nn - 1, 0), color=PAL[2], lw=1.1, ls="--", label="$c_-(N_H-1)$")
ax.set_xlabel("periods after failure $N_H$"); ax.set_ylabel("overshoot $O_H$"); ax.legend(loc="upper left", fontsize=FS - 1); panel_label(ax, "b")
ax = axes[1, 0]
bins = np.linspace(total.min(), total.max(), 60)
ax.hist(total[~failed], bins=bins, color=MUTED, alpha=0.6, lw=0, label=f"within the budget ({1 - p:.0%})")
ax.hist(total[failed], bins=bins, color=C_F, alpha=0.75, lw=0, label=f"failure ({p:.0%})")
ax.axvline(B, color="k", lw=1.0)
ax.axvline(avar, color=C_F, lw=1.1, ls="--", label="mean of the tail")
ax.set_ylim(0, ax.get_ylim()[1] * 1.5); ax.set_xlabel("total consumption $c_{<H}$"); ax.set_ylabel("paths"); ax.legend(loc="upper left", fontsize=FS - 1.5); panel_label(ax, "c")
ax = axes[1, 1]
ax.plot(rgrid, phi, color=PAL[0], lw=1.3)
ax.plot(B, avar, "o", ms=5, color=C_F)
ax.axvline(B, color="k", lw=0.7, ls=":")
ax.set_xlabel("threshold $r$"); ax.set_ylabel("$r+p^{-1}\\mathrm{E}(c_{<H}-r)^+$"); panel_label(ax, "d")
out = dict(c_minus=CM, c_plus=CP, H=H, paths=N, seed=SEED, budget=B, p=p, mean_time_after_failure=float(NH.mean()), mean_overshoot=float(OH.mean()),
           average_value_at_risk=float(avar), mean_consumption_of_failed_paths=float(total[failed].mean()), minimizing_threshold=float(rgrid[np.argmin(phi)]),
           path=dict(tau=int(tau[i]), N_H=int(NH[i]), O_H=float(OH[i])))
(ROOT / "cvar-consumption-v76.json").write_text(json.dumps(out, indent=1) + "\n")
save(fig, "cvar-consumption-v76", ROOT)
print(json.dumps(out, indent=1))
