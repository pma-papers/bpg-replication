"""Figure 3 of the paper (identity.pdf): one wear realization of the running wear-allowance example and the
exact return accounting V_m = U - I_m - m beta^tau / (1 - beta) on that path. Seed 7048; every path value is
written to running-example-figure-checks.json. The figure is an illustration of the accounting identity; it
reports no trained policy and no estimate of a failure probability. The panel descriptions are in the caption
(the journal allows no titles inside a figure).
Run: ~/.venvs/constrained-rl/bin/python running_example_figures.py   (writes identity.pdf/png beside this script)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BUDGET, THETA, BETA, PRICE, ACTION = 1.0, 0.1, 0.95, 0.3, 0.8
C_UTIL, C_LOSS, C_FAIL, C_PATH = PAL[2], PAL[1], PAL[7], PAL[0]

# one realization of the asset model (constant rate ACTION, gamma wear shocks of shape 1/2 and mean one)
rng = np.random.default_rng(7048)
znow, snow, records = 1.0, 0.0, []
for t in range(200):
    ut = (1 + snow / 2) * ACTION / 1.5
    shock = float(rng.gamma(0.5, 2.0)); w = float(rng.uniform(-1, 1)); cost = 0.1 * ACTION * shock
    znext = znow - cost
    records.append(dict(t=t, s=snow, action=ACTION, utility=ut, shock=shock, market_innovation=w, cost=cost,
                        reserve_before=znow, reserve_after=znext))
    if znext < 0:
        break
    snow = 0.8 * snow + 0.2 * w; znow = znext
else:
    raise RuntimeError("The illustration did not fail within the stated cap.")
tau = t
times = np.arange(tau + 1)
u = np.array([r["utility"] for r in records]); zp = np.array([r["reserve_after"] for r in records])
B = np.full(tau + 1, np.inf); B[:-1] = THETA * np.log1p(1 / zp[:-1]); loss = np.minimum(B[:-1], u[:-1] + PRICE)
U_path = float(np.dot(BETA ** times[:-1], u[:-1])); I_path = float(np.dot(BETA ** times[:-1], loss))
tail = float(PRICE * BETA ** tau / (1 - BETA)); Vm = U_path - I_path - tail
H = tau + 8; tt = np.arange(H)
utility_terms = np.zeros(H); loss_terms = np.zeros(H); fail_terms = np.zeros(H)
utility_terms[:tau] = BETA ** tt[:tau] * u[:-1]; loss_terms[:tau] = -BETA ** tt[:tau] * loss
fail_terms[tau:] = -PRICE * BETA ** tt[tau:]
exact = float(np.sum(utility_terms + loss_terms + fail_terms) - PRICE * BETA ** H / (1 - BETA))
assert abs(Vm - exact) < 1e-12

fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 2.3), gridspec_kw=dict(width_ratios=[1.0, 1.35, 1.0], wspace=0.55))
# (a) the wear path
ax = axs[0]
ax.plot(times, zp, marker="o", ms=3, color=C_PATH); ax.axhline(0, color=TEXT2, lw=0.8)
ax.axvline(tau, color=C_FAIL, ls=":", lw=1); ax.plot(tau, zp[-1], "o", color=C_FAIL, ms=6)
ax.annotate(f"breakdown\n$\\tau={tau}$", xy=(tau, zp[-1]), xytext=(0.3, 0.06), fontsize=FS, color=C_FAIL,
            arrowprops=dict(arrowstyle="->", color=C_FAIL))
ax.set_xlabel("action index $t$"); ax.set_ylabel("allowance after action, $z'$"); ax.set_ylim(-0.18, 1.06)
panel_label(ax, "(a)")
# (b) the discounted terms, one bar per action (hatching distinguishes the three terms without colour)
ax = axs[1]
ax.bar(tt, utility_terms, color=C_UTIL, label="utility before breakdown")
ax.bar(tt, loss_terms, color=C_LOSS, hatch="///", edgecolor="white", lw=0, label="interior loss")
ax.bar(tt, fail_terms, color=C_FAIL, hatch="...", edgecolor="white", lw=0, label="downtime price")
ax.axvline(tau - 0.5, color=TEXT2, ls=":", lw=1); ax.axhline(0, color=TEXT2, lw=0.8)
ax.annotate("tail continues", xy=(H - 1, fail_terms[-1]), xytext=(H - 9.5, -0.40), fontsize=FS, color=C_FAIL,
            arrowprops=dict(arrowstyle="->", color=C_FAIL))
ax.set_xlabel("action index $t$"); ax.set_ylabel("discounted term"); ax.set_ylim(-0.46, 0.62)
panel_label(ax, "(b)")
# (c) the exact totals
ax = axs[2]
vals = [U_path, -I_path, -tail, Vm]; labels = ["$U_\\omega$", "$-I_\\omega$", "tail", "$V_{m,\\omega}$"]
bars = ax.bar(range(4), vals, color=[C_UTIL, C_LOSS, C_FAIL, C_PATH], width=0.65)
bars[1].set(hatch="///", edgecolor="white", lw=0); bars[2].set(hatch="...", edgecolor="white", lw=0)
ax.axhline(0, color=TEXT2, lw=0.8)
for i, v in enumerate(vals):
    ax.text(i, v + (0.16 if v >= 0 else -0.2), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=FS)
ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_ylabel("discounted total"); ax.set_ylim(-tail - 1.2, U_path + 1.0)
panel_label(ax, "(c)")
legend_below(fig, *axs[1].get_legend_handles_labels(), ncol=3)
save(fig, "identity", ROOT)
payload = dict(model=dict(budget=BUDGET, beta=BETA, barrier_theta=THETA, m=PRICE, constant_action=ACTION, seed=7048),
               path=records, pathwise_totals=dict(tau=tau, U=U_path, I=I_path, failure_charge=tail, Vm=Vm, direct_value=exact,
                                                  identity_error=abs(Vm - exact)), illustration_only=True)
(ROOT / "running-example-figure-checks.json").write_text(json.dumps(payload, indent=2) + "\n")
print("PASS: wear-model Figure 3 identity; totals", payload["pathwise_totals"])
