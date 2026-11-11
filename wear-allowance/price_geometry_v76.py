"""Revision 76: the results of Sect. 3 of the paper drawn on the wear-allowance example (for the Online Resource).

A class of 93 simple rules is evaluated exactly alike on 20,000 simulated paths with common random numbers: the
constant rates a in {0, 0.05, ..., 1}, and the threshold rules that run at the rate a_hi while the remaining
allowance is at least c and at a_lo below it, a_hi in {0.5, ..., 1}, a_lo in {0, 0.1, 0.2, 0.3}, c in
{0.15, 0.3, 0.5}. The price is failure-only, so the interior loss is zero and the clipped value of a rule is its
stopped utility less nu = m/(1 - beta) times its geometric risk. Each rule is a point (risk, utility).

(a) The price as a slope: the maximizer of the clipped value at m = 0.3 is the point that the highest line of
    slope nu touches, and no rule with at most its risk has more utility (Proposition 4(i)).
(b) The price-risk law at m = 0.3 with a comparison rule: the maximizer lies on or above the line of slope nu
    through the comparison rule and below the largest utility of the class, which bounds its risk; the second
    form uses the largest utility that any policy with that risk could earn (Proposition 4(ii)).
(c) The same two bounds and the risk of the maximizer across prices.
(d) The maximizers at three prices: the risk does not rise with the price (Proposition 4(iii)).
(e) Theorem 3 against the comparison rule: the utility that a rule within delta of the largest clipped value can
    lose against the comparison rule is at most delta plus nu times the risk it leaves unused.
Output: price-geometry-v76 (pdf/png) and price-geometry-v76.json. About twenty seconds.
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

BETA, UBAR, N, TMAX, SEED = 0.95, 1.0, 20_000, 300, 11
M, DELTA, PRICES3 = 0.3, 0.3, (0.1, 0.3, 1.0)


def run(rate):
    """Geometric risk and stopped utility of the rate rule rate(z) on N paths of the wear-allowance example."""
    rng = np.random.default_rng(SEED)
    z = np.ones(N); s = np.zeros(N); alive = np.ones(N, bool); tau = np.full(N, TMAX + 1); U = np.zeros(N)
    for t in range(TMAX):
        a = rate(z)
        u = (1 + s / 2) * a / 1.5
        xi = rng.gamma(0.5, 2.0, N); w = rng.uniform(-1, 1, N)
        zn = z - 0.1 * a * xi
        fail = alive & (zn < 0); surv = alive & ~fail
        U[surv] += BETA ** t * u[surv]
        tau[fail] = t; alive = surv
        z = np.where(alive, zn, z); s = 0.8 * s + 0.2 * w
        if not alive.any():
            break
    return float(np.where(tau <= TMAX, BETA ** np.minimum(tau, TMAX), 0.0).mean()), float(U.mean())


rules = [("constant", a, None, None) for a in np.round(np.linspace(0, 1, 21), 2)]
rules += [("threshold", ah, al, c) for ah in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0) for al in (0.0, 0.1, 0.2, 0.3) for c in (0.15, 0.3, 0.5)]
pts = [run((lambda z, ah=ah: np.full_like(z, ah)) if kind == "constant" else (lambda z, ah=ah, al=al, c=c: np.where(z >= c, ah, al))) for kind, ah, al, c in rules]
q = np.array([p[0] for p in pts]); U = np.array([p[1] for p in pts]); const = np.array([r[0] == "constant" for r in rules])
nu = lambda m: m / (1 - BETA)
best = lambda m: int(np.argmax(U - nu(m) * q))
supU = float(U.max())
cand = np.flatnonzero(q <= 0.05); i0 = int(cand[np.argmax(U[cand])])           # the comparison rule: most utility at a risk of at most 0.05
q0, U0 = q[i0], U[i0]
law = lambda m: q0 + (supU - U0) / nu(m)                                          # the price-risk law with the comparison rule
law2 = lambda m: q0 + (1 - BETA) / (m + UBAR) * (UBAR * (1 - q0) / (1 - BETA) - U0)   # its second form
mg = np.geomspace(0.03, 5, 60); qhat = np.array([q[best(m)] for m in mg])
for m in mg:
    j = best(m)
    assert q[j] <= min(law(m), law2(m)) + 1e-12 and (U[q <= q[j]] <= U[j] + 1e-12).all()
assert (np.diff(qhat) <= 1e-12).all()
bh = best(M); qh, Uh = q[bh], U[bh]
Vm = U - nu(M) * q; dopt = Vm >= Vm.max() - DELTA; gap = U0 - U
assert (gap[dopt] <= DELTA + nu(M) * (q0 - q[dopt]) + 1e-12).all()

C_HAT, C_LAW, C_LAW2 = PAL[7], PAL[0], PAL[3]
xs = np.linspace(0, 1, 50)
fig = plt.figure(figsize=(TEXTW, 4.3))
gs = fig.add_gridspec(2, 3, wspace=0.55, hspace=0.55)
ax = [fig.add_subplot(gs[0, k]) for k in range(3)] + [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]


def cloud(a):
    a.plot(q[~const], U[~const], "o", ms=2.6, color=MUTED, mfc="none", mew=0.6)
    a.plot(q[const], U[const], "o-", ms=2.6, color=TEXT2, lw=0.8)
    a.set_xlim(0, 1); a.set_ylim(0, 8); a.set_xlabel("geometric risk $q_\\beta$"); a.set_ylabel("stopped utility $U$")


# (a) the price as a slope
cloud(ax[0])
ax[0].plot(xs, Uh + nu(M) * (xs - qh), color=C_HAT, lw=1.1)
ax[0].fill_between([0, qh], 0, Uh, color=C_HAT, alpha=0.10, lw=0)
ax[0].plot(qh, Uh, "o", ms=7, color=C_HAT, mfc="none", mew=1.4); panel_label(ax[0], "a")
# (b) the price-risk law at one price
cloud(ax[1])
ax[1].plot(xs, U0 + nu(M) * (xs - q0), color=C_HAT, lw=1.1)
ax[1].axhline(supU, color=C_LAW, lw=1.1, ls="--")
ax[1].plot(xs, UBAR * (1 - xs) / (1 - BETA), color=C_LAW2, lw=1.1, ls="-.")
for b_, c_ in ((law(M), C_LAW), (law2(M), C_LAW2)):
    ax[1].plot([b_, b_], [0, U0 + nu(M) * (b_ - q0)], color=c_, lw=1.0, ls=":")
ax[1].plot(q0, U0, "s", ms=5, color="k"); ax[1].plot(qh, Uh, "o", ms=7, color=C_HAT, mfc="none", mew=1.4); panel_label(ax[1], "b")
# (c) across prices
ax[2].step(mg, qhat, where="post", color="k", lw=1.4)
ax[2].plot(mg, np.minimum(1, [law(m) for m in mg]), color=C_LAW, lw=1.1, ls="--")
ax[2].plot(mg, np.minimum(1, [law2(m) for m in mg]), color=C_LAW2, lw=1.1, ls="-.")
ax[2].axhline(q0, color=MUTED, lw=0.8, ls=":")
ax[2].set_xscale("log"); ax[2].set_ylim(0, 1.04); ax[2].set_xlabel("price $m$"); ax[2].set_ylabel("geometric risk $q_\\beta$"); panel_label(ax[2], "c")
# (d) three prices
cloud(ax[3])
for m, ls in zip(PRICES3, (":", "-", "--")):
    j = best(m)
    ax[3].plot(xs, U[j] + nu(m) * (xs - q[j]), color=C_HAT, lw=1.0, ls=ls)
    ax[3].plot(q[j], U[j], "o", ms=7, color=C_HAT, mfc="none", mew=1.4)
    xl = min(q[j] + (7.0 - U[j]) / nu(m), 0.72); yl = U[j] + nu(m) * (xl - q[j])
    ax[3].annotate(f"$m={m:g}$", (xl, yl), textcoords="offset points", xytext=(5, -9), fontsize=FS - 1.5, color=C_HAT)
panel_label(ax[3], "d")
# (e) Theorem 3 against the comparison rule
ax[4].plot(q[~dopt], gap[~dopt], "o", ms=2.6, color=MUTED, mfc="none", mew=0.6)
ax[4].plot(q[dopt], gap[dopt], "o", ms=4, color=C_HAT)
ax[4].plot(xs, DELTA + nu(M) * (q0 - xs), color="k", lw=1.1)
ax[4].axhline(0, color=MUTED, lw=0.7, ls=":"); ax[4].axvline(q0, color=MUTED, lw=0.7, ls=":")
ax[4].set_xlim(0, 1); ax[4].set_ylim(-4.5, 4.0); ax[4].set_xlabel("geometric risk $q_\\beta$ of a rule"); ax[4].set_ylabel("utility lost against\nthe comparison rule"); panel_label(ax[4], "e")
handles = [Line2D([], [], color=TEXT2, marker="o", ms=2.6, lw=0.8, label="constant rates"),
           Line2D([], [], color=MUTED, marker="o", mfc="none", ms=2.6, ls="none", label="threshold rules"),
           Line2D([], [], color=C_HAT, marker="o", mfc="none", ms=7, mew=1.4, ls="none", label="maximizer of the clipped value"),
           Line2D([], [], color="k", marker="s", ms=5, ls="none", label="comparison rule"),
           Line2D([], [], color=C_HAT, lw=1.1, label="line of slope $\\nu=m/(1-\\beta)$"),
           Line2D([], [], color=C_LAW, lw=1.1, ls="--", label="price$-$risk law"),
           Line2D([], [], color=C_LAW2, lw=1.1, ls="-.", label="its second form"),
           Line2D([], [], color=C_HAT, marker="o", ms=4, ls="none", label="rules within $\\delta$ of the maximum"),
           Line2D([], [], color="k", lw=1.1, label="bound of Theorem 3")]
box = gs[1, 2].get_position(fig)
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(box.x0 - 0.05, (box.y0 + box.y1) / 2), frameon=False, fontsize=FS - 0.5, handlelength=1.8, labelspacing=0.45)
desc = lambda i: dict(kind=rules[i][0], a_hi=float(rules[i][1]), a_lo=rules[i][2], c=rules[i][3], q=float(q[i]), U=float(U[i]))
out = dict(paths=N, seed=SEED, rules=len(rules), m=M, nu=nu(M), delta=DELTA, maximizer=desc(bh), comparison_rule=desc(i0), largest_utility=supU,
           bound_price_risk_law=float(law(M)), bound_second_form=float(law2(M)),
           maximizers={f"{m:g}": desc(best(m)) for m in PRICES3}, rules_within_delta=int(dopt.sum()),
           risk_range_within_delta=[float(q[dopt].min()), float(q[dopt].max())])
(ROOT / "price-geometry-v76.json").write_text(json.dumps(out, indent=1) + "\n")
save(fig, "price-geometry-v76", ROOT)
print(json.dumps(out, indent=1))
