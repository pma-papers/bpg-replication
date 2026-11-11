"""Revision 76: the price as a shadow price on an exact frontier (for Appendix D.2 of the Online Resource; the
visual of the verification companion, redrawn).

The exact budget model: two exogenous states, calm and stressed, which switch with probability 0.2 in each period,
starting calm; the reserve takes the values 2.5, 1.5 and 0.5, starting at 2.5; a safe action earns 0.1 and consumes
nothing, a risky action earns 1 and consumes one unit with probability 0.1 when calm and 0.3 when stressed; failure
when the reserve turns negative; beta = 0.95 and a failure-only price, so the interior loss is zero and all prices
share one frontier. The class is the 64 deterministic policies (risky or safe in each of the six live states), and
every value is exact (a linear system per policy).
Shown: the policies as points (geometric risk, stopped utility), the frontier, its closed concave hull, the
maximizers of the clipped value at three prices with their lines of slope nu = m/(1 - beta), and a frontier point
that no price supports.
Output: shadow-price-frontier-v76 (pdf/png) and shadow-price-frontier-v76.json.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BETA = 0.95
PS = np.array([[0.8, 0.2], [0.2, 0.8]]); PRISK = np.array([0.1, 0.3]); ZS = [2.5, 1.5, 0.5]; UA = np.array([0.1, 1.0])
LIVE = [(s, j) for s in range(2) for j in range(3)]              # live state index 3 s + j with the reserve ZS[j]


def evaluate(r):
    """Exact stopped utility and geometric risk of the policy that takes the risky action with probability r[i]."""
    n = len(LIVE); P = np.zeros((n, n)); f = np.zeros(n); u = np.zeros(n)
    for i, (s, j) in enumerate(LIVE):
        for a, pa in ((0, 1 - r[i]), (1, r[i])):
            pc = PRISK[s] if a == 1 else 0.0
            for c, pcons in ((0, 1 - pc), (1, pc)):
                zn = ZS[j] - c
                for s2 in range(2):
                    w = pa * pcons * PS[s, s2]
                    if w == 0:
                        continue
                    if zn < 0:
                        f[i] += w
                    else:
                        P[i, 3 * s2 + ZS.index(zn)] += w; u[i] += w * UA[a]
    occ = np.linalg.inv(np.eye(n) - BETA * P)[0]
    return float(occ @ u), float(occ @ f)


CLASS = [np.array(b, float) for b in itertools.product([0, 1], repeat=6)]
name = lambda r: "".join(str(int(x)) for x in r)                  # risky (1) or safe (0) in calm 2.5, 1.5, 0.5, stressed 2.5, 1.5, 0.5
ev = [evaluate(r) for r in CLASS]
U = np.array([e[0] for e in ev]); q = np.array([e[1] for e in ev])
pareto, bestU = [], -np.inf
for i in sorted(range(len(CLASS)), key=lambda i: (q[i], -U[i])):
    if U[i] > bestU + 1e-12:
        pareto.append(i); bestU = U[i]
hull = []
for p in sorted((q[i], U[i]) for i in pareto):
    while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) >= 0:
        hull.pop()
    hull.append(p)
hullF = [hull[0]]
for p in hull[1:]:
    if p[1] > hullF[-1][1] + 1e-15:
        hullF.append(p)
hull_q = {round(h[0], 12) for h in hullF}
reached = set()
for nu in np.r_[0, np.geomspace(1e-3, 1e4, 20000)]:
    V = U - nu * q
    reached |= {round(q[j], 12) for j in np.flatnonzero(V >= V.max() - 1e-12)}
assert reached == hull_q                                           # the prices reach exactly the points of the hull
Fbar = lambda a: np.interp(a, [h[0] for h in hullF], [h[1] for h in hullF])
unsup = [i for i in pareto if round(q[i], 12) not in hull_q]
gap, iu = max((Fbar(q[i]) - U[i], i) for i in unsup)

fig, ax = plt.subplots(figsize=(0.62 * TEXTW, 2.5))
ax.plot(q, U, "o", ms=2.8, color=MUTED, mfc="none", mew=0.6, label="the 64 deterministic policies")
ax.step([q[i] for i in pareto] + [1.0], [U[i] for i in pareto] + [U[pareto[-1]]], where="post", color="k", lw=1.3, label="frontier")
ax.plot([h[0] for h in hullF] + [1.0], [h[1] for h in hullF] + [hullF[-1][1]], color=TEXT2, lw=0.9, ls="--", label="its closed concave hull")
rows = {}
for m, col, ls in ((0.05, PAL[2], ":"), (0.3, PAL[0], "-"), (1.0, PAL[7], "-.")):
    nu = m / (1 - BETA); j = int(np.argmax(U - nu * q))
    xl = np.linspace(max(0, q[j] - 0.2), min(0.36, q[j] + 0.2), 50)
    ax.plot(xl, U[j] + nu * (xl - q[j]), color=col, lw=1.0, ls=ls, label=f"slope $\\nu={nu:g}$ ($m={m:g}$)")
    ax.plot(q[j], U[j], "o", ms=6.5, color=col, mfc="none", mew=1.3)
    rows[f"{m:g}"] = dict(nu=nu, policy=name(CLASS[j]), q=float(q[j]), U=float(U[j]))
ax.plot(q[iu], U[iu], "x", ms=7, color="k", mew=1.4, label="frontier point that no price supports")
ax.set_xlim(0, 0.36); ax.set_ylim(9.0, 11.8); ax.set_xlabel("geometric risk $q_\\beta$"); ax.set_ylabel("stopped utility $U$")
ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=FS - 0.5)
out = dict(policies=len(CLASS), frontier=[dict(policy=name(CLASS[i]), q=float(q[i]), U=float(U[i])) for i in pareto], hull_points=len(hullF),
           maximizers=rows, unsupported=dict(policy=name(CLASS[iu]), q=float(q[iu]), U=float(U[iu]), hull=float(Fbar(q[iu])), gap=float(gap)))
(ROOT / "shadow-price-frontier-v76.json").write_text(json.dumps(out, indent=1) + "\n")
save(fig, "shadow-price-frontier-v76", ROOT)
print(json.dumps(out, indent=1))
