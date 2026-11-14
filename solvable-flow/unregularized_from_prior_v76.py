"""Revision 76: unregularized transport of the analytical example started from the prior.

Without regularization a unit with parameter x moves by dx = V_m'(c) phi'(x) dl, with the tilt c = int phi dmu. The
clipped value increases in the tilt (V_m' > 0 on the value curve) and phi' has the sign of x outside the plateau
|x| <= 1 and vanishes on it. Every unit outside the plateau therefore moves outward and its feature tends to one, and
the units on the plateau do not move. The limiting tilt is the mass of the prior outside the plateau,
P(|x| > 1), and the limiting clipped value is the value curve at that tilt.
The law is represented by 20000 quantiles of the prior N(x0, sigma^2) and the equation is integrated by the explicit
Euler scheme with the step 0.01. Output: unregularized-from-prior-v76.json, with the clipped value at the learning times
1, 10, 100 and 1000 and in the limit, for the four priors of the regularization paths.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / "prior-path.json").read_text())
C, V = np.array(D["curve"]["c"]), np.array(D["curve"]["V_m"])
DV = np.gradient(V, C)
assert DV.min() > 0
N, DT, TIMES = 20000, 0.01, (1, 10, 100, 1000)


def phi(x):
    y = np.maximum(np.abs(x) - 1, 1e-12)
    return np.where(np.abs(x) > 1, np.exp(-y ** -2), 0.0)


def dphi(x):
    y = np.maximum(np.abs(x) - 1, 1e-12)
    return np.where(np.abs(x) > 1, np.sign(x) * 2 * y ** -3 * np.exp(-y ** -2), 0.0)


out = {}
for x0, sigma in D["priors"]:
    x = norm.ppf((np.arange(N) + 0.5) / N, x0, sigma)
    mass = float(norm.cdf(-1, x0, sigma) + norm.sf(1, x0, sigma))
    row = dict(mass_outside_the_plateau=round(mass, 4), V_m_of_the_prior=round(float(np.interp(phi(x).mean(), C, V)), 4), V_m_at={})
    step = 0
    for t in TIMES:
        while step < round(t / DT):
            x = x + DT * np.interp(phi(x).mean(), C, DV) * dphi(x); step += 1
        row["V_m_at"][str(t)] = round(float(np.interp(phi(x).mean(), C, V)), 4)
    row["V_m_in_the_limit"] = round(float(np.interp(mass, C, V)), 4)
    out[f"N({x0:g}, {sigma ** 2:g})"] = row
    print(f"N({x0:g}, {sigma ** 2:g})", row)
(ROOT / "unregularized-from-prior-v76.json").write_text(json.dumps(out, indent=1) + "\n")
