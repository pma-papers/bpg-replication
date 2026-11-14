"""Revision 76: the minimizers of the analytical example under relative entropy as the strength decreases.

For the prior Gamma = N(0,1) and relative entropy alone, the minimizer of -V_m + kappa KL(. || Gamma) is an exponential
tilt of the prior by the feature, mu_theta = exp(theta phi - Theta(theta)) Gamma with Theta(theta) = log int exp(theta phi)
dGamma, and the exponent minimizes the function of one variable
    theta -> -V_m(c(theta)) + kappa [theta c(theta) - Theta(theta)],    c(theta) = Theta'(theta),
over theta >= 0 (the paragraph on the optimum in Sect. 6.3 of the paper). This script computes the minimizer for a
sequence of strengths decreasing to zero and reports the clipped value, its distance to the supremum of the clipped
value, the relative entropy and its product with the strength. It is the numerical check of the proposition on a
decreasing strength: the values converge to the supremum, the product tends to zero, and the relative entropy grows.
Output: decreasing-strength-limit-v76.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / "regularizer-flow.json").read_text())
C, V = np.array(D["curve"]["c"]), np.array(D["curve"]["V_m"])
SUP = float(V.max())
X = np.linspace(-60.0, 60.0, 600001)
Y = np.maximum(np.abs(X) - 1, 1e-12)
PHI = np.where(np.abs(X) > 1, np.exp(-Y ** -2), 0.0)
LOGG = -X ** 2 / 2 - 0.5 * np.log(2 * np.pi)


def tilt(theta):
    logw = theta * PHI + LOGG
    top = logw.max()                                       # the weights are scaled by their largest value, against overflow
    w = np.exp(logw - top)
    Z = np.trapezoid(w, X)
    c = np.trapezoid(PHI * w, X) / Z
    return float(c), float(theta * c - (np.log(Z) + top))  # the tilt and the relative entropy of mu_theta


GRID = np.geomspace(1e-3, 3e4, 500)
TILT = np.array([tilt(th) for th in GRID])


def minimizer(kappa):
    f = lambda th: -np.interp(tilt(th)[0], C, V) + kappa * tilt(th)[1]
    k = int(np.argmin(-np.interp(TILT[:, 0], C, V) + kappa * TILT[:, 1]))
    assert 0 < k < len(GRID) - 1, "the minimizer lies at an end of the grid of exponents"
    r = minimize_scalar(f, bounds=(GRID[k - 1], GRID[k + 1]), method="bounded", options=dict(xatol=1e-9))
    c, kl = tilt(r.x)
    v = float(np.interp(c, C, V))
    return dict(theta=round(float(r.x), 4), tilt=round(c, 6), V_m=round(v, 6), distance_to_the_supremum=round(SUP - v, 6),
                KL=round(kl, 4), strength_times_KL=round(kappa * kl, 6), objective=round(-v + kappa * kl, 6))


out = dict(supremum_of_V_m=round(SUP, 6), minimizers={f"{k:g}": minimizer(k) for k in (10, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001)})
stored = json.loads((ROOT / "prior-path.json").read_text())["runs"]
out["finite_volume_runs"] = {f"{k:g}": round(stored[f"path/x0s1/k{k:g}"]["final"]["V_m"], 6) for k in (10, 1, 0.3, 0.1, 0.03)}
(ROOT / "decreasing-strength-limit-v76.json").write_text(json.dumps(out, indent=1) + "\n")
for k, v in out["minimizers"].items():
    print(k, v, "finite volume", out["finite_volume_runs"].get(k))
