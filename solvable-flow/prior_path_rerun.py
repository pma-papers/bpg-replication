"""Revision 41: rerun the one regularization-path flow that had not converged within its horizon
(prior N(0,1/4), kappa = 0.03; prior-path.json rate_energy = NaN) with a ten times longer horizon.
Everything else is solvable-flow/prior_flow.py (same model, scheme, grid, fit)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "solvable-flow")); sys.path.insert(0, str(ROOT.parent / "lib"))
import prior_flow as pf  # noqa: E402

def one_run_long(name, kappa, x0, sigma, P, factor=10.0):
    model = pf.FirstActionModel("plateau", P["eps"], P["m"], P["beta"], P["gamma"])
    flow = pf.PriorFlow(model, kappa, x0, sigma)
    p0 = pf.initial_plateau(flow.theta, flow.h)
    lam_W = 1.0 / sigma ** 2
    l_max = factor * 40.0 / (kappa * lam_W)
    res = flow.run(p0, l_max, max_steps=40_000_000)
    rec = res["records"]; Jf = rec[-1]["J"]
    gap = np.array([x["J"] - Jf for x in rec]); ls = np.array([x["l"] for x in rec])
    dist = np.array([pf.w2(s, res["p_final"], flow.faces, flow.h) for s in res["snaps"]])
    th = np.linspace(-40, 40, 400001)
    c_G = float(np.trapezoid(pf.chi(th) * pf.gauss(th, x0, sigma), th)); stG = model.expectations(c_G)
    return dict(name=name, kappa=kappa, x0=x0, sigma=sigma, lambda_W=lam_W, kappa_lambda=kappa * lam_W, l_max=l_max, horizon_factor=factor,
                steps=res["steps"], energy_increases=res["energy_increases"], wall=res["wall"], E0=float(gap[0]), W2_0=float(dist[0]),
                rate_energy=pf.fit_rate(ls, gap), rate_W2=pf.fit_rate(ls, dist, lo=1e-4, hi=1e-1),
                prior=dict(c=c_G, V_m=stG["V_m"], q=stG["q"], U=stG["U"]),
                final={k: rec[-1][k] for k in ("J", "V_m", "U", "I_m", "q", "H", "c", "mass")},
                records=[dict(l=x["l"], gap=float(g), W2=float(d), V_m=x["V_m"], q=x["q"], H=x["H"], c=x["c"]) for x, g, d in zip(rec, gap, dist)])

if __name__ == "__main__":
    P = pf.PARAMS
    out = one_run_long("path/x0s0.5/k0.03", 0.03, 0.0, 0.5, P)
    (ROOT / "prior-path-rerun.json").write_text(json.dumps(out, indent=1) + "\n")
    print(out["name"], "steps", out["steps"], "rate", out["rate_energy"], "final", out["final"], "wall", out["wall"])
