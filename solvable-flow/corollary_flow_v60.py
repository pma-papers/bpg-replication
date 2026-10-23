"""Revision 60: solvable-flow runs for the Corollary thm:combined figure (relative entropy alone).

Reuses the flow solver of revision 38/39 (solvable-flow/regularizer_flow.py) on the plateau model with the
prior N(0,1), at the strengths kappa = margin * C_m / lambda_W for margins 1.5, 2 and 4 (the margin 1.05
run is the stored cert/KL run of solvable-flow/regularizer-flow.json). Horizon l_max = 12 / Lambda_m as in
that study. Output: corollary-flow-v60.json with the same record fields (l, gap, q, V_m, ...).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "solvable-flow"))
import regularizer_flow as RF  # noqa: E402

MARGINS = [1.5, 2.0, 4.0]


def main():
    P = RF.PARAMS
    _, consts = RF.certified(P["m"], P["beta"], P["ubar"], P["lambda_W"], margin=1.0)
    jobs = []
    for mg in MARGINS:
        kappa = mg * consts["C_m"] / P["lambda_W"]
        Lam = kappa * P["lambda_W"] - consts["C_m"]
        jobs.append((f"m{mg:g}/KL", 0.0, 2.0, kappa, 12.0 / Lam, P))
    with Pool(3) as pool:
        runs = pool.map(RF.one_run, jobs)
    out = dict(params=P, constants=consts, margins=MARGINS, runs={})
    for r in runs:
        r.pop("density", None)
        r["Lambda"] = r["kappa"] * P["lambda_W"] - consts["C_m"]
        out["runs"][r["name"]] = r
    (HERE / "corollary-flow-v60.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
