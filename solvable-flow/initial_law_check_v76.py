"""Revision 76: the runs of the analytical example from a Gaussian initial law in place of the conditioned prior.

The initial law of the paper is the prior N(0,1) conditioned on [-1/2, 1/2], which lies on the plateau, so that its
tilt is exactly zero. This check repeats the particle runs of decreasing_strength_v76.py at the fixed strength 0.03 and
under the rule from the Gaussian initial law N(0, 1/16), which has all but 6e-5 of its mass on the plateau, and compares
the learning time at which the clipped value first comes within 0.02 of that of the minimizer, and the final value.
Output: initial-law-check-v76.json.
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import decreasing_strength_v76 as m

ROOT = Path(__file__).resolve().parent
SD0 = 0.25

if __name__ == "__main__":
    jobs = [(kind, None, sigma, seed) + extra for sigma in (1.0, 2.0) for kind in ("fixed", "rule") for seed in m.SEEDS for extra in ((), (SD0,))]
    with Pool() as pool:
        res = pool.map(m.run, jobs, chunksize=1)
    out = dict(gaussian_initial_standard_deviation=SD0, units=m.N, seeds=list(m.SEEDS), runs={})
    for sigma in (1.0, 2.0):
        for kind in ("fixed", "rule"):
            for name, n_extra in (("conditioned prior", 0), ("Gaussian", 1)):
                rs = [r for r, j in zip(res, jobs) if j[2] == sigma and j[0] == kind and len(j) == 4 + n_extra]
                t = [r["reached"] for r in rs]
                row = dict(time_mean=round(float(np.mean(t)), 1), time_min=round(float(np.min(t)), 1), time_max=round(float(np.max(t)), 1),
                           final_value=round(float(np.mean([r["trace"][-1] for r in rs])), 4))
                out["runs"][f"prior N(0,{sigma ** 2:g}), {kind}, {name}"] = row
                print(f"prior N(0,{sigma ** 2:g}) {kind:6s} {name:18s}", row)
    (ROOT / "initial-law-check-v76.json").write_text(json.dumps(out, indent=1) + "\n")
