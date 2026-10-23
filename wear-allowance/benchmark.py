"""Exact benchmark of the wear-allowance example: frontier, price-risk law, margins.

Writes benchmark.json. Grid values (U, q, I) are exact for the interpolated grid model;
risks under a reduced allowance and near-exhaustion frequencies are Monte Carlo estimates
of the interpolated policies in the continuous simulator.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from dp import Grid, metrics, policy_function, solve
from model import Model, simulate

ROOT = Path(__file__).resolve().parent
N = 100_000
BASE = Model(wear_shape=0.5)
PRICES = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30]


def record(model, grid, kind, m, policy_table, policy, extra=None):
    g = metrics(grid, policy_table, m) if policy_table is not None else None
    out = dict(kind=kind, m=m, grid=g)
    for name, scale, seed in (("nominal", 1.0, 11), ("allowance90", 0.9, 12), ("allowance80", 0.8, 13)):
        r = simulate(model, policy, n=N, seed=seed, m=m, budget_scale=scale)
        out[name] = dict(risk=r["risk"], U=r["U"], I=r["I"], near=r["near"])
    out.update(extra or {})
    return out


def main():
    rows, tables = [], {}
    G = Grid(BASE)
    for m in PRICES:
        sol = solve(G, "failure", m)
        rows.append(record(BASE, G, "failure", m, sol["policy_index"], policy_function(G, sol["policy"])))
        if m in (0, 1):
            tables[f"failure-m{m}"] = sol["policy"].tolist()
    for theta in (0.03, 0.1, 0.3):
        Mb = replace(BASE, theta=theta)
        Gb = Grid(Mb)
        for m in (0.3, 1, 3, 10):
            sol = solve(Gb, "barrier", m)
            rows.append(record(Mb, Gb, "barrier", m, sol["policy_index"], policy_function(Gb, sol["policy"]),
                               dict(theta=theta)))
            if theta == 0.1 and m == 1:
                tables["barrier-m1"] = sol["policy"].tolist()
    for keep in (0.9, 0.8):
        Mt = replace(BASE, budget=keep)
        Gt = Grid(Mt)
        for m in (0.3, 1, 3, 10):
            sol = solve(Gt, "failure", m)
            inner = policy_function(Gt, sol["policy"])
            policy = lambda s, z, inner=inner, keep=keep: inner(s, np.maximum(z - (1 - keep), 0.0)) * (z > 1 - keep)
            rows.append(record(BASE, None, "tightened", m, None, policy, dict(planned_allowance=keep)))
    out = dict(model=BASE.to_dict(), n=N, rows=rows, s_grid=G.s.tolist(), z_grid=G.z.tolist(), policy_tables=tables)
    (ROOT / "benchmark.json").write_text(json.dumps(out) + "\n")
    for r in rows:
        g = r["grid"] or {}
        print(f"{r['kind']:9s} m={r['m']:<5} th={r.get('theta','-')!s:5} plan={r.get('planned_allowance','-')!s:4} "
              f"U={g.get('U', r['nominal']['U']):.3f} q={g.get('q', r['nominal']['risk']):.4f} "
              f"near5={r['nominal']['near']['0.05']:.3f} q90={r['allowance90']['risk']:.3f} q80={r['allowance80']['risk']:.3f}")


if __name__ == "__main__":
    main()
