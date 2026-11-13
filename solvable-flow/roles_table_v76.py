"""Table of the roles of the regularizers on the analytical example (added for revision 76; until then the table was
assembled by hand from the same record).

Reads roles-flow-v59.json, the record of roles_flow_v59.py, and writes roles-flow-table-v76.tex: for each
configuration the noise strength kappa_1, the strength kappa_2 of the potential, the final clipped value (mean and
between-seed standard deviation), the final spread of the units, and the stationary spread
sqrt(kappa_1 / (kappa_1 + kappa_2)) of the Ornstein-Uhlenbeck process without the value drift (zero under the
potential alone, where there is no noise). The last five rows vary the noise at kappa_2 = 1.
Run: ~/.venvs/constrained-rl/bin/python roles_table_v76.py
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {"K": "relative entropy", "P": "potential", "K+P": "sum"}


def row(name, c):
    k1, k2 = c["kappa1"], c["kappa2"]
    ou = math.sqrt(k1 / (k1 + k2)) if k1 > 0 else 0.0
    return (f"{name} & ${k1:g}$ & ${k2:g}$ & ${c['final_V_m_mean']:.4f}\\pm{c['final_V_m_sd']:.4f}$ & "
            f"${c['final_spread_mean']:.3f}$ & ${ou:.3f}$ \\\\")


if __name__ == "__main__":
    rec = json.loads((HERE / "roles-flow-v59.json").read_text())
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "Regularizer & $\\kappa_1$ (noise) & $\\kappa_2$ (potential) & final $V_m$ & final spread & OU spread \\\\", "\\midrule"]
    lines += [row(NAMES[c["name"]], c) for c in rec["configs"]]
    lines += ["\\midrule"] + [row("sum", c) for c in rec["sweep"]] + ["\\bottomrule", "\\end{tabular}"]
    (HERE / "roles-flow-table-v76.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
