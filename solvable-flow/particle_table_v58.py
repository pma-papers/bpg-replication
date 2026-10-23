"""Revision 58: the table of the particle check from particle-flow-v58.json (written beside this script as particle-flow-table-v58.tex; journal-v58/tables/particle-flow.tex at the time)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "particle-flow-v58.json").read_text())
lines = ["\\begin{tabular}{llrrr}", "\\toprule",
         "Strength & Units $n$ & final $V_m$, particles & final $V_m$, flow & largest deviation \\\\", "\\midrule"]
for run, R in S["runs"].items():
    first = True
    for n, q in R["particles"].items():
        lines.append(f"{'$\\kappa=' + format(R['kappa'], 'g') + '$' if first else ''} & ${int(n)}$ & ${q['final_V_m_mean']:.4f}\\pm{q['final_V_m_sd']:.4f}$ & ${R['pde_final']['V_m']:.4f}$ & ${q['max_abs_dev_V_m']:.4f}$ \\\\")
        first = False
lines += ["\\bottomrule", "\\end{tabular}"]
(HERE / "particle-flow-table-v58.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
