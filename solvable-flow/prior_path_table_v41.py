"""Revision 41: online table of the 28 regularization-path runs of the prior figure (panels b, c), one row per run,
in the style of solvable-flow/prior-table.tex. Reads solvable-flow/prior-path.json; the run path/x0s0.5/k0.03 is
replaced by solvable-flow/prior-path-rerun.json (ten times longer horizon) and marked with a dagger.
Output: prior-path-table-v41.tex."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V39 = ROOT.parent / "solvable-flow"
sys.path.insert(0, str(ROOT))
from regularizer_figure_v41 import sci  # noqa: E402

D = json.loads((V39 / "prior-path.json").read_text())
RERUN = ROOT / "prior-path-rerun.json"
LAB = {(0.0, 0.5): "$N(0,1/4)$", (0.0, 1.0): "$N(0,1)$", (0.0, 2.0): "$N(0,4)$", (2.0, 1.0): "$N(2,1)$"}


def main():
    replaced = None
    if RERUN.exists():
        R = json.loads(RERUN.read_text())
        run = D["runs"][R["name"]]
        for k in ("final", "rate_energy", "rate_W2", "l_max", "steps", "horizon_factor"):
            run[k] = R[k]
        replaced = R["name"]
    rows = []
    for x0, sigma in (tuple(p) for p in D["priors"]):
        for kappa in D["kappas"]:
            name = f"path/x{x0:g}s{sigma:g}/k{kappa:g}"
            r = D["runs"][name]
            dag = "\\textsuperscript{$\\dagger$}" if name == replaced else ""
            rate = sci(r["rate_energy"], 3) if r["rate_energy"] == r["rate_energy"] else "--"   # NaN: not converged
            rows.append(f"{LAB[(x0, sigma)]} & ${r['kappa']:g}${dag} & ${r['kappa_lambda']:g}$ & {rate} & "
                        f"${100 * r['prior']['q']:.2f}$ & ${r['prior']['V_m']:.4f}$ & "
                        f"${r['final']['c']:.3f}$ & ${100 * r['final']['q']:.2f}$ & ${r['final']['U']:.4f}$ & ${r['final']['V_m']:.4f}$ & {sci(r['final']['H'])} \\\\")
    note = "\\multicolumn{11}{@{}l@{}}{\\textsuperscript{$\\dagger$}rerun with a ten times longer horizon.} \\\\" if replaced else ""
    tex = ("\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n"
           " & & & & \\multicolumn{2}{c}{prior policy $\\pi_\\Gamma$} & \\multicolumn{5}{c}{minimizer $\\mu^*$} \\\\\n"
           "\\cmidrule(lr){5-6}\\cmidrule(lr){7-11}\n"
           "prior & $\\kappa$ & $\\kappa\\lambda_W$ & rate of the gap & $q_\\beta$ (\\%) & $V_m$ & $c$ & $q_\\beta$ (\\%) & $U$ & $V_m$ & $\\KL$ \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n" + note + "\n\\end{tabular}\n")
    (ROOT / "prior-path-table-v41.tex").write_text(tex)
    print(f"{len(rows)} rows; replaced run: {replaced}")


if __name__ == "__main__":
    main()
