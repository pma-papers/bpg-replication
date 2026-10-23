"""Figure and table of the regularizer-theorem test (reads regularizer-flow.json)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

D = json.loads((ROOT / "regularizer-flow.json").read_text())
LABEL = {"KL": "KL alone", "r1.5": "KL $+\\,\\omega G$, $r=3/2$", "r2": "KL $+\\,\\omega G$, $r=2$", "r3": "KL $+\\,\\omega G$, $r=3$"}
COL = {"KL": PAL[0], "r1.5": PAL[1], "r2": PAL[2], "r3": PAL[3]}
ORDER = ["KL", "r1.5", "r2", "r3"]
KAPPAS = ["cert", "k10", "k1", "k0.1"]


def sci(x, digits=2):
    """LaTeX number: plain when 0.01 <= |x| < 1000, else a.b x 10^k."""
    if x == 0:
        return "$0$"
    if 1e-2 <= abs(x) < 1e3:
        return f"${x:.{digits}g}$" if abs(x) < 100 else f"${x:.0f}$"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"


def panel_decay(ax, kname, title, xmax):
    for reg in ORDER:
        r = D["runs"][f"{kname}/{reg}"]
        l = np.array([x["l"] for x in r["records"]]); g = np.array([x["gap"] for x in r["records"]]) / r["E0"]
        pos = g > 0
        ax.plot(l[pos], g[pos], color=COL[reg], label=LABEL[reg])
    r = D["runs"][f"{kname}/KL"]
    l = np.linspace(0, xmax, 200)
    lam = r["Lambda"]
    if lam > 0:
        ax.plot(l, np.exp(-2 * lam * l), color=TEXT2, ls="--", label="theorem, $e^{-2\\Lambda_m\\ell}$")
        ax.annotate(f"theorem envelope, rate $2\\Lambda_m={2 * lam:.0f}$", (0.55 * xmax, np.exp(-2 * lam * 0.55 * xmax)),
                    textcoords="offset points", xytext=(0, -10), fontsize=6.5, color=TEXT2, ha="center")
    ax.plot(l, np.exp(-2 * r["kappa"] * l), color=MUTED, ls=":", label="$e^{-2\\kappa\\lambda_W\\ell}$")
    ax.set_yscale("log"); ax.set_ylim(1e-10, 2); ax.set_xlim(0, xmax)
    ax.set_xlabel("flow time $\\ell$"); ax.set_title(title)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.25), gridspec_kw=dict(wspace=0.4))
    ax = axes[0]
    panel_decay(ax, "cert", f"certified $\\kappa={D['kappa_cert']:.0f}$", 0.004)
    ax.set_ylabel("objective gap, relative to $E_0$"); panel_label(ax, "a")
    ax.legend(loc="lower left", fontsize=6.0, borderaxespad=0.2, labelspacing=0.3)
    ax = axes[1]
    panel_decay(ax, "k1", "$\\kappa=1$, below the threshold", 8.0); panel_label(ax, "b")
    ax = axes[2]
    ks = [D["runs"][f"{k}/KL"]["kappa"] for k in KAPPAS]
    VG = D["stats_Gamma"]["V_m"]
    for reg in ORDER:
        y = np.array([D["runs"][f"{k}/{reg}"]["final"]["V_m"] - VG for k in KAPPAS])
        ax.plot(ks, np.maximum(y, 1e-7), marker="o", ms=3.5, mec="white", mew=0.5, color=COL[reg], label=LABEL[reg])
    ax.axhline(D["sup_V_m"] - VG, color=TEXT2, ls="-", lw=0.8)
    ax.annotate("$\\sup V_m-V_m(\\Gamma)$", (ks[1], D["sup_V_m"] - VG), textcoords="offset points", xytext=(0, 3), fontsize=6.5, color=TEXT2, ha="center")
    ax.axvline(D["kappa_cert"], color=MUTED, lw=0.7, ls=":")
    ax.annotate("certified", (D["kappa_cert"], 3e-7), textcoords="offset points", xytext=(-3, 0), fontsize=6.5, color=TEXT2, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-7, 2)
    ax.set_xlabel("$\\kappa$"); ax.set_ylabel("$V_m(\\mu^*)-V_m(\\Gamma)$"); ax.set_title("improvement over the prior")
    panel_label(ax, "c")
    save(fig, "regularizer-flow", ROOT)

    # appendix table
    rows = []
    for kname in KAPPAS:
        for reg in ORDER:
            r = D["runs"][f"{kname}/{reg}"]
            lam = r["Lambda"]
            thm = sci(2 * lam) if lam > 0 else "--"
            lp = sci(r["l_predicted"]) if r.get("l_predicted") else "--"
            lo = r.get("l_observed_half_Delta")
            lo = sci(lo) if lo is not None else "--"
            rows.append(f"${r['kappa']:.4g}$ & {LABEL[reg]} & {sci(r['rate_energy'])} & {thm} & {sci(r['rate_W2'])} & "
                        f"${r['final']['V_m']:.4f}$ & ${100 * r['final']['q']:.2f}$ & {sci(r['final']['H'])} & {lp} & {lo} \\\\")
    body = "\n".join(rows)
    tex = ("\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{llrrrrrrrr}\n\\toprule\n"
           "$\\kappa$ & regularizer & rate of the gap & $2\\Lambda_m$ & rate of $W_2$ & $V_m(\\mu^*)$ & $q_\\beta$ (\\%) & $\\mathcal H(\\mu^*)$ & $\\ell_{\\rm pred}$ & $\\ell_{\\rm obs}$ \\\\\n\\midrule\n"
           + body + "\n\\bottomrule\n\\end{tabular}\n")
    (ROOT / "regularizer-table.tex").write_text(tex)
    for kname in KAPPAS:
        for reg in ORDER:
            r = D["runs"][f"{kname}/{reg}"]
            print(f"{kname:5s} {reg:5s} kappa={r['kappa']:9.4g} Lam={r['Lambda']:9.4g} rateE={r['rate_energy']:9.4g} rateW2={r['rate_W2']:9.4g} "
                  f"V_m*={r['final']['V_m']:.5f} q*={r['final']['q']:.4f} H*={r['final']['H']:.3g} steps={r['steps']} inc={r['energy_increases']} "
                  f"lpred={r.get('l_predicted')} lobs={r.get('l_observed_half_Delta')}")
    print("mu0", D["stats_mu0"]["V_m"], "Gamma", D["stats_Gamma"]["V_m"], "sup", D["sup_V_m"], "Delta", D["Delta"])


if __name__ == "__main__":
    main()
