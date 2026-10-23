"""Revision 41: figure and table of the regularizer-theorem test (reads solvable-flow/regularizer-flow.json).

Changes against solvable-flow/regularizer_figure.py (review findings):
 * the finite-difference curvature estimate of this example, C_sharp = max|V_m''| L1^2 + max|V_m'| L2, computed from the stored
   curve (curve.c, curve.V_m) with np.gradient, against the certified constant C_m of the theorem;
 * panel (a): the estimated envelope exp(-2 (kappa lambda_W - C_sharp) l) next to the certified envelope;
   y-label "objective gap";
 * table (regularizer-table-v41.tex) with a note giving C_sharp; the constants in flow-constants-v41.json.
Outputs (this directory): regularizer-flow-v41 (pdf/png), regularizer-table-v41.tex, flow-constants-v41.json,
regularizer-digests-v41.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
V39 = ROOT.parent / "solvable-flow"
sys.path.insert(0, str(ROOT.parent / "lib"))
sys.path.insert(0, str(ROOT))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, MK, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from digest_v41 import digest  # noqa: E402

D = json.loads((V39 / "regularizer-flow.json").read_text())
LABEL = {"KL": "KL alone", "r1.5": "KL $+\\,\\omega G$, $r=3/2$", "r2": "KL $+\\,\\omega G$, $r=2$", "r3": "KL $+\\,\\omega G$, $r=3$"}
COL = {"KL": PAL[0], "r1.5": PAL[1], "r2": PAL[2], "r3": PAL[3]}
MARK = {"KL": MK[0], "r1.5": MK[1], "r2": MK[2], "r3": MK[3]}   # markers beside the colours
ORDER = ["KL", "r1.5", "r2", "r3"]
KAPPAS = ["cert", "k10", "k1", "k0.1"]
LAM_W = D["params"]["lambda_W"]
DIGESTS = {}


def sci(x, digits=2):
    """LaTeX number: plain when 0.01 <= |x| < 1000, else a.b x 10^k."""
    if x == 0:
        return "$0$"
    if 1e-2 <= abs(x) < 1e3:
        if abs(x) >= 100:
            return f"${x:.0f}$"
        g = f"{x:.{digits}g}"
        return f"${float(g):.0f}$" if "e" in g else f"${g}$"   # 99.6 rounds to 100, not to 1e+02
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"


def sharp_constant():
    """C_sharp = max|V_m''| L1^2 + max|V_m'| L2 along the stored curve c -> V_m(c), derivatives by np.gradient."""
    c = np.array(D["curve"]["c"]); V = np.array(D["curve"]["V_m"])
    V1 = np.gradient(V, c); V2 = np.gradient(V1, c)
    L1, L2 = D["constants"]["L1"], D["constants"]["L2"]
    m1, m2 = float(np.max(np.abs(V1))), float(np.max(np.abs(V2)))
    return dict(C_sharp=m2 * L1 ** 2 + m1 * L2, max_abs_V1=m1, max_abs_V2=m2, L1=L1, L2=L2, L0=D["constants"]["L0"],
                c_psi=D["constants"]["c_psi"], C_m=D["constants"]["C_m"], kappa_cert=D["kappa_cert"], lambda_W=LAM_W,
                Lambda_m_cert=D["runs"]["cert/KL"]["Lambda"], n_curve_points=int(len(c)), curve_step=float(c[1] - c[0]),
                derivative="np.gradient (second-order central differences) applied twice to curve.V_m over curve.c")


def panel_decay(ax, kname, title, xmax, C_sharp=None):
    arrs = []
    for reg in ORDER:
        r = D["runs"][f"{kname}/{reg}"]
        l = np.array([x["l"] for x in r["records"]]); g = np.array([x["gap"] for x in r["records"]]) / r["E0"]
        pos = g > 0; arrs += [l[pos], g[pos]]
        ax.plot(l[pos], g[pos], color=COL[reg], marker=MARK[reg], markevery=(0.05 * ORDER.index(reg), 0.2), ms=3.5, mec="white", mew=0.4, label=LABEL[reg])
    r = D["runs"][f"{kname}/KL"]
    l = np.linspace(0, xmax, 200)
    lam = r["Lambda"]
    if lam > 0:
        ax.plot(l, np.exp(-2 * lam * l), color=TEXT2, ls="--", label="theorem, $e^{-2\\Lambda_m\\ell}$")
    if C_sharp is not None and r["kappa"] * LAM_W - C_sharp > 0:
        rate = 2 * (r["kappa"] * LAM_W - C_sharp)
        ax.plot(l, np.exp(-rate * l), color=PAL[6], ls="-.", lw=1.0, label="estimated, $e^{-2(\\kappa\\lambda_W-C_\\sharp)\\ell}$")
    ax.plot(l, np.exp(-2 * r["kappa"] * LAM_W * l), color=MUTED, ls=":", label="$e^{-2\\kappa\\lambda_W\\ell}$")
    ax.set_yscale("log"); ax.set_ylim(1e-10, 2); ax.set_xlim(0, xmax)
    ax.set_xlabel("flow time $\\ell$")
    return arrs


def main():
    K = sharp_constant()
    K["sharp_rate_by_kappa"] = {k: 2 * (D["runs"][f"{k}/KL"]["kappa"] * LAM_W - K["C_sharp"]) for k in KAPPAS}
    K["certified_rate"] = 2 * D["runs"]["cert/KL"]["Lambda"]
    K["observed_rate_KL_by_kappa"] = {k: D["runs"][f"{k}/KL"]["rate_energy"] for k in KAPPAS}
    (ROOT / "flow-constants-v41.json").write_text(json.dumps(K, indent=1) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.2), gridspec_kw=dict(wspace=0.5))
    ax = axes[0]
    arrs = panel_decay(ax, "cert", f"certified $\\kappa={D['kappa_cert']:.0f}$", 0.004, C_sharp=K["C_sharp"])
    d = digest(*arrs); DIGESTS["regularizer/a flow curves"] = dict(before=d, after=d, changed=False, note="sharp envelope added; y-label changed")
    ax.set_ylabel("objective gap"); panel_label(ax, "a")
    handles_a, labels_a = ax.get_legend_handles_labels()
    ax = axes[1]
    arrs = panel_decay(ax, "k1", "$\\kappa=1$, below the threshold", 8.0); panel_label(ax, "b")
    d = digest(*arrs); DIGESTS["regularizer/b flow curves"] = dict(before=d, after=d, changed=False)
    ax = axes[2]
    ks = [D["runs"][f"{k}/KL"]["kappa"] for k in KAPPAS]
    VG = D["stats_Gamma"]["V_m"]
    arrs = [ks]
    for reg in ORDER:
        y = np.array([D["runs"][f"{k}/{reg}"]["final"]["V_m"] - VG for k in KAPPAS]); arrs.append(y)
        ax.plot(ks, np.maximum(y, 1e-7), marker=MARK[reg], ms=3.5, mec="white", mew=0.5, color=COL[reg], label=LABEL[reg])
    d = digest(*arrs); DIGESTS["regularizer/c improvement over the prior"] = dict(before=d, after=d, changed=False)
    ax.axhline(D["sup_V_m"] - VG, color=TEXT2, ls="-", lw=0.8)
    ax.annotate("$\\sup V_m-V_m(\\Gamma)$", (ks[1], D["sup_V_m"] - VG), textcoords="offset points", xytext=(0, 3), fontsize=FS, color=TEXT2, ha="center")
    ax.axvline(D["kappa_cert"], color=MUTED, lw=0.7, ls=":")
    ax.annotate("certified", (D["kappa_cert"], 3e-7), textcoords="offset points", xytext=(-3, 0), fontsize=FS, color=TEXT2, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-7, 2)
    ax.set_xlabel("$\\kappa$"); ax.set_ylabel("$V_m(\\mu^*)-V_m(\\Gamma)$")
    panel_label(ax, "c")
    legend_below(fig, handles_a, labels_a, ncol=3)
    save(fig, "regularizer-flow-v41", ROOT)
    (ROOT / "regularizer-digests-v41.json").write_text(json.dumps(DIGESTS, indent=1) + "\n")

    # appendix table: as revision 39, with a note giving the constants and C_sharp
    rows = []
    for kname in KAPPAS:
        for reg in ORDER:
            r = D["runs"][f"{kname}/{reg}"]
            lam = r["Lambda"]
            thm = sci(2 * lam) if lam > 0 else "--"
            lp = sci(r["l_predicted"]) if r.get("l_predicted") else "--"
            lo = r.get("l_observed_half_Delta")
            lo = sci(lo) if lo is not None else "--"
            rows.append(f"${r['kappa']:.4g}$ & {LABEL[reg].replace('rel. ', 'rel.\\ ')} & {sci(r['rate_energy'])} & {thm} & {sci(r['rate_W2'])} & "
                        f"${r['final']['V_m']:.4f}$ & ${100 * r['final']['q']:.2f}$ & {sci(r['final']['H'])} & {lp} & {lo} \\\\")
    body = "\n".join(rows)
    note = (f"\\multicolumn{{10}}{{@{{}}p{{0.95\\linewidth}}@{{}}}}{{Constants of the example: $L_1={K['L1']:.3f}$, $L_2={K['L2']:.3f}$, "
            f"$C_m={K['C_m']:.0f}$, so the certified threshold is ${K['C_m'] / K['lambda_W']:.0f}$, and at the certified strength $\\kappa={K['kappa_cert']:.0f}$ used here $2\\Lambda_m={K['certified_rate']:.0f}$. "
            f"The finite-difference curvature estimate of this example, $C_\\sharp=\\max|V_m''|\\,L_1^2+\\max|V_m'|\\,L_2={K['C_sharp']:.2f}$ "
            f"($\\max|V_m'|={K['max_abs_V1']:.3f}$, $\\max|V_m''|={K['max_abs_V2']:.3f}$ along the stored curve), gives the estimated envelope rate "
            f"$2(\\kappa\\lambda_W-C_\\sharp)={K['sharp_rate_by_kappa']['cert']:.0f}$ at the certified $\\kappa$.}} \\\\")
    tex = ("\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{llrrrrrrrr}\n\\toprule\n"
           "$\\kappa$ & regularizer & rate of the gap & $2\\Lambda_m$ & rate of $W_2$ & $V_m(\\mu^*)$ & $q_\\beta$ (\\%) & $\\mathcal H(\\mu^*)$ & $\\ell_{\\rm bd}$ & $\\ell_{\\rm obs}$ \\\\\n\\midrule\n"
           + body + "\n\\bottomrule\n" + note + "\n\\end{tabular}\n")
    (ROOT / "regularizer-table-v41.tex").write_text(tex)
    print(json.dumps({k: v for k, v in K.items() if not isinstance(v, dict)}, indent=1))
    print("sharp rate by kappa:", K["sharp_rate_by_kappa"]); print("observed KL-alone rate by kappa:", K["observed_rate_KL_by_kappa"])


if __name__ == "__main__":
    main()
