"""Figure and table of the parameter-anchor study (reads wear-anchor/campaign*, and the wear campaign's
action-anchored and unanchored arms from wear-allowance/campaign for comparison)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
sys.path.insert(0, str(ROOT.parent / "solvable-flow"))   # regularizer_figure.sci (same folder before 2026-10-13)
from plotstyle import PAL, TEXT2, MUTED, TEXTW, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from regularizer_figure import sci  # noqa: E402

WEAR = ROOT.parent / "wear-allowance"
BENCH = json.loads((WEAR / "benchmark.json").read_text())


def load(stage):
    runs = {}
    for f in sorted((ROOT / "wear-anchor" / stage).glob("*/result.json")):
        k = float(f.parent.name.split("__")[0][5:])
        runs.setdefault(k, []).append(json.loads(f.read_text()))
    return runs


def wear_arm(name):
    return [json.loads(f.read_text()) for f in sorted((WEAR / "campaign").glob(f"{name}__seed*/result.json"))]


def ms(rows, get):
    x = np.array([get(r) for r in rows], dtype=float)
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else 0.0), float(x.min()), float(x.max())


def main():
    full, meanonly = load("campaign"), load("campaign-meanonly")
    exact = [r for r in BENCH["rows"] if r["kind"] == "failure" and abs(r["m"] - 0.03) < 1e-9]
    exact = exact[0]["grid"] if exact else None
    action = {eta: wear_arm(f"anchored-e{eta:g}") for eta in (0.01, 0.1, 1.0)}
    unanch = wear_arm("fixed-m0.03") or wear_arm("m0.03")
    prior_rows = [r for rows in full.values() for r in rows]
    prior = dict(risk=ms(prior_rows, lambda r: r["prior_policy"]["stochastic"]["risk"]), U=ms(prior_rows, lambda r: r["prior_policy"]["stochastic"]["U"]),
                 risk_mean=ms(prior_rows, lambda r: r["prior_policy"]["mean_action"]["risk"]), U_mean=ms(prior_rows, lambda r: r["prior_policy"]["mean_action"]["U"]))
    ks = sorted(full)
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.25), gridspec_kw=dict(wspace=0.45))
    # (a) risk and (b) utility against the anchor strength
    for ax, key, lab, panel in ((axes[0], "risk", "breakdown probability $q_\\beta$", "a"), (axes[1], "U", "stopped utility $U$", "b")):
        for k in ks:
            for r in full[k]:
                ax.plot([k], [r["stochastic"][key]], marker=".", ms=3, color=PAL[0], alpha=0.45, ls="none")
        ax.plot(ks, [ms(full[k], lambda r: r["stochastic"][key])[0] for k in ks], marker="o", ms=4.5, mec="white", mew=0.6, color=PAL[0], label="parameter anchor")
        for k, rows in meanonly.items():
            for r in rows:
                ax.plot([k], [r["stochastic"][key]], marker="v", ms=3, color=PAL[3], alpha=0.5, ls="none")
            ax.plot([k], [ms(rows, lambda r: r["stochastic"][key])[0]], marker="v", ms=5, mec="white", mew=0.6, color=PAL[3], ls="none", label="mean-only prior")
        ax.axhline(prior[key][0], color=PAL[0], lw=0.8, ls=":")
        ax.annotate("prior policy", (ks[0], prior[key][0]), textcoords="offset points", xytext=(2, 3), fontsize=6.5, color=TEXT2)
        if exact:
            ax.axhline(exact["q"] if key == "risk" else exact["U"], color=TEXT2, lw=0.8, ls="--")
            ax.annotate("exact optimum at $m=0.03$", (ks[-1], exact["q"] if key == "risk" else exact["U"]), textcoords="offset points", xytext=(0, 3), fontsize=6.5, color=TEXT2, ha="right")
        ax.set_xscale("log"); ax.set_xlabel("anchor strength $\\kappa_p$"); ax.set_ylabel(lab); panel_label(ax, panel)
    axes[0].legend(loc="center right", fontsize=6.0, borderaxespad=0.2, labelspacing=0.3)
    # (c) risk against action relative entropy to the incumbent: both anchors
    ax = axes[2]
    for k in ks:
        xs = [r["action_kl"]["A"] for r in full[k]]; ys = [r["stochastic"]["risk"] for r in full[k]]
        ax.plot(xs, ys, marker="o", ms=3.5, mec="white", mew=0.5, color=PAL[0], ls="none")
        ax.annotate(f"$\\kappa_p={k:g}$", (np.mean(xs), np.mean(ys)), textcoords="offset points", xytext=(5, 4), fontsize=6, color=TEXT2)
    for eta, rows in action.items():
        if not rows:
            continue
        xs = [r["action_kl"]["A"] for r in rows]; ys = [r["stochastic"]["risk"] for r in rows]
        ax.plot(xs, ys, marker="^", ms=3.5, mec="white", mew=0.5, color=PAL[1], ls="none")
        off = {1.0: (-3, -9), 0.1: (3, -9), 0.01: (5, 3)}[eta]
        ax.annotate(f"$\\eta={eta:g}$", (np.mean(xs), np.mean(ys)), textcoords="offset points", xytext=off, fontsize=6, color=TEXT2, ha="right" if eta == 1.0 else "left")
    for k, rows in meanonly.items():
        ax.plot([r["action_kl"]["A"] for r in rows], [r["stochastic"]["risk"] for r in rows], marker="v", ms=3.5, mec="white", mew=0.5, color=PAL[3], ls="none")
    ax.axhline(prior["risk"][0], color=PAL[0], lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("action relative entropy $\\mathcal{A}$ to the incumbent"); ax.set_ylabel("breakdown probability $q_\\beta$")
    handles = [Line2D([], [], marker="o", ls="none", color=PAL[0], ms=4, label="parameter anchor"),
               Line2D([], [], marker="^", ls="none", color=PAL[1], ms=4, label="action anchor"),
               Line2D([], [], marker="v", ls="none", color=PAL[3], ms=4, label="mean-only prior")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.0, borderaxespad=0.2, labelspacing=0.3); panel_label(ax, "c")
    save(fig, "wear-anchor", ROOT)
    # table
    lines = []
    def row(label, setting, rows, key="stochastic"):
        q = ms(rows, lambda r: r[key]["risk"]); U = ms(rows, lambda r: r[key]["U"]); A = ms(rows, lambda r: r["action_kl"]["A"])
        d = ms(rows, lambda r: r.get("distance", float("nan"))); sd = ms(rows, lambda r: math.exp(r["log_std"]))
        dist = f"${d[0]:.3f}$" if not math.isnan(d[0]) else "--"
        return f"{label} & {setting} & ${100 * q[0]:.2f}\\pm{100 * q[1]:.2f}$ & ${U[0]:.2f}\\pm{U[1]:.2f}$ & {sci(A[0]) if not math.isnan(A[0]) else '--'} & {dist} & ${sd[0]:.2f}$ \\\\"
    lines.append(row("Prior policy", "$\\theta_0$, $\\sigma_a=0.37$", prior_rows_as := [dict(stochastic=r["prior_policy"]["stochastic"], action_kl=r["prior_policy"]["action_kl"], log_std=-1.0) for r in prior_rows]))
    for i, k in enumerate(ks):
        lines.append(row("Parameter anchor" if i == 0 else "", f"$\\kappa_p={k:g}$", full[k]))
    for k, rows in meanonly.items():
        lines.append(row("Mean-only prior", f"$\\kappa_p={k:g}$", rows))
    for i, (eta, rows) in enumerate(action.items()):
        if rows:
            lines.append(row("Action anchor" if i == 0 else "", f"$\\eta={eta:g}$", rows))
    if unanch:
        lines.append(row("No anchor", "$m=0.03$", [dict(stochastic=r["stochastic"], action_kl=dict(A=float("nan")), log_std=r["log_std"]) for r in unanch]))
    tex = ("\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{llrrrrr}\n\\toprule\n"
           "Configuration & Setting & $q_\\beta$ (\\%) & $U$ & $\\mathcal A$ & $\\|\\theta-\\theta_0\\|$ & $\\sigma_a$ \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (ROOT / "wear-anchor-table.tex").write_text(tex)
    print("prior policy: risk %.4f U %.3f (mean action: risk %.4f U %.3f)" % (prior["risk"][0], prior["U"][0], prior["risk_mean"][0], prior["U_mean"][0]))
    for k in ks:
        q = ms(full[k], lambda r: r["stochastic"]["risk"]); U = ms(full[k], lambda r: r["stochastic"]["U"]); d = ms(full[k], lambda r: r["distance"]); A = ms(full[k], lambda r: r["action_kl"]["A"]); sd = ms(full[k], lambda r: math.exp(r["log_std"]))
        print(f"kappa_p={k:6g} n={len(full[k])} risk={100*q[0]:.2f}+-{100*q[1]:.2f} U={U[0]:.3f}+-{U[1]:.3f} dist={d[0]:.4f} A={A[0]:.3g} std={sd[0]:.3f}")
    for k, rows in meanonly.items():
        q = ms(rows, lambda r: r["stochastic"]["risk"]); U = ms(rows, lambda r: r["stochastic"]["U"]); sd = ms(rows, lambda r: math.exp(r["log_std"]))
        print(f"mean-only kappa_p={k:g} n={len(rows)} risk={100*q[0]:.2f}+-{100*q[1]:.2f} U={U[0]:.3f} std={sd[0]:.2f}")
    if exact:
        print("exact m=0.03: q=%.4f U=%.3f" % (exact["q"], exact["U"]))


if __name__ == "__main__":
    main()
