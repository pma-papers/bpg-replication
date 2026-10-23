"""Revision 60: analysis of the injected-parameter-noise campaign (PROTOCOL-noise-v60.md).

Reads wear-noise/campaign/k1_<k1>__k2_<k2>__seed<S>/result.json (eight seeds per
configuration) and the reused pull-only (kappa_p = 10) and unanchored-control (kappa_p = 0) runs of the
matched campaign of revision 41 (wear-allowance/wear-anchor/campaign-matched). Per configuration it reports
the mean and between-seed standard deviation of the breakdown probability q_beta and stopped utility U of
the stochastic policy, the mean-action policy's q_beta, the action relative entropy A to the incumbent
rule, the parameter distance ||theta - theta_0||, the per-parameter spread ||theta - theta_0|| / sqrt(d)
(d = number of actor parameters) and the final action dispersion sigma_a. It checks predictions R1-R4 of
the protocol and writes wear-noise-v60.json, wear-noise-v60.pdf/png and wear-noise-v60.tex (the table).
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, legend_below, save  # noqa: E402

K2 = 10.0
SWEEP = [0.0010, 0.0090, 0.1010, 0.9890]          # kappa_1 at the pull kappa_2 = 10
NOMINAL = {0.0010: 0.01, 0.0090: 0.03, 0.1010: 0.1, 0.9890: 0.3}   # the protocol's design values, sqrt(k1/(k1+k2)), for (R1)
# The zero-objective reference of injected noise under the pull: the noise carries no pull of its own, so a gradient
# pull kappa_2 holds it at sqrt(kappa_1/kappa_2) (the paper's table and figure since revision 70).
REFERENCE = {k: math.sqrt(k / K2) for k in SWEEP}
NOISE_ALONE = 0.1010
V41 = HERE.parent / "wear-allowance" / "wear-anchor" / "campaign-matched"
STAGE = HERE / "wear-noise" / "campaign"


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def ms(rows, get):
    x = np.array([get(r) for r in rows], dtype=float)
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else 0.0)


def summarize(rows, d_params):
    s = dict(n=len(rows))
    s["risk"] = ms(rows, lambda r: r["stochastic"]["risk"])
    s["U"] = ms(rows, lambda r: r["stochastic"]["U"])
    s["risk_mean_action"] = ms(rows, lambda r: r["mean_action"]["risk"])
    s["A"] = ms(rows, lambda r: r["action_kl"]["A"])
    s["distance"] = ms(rows, lambda r: r["distance"])
    s["spread"] = ms(rows, lambda r: r["distance"] / math.sqrt(d_params))
    s["sigma_a"] = ms(rows, lambda r: math.exp(r["log_std"]))
    s["train_wall"] = ms(rows, lambda r: r["train_wall"])
    return s


def sci(x):
    if x >= 100:
        return f"{x:.0f}"
    if x >= 1:
        return f"{x:.2g}"
    e = int(math.floor(math.log10(x))); m = x / 10 ** e
    return f"{m:.1f}\\times10^{{{e}}}"


def main():
    configs = {}
    for k1 in SWEEP:
        configs[("sum", k1)] = load(str(STAGE / f"k1_{k1}__k2_{K2:g}__seed*/result.json"))
    configs[("noise", NOISE_ALONE)] = load(str(STAGE / f"k1_{NOISE_ALONE}__k2_0__seed*/result.json"))
    pull = load(str(V41 / "kappa10__seed9400*/result.json"))
    control = load(str(V41 / "kappa0__seed9400*/result.json"))
    any_new = next((r for rows in configs.values() for r in rows), None)
    d_params = any_new["n_actor_params"] if any_new else 4418
    for rows in configs.values():
        for r in rows:
            assert r["n_actor_params"] == d_params and r["arm"]["m"] == 0.03 and r["arm"]["incumbent"]["std"] == 0.1

    summ = {"pull": summarize(pull, d_params), "control": summarize(control, d_params)}
    summ["pull"].update(kappa1=0.0, kappa2=K2); summ["control"].update(kappa1=0.0, kappa2=0.0)
    for (kind, k1), rows in configs.items():
        if rows:
            summ[f"{kind}_{k1}"] = dict(summarize(rows, d_params), kappa1=k1, kappa2=(K2 if kind == "sum" else 0.0),
                                        nominal=(NOMINAL[k1] if kind == "sum" else None),
                                        reference=(REFERENCE[k1] if kind == "sum" else None))
    sums = [summ[f"sum_{k1}"] for k1 in SWEEP if f"sum_{k1}" in summ]
    done = {k: v["n"] for k, v in summ.items()}
    print("runs per configuration:", done)

    # ---- predictions
    checks = {}
    if len(sums) == len(SWEEP):
        sp = [s["spread"][0] for s in sums]; nom = [s["nominal"] for s in sums]
        mono = all(a < b for a, b in zip(sp, sp[1:]))
        within = all(1 / 3 <= o / n <= 3 for o, n in zip(sp, nom) if n <= 0.1)
        checks["R1"] = dict(holds=bool(mono and within), monotone=bool(mono), within_factor_three=bool(within), spread=sp, nominal=nom, ratio=[o / n for o, n in zip(sp, nom)], spread_over_kappa1=[o / s_["kappa1"] for o, s_ in zip(sp, sums)])
        risk = [summ["pull"]["risk"][0]] + [s["risk"][0] for s in sums]; U = [summ["pull"]["U"][0]] + [s["U"][0] for s in sums]
        checks["R2"] = dict(holds=bool(all(a < b for a, b in zip(risk, risk[1:])) and all(a < b for a, b in zip(U, U[1:]))),
                            risk=risk, U=U, control_risk=summ["control"]["risk"][0], control_U=summ["control"]["U"][0])
        A = [summ["pull"]["A"][0]] + [s["A"][0] for s in sums]
        checks["R4"] = dict(holds=bool(all(a < b for a, b in zip(A, A[1:]))), A=A)
    if f"noise_{NOISE_ALONE}" in summ:
        n = summ[f"noise_{NOISE_ALONE}"]; c = summ["control"]
        checks["R3"] = dict(holds=bool(n["risk"][0] >= 0.5 * c["risk"][0]), risk=n["risk"][0], control_risk=c["risk"][0],
                            U=n["U"][0], control_U=c["U"][0], A=n["A"][0], control_A=c["A"][0],
                            A_larger=bool(n["A"][0] > c["A"][0]), distance=n["distance"][0], control_distance=c["distance"][0])
    for k, v in checks.items():
        print(k, "holds" if v["holds"] else "FALSIFIED", {kk: vv for kk, vv in v.items() if kk != "holds"})

    # ---- figure: (a) q_beta, (b) U against kappa_1 at the pull; (c) observed against nominal spread
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.2), gridspec_kw=dict(wspace=0.5))
    x0 = 1e-4   # position of kappa_1 = 0 (pull alone) on the log axis
    def seeds(rows, key):
        return [r["stochastic"][key] for r in rows]
    for ax, key, lab, scale in [(axes[0], "risk", "breakdown probability $q_\\beta$ (\\%)", 100.0), (axes[1], "U", "stopped utility $U$", 1.0)]:
        xs = [x0] + [s["kappa1"] for s in sums]; ys = [summ["pull"][key][0] * scale] + [s[key][0] * scale for s in sums]
        ax.plot(xs, ys, color=PAL[0], marker="D", ms=3.5, lw=1.0, label="pull $\\kappa_2=10$ with noise $\\kappa_1$, mean")
        for x, rows in [(x0, pull)] + [(k1, configs[("sum", k1)]) for k1 in SWEEP]:
            ax.plot([x] * len(rows), [v * scale for v in seeds(rows, key)], color=PAL[0], marker=".", ms=3, lw=0, alpha=0.5)
        ax.axhline(summ["control"][key][0] * scale, color=MUTED, ls="--", lw=0.8, label="unanchored control, mean")
        if f"noise_{NOISE_ALONE}" in summ:
            rows = configs[("noise", NOISE_ALONE)]
            ax.plot([NOISE_ALONE] * len(rows), [v * scale for v in seeds(rows, key)], color=PAL[1], marker=".", ms=3, lw=0, alpha=0.5)
            ax.plot([NOISE_ALONE], [summ[f"noise_{NOISE_ALONE}"][key][0] * scale], color=PAL[1], marker="o", ms=4, lw=0, label="noise alone, $\\kappa_2=0$, mean")
        ax.set_xscale("log"); ax.set_xlabel("noise strength $\\kappa_1$"); ax.set_ylabel(lab)
        ax.set_xticks([x0, 1e-3, 1e-2, 1e-1, 1]); ax.set_xticklabels(["$0$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
    legend_below(fig, *axes[0].get_legend_handles_labels(), ncol=1, axes=axes[:2])
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    ax = axes[2]
    if sums:
        k1 = np.array([s["kappa1"] for s in sums]); sp = np.array([s["spread"][0] for s in sums]); se = np.array([s["spread"][1] for s in sums])
        grid = np.geomspace(5e-4, 2.0, 100)
        ax.plot(grid, np.sqrt(grid / K2), color=MUTED, lw=0.8, ls=":", label="$\\sqrt{\\kappa_1/\\kappa_2}$, gradient-pull reference")
        ax.plot(grid, np.sqrt(2.0) * grid, color=MUTED, lw=0.8, ls="--", label="$\\sqrt{2}\\,\\kappa_1$, sign-drift reference")
        ax.errorbar(k1, sp, yerr=se, color=PAL[0], marker="D", ms=3.5, lw=0, elinewidth=1.0, capsize=0, label="pull with noise")
        if f"noise_{NOISE_ALONE}" in summ:
            n = summ[f"noise_{NOISE_ALONE}"]; ax.errorbar([NOISE_ALONE], [n["spread"][0]], yerr=[n["spread"][1]], color=PAL[1], marker="o", ms=4, lw=0, elinewidth=1.0, capsize=0, label="noise alone")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(5e-4, 2.0); ax.set_ylim(5e-4, 5.0)
        ax.set_xlabel("noise strength $\\kappa_1$"); ax.set_ylabel("per-parameter spread")
        legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax])
    panel_label(ax, "c")
    save(fig, "wear-noise-v60", HERE)

    # ---- table
    def row(label, s):
        k1 = f"{s['kappa1']:g}"; k2 = f"{s['kappa2']:g}"
        nom = f"{s['reference']:.3g}" if s.get("reference") else "--"
        return (f"{label} & ${k1}$ & ${k2}$ & {nom} & ${100 * s['risk'][0]:.2f}\\pm{100 * s['risk'][1]:.2f}$ & "
                f"${100 * s['risk_mean_action'][0]:.2f}$ & ${s['U'][0]:.2f}\\pm{s['U'][1]:.2f}$ & ${sci(s['A'][0])}$ & "
                f"${sci(s['spread'][0])}$ & ${s['sigma_a'][0]:.2f}$ \\\\")
    lines = ["\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
             "Configuration & $\\kappa_1$ & $\\kappa_2$ & nominal & $q_\\beta$ (\\%) & $q_\\beta^{\\mathrm{mean}}$ (\\%) & $U$ & $\\mathcal A$ & spread & $\\sigma_a$ \\\\",
             "\\midrule", row("Pull alone", summ["pull"])]
    lines += [row("Pull with noise", s) for s in sums]
    if f"noise_{NOISE_ALONE}" in summ:
        lines.append(row("Noise alone", summ[f"noise_{NOISE_ALONE}"]))
    lines += [row("Unanchored control", summ["control"]), "\\bottomrule", "\\end{tabular}"]
    (HERE / "wear-noise-v60.tex").write_text("\n".join(lines) + "\n")
    (HERE / "wear-noise-v60.json").write_text(json.dumps(dict(summary=summ, checks=checks, d_params=d_params), indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
