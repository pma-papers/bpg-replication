"""Revision 68: analysis of the decoupled-pull campaign (PROTOCOL-decoupled-v68.md, part A).

Reads wear-noise-decoupled-v68/campaign/k1_<k1>__k2_<k2>__seed<S>/result.json (eight seeds per row: kappa_2 in {10, 1}
with kappa_1 = 0.101, and pull alone kappa_1 = 0 at both pulls), and for comparison the coupled rows of
wear-noise-v60.json (the pull with noise at kappa_2 = 10, pull alone, the unanchored control). Per row it reports the
mean and between-seed standard deviation of q_beta and U of the stochastic policy, the mean-action q_beta, the action
relative entropy A, the distance ||theta - theta_0||, the per-parameter spread ||theta - theta_0|| / sqrt(d) and the
final dispersion, the predicted spread sqrt(kappa_1 / kappa_2), and the checks (D1)-(D3) and the (O16) criterion of
the protocol. Writes wear-noise-decoupled-v68.json, wear-noise-decoupled-v68.tex and wear-noise-decoupled-v68.pdf/png.
Runtime: ~/.venvs/constrained-rl/bin/python (numpy, matplotlib).
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
from plotstyle import PAL, MUTED, TEXTW, panel_label, legend_below, save  # noqa: E402

K1 = 0.101
ROWS = [("A1", 10.0, K1), ("A2", 1.0, K1), ("A3", 10.0, 0.0), ("A4", 1.0, 0.0)]
STAGE = HERE / "wear-noise-decoupled-v68" / "campaign"
COUPLED = HERE / "wear-noise-v60.json"


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def ms(rows, get):
    x = np.array([get(r) for r in rows], dtype=float)
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else 0.0)


def summarize(rows, d):
    s = dict(n=len(rows))
    s["risk"] = ms(rows, lambda r: r["stochastic"]["risk"]); s["U"] = ms(rows, lambda r: r["stochastic"]["U"])
    s["risk_mean_action"] = ms(rows, lambda r: r["mean_action"]["risk"]); s["A"] = ms(rows, lambda r: r["action_kl"]["A"])
    s["distance"] = ms(rows, lambda r: r["distance"]); s["spread"] = ms(rows, lambda r: r["distance"] / math.sqrt(d))
    s["sigma_a"] = ms(rows, lambda r: math.exp(r["log_std"])); s["train_wall"] = ms(rows, lambda r: r["train_wall"])
    s["optimizer_steps"] = ms(rows, lambda r: r["hook"]["optimizer_steps"])
    s["per_seed"] = [dict(seed=r["seed"], risk=r["stochastic"]["risk"], U=r["stochastic"]["U"], risk_mean_action=r["mean_action"]["risk"], A=r["action_kl"]["A"],
                          distance=r["distance"], spread=r["distance"] / math.sqrt(d), log_std=r["log_std"], train_wall=r["train_wall"]) for r in rows]
    return s


def sci(x):
    if x == 0:
        return "0"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 1:
        return f"{x:.2g}"
    e = int(math.floor(math.log10(x))); m = x / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.1f}\\times10^{{{e}}}"


def main():
    coupled = json.load(open(COUPLED))["summary"]
    d = json.load(open(COUPLED))["d_params"]
    summ = {}
    for label, k2, k1 in ROWS:
        rows = load(str(STAGE / f"k1_{k1:g}__k2_{k2:g}__seed*/result.json"))
        for r in rows:
            assert r["n_actor_params"] == d and r["arm"]["pull_form"] == "decoupled" and r["arm"]["m"] == 0.03
        if rows:
            summ[label] = dict(summarize(rows, d), kappa1=k1, kappa2=k2, form="decoupled",
                               predicted=(math.sqrt(k1 / k2) if k1 > 0 else 0.0), predicted_pull_in_noise=(math.sqrt(k1 / (k1 + k2)) if k1 > 0 else 0.0),
                               laplace_coupled=math.sqrt(2) * k1, envelope=1.0 / k2)
    print("runs per row:", {k: v["n"] for k, v in summ.items()})

    checks = {}
    if all(k in summ for k in ("A1", "A3")):
        s1, s0 = summ["A1"]["spread"][0], summ["A3"]["spread"][0]
        net = math.sqrt(max(s1 ** 2 - s0 ** 2, 0.0))
        checks["D1"] = dict(holds=bool(0.077 <= net <= 0.131) if s0 <= 0.077 else None, decidable=bool(s0 <= 0.077), spread_A1=s1, spread_pull_alone_A3=s0,
                            net_spread=net, predicted=summ["A1"]["predicted"], ratio_net_over_predicted=net / summ["A1"]["predicted"], ratio_raw_over_predicted=s1 / summ["A1"]["predicted"])
    if all(k in summ for k in ("A1", "A2")):
        ratio = summ["A2"]["spread"][0] / summ["A1"]["spread"][0]
        checks["D2"] = dict(holds=bool(2.0 <= ratio <= 4.0), ratio=ratio, particle_form=math.sqrt(10.0), coupled_laplace=1.0,
                            spread_A2=summ["A2"]["spread"][0], predicted_A2=summ["A2"]["predicted"], ratio_A2_over_predicted=summ["A2"]["spread"][0] / summ["A2"]["predicted"])
        if "A4" in summ:
            s0 = summ["A4"]["spread"][0]; net = math.sqrt(max(summ["A2"]["spread"][0] ** 2 - s0 ** 2, 0.0))
            checks["D2"].update(spread_pull_alone_A4=s0, net_spread_A2=net, ratio_net_over_predicted=net / summ["A2"]["predicted"])
        o16 = (0.091 <= summ["A1"]["spread"][0] <= 0.172) and (0.091 <= summ["A2"]["spread"][0] <= 0.172) and ratio < 1.5
        checks["O16_survives"] = dict(holds=bool(o16), spreads=[summ["A1"]["spread"][0], summ["A2"]["spread"][0]], ratio=ratio, band=[0.091, 0.172])
    if all(k in summ for k in ("A3", "A4")):
        a, b = summ["A3"]["spread"][0], summ["A4"]["spread"][0]
        checks["D3"] = dict(holds=bool(a <= 0.1 and b <= 1.0 and a > coupled["pull"]["spread"][0] and b > a), spread_A3=a, spread_A4=b, envelopes=[0.1, 1.0],
                            coupled_pull_alone=coupled["pull"]["spread"][0])
    for k, v in checks.items():
        print(k, v.get("holds"), {kk: vv for kk, vv in v.items() if kk != "holds"})

    # ---- figure: (a) spread against kappa_2 with the predictions; (b) risk, (c) U per seed by row
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.3), gridspec_kw=dict(wspace=0.55))
    ax = axes[0]
    grid = np.geomspace(0.5, 20, 100)
    ax.plot(grid, np.sqrt(K1 / grid), color=MUTED, lw=0.8, ls=":", label="$\\sqrt{\\kappa_1/\\kappa_2}$, particle form")
    ax.axhline(math.sqrt(2) * K1, color=MUTED, lw=0.8, ls="--", label="$\\sqrt{2}\\,\\kappa_1$, coupled through Adam")
    cs = coupled["sum_0.101"]
    ax.errorbar([10.0], [cs["spread"][0]], yerr=[cs["spread"][1]], color=PAL[1], marker="s", ms=4, lw=0, elinewidth=1.0, label="coupled, $\\kappa_1=0.101$ (recorded)")
    for label, color, marker, name in [("A1", PAL[0], "D", "decoupled, $\\kappa_1=0.101$"), ("A2", PAL[0], "D", None), ("A3", PAL[2], "o", "decoupled, pull alone"), ("A4", PAL[2], "o", None)]:
        if label in summ:
            s = summ[label]; ax.errorbar([s["kappa2"]], [s["spread"][0]], yerr=[s["spread"][1]], color=color, marker=marker, ms=4, lw=0, elinewidth=1.0, label=name)
            ax.plot([s["kappa2"]] * s["n"], [p["spread"] for p in s["per_seed"]], color=color, marker=".", ms=3, lw=0, alpha=0.4)
    ax.plot([10.0], [coupled["pull"]["spread"][0]], color=PAL[1], marker="s", ms=4, lw=0, mfc="none", label="coupled, pull alone (recorded)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.5, 20); ax.set_ylim(1e-4, 2.0)
    ax.set_xticks([1, 10]); ax.set_xticklabels(["$1$", "$10$"]); ax.set_xlabel("pull $\\kappa_2$"); ax.set_ylabel("per-parameter spread")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax])
    panel_label(ax, "a")
    order = [("A3", "pull\nalone\n$\\kappa_2{=}10$"), ("A1", "with\nnoise\n$\\kappa_2{=}10$"), ("A4", "pull\nalone\n$\\kappa_2{=}1$"), ("A2", "with\nnoise\n$\\kappa_2{=}1$")]
    for ax, key, lab, scale in [(axes[1], "risk", "breakdown probability $q_\\beta$ (\\%)", 100.0), (axes[2], "U", "stopped utility $U$", 1.0)]:
        xs, labels = [], []
        for i, (label, name) in enumerate(order):
            if label in summ:
                s = summ[label]; ys = [p[key] * scale for p in s["per_seed"]]
                ax.plot([i] * len(ys), ys, color=PAL[0], marker=".", ms=3, lw=0, alpha=0.5); ax.plot([i], [s[key][0] * scale], color=PAL[0], marker="D", ms=3.5, lw=0)
            xs.append(i); labels.append(name)
        ax.plot([1.0], [cs[key][0] * scale], color=PAL[1], marker="s", ms=4, lw=0, label="coupled, recorded mean")
        ax.plot([1.0] * len(cs.get("per_seed", [])), [p[key] * scale for p in cs.get("per_seed", [])], color=PAL[1], marker=".", ms=3, lw=0, alpha=0.4)
        ax.axhline(coupled["control"][key][0] * scale, color=MUTED, ls="--", lw=0.8, label="unanchored control, mean")
        ax.axhline(coupled["pull"][key][0] * scale, color=MUTED, ls=":", lw=0.8, label="coupled pull alone, mean")
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6); ax.set_ylabel(lab); ax.set_xlim(-0.5, 3.5)
    legend_below(fig, *axes[1].get_legend_handles_labels(), ncol=1, axes=axes[1:])
    panel_label(axes[1], "b"); panel_label(axes[2], "c")
    save(fig, "wear-noise-decoupled-v68", HERE)

    # ---- table (the coupled rows for comparison)
    def row(label, s, form, pred):
        return (f"{label} & {form} & ${s['kappa1']:g}$ & ${s['kappa2']:g}$ & {pred} & ${100 * s['risk'][0]:.2f}\\pm{100 * s['risk'][1]:.2f}$ & "
                f"${100 * s['risk_mean_action'][0]:.2f}$ & ${s['U'][0]:.2f}\\pm{s['U'][1]:.2f}$ & ${sci(s['A'][0])}$ & "
                f"${sci(s['spread'][0])}$ & ${s['sigma_a'][0]:.2f}$ \\\\")
    lines = ["\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{llrrrrrrrrr}", "\\toprule",
             "Configuration & pull & $\\kappa_1$ & $\\kappa_2$ & $\\sqrt{\\kappa_1/\\kappa_2}$ & $q_\\beta$ (\\%) & $q_\\beta^{\\mathrm{mean}}$ (\\%) & $U$ & $\\mathcal A$ & spread & $\\sigma_a$ \\\\",
             "\\midrule", row("Pull alone", coupled["pull"], "coupled", "--"), row("Pull with noise", coupled["sum_0.101"], "coupled", "0.10")]
    for label, name in [("A3", "Pull alone"), ("A1", "Pull with noise"), ("A4", "Pull alone"), ("A2", "Pull with noise")]:
        if label in summ:
            s = summ[label]; lines.append(row(name, s, "decoupled", f"{s['predicted']:.2f}" if s["predicted"] else "--"))
    lines += [row("Unanchored control", coupled["control"], "--", "--"), "\\bottomrule", "\\end{tabular}"]
    (HERE / "wear-noise-decoupled-v68.tex").write_text("\n".join(lines) + "\n")
    (HERE / "wear-noise-decoupled-v68.json").write_text(json.dumps(dict(summary=summ, checks=checks, d_params=d, kappa1=K1,
                                                                        coupled_reference={k: {kk: vv for kk, vv in coupled[k].items()} for k in ("pull", "sum_0.101", "control")}), indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
