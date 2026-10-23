"""Summary, table and figure of the lunar fuel-tank study (reads lunar/campaign/*/result.json)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

STAGE = sys.argv[1] if len(sys.argv) > 1 else "campaign"
LUNAR = ROOT / "lunar" if (ROOT / "lunar").exists() else ROOT.parent / "wear-allowance" / "lunar"
C_FIXED, C_ADAPT, C_TIGHT = PAL[0], PAL[1], PAL[2]


def load():
    protocol = json.loads((ROOT / "lunar-protocol.json").read_text())
    runs = {}
    for f in sorted((LUNAR / STAGE).glob("*/result.json")):
        arm = f.parent.name.split("__seed")[0]
        runs.setdefault(arm, []).append(json.loads(f.read_text()))
    return protocol, runs


def stat(rows, key, field):
    x = np.array([r[key][field] for r in rows], dtype=float)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0, min=float(x.min()), max=float(x.max()), n=len(x))


def summarize(protocol, runs):
    out = {}
    for arm, rows in runs.items():
        cfg = protocol["arms"][arm]
        s = dict(arm=arm, cfg=cfg, n=len(rows), steps=[r["steps"] for r in rows])
        for key in ("stochastic", "mean_action"):
            s[key] = {f: stat(rows, key, f) for f in ("risk", "landed", "crashed", "U", "length")}
            s[key]["fuel_median"] = float(np.mean([r[key]["fuel_quantiles"][2] for r in rows]))
        if cfg["kind"] == "adaptive":
            s["final_multiplier"] = stat([dict(x=dict(v=r["final_multiplier"])) for r in rows], "x", "v")
        out[arm] = s
    return out


def table(summary):
    def row(label, setting, s):
        st, ma = s["stochastic"], s["mean_action"]
        return (f"{label} & {setting} & ${100 * st['risk']['mean']:.1f}\\pm{100 * st['risk']['sd']:.1f}$ & "
                f"${100 * st['landed']['mean']:.1f}$ & ${100 * st['crashed']['mean']:.1f}$ & "
                f"${st['U']['mean']:.0f}\\pm{st['U']['sd']:.0f}$ & ${st['fuel_median']:.1f}$ & "
                f"${100 * ma['risk']['mean']:.1f}$ & ${100 * ma['landed']['mean']:.1f}$ \\\\")
    lines = []
    if "lunar-unlimited" in summary:
        lines.append(row("No tank", "--", summary["lunar-unlimited"]))
    for b, label in ((22, "Tank $b=22$"), (21, "Tank $b=21$"), (20, "Tank $b=20$")):
        first = True
        group = [s for s in summary.values() if s["cfg"]["budget"] == b]
        group.sort(key=lambda s: (0, s["cfg"]["m"], 0) if s["cfg"]["kind"] == "fixed" else (1, s["cfg"]["dual_step"], -s["cfg"]["alpha"]))
        for s in group:
            cfg = s["cfg"]
            if cfg["kind"] == "fixed":
                setting = f"$m={cfg['m']:g}$"
            else:
                setting = f"$\\alpha={cfg['alpha']:g}$, step ${cfg['dual_step']:g}$"
            lines.append(row(label if first else "", setting, s)); first = False
    body = "\n".join(lines)
    tex = ("\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
           " & & \\multicolumn{5}{c}{stochastic policy} & \\multicolumn{2}{c}{mean action} \\\\\n"
           "\\cmidrule(lr){3-7}\\cmidrule(lr){8-9}\n"
           "Tank & Setting & exhausted (\\%) & landed (\\%) & crashed (\\%) & $U$ & fuel & exh.\\ (\\%) & landed (\\%) \\\\\n\\midrule\n"
           + body + "\n\\bottomrule\n\\end{tabular}\n")
    (ROOT / "lunar-table.tex").write_text(tex)


def figure(protocol, runs, summary):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.25), gridspec_kw=dict(wspace=0.42, width_ratios=[1.0, 1.15, 1.0]))
    fixed22 = sorted([s for s in summary.values() if s["cfg"]["kind"] == "fixed" and s["cfg"]["budget"] == 22], key=lambda s: s["cfg"]["m"])
    fixed21 = sorted([s for s in summary.values() if s["cfg"]["kind"] == "fixed" and s["cfg"]["budget"] == 21], key=lambda s: s["cfg"]["m"])
    fixed20 = sorted([s for s in summary.values() if s["cfg"]["kind"] == "fixed" and s["cfg"]["budget"] == 20], key=lambda s: s["cfg"]["m"])
    adapt22 = [s for s in summary.values() if s["cfg"]["kind"] == "adaptive" and s["cfg"]["budget"] == 22]
    adapt_slow = [s for s in adapt22 if s["cfg"]["dual_step"] < 50]
    adapt_fast = [s for s in adapt22 if s["cfg"]["dual_step"] >= 50]
    # (a) exhaustion probability against the price (categorical positions)
    ax = axes[0]
    prices = sorted({s["cfg"]["m"] for s in fixed22 + fixed21 + fixed20})
    pos = {m: i for i, m in enumerate(prices)}
    for group, col, lab, dx in ((fixed22, C_FIXED, "tank $b=22$", -0.12), (fixed21, C_TIGHT, "tank $b=21$", 0.0), (fixed20, PAL[3], "tank $b=20$", 0.12)):
        for s in group:
            for r in runs[s["arm"]]:
                ax.plot([pos[s["cfg"]["m"]] + dx], [r["stochastic"]["risk"]], marker=".", ms=3, color=col, alpha=0.45, ls="none")
        ax.plot([pos[s["cfg"]["m"]] + dx for s in group], [s["stochastic"]["risk"]["mean"] for s in group], marker="o", ms=4, mec="white", mew=0.6, color=col, label=lab)
    for a_ in adapt_slow:
        ax.axhline(a_["cfg"]["alpha"], color=MUTED, lw=0.7, ls=":")
    ax.set_xticks(range(len(prices))); ax.set_xticklabels([f"{m:g}" for m in prices])
    ax.set_xlabel("price $m$"); ax.set_ylabel("fuel-exhaustion probability"); ax.set_ylim(-0.01, 0.5)
    ax.legend(loc="upper right", fontsize=6.5, borderaxespad=0.2); panel_label(ax, "a")
    # (b) stopped utility against exhaustion probability, tank 22: seeds (small) and means (large)
    ax = axes[1]
    if "lunar-unlimited" in summary:
        ax.axhline(summary["lunar-unlimited"]["stochastic"]["U"]["mean"], color=TEXT2, lw=0.8, ls="--")
        ax.annotate("no tank", (0.44, summary["lunar-unlimited"]["stochastic"]["U"]["mean"]), textcoords="offset points", xytext=(0, 3), fontsize=6.5, color=TEXT2, ha="right")
    from plotstyle import SEQ
    shade = {0.0: SEQ[1], 0.05: SEQ[2], 0.15: SEQ[3], 0.5: SEQ[5], 2.0: SEQ[6]}
    series = [(s, shade.get(s["cfg"]["m"], C_FIXED), "o", f"$m={s['cfg']['m']:g}$") for s in fixed22]
    series += [(s, C_ADAPT if s["cfg"]["alpha"] > 0.03 else PAL[3], "^", f"adaptive, $\\alpha={s['cfg']['alpha']:g}$") for s in sorted(adapt_slow, key=lambda s: -s["cfg"]["alpha"])]
    for s, col, mk, lab in series:
        xs = [r["stochastic"]["risk"] for r in runs[s["arm"]]]; ys = [r["stochastic"]["U"] for r in runs[s["arm"]]]
        ax.plot(xs, ys, marker=mk, ms=2.8, color=col, alpha=0.5, ls="none")
        ax.plot([s["stochastic"]["risk"]["mean"]], [s["stochastic"]["U"]["mean"]], marker=mk, ms=6, mec="white", mew=0.7, color=col, ls="none", label=lab)
    ax.legend(loc="lower right", fontsize=5.8, borderaxespad=0.2, labelspacing=0.25, handletextpad=0.4)
    ax.set_xlim(0, 0.45); ax.set_ylim(0, 250)
    ax.set_xlabel("fuel-exhaustion probability"); ax.set_ylabel("stopped utility $U$"); ax.set_title("tank $b=22$"); panel_label(ax, "b")
    # (c) multipliers of the adaptive arms during training
    ax = axes[2]
    for s, col in zip(sorted(adapt22, key=lambda s: (s["cfg"]["dual_step"], -s["cfg"]["alpha"])), (C_ADAPT, PAL[3], PAL[4], PAL[5])):
        traces = [np.array([(t["steps"], t["multiplier"]) for t in r["trace"]]) for r in runs[s["arm"]]]
        n = min(len(t) for t in traces)
        steps = traces[0][:n, 0] / 1e6; M = np.array([t[:n, 1] for t in traces])
        ax.plot(steps, np.maximum(M.mean(0), 1.0), color=col, label=f"tolerance ${s['cfg']['alpha']:g}$, step ${s['cfg']['dual_step']:g}$")
        ax.fill_between(steps, np.maximum(M.min(0), 1.0), np.maximum(M.max(0), 1.0), color=col, alpha=0.18, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("training steps ($10^6$)"); ax.set_ylabel("multiplier $\\nu$"); ax.legend(loc="lower right", fontsize=6.0, borderaxespad=0.2, labelspacing=0.3); panel_label(ax, "c")
    save(fig, "lunar-campaign", ROOT)


def main():
    protocol, runs = load()
    summary = summarize(protocol, runs)
    (ROOT / f"lunar-summary-{STAGE}.json").write_text(json.dumps(summary, indent=1) + "\n")
    table(summary)
    figure(protocol, runs, summary)
    for arm, s in summary.items():
        st, ma = s["stochastic"], s["mean_action"]
        extra = f" lambda={s['final_multiplier']['mean']:.1f}+-{s['final_multiplier']['sd']:.1f}" if "final_multiplier" in s else ""
        print(f"{arm:26s} n={s['n']} risk={100 * st['risk']['mean']:5.1f}+-{100 * st['risk']['sd']:4.1f} landed={100 * st['landed']['mean']:5.1f} "
              f"crashed={100 * st['crashed']['mean']:5.1f} U={st['U']['mean']:6.1f}+-{st['U']['sd']:4.1f} fuel={st['fuel_median']:.1f} | "
              f"mean-action risk={100 * ma['risk']['mean']:5.1f} landed={100 * ma['landed']['mean']:5.1f} U={ma['U']['mean']:6.1f}{extra}")


if __name__ == "__main__":
    main()
