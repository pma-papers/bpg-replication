"""Revision 41: summary, table and figure of the lunar fuel-tank study, regenerated from the archived
revision-39 campaign (fuel-tank/lunar/campaign/*/result.json, lunar-protocol.json).

Changes against fuel-tank/lunar_analyze.py (review findings):
 * panel (a): price on a symlog axis; per-seed lines connect the configurations of one seed at tank 22 (paired
   design: every configuration of a seed starts from the same no-tank controller); seed means as large markers;
   tanks 22/21/20 in distinct hues; no line across the untested prices for tanks 21 and 20;
 * panel (b): the five prices in distinct hues; the adaptive configurations with dual step 5 (added after the
   campaign) and dual step 100 (fixed before the campaign) with distinct markers;
 * panel (c): legend entries name both dual steps;
 * paired per-seed differences of the exhaustion probability (lunar-paired-v41.json/.tex); caption numbers
   (lunar-caption-v41.json); post-registration rows marked in the table (lunar-table-v41.tex).
Outputs (this directory): lunar-campaign-v41 (pdf/png), lunar-table-v41.tex, lunar-paired-v41.json,
lunar-paired-v41.tex, lunar-caption-v41.json, lunar-summary-v41.json, lunar-digests-v41.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
V39 = ROOT.parent / "fuel-tank"
LUNAR = V39 / "lunar"
sys.path.insert(0, str(ROOT.parent / "lib"))
sys.path.insert(0, str(ROOT))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, MK, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402
from digest_v41 import digest  # noqa: E402

STAGE = "campaign"
TANK_COL = {22: PAL[0], 21: PAL[2], 20: PAL[3]}
PRICE_COL = {0.0: PAL[0], 0.05: PAL[2], 0.15: PAL[4], 0.5: PAL[6], 2.0: PAL[7]}
ALPHA_COL = {0.05: PAL[1], 0.02: PAL[3]}
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]   # fuel_quantiles in fuel-tank/lunar.py
DIGESTS = {}


def post_registration(cfg):
    """Configurations added after the campaign was registered (lunar-protocol.json, protocol_note)."""
    return (cfg["kind"] == "adaptive" and cfg["dual_step"] == 5) or cfg["budget"] == 20


def load():
    protocol = json.loads((V39 / "lunar-protocol.json").read_text())
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
        s = dict(arm=arm, cfg=cfg, n=len(rows), seeds=[r["seed"] for r in rows], steps=[r["steps"] for r in rows], post_registration=post_registration(cfg))
        for key in ("stochastic", "mean_action"):
            s[key] = {f: stat(rows, key, f) for f in ("risk", "landed", "crashed", "U", "length")}
            s[key]["fuel_median"] = float(np.mean([r[key]["fuel_quantiles"][2] for r in rows]))
            s[key]["fuel_quantiles_mean"] = np.mean([r[key]["fuel_quantiles"] for r in rows], axis=0).tolist()
            s[key]["fuel_median_per_seed"] = [r[key]["fuel_quantiles"][2] for r in rows]
            s[key]["length_per_seed"] = [r[key]["length"] for r in rows]
            s[key]["risk_per_seed"] = [r[key]["risk"] for r in rows]
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
            if s["post_registration"]:
                setting += "\\textsuperscript{$\\dagger$}"
            lines.append(row(label if first else "", setting, s)); first = False
    body = "\n".join(lines)
    tex = ("\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
           " & & \\multicolumn{5}{c}{stochastic policy} & \\multicolumn{2}{c}{mean action} \\\\\n"
           "\\cmidrule(lr){3-7}\\cmidrule(lr){8-9}\n"
           "Tank & Setting & exhausted (\\%) & landed (\\%) & crashed (\\%) & $U$ & fuel & exh.\\ (\\%) & landed (\\%) \\\\\n\\midrule\n"
           + body + "\n\\bottomrule\n"
           "\\multicolumn{9}{@{}l@{}}{\\textsuperscript{$\\dagger$}added after the campaign (protocol note).} \\\\\n\\end{tabular}\n")
    (ROOT / "lunar-table-v41.tex").write_text(tex)


def paired(runs):
    """Per-seed differences of the exhaustion probability, m = 0 minus the configuration, within one tank."""
    def by_seed(arm):
        return {r["seed"]: r["stochastic"]["risk"] for r in runs.get(arm, [])}
    comps = [("lunar-b22-m0", f"lunar-b22-m{m:g}", 22, f"$m={m:g}$", False) for m in (0.05, 0.15, 0.5, 2.0)]
    comps += [("lunar-b21-m0", "lunar-b21-m0.5", 21, "$m=0.5$", False), ("lunar-b20-m0", "lunar-b20-m0.5", 20, "$m=0.5$", True)]
    comps += [("lunar-b22-m0", "lunar-b22-adaptive-a0.05-s5", 22, "$\\alpha=0.05$, dual step 5", True),
              ("lunar-b22-m0", "lunar-b22-adaptive-a0.02-s5", 22, "$\\alpha=0.02$, dual step 5", True)]
    out = []
    for a0, a1, tank, label, post in comps:
        r0, r1 = by_seed(a0), by_seed(a1)
        seeds = sorted(set(r0) & set(r1))
        if not seeds:
            continue
        d = np.array([r0[s] - r1[s] for s in seeds])
        n = len(d); mean = float(d.mean()); sd = float(d.std(ddof=1)) if n > 1 else float("nan")
        t = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
        out.append(dict(tank=tank, reference=a0, configuration=a1, label=label, post_registration=post, seeds=seeds,
                        risk_reference=[r0[s] for s in seeds], risk_configuration=[r1[s] for s in seeds],
                        diff=d.tolist(), mean=mean, sd=sd, t=t, n_positive=int((d > 0).sum()), n=n))
    lines = []
    last = None
    for c in out:
        lab = f"$b={c['tank']}$" if c["tank"] != last else ""; last = c["tank"]
        setting = c["label"] + ("\\textsuperscript{$\\dagger$}" if c["post_registration"] else "")
        lines.append(f"{lab} & {setting} & ${100 * c['mean']:.1f}$ & ${100 * c['sd']:.1f}$ & ${c['t']:.2f}$ & ${c['n_positive']}/{c['n']}$ \\\\")
    tex = ("\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{llrrrr}\n\\toprule\n"
           "Tank & Configuration & mean difference (pp) & SD (pp) & $t$ & positive \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n"
           "\\multicolumn{6}{@{}p{0.92\\linewidth}@{}}{Per-seed difference of the exhaustion probability, $m=0$ minus the configuration, "
           "in percentage points over the five seeds; $t$ is the paired $t$ statistic ($n-1$ degrees of freedom); positive counts the seeds "
           "whose exhaustion probability fell. \\textsuperscript{$\\dagger$}added after the campaign (protocol note).} \\\\\n\\end{tabular}\n")
    (ROOT / "lunar-paired-v41.tex").write_text(tex)
    (ROOT / "lunar-paired-v41.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def caption_numbers(summary):
    out = {}
    s = summary.get("lunar-b22-m0.5")
    if s:
        out["crashed_b22_m0.5"] = dict(mean=s["stochastic"]["crashed"]["mean"], rounded="%.2f" % s["stochastic"]["crashed"]["mean"],
                                       per_seed=[r for r in s["stochastic"].get("crashed_per_seed", [])])
    u = summary.get("lunar-unlimited")
    if u:
        st = u["stochastic"]
        out["no_tank_fuel"] = dict(quantiles=QUANTILES, quantile_means_over_seeds=st["fuel_quantiles_mean"], median_per_seed=st["fuel_median_per_seed"],
                                   median_range="%.1f-%.1f" % (min(st["fuel_median_per_seed"]), max(st["fuel_median_per_seed"])),
                                   note="fuel_quantiles are the 10, 25, 50, 75 and 90 percent quantiles of the fuel used per episode (lunar.py)")
    out["episode_length"] = {}
    for arm in ("lunar-b22-m0.5", "lunar-b22-adaptive-a0.05-s5", "lunar-b22-adaptive-a0.02-s5", "lunar-b21-m0.5", "lunar-b20-m0.5", "lunar-unlimited", "lunar-b22-m0"):
        if arm in summary:
            st = summary[arm]["stochastic"]
            out["episode_length"][arm] = dict(mean_over_seeds=st["length"]["mean"], per_seed=st["length_per_seed"])
    out["median_exhaustion_frame"] = ("not available: result.json stores only the mean episode length over all episodes ('length') and "
                                      "the fuel-use quantiles ('fuel_quantiles'); no episode-level data")
    (ROOT / "lunar-caption-v41.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def figure(protocol, runs, summary):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.2), gridspec_kw=dict(wspace=0.55, width_ratios=[1.15, 1.25, 0.9]))
    TANK_MK = {22: "o", 21: "s", 20: "^"}
    fixed = {b: sorted([s for s in summary.values() if s["cfg"]["kind"] == "fixed" and s["cfg"]["budget"] == b], key=lambda s: s["cfg"]["m"]) for b in (22, 21, 20)}
    adapt22 = [s for s in summary.values() if s["cfg"]["kind"] == "adaptive" and s["cfg"]["budget"] == 22]
    adapt_slow = sorted([s for s in adapt22 if s["cfg"]["dual_step"] < 50], key=lambda s: -s["cfg"]["alpha"])
    adapt_fast = sorted([s for s in adapt22 if s["cfg"]["dual_step"] >= 50], key=lambda s: -s["cfg"]["alpha"])
    # (a) exhaustion probability against the price: symlog axis, paired seeds at tank 22
    ax = axes[0]
    ax.set_xscale("symlog", linthresh=0.05, linscale=0.5)
    arrs = []
    for b, dx in ((22, -3.0), (21, 0.0), (20, 3.0)):
        col = TANK_COL[b]; group = fixed[b]
        if not group:
            continue
        tr = offset_copy(ax.transData, fig=fig, x=dx, y=0, units="points")
        by_seed = {}
        for s in group:
            for r in runs[s["arm"]]:
                by_seed.setdefault(r["seed"], []).append((s["cfg"]["m"], r["stochastic"]["risk"]))
        for seed, pts in sorted(by_seed.items()):
            pts.sort()
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; arrs += [xs, ys]
            if b == 22:   # tested at every price: connect the configurations of one seed
                ax.plot(xs, ys, color=col, lw=0.6, alpha=0.4, marker=".", ms=2.5, transform=tr, zorder=2)
            else:         # only m = 0 and m = 0.5: no line across the untested prices
                ax.plot(xs, ys, color=col, ls="none", marker=".", ms=2.5, alpha=0.5, transform=tr, zorder=2)
        ax.plot([s["cfg"]["m"] for s in group], [s["stochastic"]["risk"]["mean"] for s in group], marker=TANK_MK[b], ms=5, mec="white", mew=0.6, color=col, ls="none", transform=tr, zorder=3)
    for a_ in adapt_slow:
        ax.axhline(a_["cfg"]["alpha"], color=MUTED, lw=0.7, ls=":")
    DIGESTS["lunar/a per-seed exhaustion probabilities"] = dict(after=digest(*arrs))
    prices = sorted({s["cfg"]["m"] for g in fixed.values() for s in g})
    ax.set_xticks(prices); ax.set_xticklabels([f"{m:g}" for m in prices], rotation=45, ha="right", rotation_mode="anchor"); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.015, 3.3)
    ax.set_xlabel("price of failure risk $m$"); ax.set_ylabel("fuel-exhaustion probability"); ax.set_ylim(-0.01, 0.5)
    handles = [Line2D([], [], marker=TANK_MK[b], ms=5, mec="white", mew=0.6, color=TANK_COL[b], ls="none", label=f"tank $b={b}$, seed mean") for b in (22, 21, 20) if fixed[b]]
    handles.append(Line2D([], [], color=TANK_COL[22], lw=0.6, alpha=0.6, marker=".", ms=2.5, label="one seed, $b=22$"))
    handles_a = handles; panel_label(ax, "a")
    # (b) stopped utility against exhaustion probability, tank 22: seeds (small) and means (large)
    ax = axes[1]
    if "lunar-unlimited" in summary:
        U0 = summary["lunar-unlimited"]["stochastic"]["U"]["mean"]
        ax.axhline(U0, color=TEXT2, lw=0.8, ls="--")
        ax.annotate("no tank", (0.01, U0), textcoords="offset points", xytext=(0, 3), fontsize=9, color=TEXT2, ha="left")
    series = [(s, PRICE_COL.get(s["cfg"]["m"], PAL[0]), MK[i], f"$m={s['cfg']['m']:g}$") for i, s in enumerate(fixed[22])]
    series += [(s, ALPHA_COL[s["cfg"]["alpha"]], "^", f"$\\alpha={s['cfg']['alpha']:g}$, dual step 5") for s in adapt_slow]
    series += [(s, ALPHA_COL[s["cfg"]["alpha"]], "D", f"$\\alpha={s['cfg']['alpha']:g}$, dual step 100") for s in adapt_fast]
    arrs = []
    for s, col, mk, lab in series:
        xs = [r["stochastic"]["risk"] for r in runs[s["arm"]]]; ys = [r["stochastic"]["U"] for r in runs[s["arm"]]]; arrs += [xs, ys]
        small, big = (2.4, 5.0) if mk == "D" else (2.8, 5.5)
        ax.plot(xs, ys, marker=mk, ms=small, color=col, alpha=0.5, ls="none")
        ax.plot([s["stochastic"]["risk"]["mean"]], [s["stochastic"]["U"]["mean"]], marker=mk, ms=big, mec="white", mew=0.7, color=col, ls="none", label=lab)
    DIGESTS["lunar/b per-seed (risk, U) at tank 22"] = dict(after=digest(*arrs))
    handles_b = [Line2D([], [], marker=mk, ms=5, mec="white", mew=0.6, color=col, ls="none", label=lab) for s, col, mk, lab in series if s["cfg"]["kind"] == "fixed"]
    ax.set_xlim(0, 1.03); ax.set_ylim(0, 250)
    ax.set_xlabel("fuel-exhaustion probability"); ax.set_ylabel("stopped utility $U$"); panel_label(ax, "b")
    # (c) multipliers of the adaptive configurations during training (unchanged data; legend names both dual steps)
    ax = axes[2]
    arrs = []
    for s, col, ls in zip(sorted(adapt22, key=lambda s: (s["cfg"]["dual_step"], -s["cfg"]["alpha"])), (PAL[1], PAL[3], PAL[4], PAL[5]), ("-", "--", "-.", ":")):
        col = ALPHA_COL[s["cfg"]["alpha"]]; mk = "^" if s["cfg"]["dual_step"] < 50 else "D"
        traces = [np.array([(t["steps"], t["multiplier"]) for t in r["trace"]]) for r in runs[s["arm"]]]
        n = min(len(t) for t in traces)
        steps = traces[0][:n, 0] / 1e6; M = np.array([t[:n, 1] for t in traces]); arrs += [steps, M]
        ax.plot(steps, np.maximum(M.mean(0), 1.0), color=col, ls=ls, marker=mk, markevery=(0.1, 0.25), ms=4, mec="white", mew=0.5,
                label=f"$\\alpha={s['cfg']['alpha']:g}$, dual step ${s['cfg']['dual_step']:g}$" + (" (added after the campaign)" if s["cfg"]["dual_step"] < 50 else ""))
        ax.fill_between(steps, np.maximum(M.min(0), 1.0), np.maximum(M.max(0), 1.0), color=col, alpha=0.18, lw=0)
    d = digest(*arrs); DIGESTS["lunar/c multiplier traces"] = dict(before=d, after=d, changed=False)
    ax.set_yscale("log"); ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("training steps ($10^6$)"); ax.set_ylabel("multiplier $\\nu$"); panel_label(ax, "c")
    handles_c = ax.get_legend_handles_labels()[0]
    legend_below(fig, handles_a + handles_b + handles_c, ncol=3)
    save(fig, "lunar-campaign-v41", ROOT)


def main():
    protocol, runs = load()
    summary = summarize(protocol, runs)
    for arm, s in summary.items():
        s["stochastic"]["crashed_per_seed"] = [r["stochastic"]["crashed"] for r in runs[arm]]
    (ROOT / "lunar-summary-v41.json").write_text(json.dumps(summary, indent=1) + "\n")
    table(summary)
    pairs = paired(runs)
    cap = caption_numbers(summary)
    figure(protocol, runs, summary)
    (ROOT / "lunar-digests-v41.json").write_text(json.dumps(DIGESTS, indent=1) + "\n")
    for arm, s in summary.items():
        st, ma = s["stochastic"], s["mean_action"]
        extra = f" nu={s['final_multiplier']['mean']:.1f}+-{s['final_multiplier']['sd']:.1f}" if "final_multiplier" in s else ""
        print(f"{arm:28s} n={s['n']} risk={100 * st['risk']['mean']:5.1f}+-{100 * st['risk']['sd']:4.1f} landed={100 * st['landed']['mean']:5.1f} "
              f"crashed={100 * st['crashed']['mean']:5.1f} U={st['U']['mean']:6.1f}+-{st['U']['sd']:4.1f} fuel={st['fuel_median']:.1f} length={st['length']['mean']:.0f} | "
              f"mean-action risk={100 * ma['risk']['mean']:5.1f} landed={100 * ma['landed']['mean']:5.1f}{extra}" + ("  [post-registration]" if s["post_registration"] else ""))
    print("paired differences (m = 0 minus configuration, exhaustion probability):")
    for c in pairs:
        print(f"  b={c['tank']} {c['label']:28s} diffs={[round(100 * x, 1) for x in c['diff']]} mean={100 * c['mean']:.1f}pp sd={100 * c['sd']:.1f}pp t={c['t']:.2f} positive={c['n_positive']}/{c['n']}")
    print("caption numbers:", json.dumps(cap, indent=1))


if __name__ == "__main__":
    main()
