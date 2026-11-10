"""Revision 41: summary, table and figure of the lunar fuel-tank study, regenerated from the archived
revision-39 campaign (fuel-tank/lunar/campaign/*/result.json, lunar-protocol.json).

Changes against fuel-tank/lunar_analyze.py (review findings):
 * panel (a): price on a symlog axis; per-seed lines connect the configurations of one seed at tank 22 (paired
   design: every configuration of a seed starts from the same unconstrained controller); seed means as large markers;
   tanks 22/21/20 in distinct hues; no line across the untested prices for tanks 21 and 20;
 * panel (b): the five prices in distinct hues; the adaptive configurations with dual step 5 (added after the
   campaign) and dual step 100 (fixed before the runs) with distinct markers;
 * panel (a): seed means with bars spanning the seeds (revision 75, 2026-11-09; the per-seed points and lines were dropped);
 * panel (c): legend entries name both dual steps;
 * revision 75 (2026-11-10), after the fifty added runs (lunar-protocol.json, additions_2026_11_10): the figure has two
   rows. Top row, the fixed prices: (a) tank 22 alone, (b) exhaustion without a price and at the price 0.5 at the five
   tanks, (c) the fall in exhaustion of every pair of a priced and an unpriced run against the price times the priced
   risk of the unpriced run, m q_{beta,H}(pi_0)/(1-beta) (the priced risks are those of lunar_reeval.py, files
   priced-risk.json). Bottom row, tank 22: (d) utility against exhaustion and (e) the multipliers, the former panels
   (b) and (c). A price has one colour and one marker in every panel;
 * tables of the price times the risk without a price (lunar-price-risk-v41.tex, by tank and price;
   lunar-amount-v41.tex, pair by pair); the two existing tables list the added configurations;
 * paired per-seed differences of the exhaustion probability (lunar-paired-v41.json/.tex); caption numbers
   (lunar-caption-v41.json); post-registration rows marked in the table (lunar-table-v41.tex).
Outputs (this directory): lunar-campaign-v41 (pdf/png), lunar-table-v41.tex, lunar-paired-v41.json,
lunar-paired-v41.tex, lunar-price-risk-v41.tex, lunar-amount-v41.tex, lunar-price-risk-v41.json,
lunar-caption-v41.json, lunar-summary-v41.json, lunar-digests-v41.json.
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
TANKS = (24, 23, 22, 21, 20)
BETA = 0.999
PRICE_COL = {0.0: MUTED, 0.05: PAL[2], 0.15: PAL[4], 0.5: PAL[6], 2.0: PAL[7]}   # one colour and one marker per price in every panel; m = 0 in grey
PRICE_MK = {0.0: "P", 0.05: "X", 0.15: "*", 0.5: "D", 2.0: "v"}
ADAPT_MK = {"slow": "h", "fast": "p"}                                             # the adaptive configurations in panels (d) and (e)
ALPHA_COL = {0.05: PAL[1], 0.02: PAL[3]}
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]   # fuel_quantiles in fuel-tank/lunar.py
DIGESTS = {}


def post_registration(cfg, arm=None, protocol=None):
    """Configurations added after the first runs (lunar-protocol.json: protocol_note and additions_2026_11_10)."""
    added = protocol["stages"][STAGE].get("additions_2026_11_10", {}).get("arms", []) if protocol else []
    return (cfg["kind"] == "adaptive" and cfg["dual_step"] == 5) or cfg["budget"] == 20 or arm in added


def load():
    protocol = json.loads((V39 / "lunar-protocol.json").read_text())
    runs = {}
    for f in sorted((LUNAR / STAGE).glob("*/result.json")):
        arm = f.parent.name.split("__seed")[0]
        r = json.loads(f.read_text())
        q = f.parent / "priced-risk.json"               # written by lunar_reeval.py for the fixed-price runs
        if q.exists():
            pr = json.loads(q.read_text())
            assert abs(pr["risk"] - r["stochastic"]["risk"]) < 1e-12 and abs(pr["U"] - r["stochastic"]["U"]) < 1e-6, f.parent.name
            r["priced_risk"] = pr["priced_risk"]
        runs.setdefault(arm, []).append(r)
    return protocol, runs


def stat(rows, key, field):
    x = np.array([r[key][field] for r in rows], dtype=float)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0, min=float(x.min()), max=float(x.max()), n=len(x))


def summarize(protocol, runs):
    out = {}
    for arm, rows in runs.items():
        cfg = protocol["arms"][arm]
        s = dict(arm=arm, cfg=cfg, n=len(rows), seeds=[r["seed"] for r in rows], steps=[r["steps"] for r in rows], post_registration=post_registration(cfg, arm, protocol))
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
        lines.append(row("No fuel limit", "--", summary["lunar-unlimited"]))
    for b, label in ((b, f"Tank $b={b}$") for b in TANKS):
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
           "\\multicolumn{9}{@{}l@{}}{\\textsuperscript{$\\dagger$}added after the runs.} \\\\\n\\end{tabular}\n")
    (ROOT / "lunar-table-v41.tex").write_text(tex)


def pairs_of(runs, tank, m):
    """Per seed: exhaustion, priced risk and utility of the unpriced and of the priced run at one tank."""
    r0 = {r["seed"]: r for r in runs.get(f"lunar-b{tank}-m0", [])}
    r1 = {r["seed"]: r for r in runs.get(f"lunar-b{tank}-m{m:g}", [])}
    out = []
    for seed in sorted(set(r0) & set(r1)):
        a, b = r0[seed], r1[seed]
        amount = m * a["priced_risk"] / (1 - BETA)                       # the price times the risk without a price
        value = (b["stochastic"]["U"] - m * b["priced_risk"] / (1 - BETA)) - (a["stochastic"]["U"] - amount)
        out.append(dict(tank=tank, m=m, seed=seed, risk0=a["stochastic"]["risk"], risk1=b["stochastic"]["risk"], fall=a["stochastic"]["risk"] - b["stochastic"]["risk"],
                        priced_risk0=a["priced_risk"], priced_risk1=b["priced_risk"], amount=amount, value_gain=value,
                        U0=a["stochastic"]["U"], U1=b["stochastic"]["U"]))
    return out


def paired(runs, summary):
    """Per-seed differences of the exhaustion probability, m = 0 minus the configuration, within one tank."""
    def by_seed(arm):
        return {r["seed"]: r["stochastic"]["risk"] for r in runs.get(arm, [])}
    comps = []
    for b in TANKS:
        for m in (0.05, 0.15, 0.5, 2.0):
            arm = f"lunar-b{b}-m{m:g}"
            if arm in runs:
                comps.append((f"lunar-b{b}-m0", arm, b, f"$m={m:g}$", summary[arm]["post_registration"], m))
    comps += [("lunar-b22-m0", "lunar-b22-adaptive-a0.05-s5", 22, "$\\alpha=0.05$, dual step 5", True, None),
              ("lunar-b22-m0", "lunar-b22-adaptive-a0.02-s5", 22, "$\\alpha=0.02$, dual step 5", True, None)]
    out = []
    for a0, a1, tank, label, post, m in comps:
        r0, r1 = by_seed(a0), by_seed(a1)
        seeds = sorted(set(r0) & set(r1))
        if not seeds:
            continue
        d = np.array([r0[s] - r1[s] for s in seeds])
        n = len(d); mean = float(d.mean()); sd = float(d.std(ddof=1)) if n > 1 else float("nan")
        t = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
        amount = float(np.mean([p["amount"] for p in pairs_of(runs, tank, m)])) if m is not None else None
        out.append(dict(tank=tank, reference=a0, configuration=a1, label=label, post_registration=post, seeds=seeds,
                        risk_reference=[r0[s] for s in seeds], risk_configuration=[r1[s] for s in seeds],
                        diff=d.tolist(), mean=mean, sd=sd, t=t, n_positive=int((d > 0).sum()), n=n, amount=amount))
    lines = []
    last = None
    for c in out:
        lab = f"$b={c['tank']}$" if c["tank"] != last else ""; last = c["tank"]
        setting = c["label"] + ("\\textsuperscript{$\\dagger$}" if c["post_registration"] else "")
        amount = f"${c['amount']:.0f}$" if c["amount"] is not None else "--"
        lines.append(f"{lab} & {setting} & {amount} & ${100 * c['mean']:.1f}$ & ${100 * c['sd']:.1f}$ & ${c['t']:.2f}$ & ${c['n_positive']}/{c['n']}$ \\\\")
    tex = ("\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{llrrrrr}\n\\toprule\n"
           "Tank & Configuration & $m\\,q_{\\beta,H}(\\pi_0)/(1-\\beta)$ & mean difference (pp) & SD (pp) & $t$ & positive \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n"
           "\\multicolumn{7}{@{}p{0.96\\linewidth}@{}}{Per-seed difference of the exhaustion probability, $m=0$ minus the configuration, "
           "in percentage points over the five seeds; $t$ is the paired $t$ statistic ($n-1$ degrees of freedom); positive counts the seeds "
           "whose exhaustion probability fell. The third column is the price times the priced risk of the policy $\\pi_0$ trained without a price "
           "at the same tank and seed, in units of return, as a mean over the seeds. \\textsuperscript{$\\dagger$}added after the runs.} \\\\\n\\end{tabular}\n")
    (ROOT / "lunar-paired-v41.tex").write_text(tex)
    (ROOT / "lunar-paired-v41.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def price_risk(runs, summary):
    """The price times the risk without a price: by tank and price, and pair by pair."""
    spread = {b: summary[f"lunar-b{b}-m0"]["stochastic"]["U"]["sd"] for b in TANKS}
    prices = (0.05, 0.15, 0.5, 2.0)
    cells, allpairs = {}, []
    for b in TANKS:
        for m in prices:
            ps = pairs_of(runs, b, m)
            if not ps:
                continue
            allpairs += ps
            cells[(b, m)] = dict(tank=b, m=m, n=len(ps), fall=float(np.mean([p["fall"] for p in ps])), n_fell=int(sum(p["fall"] > 0 for p in ps)),
                                 amount=float(np.mean([p["amount"] for p in ps])), value_gain=float(np.mean([p["value_gain"] for p in ps])),
                                 n_value_gain=int(sum(p["value_gain"] > 0 for p in ps)))
    lines = []
    for b in TANKS:
        s0 = summary[f"lunar-b{b}-m0"]["stochastic"]
        q0 = float(np.mean([r["priced_risk"] for r in runs[f"lunar-b{b}-m0"]]))
        row = [f"${b}$", f"${100 * s0['risk']['mean']:.1f}$", f"${q0:.3f}$"]
        for m in prices:
            c = cells.get((b, m))
            row.append(f"${100 * c['fall']:.1f}$ (${c['amount']:.0f}$)" if c else "--")
        lines.append(" & ".join(row) + " \\\\")
    tex = ("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
           " & \\multicolumn{2}{c}{Without a price} & \\multicolumn{4}{c}{Fall in exhaustion (pp) at the price $m$} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}\n"
           "Tank $b$ & exhausted (\\%) & $q_{\\beta,H}(\\pi_0)$ & $0.05$ & $0.15$ & $0.5$ & $2$ \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (ROOT / "lunar-price-risk-v41.tex").write_text(tex)
    low = [p for p in allpairs if p["m"] < 2]
    bins = [(0, 5, "below $5$"), (5, 10, "$5$ to $10$"), (10, 20, "$10$ to $20$"), (20, 40, "$20$ to $40$"), (40, 80, "$40$ to $80$"), (80, float("inf"), "above $80$")]
    rows, binned = [], []
    for lo, hi, lab in bins:
        g = [p for p in low if lo <= p["amount"] < hi]
        if not g:
            continue
        binned.append(dict(lo=lo, hi=hi if hi < float("inf") else None, n=len(g), fall=float(np.mean([p["fall"] for p in g])), n_fell=int(sum(p["fall"] > 0 for p in g))))
        rows.append(f"$m\\le0.5$ & {lab} & ${len(g)}$ & ${100 * binned[-1]['fall']:.1f}$ & ${binned[-1]['n_fell']}$ \\\\")
    g = [p for p in allpairs if p["m"] >= 2]
    high = dict(n=len(g), fall=float(np.mean([p["fall"] for p in g])), n_fell=int(sum(p["fall"] > 0 for p in g)), amount_min=min(p["amount"] for p in g), amount_max=max(p["amount"] for p in g),
                n_value_gain=int(sum(p["value_gain"] > 0 for p in g)))
    rows.append(f"$m=2$ & ${high['amount_min']:.0f}$ to ${high['amount_max']:.0f}$ & ${high['n']}$ & ${100 * high['fall']:.1f}$ & ${high['n_fell']}$ \\\\")
    tex = ("\\begin{tabular}{llrrr}\n\\toprule\n"
           "Prices & $m\\,q_{\\beta,H}(\\pi_0)/(1-\\beta)$ & pairs & mean fall in exhaustion (pp) & pairs with a fall \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (ROOT / "lunar-amount-v41.tex").write_text(tex)
    x = np.array([p["amount"] for p in low]); y = np.array([p["fall"] for p in low])
    rank = lambda v: np.argsort(np.argsort(v))
    slopes = {}
    for m in prices:
        g = [p for p in allpairs if p["m"] == m]
        slopes[f"{m:g}"] = float(np.polyfit([p["risk0"] for p in g], [p["fall"] for p in g], 1)[0])
    q0 = {b: float(np.mean([r["priced_risk"] for r in runs[f"lunar-b{b}-m0"]])) for b in TANKS}
    above = [p for p in low if p["amount"] > spread[p["tank"]]]
    below = [p for p in low if p["amount"] <= spread[p["tank"]]]
    side = lambda g: dict(n=len(g), n_fell=int(sum(p["fall"] > 0 for p in g)), fall=float(np.mean([p["fall"] for p in g])))
    # the price 2: value at that price of the policy trained at 0.5 minus that of the policy trained at 2, same tank and seed
    at2 = []
    for b in TANKS:
        r05 = {r["seed"]: r for r in runs.get(f"lunar-b{b}-m0.5", [])}; r2 = {r["seed"]: r for r in runs.get(f"lunar-b{b}-m2", [])}
        for seed in sorted(set(r05) & set(r2)):
            v = lambda r: r["stochastic"]["U"] - 2.0 * r["priced_risk"] / (1 - BETA)
            at2.append(dict(tank=b, seed=seed, difference=v(r05[seed]) - v(r2[seed])))
    U_free = summary["lunar-unlimited"]["stochastic"]["U"]["mean"]
    weight = float(np.mean([r["priced_risk"] / r["stochastic"]["risk"] for b in TANKS for r in runs[f"lunar-b{b}-m0"] if r["stochastic"]["risk"] > 0]))
    out = dict(note="amount = m q_{beta,H}(pi_0)/(1-beta): the price times the priced risk of the policy trained without a price at the same tank and seed, in units of return; "
                    "fall = exhaustion probability of the unpriced run minus that of the priced run; value_gain = U - m q_{beta,H}/(1-beta) of the priced run minus that of the unpriced run",
               spread_of_return_without_price=spread, priced_risk_without_price=q0, cells=[cells[k] for k in sorted(cells, key=lambda k: (-k[0], k[1]))],
               binned_prices_up_to_0_5=binned, price_2=high, rank_correlation_amount_fall_prices_up_to_0_5=float(np.corrcoef(rank(x), rank(y))[0, 1]),
               slope_of_fall_on_exhaustion_without_price=slopes,
               prices_up_to_0_5_amount_above_the_spread_of_the_tank=side(above), prices_up_to_0_5_amount_at_or_below_the_spread=side(below),
               value_at_price_2_trained_at_0_5_minus_trained_at_2=dict(n=len(at2), n_positive=int(sum(a["difference"] > 0 for a in at2)),
                                                                        mean_by_tank={b: float(np.mean([a["difference"] for a in at2 if a["tank"] == b])) for b in TANKS if any(a["tank"] == b for a in at2)}, pairs=at2),
               lowest_price_with_amount_at_the_spread={b: spread[b] * (1 - BETA) / q0[b] for b in TANKS},
               price_charging_one_episode_return=dict(return_without_fuel_limit=U_free, priced_risk_per_exhaustion=weight, price=U_free * (1 - BETA) / weight),
               pairs=allpairs)
    (ROOT / "lunar-price-risk-v41.json").write_text(json.dumps(out, indent=1) + "\n")
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
    out["median_exhaustion_frame"] = ("result.json stores only the mean episode length over all episodes ('length') and the fuel-use quantiles "
                                      "('fuel_quantiles'); the exhaustion frames of the fixed-price policies are in priced-risk.json of each run (lunar_reeval.py)")
    (ROOT / "lunar-caption-v41.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def figure(protocol, runs, summary, pr):
    fig = plt.figure(figsize=(TEXTW, 4.45))
    gs = fig.add_gridspec(2, 3, wspace=0.78, hspace=0.62, width_ratios=[1.0, 1.0, 1.1])
    axes = [fig.add_subplot(gs[0, k]) for k in range(3)] + [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    fixed = {b: sorted([s for s in summary.values() if s["cfg"]["kind"] == "fixed" and s["cfg"]["budget"] == b], key=lambda s: s["cfg"]["m"]) for b in TANKS}
    adapt22 = [s for s in summary.values() if s["cfg"]["kind"] == "adaptive" and s["cfg"]["budget"] == 22]
    adapt_slow = sorted([s for s in adapt22 if s["cfg"]["dual_step"] < 50], key=lambda s: -s["cfg"]["alpha"])
    adapt_fast = sorted([s for s in adapt22 if s["cfg"]["dual_step"] >= 50], key=lambda s: -s["cfg"]["alpha"])
    # (a) tank 22 alone: exhaustion probability against the price (symlog axis), seed means with bars spanning the seeds
    ax = axes[0]
    ax.set_xscale("symlog", linthresh=0.05, linscale=0.5)
    arrs = []
    for s in fixed[22]:
        m = s["cfg"]["m"]; x = [r["stochastic"]["risk"] for r in runs[s["arm"]]]; arrs += [[m], x]
        mean = s["stochastic"]["risk"]["mean"]
        ax.errorbar([m], [mean], yerr=[[mean - min(x)], [max(x) - mean]], fmt="none", ecolor=PRICE_COL[m], elinewidth=0.8, capsize=2, alpha=0.85, zorder=2)
        ax.plot([m], [mean], marker=PRICE_MK[m], ms=7.5 if PRICE_MK[m] == "*" else 5.5, mec="white", mew=0.6, color=PRICE_COL[m], ls="none", zorder=3)
    for a_ in adapt_slow:
        ax.axhline(a_["cfg"]["alpha"], color=MUTED, lw=0.7, ls=":")
    DIGESTS["lunar/a per-seed exhaustion probabilities at tank 22"] = dict(after=digest(*arrs))
    prices = [s["cfg"]["m"] for s in fixed[22]]
    ax.set_xticks(prices); ax.set_xticklabels([f"{m:g}" for m in prices], rotation=45, ha="right", rotation_mode="anchor"); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.015, 3.3)
    ax.set_xlabel("price $m$"); ax.set_ylabel("exhaustion probability"); ax.set_ylim(-0.01, 0.5); panel_label(ax, "a")
    # (b) the five tanks: exhaustion without a price and at the price 0.5, seeds (small) and means (large)
    ax = axes[1]
    arrs = []
    for b in TANKS:
        for m, dx in ((0.0, -0.17), (0.5, 0.17)):
            s = summary[f"lunar-b{b}-m{m:g}"]; x = [r["stochastic"]["risk"] for r in runs[s["arm"]]]; arrs += [[b, m], x]
            ax.plot([b - dx] * len(x), x, marker=PRICE_MK[m], ms=2.6, color=PRICE_COL[m], alpha=0.55, ls="none")
            ax.plot([b - dx], [s["stochastic"]["risk"]["mean"]], marker=PRICE_MK[m], ms=5.5, mec="white", mew=0.6, color=PRICE_COL[m], ls="none", zorder=3)
    DIGESTS["lunar/b per-seed exhaustion probabilities at the five tanks, prices 0 and 0.5"] = dict(after=digest(*arrs))
    ax.set_xticks(list(TANKS)); ax.set_xlim(24.6, 19.4)
    ax.set_xlabel("tank $b$"); ax.set_ylabel("exhaustion probability"); ax.set_ylim(-0.01, 0.5); panel_label(ax, "b")
    # (c) every pair of a priced and an unpriced run: fall in exhaustion against the price times the risk without a price
    ax = axes[2]
    spread = pr["spread_of_return_without_price"]
    ax.axvspan(min(spread.values()), max(spread.values()), color=MUTED, alpha=0.30, lw=0)
    ax.axhline(0, color=TEXT2, lw=0.6)
    FLOOR = -0.125
    arrs = []; clipped = []
    for m in (0.05, 0.15, 0.5, 2.0):
        g = [q for q in pr["pairs"] if q["m"] == m]
        x = [q["amount"] for q in g]; y = [q["fall"] for q in g]; arrs += [x, y]
        clipped += [q for q in g if q["fall"] < FLOOR]
        ax.plot(x, np.maximum(y, FLOOR), marker=PRICE_MK[m], ms=3.6 if PRICE_MK[m] != "*" else 5.0, mec="white", mew=0.3, color=PRICE_COL[m], alpha=0.9, ls="none", clip_on=False)
    DIGESTS["lunar/c pairs (price times risk without a price, fall in exhaustion)"] = dict(after=digest(*arrs))
    pr["pairs_drawn_at_the_lower_edge_of_panel_c"] = [dict(tank=q["tank"], m=q["m"], seed=q["seed"], fall=q["fall"]) for q in clipped]
    ax.set_xscale("log"); ax.set_xlim(0.8, 700); ax.set_ylim(FLOOR - 0.015, 0.33)
    ax.set_xlabel("$m\\,q_{\\beta,H}(\\pi_0)/(1-\\beta)$"); ax.set_ylabel("fall in exhaustion"); ax.set_xticks([1, 10, 100]); panel_label(ax, "c")
    # (d) stopped utility against exhaustion probability, tank 22: seeds (small) and means (large)
    ax = axes[3]
    if "lunar-unlimited" in summary:
        U0 = summary["lunar-unlimited"]["stochastic"]["U"]["mean"]
        ax.axhline(U0, color=TEXT2, lw=0.8, ls="--")
        ax.annotate("no fuel limit", (1.0, U0), textcoords="offset points", xytext=(0, 3), fontsize=FS, color=TEXT2, ha="right")
    series = [(s, PRICE_COL[s["cfg"]["m"]], PRICE_MK[s["cfg"]["m"]], f"$m={s['cfg']['m']:g}$") for s in fixed[22]]
    series += [(s, ALPHA_COL[s["cfg"]["alpha"]], ADAPT_MK["slow"], f"$\\alpha={s['cfg']['alpha']:g}$, dual step $5$") for s in adapt_slow]
    series += [(s, ALPHA_COL[s["cfg"]["alpha"]], ADAPT_MK["fast"], f"$\\alpha={s['cfg']['alpha']:g}$, dual step $100$") for s in adapt_fast]
    arrs = []
    for s, col, mk, lab in series:
        xs = [r["stochastic"]["risk"] for r in runs[s["arm"]]]; ys = [r["stochastic"]["U"] for r in runs[s["arm"]]]; arrs += [xs, ys]
        small, big = (2.4, 5.0) if mk in ("D", "h", "p") else (2.8, 5.5)
        ax.plot(xs, ys, marker=mk, ms=small, color=col, alpha=0.5, ls="none")
        ax.plot([s["stochastic"]["risk"]["mean"]], [s["stochastic"]["U"]["mean"]], marker=mk, ms=big, mec="white", mew=0.7, color=col, ls="none", label=lab)
    DIGESTS["lunar/d per-seed (risk, U) at tank 22"] = dict(after=digest(*arrs))
    handles_price = [Line2D([], [], marker=mk, ms=5, mec="white", mew=0.6, color=col, ls="none", label=lab) for s, col, mk, lab in series if s["cfg"]["kind"] == "fixed"]
    ax.set_xlim(0, 1.03); ax.set_ylim(0, 260)
    ax.set_xlabel("exhaustion probability"); ax.set_ylabel("stopped utility $U$"); panel_label(ax, "d")
    # (e) multipliers of the adaptive configurations during training
    ax = axes[4]
    arrs = []
    for s, ls in zip(sorted(adapt22, key=lambda s: (s["cfg"]["dual_step"], -s["cfg"]["alpha"])), ("-", "--", "-.", ":")):
        col = ALPHA_COL[s["cfg"]["alpha"]]; mk = ADAPT_MK["slow"] if s["cfg"]["dual_step"] < 50 else ADAPT_MK["fast"]
        traces = [np.array([(t["steps"], t["multiplier"]) for t in r["trace"]]) for r in runs[s["arm"]]]
        n = min(len(t) for t in traces)
        steps = traces[0][:n, 0] / 1e6; M = np.array([t[:n, 1] for t in traces]); arrs += [steps, M]
        ax.plot(steps, np.maximum(M.mean(0), 1.0), color=col, ls=ls, marker=mk, markevery=(0.1, 0.25), ms=4, mec="white", mew=0.5,
                label=f"$\\alpha={s['cfg']['alpha']:g}$, dual step ${s['cfg']['dual_step']:g}$")
        ax.fill_between(steps, np.maximum(M.min(0), 1.0), np.maximum(M.max(0), 1.0), color=col, alpha=0.18, lw=0)
    d = digest(*arrs); DIGESTS["lunar/e multiplier traces"] = dict(before=d, after=d, changed=False)
    ax.set_yscale("log"); ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("training steps ($10^6$)"); ax.set_ylabel("multiplier $m/(1-\\beta)$"); panel_label(ax, "e")
    handles_adapt = ax.get_legend_handles_labels()[0]
    # one legend for the five panels, in the free cell of the second row
    box = gs[1, 2].get_position(fig)
    fig.legend(handles_price + handles_adapt, [h.get_label() for h in handles_price + handles_adapt], loc="center left", bbox_to_anchor=(box.x0 - 0.045, (box.y0 + box.y1) / 2),
               frameon=False, fontsize=FS, handlelength=1.8, labelspacing=0.45, handletextpad=0.5)
    save(fig, "lunar-campaign-v41", ROOT)


def main():
    protocol, runs = load()
    summary = summarize(protocol, runs)
    for arm, s in summary.items():
        s["stochastic"]["crashed_per_seed"] = [r["stochastic"]["crashed"] for r in runs[arm]]
    (ROOT / "lunar-summary-v41.json").write_text(json.dumps(summary, indent=1) + "\n")
    table(summary)
    pairs = paired(runs, summary)
    pr = price_risk(runs, summary)
    cap = caption_numbers(summary)
    figure(protocol, runs, summary, pr)
    (ROOT / "lunar-price-risk-v41.json").write_text(json.dumps(pr, indent=1) + "\n")
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
