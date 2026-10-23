"""Revision 41: figure and table of the parameter-anchor study.

Reads the revision-39 campaigns (wear-allowance/wear-anchor/campaign: prior dispersion 0.37, mismatched;
campaign-meanonly), the revision-41 matched-prior campaign (wear-anchor/campaign-matched: prior dispersion 0.1,
the incumbent rule itself; kappa_p = 0 is the unanchored control), and the wear campaign's action-anchored and
unanchored configurations (wear-allowance/campaign) for comparison.

Changes against wear-allowance/anchor_analyze.py (review findings): matched-prior configurations with a distinct
marker in every panel and in the table; kappa_p = 0 on a symlog axis; collision-free labels with leader lines in
panel (c); exact optimum at m = 0.03 under the re-evaluated convention (the DP policy re-evaluated in the
simulator, as for the learned policies); house-rule labels. Missing or partial results are skipped; the counts
found are printed. Outputs: wear-anchor-v41 (pdf/png), wear-anchor-table-v41.tex, wear-anchor-summary-v41.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
V39 = ROOT.parent / "wear-allowance"
sys.path.insert(0, str(ROOT.parent / "solvable-flow"))   # regularizer_figure_v41.sci
WEAR = ROOT.parent / "wear-allowance"
sys.path.insert(0, str(ROOT.parent / "lib"))
sys.path.insert(0, str(ROOT))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from regularizer_figure_v41 import sci  # noqa: E402

BENCH = json.loads((WEAR / "benchmark.json").read_text())
C_FULL, C_MATCHED, C_ACTION, C_MEANONLY = PAL[0], PAL[2], PAL[1], PAL[3]
M_FULL, M_MATCHED, M_ACTION, M_MEANONLY = "o", "D", "^", "v"


def load(folder):
    runs, bad = {}, []
    for f in sorted(folder.glob("*/result.json")):
        try:
            r = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            bad.append(f.parent.name); continue
        k = float(f.parent.name.split("__")[0][5:])
        runs.setdefault(k, []).append(r)
    return runs, bad


def wear_arm(name):
    return [json.loads(f.read_text()) for f in sorted((WEAR / "campaign").glob(f"{name}__seed*/result.json"))]


def ms(rows, get):
    x = np.array([get(r) for r in rows], dtype=float)
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else 0.0), float(x.min()), float(x.max())


def prior_stats(rows):
    return dict(risk=ms(rows, lambda r: r["prior_policy"]["stochastic"]["risk"]), U=ms(rows, lambda r: r["prior_policy"]["stochastic"]["U"]),
                risk_mean=ms(rows, lambda r: r["prior_policy"]["mean_action"]["risk"]), U_mean=ms(rows, lambda r: r["prior_policy"]["mean_action"]["U"]))


def place_labels(fig, ax, items, avoid_xy, extra_boxes=()):
    """Greedy collision-free placement of point labels: tries offsets (points) in order, draws a leader line for the
    larger ones, and rejects positions that overlap other labels, any data point or the axes boundary."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(rend)
    pts = ax.transData.transform(np.array(avoid_xy)) if len(avoid_xy) else np.zeros((0, 2))
    boxes = [Bbox.from_bounds(px - 4, py - 4, 8, 8) for px, py in pts] + list(extra_boxes)
    occupied = []
    # candidates: close positions first, then a ladder of offsets (mostly upward, where panel (c) is empty) with leader lines
    near = [(7, 4), (7, -11), (-7, 4), (-7, -11), (0, 11), (0, -13)]
    far = sorted({(dx, dy) for dx in range(-80, 81, 8) for dy in (14, -16, 22, -24, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126, 134)},
                 key=lambda p: abs(p[0]) + abs(p[1]))
    tol = 3.0
    for x, y, text, color in items:
        placed = False
        for dx, dy in near + far:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            va = "bottom" if dy > 0 else "top"
            leader = dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=2.5) if abs(dx) + abs(dy) > 12 else None
            ann = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=9, color=color, ha=ha, va=va, arrowprops=leader, zorder=6)
            ann.update_positions(rend)
            bb = Text.get_window_extent(ann, rend).expanded(1.08, 1.15)   # the text box alone, not the leader line
            inside = axbb.x0 - tol <= bb.x0 and bb.x1 <= axbb.x1 + tol and axbb.y0 - tol <= bb.y0 and bb.y1 <= axbb.y1 + tol
            if inside and not any(bb.overlaps(o) for o in occupied) and not any(bb.overlaps(b) for b in boxes):
                occupied.append(bb); placed = True; break
            ann.remove()
        if not placed:
            print(f"place_labels: no free position for {text!r}; placed at the default offset")
            ax.annotate(text, (x, y), textcoords="offset points", xytext=near[0], fontsize=9, color=color, zorder=6)


def main():
    full, bad_full = load(V39 / "wear-anchor" / "campaign")
    meanonly, bad_mo = load(V39 / "wear-anchor" / "campaign-meanonly")
    matched, bad_m = load(ROOT / "wear-anchor" / "campaign-matched")
    counts = dict(full={k: len(v) for k, v in sorted(full.items())}, meanonly={k: len(v) for k, v in sorted(meanonly.items())},
                  matched={k: len(v) for k, v in sorted(matched.items())}, unreadable=bad_full + bad_mo + bad_m)
    print("results found: revision-39 campaign", counts["full"], "| mean-only", counts["meanonly"], "| matched (revision 41)", counts["matched"],
          f"({sum(counts['matched'].values())} of 32 expected)", "| unreadable:", counts["unreadable"])
    exact_row = [r for r in BENCH["rows"] if r["kind"] == "failure" and abs(r["m"] - 0.03) < 1e-9]
    exact = dict(q=exact_row[0]["nominal"]["risk"], U=exact_row[0]["nominal"]["U"], q_grid=exact_row[0]["grid"]["q"], U_grid=exact_row[0]["grid"]["U"]) if exact_row else None
    action = {eta: wear_arm(f"anchored-e{eta:g}") for eta in (0.01, 0.1, 1.0)}
    unanch = wear_arm("fixed-m0.03") or wear_arm("m0.03")
    rows_full = [r for rows in full.values() for r in rows]; rows_matched = [r for rows in matched.values() for r in rows]
    prior_full = prior_stats(rows_full) if rows_full else None
    prior_matched = prior_stats(rows_matched) if rows_matched else None
    ks_full, ks_matched = sorted(full), sorted(matched)
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.2), gridspec_kw=dict(wspace=0.5))
    # (a) risk and (b) utility against the anchor strength
    for ax, key, lab, panel in ((axes[0], "risk", "breakdown probability $q_\\beta$", "a"), (axes[1], "U", "stopped utility $U$", "b")):
        for data, col, mk, big, label in ((full, C_FULL, M_FULL, 4.5, "anchor, $\\sigma_a=0.37$"),
                                          (matched, C_MATCHED, M_MATCHED, 4.0, "matched, $\\sigma_a=0.1$")):
            ks = sorted(data)
            if not ks:
                continue
            for k in ks:
                for r in data[k]:
                    ax.plot([k], [r["stochastic"][key]], marker=mk, ms=2.6, color=col, alpha=0.45, ls="none")
            ax.plot(ks, [ms(data[k], lambda r: r["stochastic"][key])[0] for k in ks], marker=mk, ms=big, mec="white", mew=0.6, color=col, label=label)
        for k, rows in meanonly.items():
            for r in rows:
                ax.plot([k], [r["stochastic"][key]], marker=M_MEANONLY, ms=3, color=C_MEANONLY, alpha=0.5, ls="none")
            ax.plot([k], [ms(rows, lambda r: r["stochastic"][key])[0]], marker=M_MEANONLY, ms=5, mec="white", mew=0.6, color=C_MEANONLY, ls="none", label="mean-only prior")
        if prior_full:
            ax.axhline(prior_full[key][0], color=C_FULL, lw=0.8, ls=":")
        if prior_matched:
            ax.axhline(prior_matched[key][0], color=C_MATCHED, lw=0.8, ls=":")
        ax.set_ylim(-0.02, 0.43) if key == "risk" else ax.set_ylim(2.45, 5.5)
        if exact:
            ax.axhline(exact["q"] if key == "risk" else exact["U"], color=TEXT2, lw=0.8, ls="--")
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.45); ax.set_xticks([0, 1, 10, 100]); ax.set_xticklabels(["0", "1", "10", "100"])
        ax.xaxis.set_minor_locator(plt.NullLocator()); ax.set_xlim(-0.25, 300)
        ax.set_xlabel("anchor strength $\\kappa_p$"); ax.set_ylabel(lab); panel_label(ax, panel)
    # one legend for the figure (panel a): the action-anchored configurations appear in panel (c) only
    handles = [Line2D([], [], marker=M_FULL, color=C_FULL, ms=4.5, mec="white", mew=0.6, label="anchor, $\\sigma_a=0.37$"),
               Line2D([], [], marker=M_MATCHED, color=C_MATCHED, ms=4.0, mec="white", mew=0.6, label="matched, $\\sigma_a=0.1$"),
               Line2D([], [], marker=M_MEANONLY, ls="none", color=C_MEANONLY, ms=5, mec="white", mew=0.6, label="mean-only prior"),
               Line2D([], [], marker=M_ACTION, ls="none", color=C_ACTION, ms=4, mec="white", mew=0.5, label="action regularization, panel (c)")]
    legend_below(fig, handles, ncol=2)
    # (c) risk against the action relative entropy to the action reference: all anchors, labelled by hand
    ax = axes[2]

    def label(text, xy, dxy, color, ha="left", va="center"):
        ax.annotate(text, xy, textcoords="offset points", xytext=dxy, fontsize=FS, color=color, ha=ha, va=va, zorder=6)

    for k in ks_full:
        xs = [r["action_kl"]["A"] for r in full[k]]; ys = [r["stochastic"]["risk"] for r in full[k]]
        ax.plot(xs, ys, marker=M_FULL, ms=3.5, mec="white", mew=0.5, color=C_FULL, ls="none")
        label(f"$\\kappa_p={k:g}$", (float(np.mean(xs)), float(np.mean(ys))), {1.0: (7, 0), 10.0: (8, 4), 100.0: (8, -8)}.get(k, (7, 0)), C_FULL)
    for k in ks_matched:
        xs = [r["action_kl"]["A"] for r in matched[k]]; ys = [r["stochastic"]["risk"] for r in matched[k]]
        ax.plot(xs, ys, marker=M_MATCHED, ms=3.2, mec="white", mew=0.5, color=C_MATCHED, ls="none")
        if k in (0.0, 1.0):   # the two lower clusters are too close to the action-regularized ones for labels: the caption gives the order
            dxy, ha = {0.0: ((7, 0), "left"), 1.0: ((-8, 0), "right")}[k]
            label(f"$\\kappa_p={k:g}$ m.", (float(np.mean(xs)), float(np.mean(ys))), dxy, C_MATCHED, ha=ha)
    for eta, rows in action.items():
        if not rows:
            continue
        xs = [r["action_kl"]["A"] for r in rows]; ys = [r["stochastic"]["risk"] for r in rows]
        ax.plot(xs, ys, marker=M_ACTION, ms=3.5, mec="white", mew=0.5, color=C_ACTION, ls="none")
    for k, rows in meanonly.items():
        xs = [r["action_kl"]["A"] for r in rows]; ys = [r["stochastic"]["risk"] for r in rows]
        ax.plot(xs, ys, marker=M_MEANONLY, ms=3.5, mec="white", mew=0.5, color=C_MEANONLY, ls="none")
    if prior_full:
        ax.axhline(prior_full["risk"][0], color=C_FULL, lw=0.8, ls=":")
    if prior_matched:
        ax.axhline(prior_matched["risk"][0], color=C_MATCHED, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlim(2e-3, 3e5); ax.set_ylim(-0.03, 0.47)
    ax.set_xlabel("action relative entropy $\\mathcal{A}$"); ax.set_ylabel("breakdown probability $q_\\beta$")
    panel_label(ax, "c")   # markers as in the legend of panel (a)
    save(fig, "wear-anchor-v41", ROOT)
    # table
    lines = []

    def row(label, setting, rows, key="stochastic"):
        q = ms(rows, lambda r: r[key]["risk"]); U = ms(rows, lambda r: r[key]["U"]); A = ms(rows, lambda r: r["action_kl"]["A"])
        d = ms(rows, lambda r: r.get("distance", float("nan"))); sd = ms(rows, lambda r: math.exp(r["log_std"]))
        dist = f"${d[0]:.3f}$" if not math.isnan(d[0]) else "--"
        return f"{label} & {setting} & ${100 * q[0]:.2f}\\pm{100 * q[1]:.2f}$ & ${U[0]:.2f}\\pm{U[1]:.2f}$ & {sci(A[0]) if not math.isnan(A[0]) else '--'} & {dist} & ${sd[0]:.2f}$ \\\\"

    def prior_rows(rows, logstd):
        return [dict(stochastic=r["prior_policy"]["stochastic"], action_kl=r["prior_policy"]["action_kl"], log_std=logstd) for r in rows]
    summary = dict(counts=counts, exact_m003=exact)
    if rows_full:
        lines.append(row("Prior policy", "$\\theta_0$, $\\sigma_a=0.37$", prior_rows(rows_full, -1.0)))
        for i, k in enumerate(ks_full):
            lines.append(row("Parameter anchor" if i == 0 else "", f"$\\kappa_p={k:g}$", full[k]))
        summary["prior_full"] = prior_full
        summary["full"] = {f"{k:g}": dict(n=len(full[k]), q=ms(full[k], lambda r: r["stochastic"]["risk"]), U=ms(full[k], lambda r: r["stochastic"]["U"]),
                                          A=ms(full[k], lambda r: r["action_kl"]["A"]), distance=ms(full[k], lambda r: r["distance"])) for k in ks_full}
    for k, rows in meanonly.items():
        lines.append(row("Mean-only prior", f"$\\kappa_p={k:g}$", rows))
    if rows_matched:
        lines.append(row("Prior policy, matched", "$\\theta_0$, $\\sigma_a=0.1$ (incumbent)", prior_rows(rows_matched, rows_matched[0]["arm"]["prior_logstd"])))
        for i, k in enumerate(ks_matched):
            setting = f"$\\kappa_p={k:g}$" + (" (unanchored control)" if k == 0 else "")
            lines.append(row("Matched prior" if i == 0 else "", setting, matched[k]))
        summary["prior_matched"] = prior_matched
        summary["matched"] = {f"{k:g}": dict(n=len(matched[k]), seeds=[r["seed"] for r in matched[k]], q=ms(matched[k], lambda r: r["stochastic"]["risk"]),
                                             U=ms(matched[k], lambda r: r["stochastic"]["U"]), A=ms(matched[k], lambda r: r["action_kl"]["A"]),
                                             distance=ms(matched[k], lambda r: r["distance"]), sigma_a=ms(matched[k], lambda r: math.exp(r["log_std"]))) for k in ks_matched}
    for i, (eta, rows) in enumerate(action.items()):
        if rows:
            lines.append(row("Action regularization" if i == 0 else "", f"$\\eta={eta:g}$", rows))
    if unanch:
        lines.append(row("No anchor", "$m=0.03$", [dict(stochastic=r["stochastic"], action_kl=dict(A=float("nan")), log_std=r["log_std"]) for r in unanch]))
    note = "\\multicolumn{7}{@{}l@{}}{Matched prior: prior policy equal to the action reference (incumbent rule, dispersion $0.1$); other rows: earlier campaigns.} \\\\"
    tex = ("\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{llrrrrr}\n\\toprule\n"
           "Configuration & Setting & $q_\\beta$ (\\%) & $U$ & $\\mathcal A$ & $\\|\\theta-\\theta_0\\|$ & $\\sigma_a$ \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n" + note + "\n\\end{tabular}\n")
    (ROOT / "wear-anchor-table-v41.tex").write_text(tex)
    (ROOT / "wear-anchor-summary-v41.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    if prior_full:
        print("prior policy (0.37): risk %.4f U %.3f" % (prior_full["risk"][0], prior_full["U"][0]))
    if prior_matched:
        print("prior policy (matched, 0.1): risk %.4f U %.3f (mean action: risk %.4f U %.3f)" % (prior_matched["risk"][0], prior_matched["U"][0], prior_matched["risk_mean"][0], prior_matched["U_mean"][0]))
    for name, data in (("revision-39", full), ("matched", matched)):
        for k in sorted(data):
            q = ms(data[k], lambda r: r["stochastic"]["risk"]); U = ms(data[k], lambda r: r["stochastic"]["U"]); d = ms(data[k], lambda r: r["distance"]); A = ms(data[k], lambda r: r["action_kl"]["A"]); sd = ms(data[k], lambda r: math.exp(r["log_std"]))
            print(f"{name:12s} kappa_p={k:6g} n={len(data[k])} risk={100*q[0]:.2f}+-{100*q[1]:.2f} U={U[0]:.3f}+-{U[1]:.3f} dist={d[0]:.4f} A={A[0]:.3g} std={sd[0]:.3f}")
    if exact:
        print("exact m=0.03: re-evaluated q=%.4f U=%.3f (grid q=%.4f U=%.3f)" % (exact["q"], exact["U"], exact["q_grid"], exact["U_grid"]))


if __name__ == "__main__":
    main()
