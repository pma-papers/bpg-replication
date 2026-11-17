"""Revision 77: regularization strengths lowered by a schedule against strengths held fixed, on the PPO controller of
the wear-allowance example at the price m = 0.03 (runs of wear_schedules_v76.py with --stage schedules-v77).

Relative entropy to the Gaussian prior N(theta_0, sigma^2) with strength kappa acts on a parameter as a pull of
strength kappa_2 = kappa / sigma^2 toward theta_0 and as noise of strength kappa_1 = kappa, so lowering the strength
of the relative entropy lowers both with kappa_1 / kappa_2 = sigma^2 held. The runs lower the pull through
100, 10, 1, 0.1 and then remove it, with the noise tied to it (sigma = 0.03 and 0.1) or without noise, and compare
with runs in which one pair of strengths is held fixed.

Reads wear-schedules-v76/schedules-v77/*/result.json and benchmark.json.
Writes wear-schedules-v77.json, wear-schedules-table-v77.tex and the figure wear-schedules-v77 (pdf/png):
rows breakdown probability, stopped utility and spread of the actor parameters around theta_0; columns the three
ways of lowering; thin lines the seeds, thick lines their means, short horizontal bars the means of the runs with the
strengths of that stage held fixed, vertical lines the changes of the strength.
Runtime ~/.venvs/constrained-rl/bin/python.
"""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from plotstyle import PAL, MUTED, TEXT2, FS, TEXTW, panel_label, plt  # noqa: E402

M = 0.03
runs = defaultdict(list)
for f in sorted(glob.glob(str(HERE / "wear-schedules-v76/schedules-v77/*/result.json"))):
    r = json.load(open(f)); a = r["arm"]
    runs[(a["optimizer"], round(r["steps"] / 1e6), a["kappa2"], a["kappa1"])].append(r)
bench = json.load(open(HERE / "benchmark.json"))
exact = next(r["nominal"] for r in bench["rows"] if r["kind"] == "failure" and abs(r["m"] - M) < 1e-12)
prior = dict(risk=0.0108, U=2.73)
N_PAR = runs[next(iter(runs))][0]["n_actor_params"]
K2 = "100@0.1,10@0.2,1@0.35,0.1@0.5,0@1"
STAGES = [(0.0, 0.1, 100.0), (0.1, 0.2, 10.0), (0.2, 0.35, 1.0), (0.35, 0.5, 0.1), (0.5, 1.0, 0.0)]     # start, end (fractions), pull strength


def tied(sig2):
    return ",".join(f"{sig2 * v:g}@{e}" for v, e in ((100, 0.1), (10, 0.2), (1, 0.35), (0.1, 0.5))) + ",0@1"


ARMS = [("pull lowered, no noise", ("sgd", 16, K2, "0"), 0.0),
        ("relative entropy lowered, prior spread $0.03$", ("sgd", 16, K2, tied(9e-4)), 9e-4),
        ("relative entropy lowered, prior spread $0.1$", ("sgd", 16, K2, tied(1e-2)), 1e-2)]
TITLE = {"pull lowered, no noise": "pull lowered,\nno noise", "relative entropy lowered, prior spread $0.03$": "relative entropy lowered,\nprior spread $0.03$",
         "relative entropy lowered, prior spread $0.1$": "relative entropy lowered,\nprior spread $0.1$"}
NONE = ("sgd", 16, "0", "0")
ADAM = [("relative entropy lowered under Adam with the coupled anchor, $\\kappa_1=0.03^2\\kappa_2$", ("adam-coupled", 16, K2, tied(9e-4)), 9e-4),
        ("relative entropy lowered under Adam with the coupled anchor, $\\kappa_1=0.1^2\\kappa_2$", ("adam-coupled", 16, K2, tied(1e-2)), 1e-2)]


def ev(r, key):
    return np.array([e["steps"] for e in r["evals"]]) / 1e6, np.array([e[key] for e in r["evals"]])


def spread(r):
    return np.array([t["steps"] for t in r["trace"]]) / 1e6, np.array([t["distance"] for t in r["trace"]]) / np.sqrt(N_PAR)


def stage_end(r, lo, hi, key, total):
    """Mean of the evaluations in the last quarter of the stage [lo, hi) of a run of `total` million steps."""
    x, y = ev(r, key); a = (hi - 0.25 * (hi - lo)) * total; b = hi * total
    sel = (x >= a) & (x <= b + 1e-9)
    return float(y[sel].mean()), float(y[sel].std())


def fixed(k2, sig2):
    key = ("sgd", 8, f"{k2:g}", f"{sig2 * k2:g}" if sig2 > 0 else "0")
    return runs.get(key, [])


def end_of_run(r, key):
    """Mean and standard deviation over time of the evaluations in the last quarter of a run."""
    x, y = ev(r, key); sel = x >= 0.75 * x[-1]
    return float(y[sel].mean()), float(y[sel].std())


out = dict(exact_optimum_without_regularizer=dict(risk=exact["risk"], U=exact["U"]), incumbent=prior, n_actor_params=N_PAR, stages=STAGES, arms={}, fixed={}, other={})
rows = []
for name, key, sig2 in ARMS + ADAM:
    R = runs.get(key, [])
    if not R:
        continue
    total = R[0]["steps"] / 1e6
    a = dict(seeds=[r["seed"] for r in R], stages=[])
    for lo, hi, k2 in STAGES:
        se = dict(kappa2=k2, kappa1=sig2 * k2)
        for q in ("risk", "U"):
            v = [stage_end(r, lo, hi, q, total) for r in R]
            se[q] = [x[0] for x in v]; se[q + "_time_sd"] = [x[1] for x in v]
        sp = []
        for r in R:
            x, y = spread(r); sel = (x >= (hi - 0.25 * (hi - lo)) * total) & (x <= hi * total); sp.append(float(y[sel].mean()))
        se["spread"] = sp
        se["dispersion"] = [float(np.exp(np.mean([tt["log_std"] for tt in r["trace"] if (hi - 0.25 * (hi - lo)) * total <= tt["steps"] / 1e6 <= hi * total]))) for r in R]
        F = fixed(k2, sig2) if (k2 > 0 and key[0] == "sgd") else []
        if F:
            se["fixed"] = {q: [end_of_run(r, q)[0] for r in F] for q in ("risk", "U")}
            se["fixed"]["risk_time_sd"] = [end_of_run(r, "risk")[1] for r in F]
            se["fixed"]["spread"] = [float(spread(r)[1][-50:].mean()) for r in F]
        a["stages"].append(se)
    out["arms"][name] = a
for key, R in sorted(runs.items()):
    if key[1] == 8 and "@" not in key[2] and "@" not in key[3]:
        out["fixed"][f"kappa2={key[2]}, kappa1={key[3]}"] = dict(
            risk=[end_of_run(r, "risk")[0] for r in R], risk_time_sd=[end_of_run(r, "risk")[1] for r in R], U=[end_of_run(r, "U")[0] for r in R],
            U_time_sd=[end_of_run(r, "U")[1] for r in R], spread=[float(spread(r)[1][-50:].mean()) for r in R], log_std=[r["log_std"] for r in R])
    elif key not in [k for _, k, _ in ARMS + ADAM]:
        out["other"][" | ".join(map(str, key))] = dict(risk=[end_of_run(r, "risk")[0] for r in R], risk_time_sd=[end_of_run(r, "risk")[1] for r in R],
                                                        U=[end_of_run(r, "U")[0] for r in R], spread=[float(spread(r)[1][-50:].mean()) for r in R], log_std=[r["log_std"] for r in R])
# how long a run at a fixed pull takes to settle, how long the last stage takes, and how much a run varies in time
out["settle"] = {}
for k2 in ("100", "10", "1", "0.1"):
    R = runs.get(("sgd", 8, k2, "0"), []); ts = []
    for r in R:
        x, U = ev(r, "U"); Uf = U[x >= 6].mean()
        ts.append(float(x[np.argmax((U - U[0]) >= 0.9 * (Uf - U[0]))]) if abs(Uf - U[0]) > 0.03 else 0.0)
    out["settle"][f"pull {k2}"] = dict(steps_to_90_percent_of_the_change_of_U_millions=ts)
if runs.get(NONE):
    out["settle"]["no regularizer"] = dict(steps_to_90_percent_of_the_change_of_U_millions=[float(ev(r, "U")[0][np.argmax((ev(r, "U")[1] - ev(r, "U")[1][0]) >= 0.9 * (ev(r, "U")[1][-10:].mean() - ev(r, "U")[1][0]))]) for r in runs[NONE]],
                                           steps_to_U_5_25_millions=[float(ev(r, "U")[0][np.argmax(ev(r, "U")[1] >= 5.25)]) for r in runs[NONE]])
for name, key, sig2 in ARMS:
    if runs.get(key):
        out["arms"][name]["steps_after_the_removal_to_U_5_25_millions"] = [float(ev(r, "U")[0][np.argmax((ev(r, "U")[1] >= 5.25) & (ev(r, "U")[0] >= 8))]) - 8.0 for r in runs[key]]
out["range_in_time"] = {" | ".join(map(str, k)): [(float(ev(r, "risk")[1][-10:].min()), float(ev(r, "risk")[1][-10:].max())) for r in R]
                        for k, R in sorted(runs.items()) if k[1] == 8}
(HERE / "wear-schedules-v77.json").write_text(json.dumps(out, indent=1) + "\n")


def ms(v, scale=1.0, d=2):
    v = np.array(v, float) * scale
    return f"${v.mean():.{d}f}\\pm{v.std(ddof=1) if len(v) > 1 else 0:.{d}f}$"


# table: the end of each stage of a lowered strength next to the runs with that strength held fixed
lines = []
for name, a in out["arms"].items():
    lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{{name[0].upper() + name[1:]}}} \\\\")
    for se in a["stages"]:
        k1 = se["kappa1"]
        k = f"${se['kappa2']:g}$ & " + ("$9\\times10^{-5}$" if abs(k1 - 9e-5) < 1e-12 else f"${k1:g}$")
        fx = se.get("fixed")
        lines.append(f"{k} & {ms(se['risk'], 100, 1)} & {ms(se['U'])} & ${np.mean(se['spread']):.3f}$ & ${np.mean(se['dispersion']):.3f}$ & " + (f"{ms(fx['risk'], 100, 1)} & {ms(fx['U'])}" if fx else "-- & --") + " \\\\")
    lines.append("\\addlinespace")
(HERE / "wear-schedules-table-v77.tex").write_text(
    "\\begin{tabular}{@{}rrrrrrrr@{}}\n\\toprule\n & & \\multicolumn{4}{c}{End of the stage, strength lowered} & \\multicolumn{2}{c}{Strength held fixed} \\\\\n\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\n"
    "Pull $\\kappa_2$ & Noise $\\kappa_1$ & $q_\\beta$ (\\%) & $U$ & spread & $\\sigma_a$ & $q_\\beta$ (\\%) & $U$ \\\\\n\\midrule\n" + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

have = [(n, k, s) for n, k, s in ARMS if runs.get(k)]
if have:
    fig, axes = plt.subplots(3, len(have), figsize=(TEXTW, 4.5), sharex=True, squeeze=False, gridspec_kw=dict(hspace=0.16, wspace=0.34))
    for j, (name, key, sig2) in enumerate(have):
        R = runs[key]; total = R[0]["steps"] / 1e6
        for i, (q, scale) in enumerate((("risk", 100.0), ("U", 1.0), ("spread", 1.0))):
            ax = axes[i, j]
            cur = []
            for r in R:
                x, y = spread(r) if q == "spread" else ev(r, q)
                ax.plot(x, y * scale, color=PAL[0], lw=0.5, alpha=0.45); cur.append(np.interp(np.linspace(0, total, 161), x, y * scale))
            ax.plot(np.linspace(0, total, 161), np.mean(cur, 0), color=PAL[0], lw=1.3)
            if runs.get(NONE) and q != "spread":
                cu = [np.interp(np.linspace(0, total, 161), *ev(r, q)) * scale for r in runs[NONE]]
                ax.plot(np.linspace(0, total, 161), np.mean(cu, 0), color=MUTED, lw=1.0, ls=(0, (4, 2)))
            for lo, hi, k2 in STAGES:
                if lo > 0:
                    ax.axvline(lo * total, color=MUTED, lw=0.5, ls=":")
                F = fixed(k2, sig2) if k2 > 0 else []
                if F:
                    v = np.mean([float(spread(r)[1][-50:].mean()) if q == "spread" else end_of_run(r, q)[0] * scale for r in F])
                    ax.plot([lo * total, hi * total], [v, v], color=PAL[3], lw=1.6, solid_capstyle="butt", zorder=5)
            if q == "risk":
                ax.axhline(100 * exact["risk"], color=TEXT2, lw=0.7, ls="--"); ax.axhline(100 * prior["risk"], color=TEXT2, lw=0.7, ls=":")
            if q == "U":
                ax.axhline(exact["U"], color=TEXT2, lw=0.7, ls="--"); ax.axhline(prior["U"], color=TEXT2, lw=0.7, ls=":")
            if q == "spread":
                ax.set_yscale("log"); ax.set_ylim(1e-4, 3)
                if sig2 > 0:
                    ax.axhline(np.sqrt(sig2), color=TEXT2, lw=0.7, ls=":")
            if j == 0:
                ax.set_ylabel(("breakdown prob. (%)", "stopped utility $U$", r"spread $\|x-x_0\|/\sqrt{d}$")[i])
            if i == 0:
                ax.set_title(TITLE[name], fontsize=FS - 1.5, pad=9)
            if i == 2:
                ax.set_xlabel("steps (millions)")
            panel_label(ax, "abcdefghi"[i * len(have) + j])
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=PAL[0], lw=1.3, label="strength lowered in steps"), Line2D([], [], color=PAL[3], lw=1.6, label="strength of the stage held fixed"),
         Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 2)), label="no regularizer")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=FS - 1, bbox_to_anchor=(0.5, -0.04))
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"wear-schedules-v77.{ext}", bbox_inches="tight", dpi=200)

for name, a in out["arms"].items():
    print(name, "seeds", a["seeds"])
    for se in a["stages"]:
        fx = se.get("fixed")
        print(f"  pull {se['kappa2']:<5g} noise {se['kappa1']:<8g} risk {100 * np.mean(se['risk']):5.1f} {[round(100 * x, 1) for x in se['risk']]} (sd in time {100 * np.mean(se['risk_time_sd']):.1f})  U {np.mean(se['U']):.2f}  spread {np.mean(se['spread']):.4f}"
              + (f"   | fixed: risk {100 * np.mean(fx['risk']):5.1f} (sd in time {100 * np.mean(fx['risk_time_sd']):.1f}) U {np.mean(fx['U']):.2f} spread {np.mean(fx['spread']):.4f}" if fx else ""))
for k, v in list(out["fixed"].items()) + list(out["other"].items()):
    print(f"{k:70s} n {len(v['risk'])} risk {100 * np.mean(v['risk']):5.1f} (sd in time {100 * np.mean(v['risk_time_sd']):.1f})  U {np.mean(v['U']):.2f}  spread {np.mean(v['spread']):.4f}  log_std {np.mean(v['log_std']):.2f}")
print("exact optimum without regularizer", exact["risk"], exact["U"])
