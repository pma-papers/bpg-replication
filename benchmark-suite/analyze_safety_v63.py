"""Analysis of the Safety-Gymnasium campaign (PROTOCOL-safety-v63): one table, one figure, a JSON summary,
and the check of predictions S1-S5. Reads safety/campaign and safety/campaign-a0.2 (launcher amendment)."""
import glob, json, statistics as st, sys
from pathlib import Path
import numpy as np
from scipy.stats import beta as Beta
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import PAL, MUTED, TEXTW, FS, panel_label, legend_below, save
ERR = 0.05 / 72      # the family error over all 72 reported candidates (the paper since revision 70; 0.05/64 omitted
                     # the eight candidates added later)
rows = {}
CAMPAIGNS = ("campaign", "campaign-a0.2", "campaign-cmdp-s0.01")   # the v63 campaign and its added arms; later campaigns have their own analysis
for f in sorted(g for c in CAMPAIGNS for g in glob.glob(str(HERE / "safety" / c / "*" / "result.json"))):
    r = json.load(open(f)); a = r["arm"]
    key = (a["kind"], a["m"] if a["kind"] == "fixed" else (a["alpha"] if a["kind"] == "adaptive" else a["cost_limit"]))
    if a["kind"] == "cmdp": key = ("cmdp_s%g" % a["dual_step"], a["cost_limit"])
    rows.setdefault(key, []).append(r)
def ub(s):
    x = round(s["exhausted"] * s["n"]); n = s["n"]
    return 1.0 if x == n else float(Beta.ppf(1 - ERR, x + 1, n - x))
summ = {}
for k, rs in rows.items():
    rs.sort(key=lambda r: r["seed"])
    ex = [r["stochastic"]["exhausted"] for r in rs]; ret = [r["stochastic"]["mean_return"] for r in rs]
    cost = [r["stochastic"]["mean_cost"] for r in rs]; q90 = [r["stochastic"]["cost_q90"] for r in rs]
    mult = [r["final_multiplier"] for r in rs]; mx = [max(p["multiplier"] for p in r["trace"]) for r in rs]
    ubs = [ub(r["stochastic"]) for r in rs]
    summ["%s_%g" % k] = dict(kind=k[0], value=k[1], n=len(rs), exhausted=(st.mean(ex), st.pstdev(ex)), seeds_exhausted=ex,
        ret=(st.mean(ret), st.pstdev(ret)), cost=(st.mean(cost), st.pstdev(cost)), q90=(st.mean(q90), st.pstdev(q90)),
        final_multiplier=(st.mean(mult), st.pstdev(mult)), max_multiplier=(st.mean(mx), st.pstdev(mx)),
        upper_bounds=ubs, pass_020=sum(u <= 0.2 for u in ubs), pass_005=sum(u <= 0.05 for u in ubs))
json.dump(summ, open(HERE / "safety-summary-v63.json", "w"), indent=1)
# ---------------- predictions
S = {}
fx = [summ["fixed_%g" % m] for m in (0, 0.03, 0.1, 0.3, 1)]
ex = [s["exhausted"][0] for s in fx]; sd = [s["exhausted"][1] / 8 ** 0.5 for s in fx]
S["S1"] = all(ex[i + 1] <= ex[i] + 2 * sd[i] for i in range(4)), "means %s" % [round(e, 3) for e in ex]
S["S2"] = fx[4]["ret"][0] < 0.5 * fx[0]["ret"][0], "return at m=1 %.2f against control %.2f" % (fx[4]["ret"][0], fx[0]["ret"][0])
c = summ["cmdp_s0.5_25"]; c2 = summ.get("cmdp_s0.01_25")
S["S3"] = c["exhausted"][0] >= 1 / 3, "CMDP exhaustion %.3f, mean cost %.1f, return %.2f, max multiplier %.0f" % (c["exhausted"][0], c["cost"][0], c["ret"][0], c["max_multiplier"][0])
p02 = {k: v["pass_020"] for k, v in summ.items()}; p005 = {k: v["pass_005"] for k, v in summ.items()}
S["S4"] = (any(v["pass_020"] > 0 for k, v in summ.items() if k.startswith("fixed")) and all(v["pass_005"] == 0 for v in summ.values())), "pass at 0.2: %s; at 0.05: %s; value certificates vacuous (bound capped at one)" % (p02, p005)
a05, a2 = summ["adaptive_0.05"], summ["adaptive_0.2"]
S["S5"] = a05["exhausted"][0] > 0.05 and abs(a2["exhausted"][0] - 0.2) < 0.05, "alpha=0.05: exhaustion %.3f, final multiplier %.1f (max %.1f); alpha=0.2: exhaustion %.3f, final multiplier %.1f" % (a05["exhausted"][0], a05["final_multiplier"][0], a05["max_multiplier"][0], a2["exhausted"][0], a2["final_multiplier"][0])
for k, (ok, why) in S.items(): print(k, "HOLDS" if ok else "FALSIFIED", "|", why)
# ---------------- table
def pm(t, f="%.1f"): return ("$" + f + "\\pm" + f + "$") % (t[0], t[1])
L = ["\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{llrrrrrcc}", "\\toprule",
     "Configuration & Setting & exhausted (\\%) & return & mean cost & cost $q_{90}$ & final multiplier & pass $0.2$ & pass $0.05$ \\\\", "\\midrule"]
def row(name, setting, s, nu):
    L.append("%s & %s & %s & %s & %s & %s & %s & %d/8 & %d/8 \\\\" % (name, setting, pm((100 * s["exhausted"][0], 100 * s["exhausted"][1])), pm(s["ret"]), pm(s["cost"]), pm(s["q90"]), nu, s["pass_020"], s["pass_005"]))
row("Unpriced", "$m=0$", summ["fixed_0"], "--")
for m in (0.03, 0.1, 0.3, 1): row("Fixed price", "$m=%g$" % m, summ["fixed_%g" % m], "--")
row("Adaptive multiplier", "$\\alpha=0.05$", a05, pm(a05["final_multiplier"]))
row("Adaptive multiplier", "$\\alpha=0.2$", a2, pm(a2["final_multiplier"]))
row("Expected cost, dual step $0.5$", "$d=25$", c, "$0$ (max %s)" % pm(c["max_multiplier"], "%.0f"))
if c2: row("Expected cost, dual step $0.01$ (added)", "$d=25$", c2, "$0$ (max %s)" % pm(c2["max_multiplier"]))
L += ["\\bottomrule", "\\end{tabular}"]
(HERE.parent / "output" / "tables" / "safety.tex").write_text("\n".join(L) + "\n")
# ---------------- figure
fig, axes = plt.subplots(2, 2, figsize=(TEXTW, 3.9), gridspec_kw=dict(wspace=0.35, hspace=0.5)); axes = axes.ravel()
ms = [0, 0.03, 0.1, 0.3, 1]; x0 = 0.01
xs = [x0 if m == 0 else m for m in ms]
arms = [(a05, PAL[1], "o", "adaptive, $\\alpha=0.05$"), (a2, PAL[3], "s", "adaptive, $\\alpha=0.2$"), (c, PAL[2], "^", "expected cost, step $0.5$")]
if c2: arms.append((c2, PAL[4], "v", "expected cost, step $0.01$"))
for ax, key, lab in ((axes[0], "exhausted", "exhaustion probability"), (axes[1], "ret", "mean return")):
    for m, x in zip(ms, xs):
        rs = rows[("fixed", float(m))]; vals = [r["stochastic"]["exhausted" if key == "exhausted" else "mean_return"] for r in rs]
        ax.plot([x] * len(vals), vals, color=PAL[0], marker=".", ms=3, lw=0, alpha=0.5)
    ax.plot(xs, [summ["fixed_%g" % m][key][0] for m in ms], color=PAL[0], marker="D", ms=3.5, lw=1.0, label="fixed price")
    for s_, col, mk, lab2 in arms:
        ax.axhline(s_[key][0], color=col, lw=0.8, ls="--"); ax.plot([2.2], [s_[key][0]], color=col, marker=mk, ms=4, lw=0, label=lab2)
    ax.set_xscale("log"); ax.set_xlim(0.006, 3.5); ax.set_xticks([x0, 0.03, 0.1, 0.3, 1]); ax.set_xticklabels(["$0$", "$0.03$", "$0.1$", "$0.3$", "$1$"]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("price $m$"); ax.set_ylabel(lab)
axes[0].axhline(0.2, color=MUTED, lw=0.6, ls=":"); axes[0].axhline(0.05, color=MUTED, lw=0.6, ls=":")
legend_ab = axes[0].get_legend_handles_labels()
def traces(k):
    tr = [np.array([p["multiplier"] for p in r["trace"]]) for r in rows[k]]; n = min(len(t) for t in tr)
    return np.array([p["steps"] for p in rows[k][0]["trace"]])[:n] / 1e6, np.stack([t[:n] for t in tr])
ax = axes[2]
for k, col, ls, lab2 in ((("adaptive", 0.05), PAL[1], "-", "$\\alpha=0.05$"), (("adaptive", 0.2), PAL[3], "--", "$\\alpha=0.2$")):
    st_, T = traces(k); ax.plot(st_, T.mean(0), color=col, lw=1.0, ls=ls, label=lab2); ax.fill_between(st_, T.min(0), T.max(0), color=col, alpha=0.15, lw=0)
ax.set_xlabel("steps (millions)"); ax.set_ylabel("adaptive multiplier $\\nu$")
ax = axes[3]
cm = [(("cmdp_s0.5", 25.0), PAL[2], "-", "step $0.5$")] + ([(("cmdp_s0.01", 25.0), PAL[4], "--", "step $0.01$")] if c2 else [])
for k, col, ls, lab2 in cm:
    st_, T = traces(k); ax.plot(st_, T.mean(0), color=col, lw=1.0, ls=ls, label=lab2); ax.fill_between(st_, T.min(0), T.max(0), color=col, alpha=0.15, lw=0)
ax.set_xlabel("steps (millions)"); ax.set_ylabel("expected-cost multiplier $\\lambda$"); ax.set_yscale("symlog", linthresh=1.0)
for ax, L_ in zip(axes, "abcd"): panel_label(ax, L_)
h, l = legend_ab
h2, l2 = axes[2].get_legend_handles_labels(); h3, l3 = axes[3].get_legend_handles_labels()
legend_below(fig, h + h2 + h3, l + ["(c) " + x for x in l2] + ["(d) " + x for x in l3], ncol=3)
save(fig, "safety", HERE.parent / "output" / "figures")
print("written: table, figure, summary")
