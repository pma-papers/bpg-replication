"""Revision 77: the pull lowered by a rule on the PPO controller of the wear-allowance example (runs of
wear_schedules_v76.py --rule, stage schedules-v77), against stages fixed in advance, strengths held fixed and
training without a regularizer.

The rule: the pull starts at 100. After every evaluation (every 100,000 steps, 4000 episodes) the value
U - m q / (1 - beta) of the current policy is recorded. When the mean over the last four evaluations differs from
the mean over the four before them by less than 0.04, the pull is divided by 10, and below 0.01 it is removed. With
relative entropy the noise strength is 0.03^2 times the pull.

Reads wear-schedules-v76/schedules-v77/*/result.json and benchmark.json.
Writes wear-rule-v77.json, wear-rule-table-v77.tex and the figure wear-rule-v77 (pdf/png):
(a) the pull against the training steps, one line per seed, with the stages fixed in advance in grey;
(b) the stopped utility of the current policy: the rule (seeds and mean), the stages fixed in advance (mean) and no
regularizer (mean); ticks at the right are the runs with the pull held fixed.
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

M, BETA = 0.03, 0.95
runs = defaultdict(list)
for f in sorted(glob.glob(str(HERE / "wear-schedules-v76/schedules-v77/*/result.json"))):
    r = json.load(open(f)); a = r["arm"]
    runs[(a["optimizer"], round(r["steps"] / 1e6), a["kappa2"], a["kappa1"], a.get("rule", ""))].append(r)
bench = json.load(open(HERE / "benchmark.json"))
exact = next(r["nominal"] for r in bench["rows"] if r["kind"] == "failure" and abs(r["m"] - M) < 1e-12)
N_PAR = runs[next(iter(runs))][0]["n_actor_params"]
K2 = "100@0.1,10@0.2,1@0.35,0.1@0.5,0@1"
RULES = [("pull lowered by the rule, no noise", "start=100,factor=10,min=0.01,window=4,tol=0.04,sigma2=0"),
         ("relative entropy lowered by the rule, prior spread $0.03$", "start=100,factor=10,min=0.01,window=4,tol=0.04,sigma2=0.0009")]
PULLS = [100.0, 10.0, 1.0, 0.1, 0.01, 0.0]


def ev(r, key):
    return np.array([e["steps"] for e in r["evals"]]) / 1e6, np.array([e[key] for e in r["evals"]])


def reach(r, level=5.25):
    x, U = ev(r, "U"); ok = U >= level
    return float(x[np.argmax(ok)]) if ok.any() else float("nan")


def fixed_end(k2, key):
    R = runs.get(("sgd", 8, f"{k2:g}", "0", ""), [])
    return [float(ev(r, key)[1][ev(r, key)[0] >= 6].mean()) for r in R]


out = dict(rule=dict(start=100, factor=10, min=0.01, window_evaluations=4, tolerance=0.04, evaluation_every_steps=100000), arms={})
for name, rule in RULES:
    R = runs.get(("sgd", 20, "rule", "rule", rule), [])
    if not R:
        continue
    a = dict(seeds=[r["seed"] for r in R], changes=[[(c["steps"] / 1e6, c["kappa2"]) for c in r["rule_changes"]] for r in R], stages=[],
             steps_to_U_5_25_millions=[reach(r) for r in R], final=dict(risk=[float(ev(r, "risk")[1][-10:].mean()) for r in R], U=[float(ev(r, "U")[1][-10:].mean()) for r in R]))
    for i, k2 in enumerate(PULLS):
        se = dict(kappa2=k2, length=[], risk=[], U=[], spread=[])
        for r in R:
            ch = [0.0] + [c["steps"] / 1e6 for c in r["rule_changes"]] + [r["steps"] / 1e6]
            if i >= len(ch) - 1:
                continue
            lo, hi = ch[i], ch[i + 1]
            x, q = ev(r, "risk"); _, U = ev(r, "U"); sel = (x > max(lo, hi - 0.4)) & (x <= hi + 1e-9) if k2 > 0 else (x >= hi - 1.0)
            se["length"].append(hi - lo); se["risk"].append(float(q[sel].mean())); se["U"].append(float(U[sel].mean()))
            tx = np.array([t["steps"] for t in r["trace"]]) / 1e6; d = np.array([t["distance"] for t in r["trace"]]) / np.sqrt(N_PAR)
            se["spread"].append(float(d[(tx > hi - 0.2) & (tx <= hi)].mean()))
        if k2 > 0:
            se["fixed"] = dict(risk=fixed_end(k2, "risk"), U=fixed_end(k2, "U"))
        a["stages"].append(se)
    out["arms"][name] = a
ref = dict(fixed_stages=runs.get(("sgd", 16, K2, "0", ""), []), none=runs.get(("sgd", 16, "0", "0", ""), []))
out["steps_to_U_5_25_millions"] = dict(stages_fixed_in_advance=[reach(r) for r in ref["fixed_stages"]], no_regularizer=[reach(r) for r in ref["none"]])
(HERE / "wear-rule-v77.json").write_text(json.dumps(out, indent=1) + "\n")


def ms(v, scale=1.0, d=2):
    v = np.array(v, float) * scale
    return f"${v.mean():.{d}f}\\pm{v.std(ddof=1) if len(v) > 1 else 0:.{d}f}$" if len(v) else "--"


lines = []
for name, a in out["arms"].items():
    lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{{name[0].upper() + name[1:]}}} \\\\")
    for se in a["stages"]:
        if not se["length"]:
            continue
        fx = se.get("fixed")
        ln = f"${min(se['length']):.1f}$ to ${max(se['length']):.1f}$" if se["kappa2"] > 0 else "to the end"
        lines.append(f"${se['kappa2']:g}$ & {ln} & {ms(se['risk'], 100, 1)} & {ms(se['U'])} & " + (f"{ms(fx['risk'], 100, 1)} & {ms(fx['U'])}" if fx and fx["U"] else "-- & --") + " \\\\")
    lines.append("\\addlinespace")
(HERE / "wear-rule-table-v77.tex").write_text(
    "\\begin{tabular}{@{}rrrrrr@{}}\n\\toprule\n & & \\multicolumn{2}{c}{End of the stage under the rule} & \\multicolumn{2}{c}{Pull held fixed} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
    "Pull $\\kappa_2$ & Length of the stage ($10^6$ steps) & $q_\\beta$ (\\%) & $U$ & $q_\\beta$ (\\%) & $U$ \\\\\n\\midrule\n" + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

name, rule = RULES[0]
R = runs.get(("sgd", 20, "rule", "rule", rule), [])
if R:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTW, 2.3), gridspec_kw=dict(wspace=0.32))
    total = R[0]["steps"] / 1e6
    for i, r in enumerate(R):
        ch = [0.0] + [c["steps"] / 1e6 for c in r["rule_changes"]]; val = [100.0] + [max(c["kappa2"], 3e-3) for c in r["rule_changes"]]
        ax1.step(ch + [total], val + [val[-1]], where="post", color=PAL[i], lw=1.0)
        x, U = ev(r, "U"); ax2.plot(x, U, color=PAL[0], lw=0.5, alpha=0.45)
    ax1.step([0, 1.6, 3.2, 5.6, 8.0, 16.0], [100, 10, 1, 0.1, 3e-3, 3e-3], where="post", color=MUTED, lw=1.0, ls=(0, (4, 2)))
    ax1.set_yscale("log"); ax1.set_ylim(2e-3, 300); ax1.set_yticks([3e-3, 1e-2, 1e-1, 1, 10, 100]); ax1.set_yticklabels(["0", "0.01", "0.1", "1", "10", "100"])
    ax1.set_xlabel("steps (millions)"); ax1.set_ylabel(r"pull $\kappa_2$"); ax1.minorticks_off()
    grid = np.linspace(0, total, 201)
    ax2.plot(grid, np.mean([np.interp(grid, *ev(r, "U")) for r in R], 0), color=PAL[0], lw=1.3, label="rule")
    for key, col, ls, lab in (("fixed_stages", PAL[3], "-", "stages fixed in advance"), ("none", MUTED, (0, (4, 2)), "no regularizer")):
        if ref[key]:
            g2 = np.linspace(0, 16, 161); ax2.plot(g2, np.mean([np.interp(g2, *ev(r, "U")) for r in ref[key]], 0), color=col, lw=1.1, ls=ls, label=lab)
    for k2 in (100.0, 10.0, 1.0, 0.1, 0.01):
        v = fixed_end(k2, "U")
        if v:
            ax2.plot([total * 1.005, total * 1.035], [np.mean(v)] * 2, color=TEXT2, lw=1.4, clip_on=False, solid_capstyle="butt")
    ax2.axhline(exact["U"], color=TEXT2, lw=0.7, ls="--")
    ax2.set_xlim(0, total); ax2.set_xlabel("steps (millions)"); ax2.set_ylabel("stopped utility $U$")
    ax2.legend(frameon=False, fontsize=FS - 2, loc="lower right")
    panel_label(ax1, "a"); panel_label(ax2, "b")
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"wear-rule-v77.{ext}", bbox_inches="tight", dpi=200)

for name, a in out["arms"].items():
    print(name, "seeds", a["seeds"])
    for s, ch in zip(a["seeds"], a["changes"]):
        print("   seed", s, "changes (million steps, new pull):", [(round(x, 1), k) for x, k in ch])
    for se in a["stages"]:
        if se["length"]:
            fx = se.get("fixed")
            print(f"   pull {se['kappa2']:<5g} length {np.round(se['length'], 1)}  risk {100 * np.mean(se['risk']):5.1f}  U {np.mean(se['U']):.2f}  spread {np.mean(se['spread']):.4f}"
                  + (f"   | fixed: risk {100 * np.mean(fx['risk']):5.1f} U {np.mean(fx['U']):.2f}" if fx and fx["U"] else ""))
    print("   steps to U >= 5.25:", np.round(a["steps_to_U_5_25_millions"], 1), " final", {k: round(float(np.mean(v)), 3) for k, v in a["final"].items()})
print("steps to U >= 5.25: stages fixed in advance", np.round(out["steps_to_U_5_25_millions"]["stages_fixed_in_advance"], 1), " no regularizer", np.round(out["steps_to_U_5_25_millions"]["no_regularizer"], 1))
