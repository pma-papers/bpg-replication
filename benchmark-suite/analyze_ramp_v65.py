"""Analysis of the ramped-price campaign (PROTOCOL-ramp-v65.md): one table, one figure, a JSON summary and the
check of predictions R1-R4, over all 104 policies (72 of the first campaign, re-evaluated with failure times,
and 32 new runs).

Conventions fixed here, where the protocol leaves them open (reported in the note):
  - "within two standard errors" in R3 is the rule of the first campaign's S1: a later arm's mean may exceed
    the earlier arm's mean by at most twice the earlier arm's standard error (between-seed sd / sqrt(8));
  - the "95% bootstrap lower bound" of R4 is one-sided: the 5th percentile of the difference of the two
    policies' mean stopped utilities, each resampled over its 200 episodes, 2000 resamples (seed 65);
    a Bonferroni version over the candidate comparisons of each seed is reported as a robustness check;
  - "Proposition 3.3(i)" of the protocol is Proposition 3.4(i) of revision 65 (same statement).
Outputs go to results-ramp-v65/; nothing in the manuscript folders is written.
"""
import glob, json, statistics as st, sys
from pathlib import Path
import numpy as np
from scipy.stats import beta as Beta
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import PAL, SEQ, MUTED, TEXTW, panel_label, save

OUT = HERE / "results-ramp-v65"; OUT.mkdir(exist_ok=True)
ERR, BETA, H = 0.05 / 64, 0.999, 1000
FIRST_STEP_CHARGE = (1 - BETA ** H) / (1 - BETA)        # charge per unit price for a failure at t = 0


def key_of(a):
    k = a["kind"]
    if k in ("fixed", "ramp"):
        return (k, a["m"])
    if k == "adaptive":
        return ("adaptive", a["alpha"])
    return ("cmdp", a["dual_step"])


# ---------------------------------------------------------------- load all 104 policies
pol = []
for f in sorted(glob.glob(str(HERE / "safety" / "campaign" / "*" / "eval-times-v65.json")) +
                glob.glob(str(HERE / "safety" / "campaign-a0.2" / "*" / "eval-times-v65.json")) +
                glob.glob(str(HERE / "safety" / "campaign-cmdp-s0.01" / "*" / "eval-times-v65.json"))):
    r = json.load(open(f)); orig = json.load(open(Path(f).parent / "result.json"))
    assert r["reproduces_v63_evaluation"], f
    pol.append(dict(key=key_of(r["arm"]), seed=r["seed"], s=r["stochastic"], trace=orig["trace"], campaign="first"))
for f in sorted(glob.glob(str(HERE / "safety" / "campaign-v65" / "*" / "result.json"))):
    r = json.load(open(f))
    pol.append(dict(key=key_of(r["arm"]), seed=r["seed"], s=r["stochastic"], trace=r["trace"], campaign="ramp-v65"))
assert len(pol) == 104, len(pol)

# per-episode quantities; check the stored summaries
for p in pol:
    ep = p["s"]["episodes"]
    tau = np.array([e["tau"] if e["failed"] else -1 for e in ep])
    p["U_ep"] = np.array([e["U"] for e in ep])
    p["qt_ep"] = np.where(tau >= 0, BETA ** np.clip(tau, 0, None) - BETA ** H, 0.0) * (tau <= H - 1)
    p["qb_ep"] = np.where(tau >= 0, BETA ** np.clip(tau, 0, None), 0.0)
    assert abs(p["qt_ep"].mean() - p["s"]["q_tilde"]) < 1e-9 and abs(p["U_ep"].mean() - p["s"]["stopped_utility"]) < 1e-9
    assert abs(p["qb_ep"].mean() - p["s"]["q_beta"]) < 1e-9


def ub(s):
    x = round(s["exhausted"] * s["n"]); n = s["n"]
    return 1.0 if x == n else float(Beta.ppf(1 - ERR, x + 1, n - x))


groups = {}
for p in pol:
    groups.setdefault(p["key"], []).append(p)
for v in groups.values():
    v.sort(key=lambda p: p["seed"]); assert len(v) == 8


def ms(vals):
    return (st.mean(vals), st.pstdev(vals))


summ = {}
for k, v in groups.items():
    S = [p["s"] for p in v]; ubs = [ub(s) for s in S]
    summ[f"{k[0]}_{k[1]:g}"] = dict(kind=k[0], value=k[1], exhausted=ms([s["exhausted"] for s in S]),
        q_tilde=ms([s["q_tilde"] for s in S]), q_beta=ms([s["q_beta"] for s in S]), U=ms([s["stopped_utility"] for s in S]),
        ret=ms([s["mean_return"] for s in S]), cost=ms([s["mean_cost"] for s in S]),
        pass_020=sum(u <= 0.2 for u in ubs), pass_005=sum(u <= 0.05 for u in ubs),
        seeds={p["seed"]: dict(exhausted=p["s"]["exhausted"], ret=p["s"]["mean_return"], U=p["s"]["stopped_utility"],
                               q_tilde=p["s"]["q_tilde"]) for p in v})
se = lambda name, field: summ[name][field][1] / 8 ** 0.5
mean = lambda name, field: summ[name][field][0]

# ---------------------------------------------------------------- predictions
R = {}
r1_ret, r1_ex = mean("ramp_0.03", "ret"), mean("ramp_0.03", "exhausted")
R["R1"] = dict(verdict="holds" if (r1_ret >= 5 and r1_ex <= 0.15) else ("falsified" if r1_ret < 3.5 else "not confirmed, not falsified"),
               detail=f"ramped m=0.03: mean return {r1_ret:.2f}, mean exhaustion {r1_ex:.3f} (fixed m=0.03: {mean('fixed_0.03','ret'):.2f}, {mean('fixed_0.03','exhausted'):.3f})")
g01 = mean("ramp_0.01", "ret") - mean("fixed_0.01", "ret"); g03 = mean("ramp_0.03", "ret") - mean("fixed_0.03", "ret")
R["R2"] = dict(verdict="holds" if g01 < g03 else "falsified", detail=f"ramp minus fixed mean return: {g01:.2f} at m=0.01, {g03:.2f} at m=0.03")
order = ["fixed_0", "fixed_0.003", "fixed_0.01", "fixed_0.03"]
mono = {f: all(mean(order[i + 1], f) <= mean(order[i], f) + 2 * se(order[i], f) for i in range(3)) for f in ("exhausted", "ret")}
between = mean("fixed_0", "exhausted") > mean("fixed_0.003", "exhausted") > mean("fixed_0.03", "exhausted")
R["R3"] = dict(verdict="holds" if (mono["exhausted"] and mono["ret"] and between) else
               ("falsified" if not (mono["exhausted"] and mono["ret"]) else "not confirmed, not falsified"),
               detail="exhaustion " + ", ".join(f"{o.split('_')[1]}: {mean(o,'exhausted'):.3f}+-{se(o,'exhausted'):.3f}" for o in order)
               + "; return " + ", ".join(f"{o.split('_')[1]}: {mean(o,'ret'):.2f}+-{se(o,'ret'):.2f}" for o in order),
               monotone=mono, strictly_between=between)

# R4: Proposition 3.4(i) diagnostic with a one-sided bootstrap lower bound
B = 2000; rng = np.random.default_rng(65)
boot = np.empty((len(pol), B))
for i, p in enumerate(pol):
    idx = rng.integers(0, len(p["U_ep"]), (B, len(p["U_ep"])))
    boot[i] = p["U_ep"][idx].mean(1)
Bb = 20000; rngb = np.random.default_rng(650)
bootb = np.empty((len(pol), Bb))
for i, p in enumerate(pol):
    idx = rngb.integers(0, len(p["U_ep"]), (Bb, len(p["U_ep"])))
    bootb[i] = p["U_ep"][idx].mean(1)


def diagnostic(name):
    k = (name.split("_")[0], float(name.split("_")[1]))
    res = {}
    for i, p in enumerate(pol):
        if p["key"] != k:
            continue
        cands = [j for j, o in enumerate(pol) if j != i and o["s"]["q_tilde"] <= p["s"]["q_tilde"]
                 and o["s"]["stopped_utility"] > p["s"]["stopped_utility"]]
        lbs = {j: float(np.percentile(boot[j] - boot[i], 5)) for j in cands}
        ncomp = sum(1 for j, o in enumerate(pol) if j != i and o["s"]["q_tilde"] <= p["s"]["q_tilde"])
        level = 5 / max(ncomp, 1)
        lbs_b = {j: float(np.percentile(bootb[j] - bootb[i], level)) for j in cands}
        best = max(lbs.items(), key=lambda kv: kv[1], default=(None, 0.0))
        bestb = max(lbs_b.items(), key=lambda kv: kv[1], default=(None, 0.0))
        res[p["seed"]] = dict(q_tilde=p["s"]["q_tilde"], U=p["s"]["stopped_utility"], candidates=len(cands), comparisons=ncomp,
                              detected=best[1] > 0, bound=max(best[1], 0.0),
                              by=(f"{pol[best[0]]['key'][0]} {pol[best[0]]['key'][1]:g} seed {pol[best[0]]['seed']}" if best[0] is not None and best[1] > 0 else None),
                              detected_bonferroni=bestb[1] > 0, bound_bonferroni=max(bestb[1], 0.0))
    return res


D_fixed, D_ramp = diagnostic("fixed_0.03"), diagnostic("ramp_0.03")
n_det = sum(d["detected"] for d in D_fixed.values()); n_det_b = sum(d["detected_bonferroni"] for d in D_fixed.values())
med_fixed = float(np.median([d["bound"] for d in D_fixed.values()])); med_ramp = float(np.median([d["bound"] for d in D_ramp.values()]))
R["R4"] = dict(verdict="holds" if (n_det >= 6 and med_ramp < med_fixed) else ("falsified" if (n_det < 6 or med_ramp > med_fixed) else "not confirmed, not falsified"),
               detail=f"fixed m=0.03: detected in {n_det} of 8 seeds (Bonferroni: {n_det_b}); median bound {med_fixed:.2f} (ramp {med_ramp:.2f})",
               fixed=D_fixed, ramp=D_ramp, median_fixed=med_fixed, median_ramp=med_ramp,
               detected_bonferroni_ramp=sum(d["detected_bonferroni"] for d in D_ramp.values()))
objective = {}
for m in (0.01, 0.03):
    nu = m / (1 - BETA)
    for kind in ("fixed", "ramp"):
        v = groups[(kind, m)]
        J = [p["s"]["stopped_utility"] - nu * p["s"]["q_tilde"] for p in v]
        objective[f"{kind}_{m:g}"] = dict(nu=nu, mean=st.mean(J), sd=st.pstdev(J), seeds={p["seed"]: j for p, j in zip(v, J)})
    a = groups[("adaptive", 0.05)]
    J = [p["s"]["stopped_utility"] - nu * p["s"]["q_tilde"] for p in a]
    objective[f"adaptive_0.05_at_m{m:g}"] = dict(nu=nu, mean=st.mean(J), sd=st.pstdev(J))
R["objective"] = objective
for k, v in R.items():
    if k == "objective":
        print("objective U - nu*qt:", {kk: (round(vv["mean"], 2), round(vv["sd"], 2)) for kk, vv in v.items()})
    else:
        print(k, v["verdict"].upper(), "|", v["detail"])

# ---------------------------------------------------------------- price paths (charge for a failure at the first step)
paths = {}
for name, field in [("ramp_0.03", "price"), ("ramp_0.01", "price"), ("adaptive_0.05", "multiplier"), ("adaptive_0.2", "multiplier")]:
    k = (name.split("_")[0], float(name.split("_")[1]))
    tr = [p["trace"] for p in groups[k]]
    steps = np.array([q["steps"] for q in tr[0]])
    vals = np.array([[q[field] for q in t[:len(steps)]] for t in tr])
    charge = vals * (FIRST_STEP_CHARGE if field == "price" else 1.0)
    paths[name] = dict(steps=steps, mean=charge.mean(0), lo=charge.min(0), hi=charge.max(0))

json.dump(dict(conventions=__doc__, summary=summ, predictions=R, first_step_charge_per_unit_price=FIRST_STEP_CHARGE,
               final_first_step_charge={k: float(v["mean"][-1]) for k, v in paths.items()}),
          open(OUT / "ramp-summary-v65.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- table
names = [("fixed_0", "fixed price", "$m=0$ (unpriced)"), ("fixed_0.003", "fixed price", "$m=0.003$"), ("fixed_0.01", "fixed price", "$m=0.01$"),
         ("fixed_0.03", "fixed price", "$m=0.03$"), ("fixed_0.1", "fixed price", "$m=0.1$"), ("fixed_0.3", "fixed price", "$m=0.3$"),
         ("fixed_1", "fixed price", "$m=1$"), ("ramp_0.01", "ramped price", "$m=0.01$"), ("ramp_0.03", "ramped price", "$m=0.03$"),
         ("adaptive_0.05", "adaptive multiplier", "$\\alpha=0.05$"), ("adaptive_0.2", "adaptive multiplier", "$\\alpha=0.2$"),
         ("cmdp_0.5", "expected cost", "step $0.5$"), ("cmdp_0.01", "expected cost", "step $0.01$")]
pm = lambda t, f: ("$" + f + "\\pm" + f + "$") % (t[0], t[1])
L = ["\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llrrrrrcc}", "\\toprule",
     "Arm & Setting & exhausted (\\%) & $\\tilde q$ & $q_\\beta$ & $U$ & return & pass $0.2$ & pass $0.05$\\\\", "\\midrule"]
for n_, arm, sett in names:
    s = summ[n_]
    L.append(f"{arm} & {sett} & {pm((100*s['exhausted'][0], 100*s['exhausted'][1]), '%.1f')} & {pm(s['q_tilde'], '%.3f')} & "
             f"{pm(s['q_beta'], '%.3f')} & {pm(s['U'], '%.2f')} & {pm(s['ret'], '%.2f')} & {s['pass_020']} & {s['pass_005']}\\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "ramp-table-v65.tex").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 2.9), gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})
ax = axs[0]
fixed_ms = [0, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
for i, m in enumerate(fixed_ms):
    v = groups[("fixed", float(m))]
    ax.scatter([p["s"]["q_tilde"] for p in v], [p["s"]["stopped_utility"] for p in v], s=9, color=SEQ[i], zorder=3,
               label="fixed $m$ (light to dark: 0 to 1)" if i == 3 else None)
style = {("ramp", 0.03): (PAL[1], "^", "ramped $m=0.03$"), ("ramp", 0.01): (PAL[3], "^", "ramped $m=0.01$"),
         ("adaptive", 0.05): (PAL[2], "s", "adaptive $\\alpha=0.05$"), ("adaptive", 0.2): (PAL[5], "s", "adaptive $\\alpha=0.2$"),
         ("cmdp", 0.5): (PAL[7], "x", "expected cost"), ("cmdp", 0.01): (PAL[7], "x", None)}
for k, (c, mk, lab) in style.items():
    v = groups[k]
    ax.scatter([p["s"]["q_tilde"] for p in v], [p["s"]["stopped_utility"] for p in v], s=12, color=c, marker=mk, zorder=4, label=lab)
ax.set(xlabel="priced risk $\\tilde q$", ylabel="stopped utility $U$")
ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, handletextpad=0.2, columnspacing=0.8)
panel_label(ax, "a")
ax = axs[1]
for name, c, lab in [("ramp_0.03", PAL[1], "ramped $m=0.03$"), ("ramp_0.01", PAL[3], "ramped $m=0.01$"), ("adaptive_0.05", PAL[2], "adaptive $\\alpha=0.05$")]:
    pth = paths[name]
    ax.fill_between(pth["steps"] / 1e6, pth["lo"], pth["hi"], color=c, alpha=0.18, lw=0)
    ax.plot(pth["steps"] / 1e6, pth["mean"], color=c, label=lab)
ax.axhline(0.03 * FIRST_STEP_CHARGE, color=MUTED, ls="--", lw=0.9, label="fixed $m=0.03$")
ax.set(xlabel="training steps (millions)", ylabel="charge for a failure at $t=0$")
ax.legend(fontsize=6, loc="upper left")
panel_label(ax, "b")
ax = axs[2]
seeds = sorted(D_fixed)
x = np.arange(len(seeds))
ax.bar(x - 0.2, [D_fixed[s]["bound"] for s in seeds], width=0.4, color=SEQ[3], label="fixed $m=0.03$")
ax.bar(x + 0.2, [D_ramp[s]["bound"] for s in seeds], width=0.4, color=PAL[1], label="ramped $m=0.03$")
ax.set(xticks=x, xticklabels=[str(s)[-1] for s in seeds], xlabel="seed (9500$k$)", ylabel="lower bound on $\\delta$")
ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
panel_label(ax, "c", y=1.12)
fig.tight_layout()
save(fig, "ramp-v65", OUT)
print("written to", OUT)
