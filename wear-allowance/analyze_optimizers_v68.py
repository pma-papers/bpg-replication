"""Revision 68: analysis of the three-optimizer campaign (PROTOCOL-optimizers-v68.md, part A).

Reads the learning runs of the three arms (sgd, adam-coupled, adam-decoupled) over the rows C1-C6, reused records
included, the zero-objective control (direct simulation) and the per-coordinate records, and writes
wear-optimizers-v68.json (per seed, per arm and row, the checks (E1)-(E6) and (D1)-(D3), the regularizer-bias and
optimizer-effect comparisons), wear-optimizers-v68.tex (the table) and wear-optimizers-v68.pdf/png (three panels).
The amendment (PROTOCOL-optimizers-v68-AMENDMENT.md) adds the rows C7-C9 of the revision-60 study under the sgd and
adam-decoupled arms (coupled: the recorded runs), their zero-objective control, the checks (F1)-(F4), and part B, the
application over six seeds per arm (wear-optimizers-v68-application.tex and the key "application" of the JSON).
Frontier and price-risk bound follow the wear-allowance analysis (analyze_v41.py): the failure-only dynamic-programming
policies re-evaluated in the simulator, interpolated linearly in log q; the bound with the idle policy as comparison
and the measured optimization error. Runtime: ~/.venvs/constrained-rl/bin/python (numpy, scipy, matplotlib).
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib")); sys.path.insert(0, str(HERE))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plotstyle import PAL, TEXT2, MUTED, TEXTW, panel_label, legend_below, save  # noqa: E402
from analyze_v41 import frontier, price_risk_bound  # noqa: E402

M = 0.03
ROWS = {"C1": (10.0, 0.009), "C2": (10.0, 0.101), "C3": (1.0, 0.101), "C4": (10.0, 0.0), "C5": (1.0, 0.0), "C6": (0.0, 0.0),
        "C7": (10.0, 0.001), "C8": (10.0, 0.989), "C9": (0.0, 0.101)}          # C7-C9: the amendment (revision-60 rows)
ZERO_ROWS = [r for r in ROWS if r != "C6"]
APP = HERE.parent / "application"
ARMS = ["sgd", "adam-coupled", "adam-decoupled"]
LR_ADAM = 3e-4
NEW = HERE / "wear-optimizers-v68" / "campaign"
COORD = HERE / "wear-optimizers-v68" / "derived-coordinates"
D = 4418


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(str(pattern)))]


def sgd_lr():
    txt = (HERE / "PROTOCOL-optimizers-v68.md").read_text()
    import re
    m = re.search(r"lr_sgd = ([0-9.eE+-]+)", txt)
    return float(m.group(1))


def sources(arm, row, lr):
    """Where the learning runs of (arm, row) are: new campaign records or the reused ones."""
    k2, k1 = ROWS[row]
    if arm == "sgd":
        return [NEW / f"sgd__lr{lr:g}__k1_{k1:g}__k2_{k2:g}__seed*/result.json"]
    if row == "C6":
        return [HERE / "wear-anchor" / "campaign-matched" / "kappa0__seed9400*/result.json"]
    if arm == "adam-coupled":
        if row in ("C1", "C2", "C7", "C8", "C9"):
            return [HERE / "wear-noise" / "campaign" / f"k1_{k1:g}__k2_{k2:g}__seed9400*/result.json"]
        if row == "C4":
            return [HERE / "wear-anchor" / "campaign-matched" / "kappa10__seed9400*/result.json"]
        return [NEW / f"adam-coupled__lr{LR_ADAM:g}__k1_{k1:g}__k2_{k2:g}__seed*/result.json"]
    if arm == "adam-decoupled":
        if row in ("C1", "C7", "C8", "C9"):
            return [NEW / f"adam-decoupled__lr{LR_ADAM:g}__k1_{k1:g}__k2_{k2:g}__seed*/result.json"]
        return [HERE / "wear-noise-decoupled-v68" / "campaign" / f"k1_{k1:g}__k2_{k2:g}__seed*/result.json"]


def per_run(r, bench, U_ex):
    s = r["stochastic"]
    q, U, I = s["risk"], s["U"], s["I"]
    law = price_risk_bound(bench, "failure", M, s)
    uex = U_ex(q)
    out = dict(seed=r["seed"], risk=q, U=U, I=I, U_net=U - I, risk_mean_action=r["mean_action"]["risk"], A=r["action_kl"]["A"],
               distance=r["distance"], spread=r["distance"] / math.sqrt(D), sigma_a=math.exp(r["log_std"]), log_std=r["log_std"],
               U_frontier=uex, gap=uex - U, gap_rel=(uex - U) / uex, bound=law["bound"], bound_slack=law["bound"] - q, bound_factor=(law["bound"] / q if q > 0 else float("inf")),
               delta=law["delta"], train_wall=r.get("train_wall"), flow_time=(r.get("hook") or {}).get("lr_sum_flow_time"),
               prior_risk=r["prior_policy"]["stochastic"]["risk"], prior_U=r["prior_policy"]["stochastic"]["U"])
    return out


def ms(xs):
    x = np.array(xs, dtype=float)
    return [float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0]


def summarize(runs):
    keys = ["risk", "U", "I", "U_net", "risk_mean_action", "A", "distance", "spread", "sigma_a", "U_frontier", "gap", "gap_rel", "bound", "bound_slack", "bound_factor", "delta", "prior_risk", "prior_U"]
    s = {k: ms([r[k] for r in runs]) for k in keys if all(r.get(k) is not None and np.isfinite(r[k]) for r in runs)}
    s["n"] = len(runs); s["per_seed"] = runs
    return s


def flow_time(lr, k=None, steps=8_000_000, rollout=32 * 256, per_rollout=80):
    """Sum of the schedule's learning rates over the first k rollouts (all 977 by default): lr (1 - j rollout / steps), clipped at zero."""
    n = math.ceil(steps / rollout) if k is None else k
    return lr * per_rollout * sum(max(1.0 - j * rollout / steps, 0.0) for j in range(1, n + 1))


def predicted(arm, row, lr):
    k2, k1 = ROWS[row]
    if k1 == 0:
        return dict(particle_form=0.0, adam_laplace=0.0)
    pf = math.sqrt(k1 / k2) if k2 > 0 else None
    out = dict(particle_form=pf, particle_form_exact_at_lr0=(math.sqrt(k1 / k2 / (1 - lr * k2 / 2)) if k2 > 0 else None), adam_laplace=math.sqrt(2) * k1)
    if k2 == 0:                                      # no pull: a Gaussian random walk, spread sqrt(2 kappa_1 sum_t lr_t) after the run
        out["random_walk"] = math.sqrt(2 * k1 * flow_time(lr))
    return out


def pred_value(arm, s):
    """The predicted spread a row is compared with: the random walk without a pull, Adam's Laplace value in the coupled arm, else the particle form."""
    p = s["predicted"]
    if s["kappa2"] == 0:
        return p.get("random_walk")
    return p["adam_laplace"] if arm == "adam-coupled" else p["particle_form"]


def main():
    lr_sgd = sgd_lr()
    bench = json.loads((HERE / "benchmark.json").read_text())
    U_ex = frontier(bench, "reeval")[3]
    LR = {"sgd": lr_sgd, "adam-coupled": LR_ADAM, "adam-decoupled": LR_ADAM}

    # ---- learning runs
    learn = {}
    for arm in ARMS:
        for row in ROWS:
            rs = []
            for pat in sources(arm, row, LR[arm]):
                rs += load(pat)
            if rs:
                learn[(arm, row)] = dict(summarize([per_run(r, bench, U_ex) for r in rs]), arm=arm, row=row, kappa2=ROWS[row][0], kappa1=ROWS[row][1], lr=LR[arm],
                                         predicted=predicted(arm, row, LR[arm]), reused=not any(str(NEW) in str(p) for p in sources(arm, row, LR[arm])))
    # ---- zero-objective control
    zero = {}
    for arm in ARMS:
        for row in ZERO_ROWS:
            k2, k1 = ROWS[row]
            rs = load(NEW / f"{arm}__lr{LR[arm]:g}__k1_{k1:g}__k2_{k2:g}__zerograd-direct__seed*/result.json")
            if rs:
                z = dict(summarize([per_run(r, bench, U_ex) for r in rs]), arm=arm, row=row, kappa2=k2, kappa1=k1, lr=LR[arm], predicted=predicted(arm, row, LR[arm]))
                z["kurtosis"] = ms([r["kurtosis"] for r in rs]); z["mean_abs_over_rms"] = ms([r["mean_abs_over_rms"] for r in rs])
                z["spread_trace_mean"] = np.mean([r["spread_trace_per_rollout"] for r in rs], axis=0).tolist()
                if k2 == 0:                          # the random walk along the run: sqrt(2 kappa_1 S_k) after rollout k
                    tr = np.array(z["spread_trace_mean"]); pr = np.array([math.sqrt(2 * k1 * flow_time(LR[arm], k)) for k in range(1, len(tr) + 1)])
                    z["random_walk_trace"] = pr.tolist(); z["trace_ratio_max_dev_from_10th_rollout"] = float(np.max(np.abs(tr[9:] / pr[9:] - 1)))
                zero[(arm, row)] = z
    checks_ppo = {}
    for arm in ARMS:
        rs = load(NEW / f"{arm}__lr{LR[arm]:g}__k1_0.101__k2_10__zerograd__seed94001/result.json")
        if rs and (arm, "C2") in zero:
            direct = [p for p in zero[(arm, "C2")]["per_seed"] if p["seed"] == 94001]
            if direct:
                checks_ppo[arm] = dict(ppo_spread=rs[0]["spread"], direct_spread=direct[0]["spread"], ppo_A=rs[0]["action_kl"]["A"], direct_A=direct[0]["A"],
                                       agree=bool(math.isclose(rs[0]["spread"], direct[0]["spread"], rel_tol=1e-6)))
    print("learning runs:", {f"{a}/{r}": v["n"] for (a, r), v in learn.items()})
    print("zero-objective:", {f"{a}/{r}": v["n"] for (a, r), v in zero.items()})

    # ---- checks
    def sp(tab, arm, row):
        return tab[(arm, row)]["spread"][0] if (arm, row) in tab else None
    checks = {}
    for arm in ("sgd", "adam-decoupled"):
        res = {}
        for row in ("C1", "C2", "C3"):
            s = sp(zero, arm, row)
            if s is not None:
                pf = zero[(arm, row)]["predicted"]["particle_form"]; k = zero[(arm, row)]["kurtosis"][0]
                res[row] = dict(spread=s, predicted=pf, ratio=s / pf, within_10pct=bool(abs(s / pf - 1) <= 0.10), kurtosis=k, gaussian=bool(2.7 <= k <= 3.3))
        for row in ("C4", "C5"):
            s = sp(zero, arm, row)
            if s is not None:
                res[row] = dict(spread=s, zero=bool(s == 0.0))
        if res:
            res["holds"] = all(v.get("within_10pct", True) and v.get("gaussian", True) and v.get("zero", True) for v in res.values())
            checks[f"E1_{arm}"] = res
    res = {}
    for row in ("C1", "C2", "C3"):
        s = sp(zero, "adam-coupled", row)
        if s is not None:
            lap = zero[("adam-coupled", row)]["predicted"]["adam_laplace"]; k = zero[("adam-coupled", row)]["kurtosis"][0]
            tol = 1.5 if row == "C1" else 1.25
            res[row] = dict(spread=s, laplace=lap, ratio=s / lap, within=bool(1 / tol <= s / lap <= tol), kurtosis=k, heavy_tailed=bool(k > 4))
    if "C2" in res and "C3" in res:
        res["C3_over_C2"] = res["C3"]["spread"] / res["C2"]["spread"]; res["independent_of_kappa2"] = bool(0.8 <= res["C3_over_C2"] <= 1.25)
    for row in ("C4", "C5"):
        s = sp(zero, "adam-coupled", row)
        if s is not None:
            res[row] = dict(spread=s, below_1e3=bool(s < 1e-3))
    if res:
        res["holds"] = all(v.get("within", True) and v.get("heavy_tailed", True) and v.get("below_1e3", True) for v in res.values() if isinstance(v, dict)) and res.get("independent_of_kappa2", True)
        checks["E1_adam-coupled"] = res
    for arm, name in (("sgd", "E2"), ("adam-decoupled", "E3")):
        res = {}
        for row, s0row in (("C1", "C4"), ("C2", "C4"), ("C3", "C5")):
            s, s0 = sp(learn, arm, row), sp(learn, arm, s0row)
            if s is not None and s0 is not None:
                sigma = learn[(arm, row)]["predicted"]["particle_form"]; target = math.sqrt(sigma ** 2 + s0 ** 2)
                res[row] = dict(spread=s, sigma=sigma, s0=s0, target=target, ratio=s / target, within_1_3=bool(1 / 1.3 <= s / target <= 1.3), ratio_to_sigma=s / sigma)
        if arm == "adam-decoupled" and sp(learn, arm, "C2") and sp(learn, arm, "C3"):
            res["C3_over_C2"] = sp(learn, arm, "C3") / sp(learn, arm, "C2"); res["ratio_2_to_4"] = bool(2 <= res["C3_over_C2"] <= 4)
        if res:
            res["holds"] = all(v["within_1_3"] for v in res.values() if isinstance(v, dict)) and res.get("ratio_2_to_4", True)
            checks[name] = res
    s2, s3 = sp(learn, "adam-coupled", "C2"), sp(learn, "adam-coupled", "C3")
    if s2 and s3:
        lap = math.sqrt(2) * 0.101
        checks["E4"] = dict(spread_C2=s2, spread_C3=s3, C3_over_C2=s3 / s2, independent=bool(1 / 1.5 <= s3 / s2 <= 1.5), laplace=lap, ratios_to_laplace=[s2 / lap, s3 / lap],
                            holds=bool(1 / 1.5 <= s3 / s2 <= 1.5 and 1 / 1.5 <= s2 / lap <= 1.5 and 1 / 1.5 <= s3 / lap <= 1.5))
    # first protocol's checks on the decoupled arm
    a1, a2, a3, a4 = (sp(learn, "adam-decoupled", r) for r in ("C2", "C3", "C4", "C5"))
    if a1 and a3:
        net = math.sqrt(max(a1 ** 2 - a3 ** 2, 0)); checks["D1"] = dict(holds=bool(0.077 <= net <= 0.131) if a3 <= 0.077 else None, net_spread=net, spread=a1, s0=a3, predicted=0.1005)
    if a1 and a2:
        checks["D2"] = dict(holds=bool(2 <= a2 / a1 <= 4), ratio=a2 / a1); checks["O16_survives_decoupled"] = dict(holds=bool(0.091 <= a1 <= 0.172 and 0.091 <= a2 <= 0.172 and a2 / a1 < 1.5), spreads=[a1, a2])
    if a3 is not None and a4 is not None:
        checks["D3"] = dict(holds=bool(a3 <= 0.1 and a4 <= 1.0 and a3 > 3.2e-4 and a4 > a3), s0=[a3, a4])
    # ---- the amendment's checks (F1)-(F4) on the rows C7-C9
    for arm in ("sgd", "adam-decoupled"):
        res = {}
        for row in ("C7", "C8", "C9"):
            if (arm, row) in zero:
                z = zero[(arm, row)]; s = z["spread"][0]; pf = pred_value(arm, z); k = z["kurtosis"][0]
                res[row] = dict(spread=s, predicted=pf, ratio=s / pf, within_10pct=bool(abs(s / pf - 1) <= 0.10), kurtosis=k, gaussian=bool(2.7 <= k <= 3.3))
                if row == "C9":
                    res[row]["trace_max_dev"] = z["trace_ratio_max_dev_from_10th_rollout"]; res[row]["trace_within_5pct"] = bool(z["trace_ratio_max_dev_from_10th_rollout"] <= 0.05)
        if res:
            res["holds"] = all(v["within_10pct"] and v["gaussian"] and v.get("trace_within_5pct", True) for v in res.values() if isinstance(v, dict))
            checks[f"F1_{arm}"] = res
    for arm, name in (("sgd", "F2"), ("adam-decoupled", "F3")):
        res = {}
        for row in ("C7", "C8", "C9"):
            if (arm, row) in learn:
                s = sp(learn, arm, row); s0 = sp(learn, arm, "C4") if row != "C9" else sp(learn, arm, "C6")
                if s0 is None:
                    continue
                sigma = pred_value(arm, learn[(arm, row)]); target = math.sqrt(sigma ** 2 + s0 ** 2)
                res[row] = dict(spread=s, sigma=sigma, s0=s0, target=target, ratio=s / target, within_1_3=bool(1 / 1.3 <= s / target <= 1.3), ratio_to_sigma=s / sigma)
        if res:
            res["holds"] = all(v["within_1_3"] for v in res.values() if isinstance(v, dict))
            checks[name] = res
    seq = {}
    for arm in ARMS:
        rows = [r for r in ("C4", "C7", "C1", "C2", "C8") if (arm, r) in learn]
        if len(rows) == 5:
            q = [learn[(arm, r)]["risk"][0] for r in rows]; sp_ = [learn[(arm, r)]["spread"][0] for r in rows]; A = [learn[(arm, r)]["A"][0] for r in rows]
            seq[arm] = dict(rows=rows, kappa1=[ROWS[r][1] for r in rows], spread=sp_, risk=q, U=[learn[(arm, r)]["U"][0] for r in rows], A=A, sigma_a=[learn[(arm, r)]["sigma_a"][0] for r in rows],
                            risk_range=[[min(p["risk"] for p in learn[(arm, r)]["per_seed"]), max(p["risk"] for p in learn[(arm, r)]["per_seed"])] for r in rows],
                            spread_increasing=bool(all(np.diff(sp_) > 0)), A_increasing=bool(all(np.diff(A) > 0)), risk_increasing=bool(all(np.diff(q) > 0)),
                            noise_alone=(dict(risk=learn[(arm, "C9")]["risk"][0], U=learn[(arm, "C9")]["U"][0], A=learn[(arm, "C9")]["A"][0], spread=learn[(arm, "C9")]["spread"][0],
                                              unregularized_risk=learn[(arm, "C6")]["risk"][0], unregularized_U=learn[(arm, "C6")]["U"][0]) if (arm, "C9") in learn and (arm, "C6") in learn else None))
    if seq:
        checks["F4"] = seq

    # ---- comparisons (E5)
    comparisons = {"regularizer_bias": {}, "optimizer_effect": {}}
    if ("sgd", "C6") in learn:
        ref = {p["seed"]: p for p in learn[("sgd", "C6")]["per_seed"]}
        for row in ("C1", "C2", "C3", "C4", "C5", "C7", "C8", "C9"):
            if ("sgd", row) in learn:
                pairs = [(p, ref[p["seed"]]) for p in learn[("sgd", row)]["per_seed"] if p["seed"] in ref]
                if pairs:
                    comparisons["regularizer_bias"][row] = {k: ms([a[k] - b[k] for a, b in pairs]) for k in ("risk", "U", "gap", "gap_rel", "A", "bound_slack")}
                    comparisons["regularizer_bias"][row]["n_pairs"] = len(pairs)
    for arm in ("adam-coupled", "adam-decoupled"):
        for row in ROWS:
            if (arm, row) in learn and ("sgd", row) in learn:
                a = {p["seed"]: p for p in learn[(arm, row)]["per_seed"]}; b = {p["seed"]: p for p in learn[("sgd", row)]["per_seed"]}
                common = sorted(set(a) & set(b))
                comparisons["optimizer_effect"][f"{arm}/{row}"] = dict(
                    gap_diff_means=learn[(arm, row)]["gap"][0] - learn[("sgd", row)]["gap"][0], gap_rel_diff_means=learn[(arm, row)]["gap_rel"][0] - learn[("sgd", row)]["gap_rel"][0],
                    risk_diff_means=learn[(arm, row)]["risk"][0] - learn[("sgd", row)]["risk"][0], U_diff_means=learn[(arm, row)]["U"][0] - learn[("sgd", row)]["U"][0],
                    paired_gap_diff=ms([a[s]["gap"] - b[s]["gap"] for s in common]) if common else None, n_pairs=len(common),
                    se_of_mean_difference=math.sqrt(learn[(arm, row)]["gap"][1] ** 2 / learn[(arm, row)]["n"] + learn[("sgd", row)]["gap"][1] ** 2 / learn[("sgd", row)]["n"]))

    # ---- per-coordinate diagnostic (E6)
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
    coords = {}
    files = {("adam-coupled", "C2"): sorted(COORD.glob("wear-noise__campaign__k1_0.101__k2_10__seed*.npz")), ("adam-coupled", "C1"): sorted(COORD.glob("wear-noise__campaign__k1_0.009__k2_10__seed*.npz")),
             ("adam-coupled", "C4"): sorted(COORD.glob("wear-anchor__campaign-matched__kappa10__seed*.npz")),
             ("adam-decoupled", "C2"): sorted(COORD.glob("wear-noise-decoupled-v68__campaign__k1_0.101__k2_10__seed*.npz")),
             ("adam-decoupled", "C3"): sorted(COORD.glob("wear-noise-decoupled-v68__campaign__k1_0.101__k2_1__seed*.npz")),
             ("adam-decoupled", "C4"): sorted(COORD.glob("wear-noise-decoupled-v68__campaign__k1_0__k2_10__seed*.npz")),
             ("adam-decoupled", "C5"): sorted(COORD.glob("wear-noise-decoupled-v68__campaign__k1_0__k2_1__seed*.npz"))}
    for arm in ARMS:
        for row in ROWS:
            k2, k1 = ROWS[row]
            fs = sorted(NEW.glob(f"{arm}__lr{LR[arm]:g}__k1_{k1:g}__k2_{k2:g}__seed*/coordinates.npz"))
            if fs:
                files.setdefault((arm, row), []); files[(arm, row)] += fs
    for (arm, row), fs in files.items():
        if not fs:
            continue
        rho_v, rho_g, dec = [], [], []
        for f in fs:
            z = np.load(f, allow_pickle=False); x = np.abs(z["x"]); v = z["adam_v"]; g2 = z["g2_sum"]
            if np.all(np.isfinite(v)):
                sv = np.sqrt(v); rho_v.append(spearman(x, sv))
                bins = np.quantile(sv, np.linspace(0, 1, 11)); idx = np.clip(np.searchsorted(bins, sv, side="right") - 1, 0, 9)
                dec.append([float(np.median(x[idx == i])) for i in range(10)])
            if np.all(np.isfinite(g2)) and g2.sum() > 0:
                rho_g.append(spearman(x, np.sqrt(g2)))
        coords[f"{arm}/{row}"] = dict(n=len(fs), rank_corr_abs_x_vs_sqrt_adam_v=(ms(rho_v) if rho_v else None), rank_corr_abs_x_vs_rms_objective_grad=(ms(rho_g) if rho_g else None),
                                      median_abs_x_by_decile_of_sqrt_v=(np.mean(dec, axis=0).tolist() if dec else None))
    for k, v in checks.items():
        print(k, v.get("holds"), {kk: vv for kk, vv in v.items() if kk != "holds"})
    for k, v in coords.items():
        print("coords", k, v["rank_corr_abs_x_vs_sqrt_adam_v"], v["rank_corr_abs_x_vs_rms_objective_grad"])

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.4), gridspec_kw=dict(wspace=0.65))
    col = {"sgd": PAL[2], "adam-coupled": PAL[1], "adam-decoupled": PAL[0]}; mk = {"sgd": "o", "adam-coupled": "s", "adam-decoupled": "D"}
    ax = axes[0]
    grid = np.geomspace(0.5, 20, 50)
    amended = any((a, r) in learn for a in ARMS for r in ("C7", "C8", "C9"))
    ax.plot(grid, np.sqrt(0.101 / grid), color=MUTED, lw=0.8, ls=":", label=("$\\sqrt{\\kappa_1/\\kappa_2}$ at each $\\kappa_1$" if amended else "$\\sqrt{\\kappa_1/\\kappa_2}$ at $\\kappa_1=0.101$"))
    ax.plot(grid, np.sqrt(0.009 / grid), color=MUTED, lw=0.8, ls=":")
    ax.axhline(math.sqrt(2) * 0.101, color=MUTED, lw=0.8, ls="--", label="$\\sqrt{2}\\,\\kappa_1$")
    ax.axhline(math.sqrt(2) * 0.009, color=MUTED, lw=0.8, ls="--")
    X0 = 0.25                                                      # the position of kappa_2 = 0 (no pull) on the logarithmic axis
    if amended:
        for k1 in (0.001, 0.989):
            ax.plot(grid, np.sqrt(k1 / grid), color=MUTED, lw=0.8, ls=":")
        for arm in ("sgd", "adam-decoupled"):
            rw = predicted(arm, "C9", LR[arm])["random_walk"]; x = X0 * {"sgd": 0.88, "adam-coupled": 1.0, "adam-decoupled": 1.13}[arm]
            ax.plot([x / 1.1, x * 1.1], [rw, rw], color=MUTED, lw=0.8, ls="-.", label=("random walk $\\sqrt{2\\kappa_1\\sum_t\\alpha_t}$" if arm == "sgd" else None))
    for arm in ARMS:
        for row in ("C1", "C2", "C3", "C4", "C5", "C7", "C8", "C9"):
            if (arm, row) in learn:
                s = learn[(arm, row)]; x = (s["kappa2"] if s["kappa2"] > 0 else X0) * {"sgd": 0.88, "adam-coupled": 1.0, "adam-decoupled": 1.13}[arm]
                ax.errorbar([x], [max(s["spread"][0], 1.2e-4)], yerr=[s["spread"][1]], color=col[arm], marker=mk[arm], ms=4, lw=0, elinewidth=1.0, label=(f"{arm}, learning" if row == "C2" else None))
            if (arm, row) in zero:
                s = zero[(arm, row)]; x = (s["kappa2"] if s["kappa2"] > 0 else X0) * {"sgd": 0.88, "adam-coupled": 1.0, "adam-decoupled": 1.13}[arm]
                ax.plot([x], [max(s["spread"][0], 1.2e-4)], color=col[arm], marker=mk[arm], ms=4, lw=0, mfc="none", label=(f"{arm}, zero objective" if row == "C2" else None))
    ax.set_xscale("log"); ax.set_yscale("log")
    if amended:
        ax.set_xlim(0.18, 20); ax.set_ylim(1e-4, 5.0); ax.set_xticks([X0, 1, 10]); ax.set_xticklabels(["$0$", "$1$", "$10$"])
    else:
        ax.set_xlim(0.5, 20); ax.set_ylim(1e-4, 2.0); ax.set_xticks([1, 10]); ax.set_xticklabels(["$1$", "$10$"])
    ax.set_xlabel("pull $\\kappa_2$"); ax.set_ylabel("per-parameter spread"); panel_label(ax, "a")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax])
    ax = axes[1]
    m_ex, q_ex, U_arr, _ = frontier(bench, "reeval"); o = np.argsort(q_ex)
    ax.plot(q_ex[o] * 100, U_arr[o], color=TEXT2, lw=1.0, label="exact frontier")
    for arm in ARMS:
        for row in ("C2", "C3", "C4", "C6", "C8", "C9"):
            if (arm, row) in learn:
                s = learn[(arm, row)]
                ax.plot([p["risk"] * 100 for p in s["per_seed"]], [p["U"] for p in s["per_seed"]], color=col[arm], marker=mk[arm], ms=3, lw=0, alpha={"C2": 0.9, "C3": 0.5, "C4": 0.9, "C6": 0.3, "C8": 0.5, "C9": 0.3}[row],
                        mfc=("none" if row in ("C3", "C6", "C8", "C9") else col[arm]), label=(arm if row == "C2" else None))
    if learn:
        anyrun = next(iter(learn.values()))["per_seed"][0]; ax.plot([anyrun["prior_risk"] * 100], [anyrun["prior_U"]], color=TEXT2, marker="*", ms=6, lw=0, label="prior policy")
    ax.set_xscale("log"); ax.set_xlabel("breakdown probability $q_\\beta$ (%)"); ax.set_ylabel("stopped utility $U$"); panel_label(ax, "b")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax])
    ax = axes[2]
    for key, c, lab in (("adam-coupled/C2", PAL[1], "coupled, $\\kappa_2=10$"), ("adam-decoupled/C2", PAL[0], "decoupled, $\\kappa_2=10$"), ("adam-coupled/C3", PAL[1], "coupled, $\\kappa_2=1$")):
        if key in coords and coords[key]["median_abs_x_by_decile_of_sqrt_v"]:
            ax.plot(range(1, 11), coords[key]["median_abs_x_by_decile_of_sqrt_v"], color=c, marker=".", lw=1.0, ls=("-" if key.endswith("C2") else "--"), label=lab)
    ax.set_yscale("log"); ax.set_xlabel("decile of $\\sqrt{v_i}$"); ax.set_ylabel("median $|\\theta_i-\\theta_{0,i}|$"); panel_label(ax, "c")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax])
    save(fig, "wear-optimizers-v68", HERE)

    # ---- table
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
    lines = ["\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llrrrrrrrrrr}", "\\toprule",
             "Arm & $\\kappa_1$ & $\\kappa_2$ & $q_\\beta$ (\\%) & $U$ & gap & slack (\\%) & $\\mathcal A$ & spread & predicted & ratio & $\\sigma_a$ \\\\", "\\midrule"]
    for arm in ARMS:
        for row in ROWS:
            if (arm, row) in learn:
                s = learn[(arm, row)]; pf = pred_value(arm, s)
                pred = (f"{pf:.3f}" if pf else "--"); ratio = (f"{s['spread'][0] / pf:.2f}" if pf else "--")
                lines.append(f"{arm} & ${s['kappa1']:g}$ & ${s['kappa2']:g}$ & ${100 * s['risk'][0]:.2f}\\pm{100 * s['risk'][1]:.2f}$ & ${s['U'][0]:.2f}\\pm{s['U'][1]:.2f}$ & ${s['gap'][0]:.2f}$ & "
                             f"${100 * s['bound_slack'][0]:.1f}$ & ${sci(s['A'][0])}$ & ${sci(s['spread'][0])}$ & {pred} & {ratio} & ${s['sigma_a'][0]:.2f}$ \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\midrule\\multicolumn{12}{l}{zero objective (direct simulation)}\\\\"
    for arm in ARMS:
        for row in ZERO_ROWS:
            if (arm, row) in zero:
                s = zero[(arm, row)]; pf = pred_value(arm, s)
                lines.append(f"{arm} & ${s['kappa1']:g}$ & ${s['kappa2']:g}$ & ${100 * s['risk'][0]:.2f}\\pm{100 * s['risk'][1]:.2f}$ & ${s['U'][0]:.2f}\\pm{s['U'][1]:.2f}$ & ${s['gap'][0]:.2f}$ & "
                             f"${100 * s['bound_slack'][0]:.1f}$ & ${sci(s['A'][0])}$ & ${sci(s['spread'][0])}$ & {(f'{pf:.3f}' if pf else '--')} & {(f'{s['spread'][0] / pf:.2f}' if pf else '--')} & ${s['sigma_a'][0]:.2f}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (HERE / "wear-optimizers-v68.tex").write_text("\n".join(lines) + "\n")
    app = application()
    out = dict(lr=LR, rows=ROWS, learning={f"{a}/{r}": v for (a, r), v in learn.items()}, zero_objective={f"{a}/{r}": v for (a, r), v in zero.items()},
               ppo_zero_grad_checks=checks_ppo, checks=checks, comparisons=comparisons, coordinates=coords, d_params=D, application=app)
    (HERE / "wear-optimizers-v68.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(lines))


# ---- part B: the application over six seeds per arm (the amendment's (F7))
APP_SOURCES = {("adam-coupled", 10): [APP / "runs" / f"anchored_F50_k10_s{s}" for s in range(1, 7)],
               ("adam-coupled", 0): [APP / "runs" / f"anchored_F50_k0_s{s}" for s in range(1, 7)],
               ("adam-decoupled", 10): [APP / "runs-decoupled-v68" / "anchored-decoupled_F50_k10_s1"] + [APP / "runs-optimizers-v68" / f"adam-decoupled_lr0.0003_anchored_F50_k10_s{s}" for s in range(2, 7)],
               ("adam-decoupled", 0): [APP / "runs-optimizers-v68" / f"adam-decoupled_lr0.0003_anchored_F50_k0_s{s}" for s in range(1, 7)],
               ("sgd", 10): [APP / "runs-optimizers-v68" / f"sgd_lr0.01_anchored_F50_k10_s{s}" for s in range(1, 7)],
               ("sgd", 0): [APP / "runs-optimizers-v68" / f"sgd_lr0.01_anchored_F50_k0_s{s}" for s in range(1, 7)]}
A5 = dict(miss_tol=0.005, cost_tol=0.4, V_tol=0.5)


def app_run(d):
    f = d / "result.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text()); s = r["results"]["stochastic"]; pp = r["prior_policy"]
    return dict(seed=r["seed"], tag=d.name, lr=r["lr"], kappa_p=r["kappa_p"], distance=r["anchor_distance"], miss=s["p_T"], cost=s["summary"]["abatement_cost"], U=s["U"], V=s["V"],
                delta=r["delta"], gap=r["gap"], upper=s["validation"]["upper"], passed=bool(s["validation"]["passed"]), miss_mean_action=r["results"]["mean_action"]["p_T"],
                sigma_a=math.exp(r["final_log_std"]), prior=dict(miss=pp["miss"], cost=pp["abatement_cost"], U=pp["U"], V=pp["V"]), versions=r.get("versions"),
                form=(r.get("anchor") or {}).get("form", "coupled (record)"), contraction=(r.get("anchor") or {}).get("contraction_per_step"))


def application():
    tab = {}
    for (arm, kp), dirs in APP_SOURCES.items():
        runs = [x for x in (app_run(d) for d in dirs) if x is not None]
        if not runs:
            continue
        keys = ["distance", "miss", "cost", "U", "V", "delta", "gap", "upper", "miss_mean_action", "sigma_a"]
        t = {k: ms([x[k] for x in runs]) for k in keys}
        t.update(arm=arm, kappa_p=kp, lr=runs[0]["lr"], n=len(runs), passed=sum(x["passed"] for x in runs), per_seed=runs, prior=runs[0]["prior"])
        if kp == 10:
            pr = runs[0]["prior"]
            t["A5"] = dict(miss_within=bool(abs(t["miss"][0] - pr["miss"]) <= A5["miss_tol"]), cost_within=bool(abs(t["cost"][0] - pr["cost"]) <= A5["cost_tol"]),
                           V_within=bool(abs(t["V"][0] - pr["V"]) <= A5["V_tol"]), miss_diff=t["miss"][0] - pr["miss"], cost_diff=t["cost"][0] - pr["cost"], V_diff=t["V"][0] - pr["V"])
            t["A5"]["holds"] = t["A5"]["miss_within"] and t["A5"]["cost_within"]
        tab[(arm, kp)] = t
    effects = {"regularizer": {}, "optimizer": {}, "control_closer_to_prior_in_V": {}}
    for arm in ARMS:
        if (arm, 10) in tab and (arm, 0) in tab:
            a = {x["seed"]: x for x in tab[(arm, 10)]["per_seed"]}; c = {x["seed"]: x for x in tab[(arm, 0)]["per_seed"]}; common = sorted(set(a) & set(c))
            effects["regularizer"][arm] = {k: ms([a[s][k] - c[s][k] for s in common]) for k in ("miss", "cost", "V", "distance")}; effects["regularizer"][arm]["n_pairs"] = len(common)
            pr = tab[(arm, 10)]["prior"]
            effects["control_closer_to_prior_in_V"][arm] = dict(anchored_abs_V_diff=abs(tab[(arm, 10)]["V"][0] - pr["V"]), control_abs_V_diff=abs(tab[(arm, 0)]["V"][0] - pr["V"]),
                                                               anchored_moves_further_than_control=bool(abs(tab[(arm, 10)]["V"][0] - pr["V"]) > abs(tab[(arm, 0)]["V"][0] - pr["V"])))
    for arm in ("adam-coupled", "adam-decoupled"):
        for kp in (10, 0):
            if (arm, kp) in tab and ("sgd", kp) in tab:
                a = {x["seed"]: x for x in tab[(arm, kp)]["per_seed"]}; b = {x["seed"]: x for x in tab[("sgd", kp)]["per_seed"]}; common = sorted(set(a) & set(b))
                effects["optimizer"][f"{arm}/k{kp}"] = {k: ms([a[s][k] - b[s][k] for s in common]) for k in ("miss", "cost", "V", "distance")}; effects["optimizer"][f"{arm}/k{kp}"]["n_pairs"] = len(common)
    # reproduction checks: the decoupled seed 1 through the three-optimizer runner against the first protocol's record; the
    # decoupled residual control against the recorded Adam control (the same arithmetic, the toolbox version apart)
    repro = {}
    x = app_run(APP / "runs-optimizers-v68" / "adam-decoupled_lr0.0003_anchored_F50_k10_s1"); y = app_run(APP / "runs-decoupled-v68" / "anchored-decoupled_F50_k10_s1")
    if x and y:
        repro["decoupled_seed1_rerun"] = dict(rerun={k: x[k] for k in ("distance", "miss", "cost", "V")}, record={k: y[k] for k in ("distance", "miss", "cost", "V")},
                                              identical=bool(all(math.isclose(x[k], y[k], rel_tol=1e-9, abs_tol=1e-12) for k in ("distance", "miss", "cost", "V"))))
    if ("adam-decoupled", 0) in tab and ("adam-coupled", 0) in tab:
        a = {x["seed"]: x for x in tab[("adam-decoupled", 0)]["per_seed"]}; c = {x["seed"]: x for x in tab[("adam-coupled", 0)]["per_seed"]}
        rows = {s: dict(rerun={k: a[s][k] for k in ("distance", "miss", "cost", "V")}, record={k: c[s][k] for k in ("distance", "miss", "cost", "V")},
                        identical=bool(all(math.isclose(a[s][k], c[s][k], rel_tol=1e-9, abs_tol=1e-12) for k in ("distance", "miss", "cost", "V")))) for s in sorted(set(a) & set(c))}
        repro["decoupled_control_vs_recorded_adam_control"] = dict(per_seed=rows, all_identical=bool(rows) and all(v["identical"] for v in rows.values()))
    # the table
    lines = ["\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llrrrrrrrrr}", "\\toprule",
             "Arm & $\\kappa_p$ & lr & $n$ & anchor distance & miss $p_{51}$ & cost & $U$ & $V$ & validated upper & pass \\\\", "\\midrule"]
    if tab:
        pr = next(iter(tab.values()))["prior"]
        lines.append(f"prior policy & & & & $0$ & ${pr['miss']:.4f}$ & ${pr['cost']:.3f}$ & ${pr['U']:.3f}$ & ${pr['V']:.3f}$ & & \\\\")
    for arm in ARMS:
        for kp in (10, 0):
            if (arm, kp) in tab:
                t = tab[(arm, kp)]
                lines.append(f"{arm} & ${kp}$ & ${t['lr']:g}$ & {t['n']} & ${t['distance'][0]:.4f}\\pm{t['distance'][1]:.4f}$ & ${t['miss'][0]:.4f}\\pm{t['miss'][1]:.4f}$ & ${t['cost'][0]:.3f}\\pm{t['cost'][1]:.3f}$ & "
                             f"${t['U'][0]:.3f}$ & ${t['V'][0]:.3f}\\pm{t['V'][1]:.3f}$ & ${t['upper'][0]:.4f}$ & {t['passed']}/{t['n']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (HERE / "wear-optimizers-v68-application.tex").write_text("\n".join(lines) + "\n")
    for (arm, kp), t in tab.items():
        print(f"application {arm} k{kp}: n={t['n']} distance={t['distance'][0]:.4f} miss={t['miss'][0]:.4f} cost={t['cost'][0]:.3f} V={t['V'][0]:.3f} pass={t['passed']}/{t['n']}", (t.get("A5") or {}).get("holds"))
    for k, v in repro.items():
        print("application reproduction", k, v.get("identical", v.get("all_identical")))
    return dict(table={f"{a}/k{k}": v for (a, k), v in tab.items()}, effects=effects, reproduction=repro, A5_band=A5)


if __name__ == "__main__":
    main()
