"""Revision 41: summaries, table and figures of the wear-allowance study, regenerated from the archived
revision-38 campaign (wear-allowance/campaign/*/result.json, benchmark.json, protocol.json).

Changes against wear-allowance/analyze.py (review findings):
 * frontier convention: the learned policies are evaluated in the continuous simulator, so the exact
   frontier, the exact barrier curve and the price-risk law use the DP policies re-evaluated in the same
   simulator (benchmark rows[*]["nominal"]); the gap is computed against this frontier (gap_reeval) and,
   for the record, against the grid values (gap_grid, the revision-38 convention);
 * table: interior loss I_m, gap mean and max, action relative entropy A, mean-action risk; footnote on
   the two-cluster sample at m = 0.1;
 * price-risk panel: the m = 0.1 configuration as individual seeds; house-rule axis labels;
 * multiplier trace: y-label in price units, (1-beta)nu.
Revision 68: the table in the paper's terms (tolerance, action-regularized, the incumbent-rule row, the interior
loss as I_m in calligraphic type) and with the column "bound": the price-risk law with the idle policy as comparison
and the measured optimization error, per seed and averaged (details at price_risk_bound); figures at the journal's
print width with sans-serif lettering, line styles and markers beside colours, no panel titles.
Outputs (this directory): wear-campaign-table-v41.tex, wear-benchmark-conventions.tex, wear-frontier-v41,
wear-margin-v41, wear-dual-v41 (pdf/png), wear-summary-v41.json, wear-digests-v41.json.
Run with ~/.venvs/constrained-rl/bin/python (gymnasium is not needed: the incumbent rule is copied below).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parent
V38 = ROOT.parent / "wear-allowance"
sys.path.insert(0, str(ROOT.parent / "lib"))
sys.path.insert(0, str(V38))
sys.path.insert(0, str(ROOT))
from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, LS, MK, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from model import Model, simulate  # noqa: E402
from digest_v41 import digest  # noqa: E402
try:
    from env import Incumbent  # noqa: E402
except ImportError:  # gymnasium absent in the plotting environment; the rule itself needs only numpy
    class Incumbent:
        """Copy of wear-allowance/env.py: rate min(1, z / (span b)) with Gaussian dispersion before clipping."""

        def __init__(self, model, span=1.5, std=0.2):
            self.model, self.span, self.std = model, span, std

        def mean(self, z):
            rate = np.minimum(1.0, np.maximum(np.asarray(z), 0.0) / (self.span * self.model.budget))
            return 2.0 * rate - 1.0

        def policy(self, seed=0):
            rng = np.random.default_rng(seed)

            def policy(s, z):
                x = self.mean(z) + self.std * rng.standard_normal(len(np.atleast_1d(z)))
                return (np.clip(np.asarray(x, dtype=float), -1.0, 1.0) + 1.0) / 2.0
            return policy

STAGE = "campaign"
BETA = 0.95
C_FIXED, C_ADAPT, C_BARRIER, C_ANCHOR = PAL[0], PAL[1], PAL[2], PAL[3]
DIGESTS = {}


def cp_upper(k, n, err):
    return 1.0 if k == n else float(beta_dist.ppf(1 - err, k + 1, n - k))


def kl_bin(p, q):
    eps = 1e-300
    return p * np.log(max(p, eps) / q) + (1 - p) * np.log(max(1 - p, eps) / (1 - q))


def kl_upper(q_ref, A):
    """Largest p with kl(p || q_ref) <= A (the bound of the action-regularization theorem)."""
    lo, hi = q_ref, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if kl_bin(mid, q_ref) <= A else (lo, mid)
    return lo


def load():
    bench = json.loads((V38 / "benchmark.json").read_text())
    protocol = json.loads((V38 / "protocol.json").read_text())
    runs = {}
    for f in sorted((V38 / STAGE).glob("*/result.json")):
        arm = f.parent.name.split("__seed")[0]
        runs.setdefault(arm, []).append(json.loads(f.read_text()))
    return bench, protocol, runs


def frontier(bench, conv="reeval"):
    """Failure-only DP policies: price m, breakdown probability q, stopped utility U and the interpolant
    U_exact(q), linear in log q. conv='grid': DP grid values (revision-38 convention); conv='reeval': the same
    policies re-evaluated in the continuous simulator (benchmark 'nominal'), the convention of the learned policies."""
    rows = [r for r in bench["rows"] if r["kind"] == "failure"]
    m = np.array([r["m"] for r in rows])
    if conv == "grid":
        q = np.array([r["grid"]["q"] for r in rows]); U = np.array([r["grid"]["U"] for r in rows])
    else:
        q = np.array([r["nominal"]["risk"] for r in rows]); U = np.array([r["nominal"]["U"] for r in rows])
    order = np.argsort(q)
    return m, q, U, (lambda x: float(np.interp(np.log(max(x, q.min())), np.log(q[order]), U[order])))


def bench_point(r, conv):
    if conv == "grid" and r["grid"]:
        return r["grid"]["q"], r["grid"]["U"]
    return r["nominal"]["risk"], r["nominal"]["U"]


def price_risk_bound(bench, kind, m, s):
    """The price-risk law with the idle policy as comparison policy and the measured optimization error, for one
    learned policy with stochastic record s: q <= q(idle) + (1-beta)/m [sup U - U(idle) + I_m(idle) + delta].
    The idle policy has q = 0 and U = 0; sup U over the represented class is bounded by the exact unpriced optimum
    re-evaluated in the simulator; I_m(idle) is the interior loss of standing still, min{B(b), m} per period over the
    geometric horizon, for the barrier objective and zero for the failure-only price; delta is the shortfall of the
    policy's objective V_m = U - I_m - m q/(1-beta) (I_m only for the barrier objective) from the exact optimum's,
    floored at zero (a negative shortfall is Monte Carlo error of the benchmark). Capped at one."""
    model = Model(**bench["model"])
    U0 = [r for r in bench["rows"] if r["kind"] == "failure" and r["m"] == 0][0]["nominal"]["U"]
    ex = [r for r in bench["rows"] if r["kind"] == ("barrier" if kind == "barrier" else "failure") and r["m"] == m
          and (kind != "barrier" or r.get("theta") == model.theta)][0]["nominal"]
    barrier_obj = kind == "barrier"
    I_idle = min(model.theta * np.log1p(1.0), m) / (1 - BETA) if barrier_obj else 0.0
    V_ex = ex["U"] - (ex["I"] if barrier_obj else 0.0) - m * ex["risk"] / (1 - BETA)
    V = s["U"] - (s["I"] if barrier_obj else 0.0) - m * s["risk"] / (1 - BETA)
    delta = max(0.0, V_ex - V)
    return dict(bound=min(1.0, (1 - BETA) / m * (U0 + I_idle + delta)), delta=delta, delta_raw=V_ex - V, I_idle=I_idle, sup_U=U0)


def summarize(bench, protocol, runs):
    U_grid = frontier(bench, "grid")[3]; U_re = frontier(bench, "reeval")[3]
    n_candidates = sum(len(v) for v in runs.values())
    out = {}
    for arm in protocol["arms"]:
        if arm not in runs:
            continue
        rs = runs[arm]
        cfg = protocol["arms"][arm]
        q = np.array([r["stochastic"]["risk"] for r in rs]); U = np.array([r["stochastic"]["U"] for r in rs])
        n = rs[0]["stochastic"]["n"]
        ups = [cp_upper(r["stochastic"]["failures"], n, 0.05 / n_candidates) for r in rs]
        gap_grid = np.array([(U_grid(a) - b) / U_grid(a) for a, b in zip(q, U)])
        gap_re = np.array([(U_re(a) - b) / U_re(a) for a, b in zip(q, U)])
        s = dict(kind=cfg["kind"], cfg=cfg, seeds=len(rs), seed_ids=[r["seed"] for r in rs],
                 q_mean=q.mean(), q_sd=q.std(ddof=1), q_min=q.min(), q_max=q.max(), U_mean=U.mean(), U_sd=U.std(ddof=1),
                 gap_grid=gap_grid.tolist(), gap_grid_mean=gap_grid.mean(), gap_grid_max=gap_grid.max(),
                 gap_reeval=gap_re.tolist(), gap_reeval_mean=gap_re.mean(), gap_reeval_max=gap_re.max(),
                 pass05=int(sum(u <= 0.05 for u in ups)), pass01=int(sum(u <= 0.01 for u in ups)),
                 q90=float(np.mean([r["robust90"]["risk"] for r in rs])), q80=float(np.mean([r["robust80"]["risk"] for r in rs])),
                 near05=float(np.mean([r["stochastic"]["near"]["0.05"] for r in rs])),
                 I=[r["stochastic"]["I"] for r in rs], I_mean=float(np.mean([r["stochastic"]["I"] for r in rs])),
                 mean_action_q=float(np.mean([r["mean_action"]["risk"] for r in rs])),
                 mean_action_U=float(np.mean([r["mean_action"]["U"] for r in rs])),
                 multiplier=[r["final_multiplier"] for r in rs], q=q.tolist(), U=U.tolist())
        if cfg["kind"] == "anchored":
            s["A"] = [r["action_kl"]["A"] for r in rs]; s["A_mean"] = float(np.mean(s["A"]))
        if cfg["kind"] in ("fixed", "barrier") and cfg["m"] > 0:
            law = [price_risk_bound(bench, cfg["kind"], cfg["m"], r["stochastic"]) for r in rs]
            s["bound"] = [x["bound"] for x in law]; s["bound_mean"] = float(np.mean(s["bound"]))
            s["delta"] = [x["delta"] for x in law]; s["delta_mean"] = float(np.mean(s["delta"])); s["delta_raw_min"] = float(min(x["delta_raw"] for x in law))
            s["bound_factor_mean"] = s["bound_mean"] / s["q_mean"]; s["bound_factor_min"] = float(min(b / qq for b, qq in zip(s["bound"], q)))
            s["bound_I_idle"] = law[0]["I_idle"]; s["bound_sup_U"] = law[0]["sup_U"]
        out[arm] = s
    return out, n_candidates


def incumbent_stats(summary):
    """Breakdown probability and stopped utility of the incumbent rule (the action reference of the
    action-regularized configurations), simulated once; used by the table and by the multiplier figure."""
    anchored = {a: s for a, s in summary.items() if s["kind"] == "anchored"}
    if not anchored:
        return None
    cfg = next(iter(anchored.values()))["cfg"]
    model = Model(**cfg["model"]); inc = Incumbent(model, **cfg["incumbent"])
    ref = simulate(model, inc.policy(seed=3), n=200_000, seed=77)
    return dict(q=ref["risk"], U=ref["U"], span=cfg["incumbent"]["span"], model=model, incumbent=inc)


def table(summary, inc=None):
    def price(c):
        if c["kind"] == "adaptive":
            return f"tolerance ${c['alpha']:g}$" + (f", step ${c['dual_step']:g}$" if c["dual_step"] != 5 else "")
        if c["kind"] == "anchored":
            return f"$m=0.03$, $\\eta={c['eta']:g}$"
        return f"$m={c['m']:g}$"
    names = {"fixed": "Fixed price", "barrier": "Fixed price, barrier", "adaptive": "Adaptive multiplier", "anchored": "Action-regularized"}
    lines = []
    last = None
    for arm, s in summary.items():
        if arm.startswith("_"):
            continue
        kind = s["kind"]
        label = names[kind] if kind != last else ""
        last = kind
        setting = price(s["cfg"])
        if kind == "fixed" and s["cfg"]["m"] == 0.1:
            setting += "\\textsuperscript{$\\dagger$}"
        A = f"${s['A_mean']:.3g}$" if "A_mean" in s else "--"
        bound = f"${100*s['bound_mean']:.1f}$" if "bound_mean" in s else "--"
        lines.append(f"{label} & {setting} & ${100*s['q_mean']:.2f}\\pm{100*s['q_sd']:.2f}$ & {bound} & ${s['U_mean']:.2f}\\pm{s['U_sd']:.2f}$ & "
                     f"${s['I_mean']:.2f}$ & ${100*s['gap_reeval_mean']:.1f}$ (${100*s['gap_reeval_max']:.1f}$) & ${s['pass05']}/{s['seeds']}$ & "
                     f"${100*s['near05']:.1f}$ & ${100*s['q90']:.1f}$ & {A} & ${100*s['mean_action_q']:.2f}$ \\\\")
    if inc:
        lines += ["\\midrule", f"Incumbent rule & $a=\\min\\{{1,z/({inc['span']:g}b)\\}}$ & ${100*inc['q']:.2f}$ & -- & ${inc['U']:.2f}$ & -- & -- & -- & -- & -- & $0$ & -- \\\\"]
    note = ("\\multicolumn{12}{@{}l@{}}{\\textsuperscript{$\\dagger$}three seeds remained at the unpriced optimum ($q_\\beta=0.59$), "
            "four reached $0.11$ to $0.17$ and one $0.33$; the mean is of a two-cluster sample.} \\\\")
    (ROOT / "wear-campaign-table-v41.tex").write_text(
        "\\setlength{\\tabcolsep}{2.5pt}\\begin{tabular}{llrrrrrrrrrr}\n\\toprule\n"
        "Configuration & Setting & $q_\\beta$ (\\%) & bound (\\%) & $U$ & $\\mathcal I_m$ & gap (\\%) & pass & near (\\%) & at $0.9b$ (\\%) & $\\mathcal A$ & mean act.\\ $q_\\beta$ (\\%) \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n" + note + "\n\\end{tabular}\n")


def conventions_table(bench):
    """Online table: the DP policies' breakdown probability and stopped utility on the grid and re-evaluated
    in the continuous simulator (the benchmark's discretization error)."""
    rows = [r for r in bench["rows"] if r["kind"] == "failure"]

    def pq(q):
        return f"{100*q:.2f}" if 100 * q >= 1 else f"{100*q:.3f}"
    lines = [f"${r['m']:g}$ & ${pq(r['grid']['q'])}$ & ${pq(r['nominal']['risk'])}$ & ${r['grid']['U']:.3f}$ & ${r['nominal']['U']:.3f}$ \\\\" for r in rows]
    (ROOT / "wear-benchmark-conventions.tex").write_text(
        "\\setlength{\\tabcolsep}{2.5pt}\\begin{tabular}{rrrrr}\n\\toprule\n"
        "$m$ & $q_\\beta$ grid (\\%) & $q_\\beta$ re-evaluated (\\%) & $U$ grid & $U$ re-evaluated \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    return [dict(m=r["m"], q_grid=r["grid"]["q"], q_reeval=r["nominal"]["risk"], U_grid=r["grid"]["U"], U_reeval=r["nominal"]["U"],
                 dU_rel=(r["grid"]["U"] - r["nominal"]["U"]) / r["nominal"]["U"], dq_rel=(r["grid"]["q"] - r["nominal"]["risk"]) / r["nominal"]["risk"]) for r in rows]


def frontier_arrays(bench, conv):
    m_ex, q_ex, U_ex, _ = frontier(bench, conv)
    bar = [r for r in bench["rows"] if r["kind"] == "barrier" and r.get("theta") == 0.1]
    bq = np.array([bench_point(r, conv)[0] for r in bar]); bU = np.array([bench_point(r, conv)[1] for r in bar])
    pos = m_ex > 0
    bound = np.minimum(1, (1 - BETA) * U_ex[m_ex == 0][0] / m_ex[pos])
    return m_ex, q_ex, U_ex, bq, bU, bound


def fig_frontier(bench, summary):
    m_ex, q_ex, U_ex, bq, bU, bound = frontier_arrays(bench, "reeval")
    old = frontier_arrays(bench, "grid")
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.1), gridspec_kw=dict(wspace=0.62, width_ratios=[0.9, 1.2, 1.0]))
    # (a) exact optimal policies at the mean price (unchanged)
    ax = axes[0]
    z = np.array(bench["z_grid"]); mid = len(bench["s_grid"]) // 2
    arrs = []
    for key, lab, col, ls in (("failure-m0", "unpriced", MUTED, ":"), ("failure-m1", "price $m=1$", C_FIXED, "-"), ("barrier-m1", "barrier, $m=1$", C_BARRIER, "--")):
        y = np.array(bench["policy_tables"][key])[mid]; arrs.append(y)
        ax.plot(z, y, color=col, ls=ls, label=lab)
    d = digest(z, *arrs); DIGESTS["wear-frontier/a policy tables"] = dict(before=d, after=d, changed=False)
    ax.set_xlabel("remaining allowance $z/b$"); ax.set_ylabel("operating rate"); ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.05)
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax]); panel_label(ax, "a")
    # (b) frontier: exact curves under the re-evaluated convention
    ax = axes[1]
    ax.plot(q_ex, U_ex, color=TEXT2, lw=1.2, label="exact frontier", zorder=1)
    for mm in (0.1, 1, 10):
        i = int(np.argmin(np.abs(m_ex - mm)))
        ax.annotate(f"$m={mm:g}$", (q_ex[i], U_ex[i]), textcoords="offset points", xytext={0.1: (-40, -12), 1: (-38, 3), 10: (-44, -4)}[mm], fontsize=FS, color=TEXT2)
        ax.plot([q_ex[i]], [U_ex[i]], marker="|", color=TEXT2, ms=5)
    ax.plot(bq, bU, color=C_BARRIER, lw=1.0, ls="--", label="exact, barrier", zorder=1)
    DIGESTS["wear-frontier/b exact frontier and barrier curve"] = dict(before=digest(old[1], old[2], old[3], old[4]), after=digest(q_ex, U_ex, bq, bU), changed=True)
    learned = []
    for kind, col, mk, lab in (("fixed", C_FIXED, "o", "fixed price"), ("adaptive", C_ADAPT, "s", "adaptive multiplier"), ("barrier", C_BARRIER, "^", "fixed price, barrier")):
        first = True
        for arm, s in summary.items():
            if s["kind"] != kind or (kind == "adaptive" and s["cfg"]["dual_step"] != 5):
                continue
            learned += [s["q_mean"], s["U_mean"], s["q_min"], s["q_max"]]
            ax.errorbar(s["q_mean"], s["U_mean"], xerr=[[s["q_mean"] - s["q_min"]], [s["q_max"] - s["q_mean"]]], fmt=mk, color=col, ms=4,
                        mec="white", mew=0.6, elinewidth=0.8, label=lab if first else None, zorder=3)
            first = False
    d = digest(learned); DIGESTS["wear-frontier/b learned points"] = dict(before=d, after=d, changed=False)
    ax.set_xscale("log"); ax.set_xlabel("breakdown probability $q_\\beta$"); ax.set_ylabel("stopped utility $U$")
    ax.axvline(0.05, color=MUTED, lw=0.7, ls=":"); legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax]); panel_label(ax, "b")
    # (c) price-risk law: exact curve and bound re-evaluated; m = 0.1 as individual seeds
    ax = axes[2]
    pos = m_ex > 0
    ax.plot(m_ex[pos], q_ex[pos], color=TEXT2, label="exact")
    ax.plot(m_ex[pos], bound, color=MUTED, ls="--", label="upper bound")
    DIGESTS["wear-frontier/c exact curve and bound"] = dict(before=digest(old[0][pos], old[1][pos], old[5]), after=digest(m_ex[pos], q_ex[pos], bound), changed=True)
    xs, ys, lo, hi, sx, sy = [], [], [], [], [], []
    xs_old, ys_old, lo_old, hi_old = [], [], [], []
    for arm, s in summary.items():
        if s["kind"] == "fixed" and s["cfg"]["m"] > 0:
            xs_old.append(s["cfg"]["m"]); ys_old.append(s["q_mean"]); lo_old.append(s["q_mean"] - s["q_min"]); hi_old.append(s["q_max"] - s["q_mean"])
            if s["cfg"]["m"] == 0.1:
                sx += [s["cfg"]["m"]] * len(s["q"]); sy += s["q"]
            else:
                xs.append(s["cfg"]["m"]); ys.append(s["q_mean"]); lo.append(s["q_mean"] - s["q_min"]); hi.append(s["q_max"] - s["q_mean"])
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=C_FIXED, ms=4, mec="white", mew=0.6, elinewidth=0.8, label="learned, mean and range")
    ax.plot(sx, sy, marker="o", ms=2.6, color=C_FIXED, mec="white", mew=0.3, ls="none", label="learned seeds, $m=0.1$", zorder=4)
    DIGESTS["wear-frontier/c learned points (plotted arrays)"] = dict(before=digest(xs_old, ys_old, lo_old, hi_old), after=digest(xs, ys, lo, hi, sx, sy), changed=True,
                                                                     note="representation change only: m = 0.1 shown as seeds instead of mean and range")
    d = digest([s["q"] for a, s in summary.items() if s["kind"] == "fixed" and s["cfg"]["m"] > 0])
    DIGESTS["wear-frontier/c learned per-seed q (underlying data)"] = dict(before=d, after=d, changed=False)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("price of failure risk $m$"); ax.set_ylabel("breakdown probability $q_\\beta$")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax]); panel_label(ax, "c")
    save(fig, "wear-frontier-v41", ROOT)


def fig_margin(bench, summary):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, key, lab in ((axes[0], "allowance90", "a"), (axes[1], "allowance80", "b")):
        sets = (("failure", None, TEXT2, "-", "failure-only prices"), ("barrier", 0.1, C_BARRIER, "-.", "interior barrier"),
                ("tightened", 0.9, C_ANCHOR, "--", "plan with $0.9\\,b$"), ("tightened", 0.8, PAL[6], ":", "plan with $0.8\\,b$"))
        before, after = [], []
        for kind, par, col, ls, name in sets:
            rows = [r for r in bench["rows"] if r["kind"] == kind and (par is None or r.get("theta", r.get("planned_allowance")) == par) and r["m"] >= 0.3]
            U = [r["nominal"]["U"] for r in rows]
            U_old = [r["grid"]["U"] if r["grid"] else r["nominal"]["U"] for r in rows]
            y = [max(r[key]["risk"], 1e-4) for r in rows]
            before += [U_old, y]; after += [U, y]
            ax.plot(U, y, color=col, ls=ls, marker=".", ms=3, label=name)
        DIGESTS[f"wear-margin/{lab} exact curves"] = dict(before=digest(*before), after=digest(*after), changed=True)
        pts = []
        for arm, s in summary.items():
            if s["kind"] in ("fixed", "barrier") and s["cfg"]["m"] >= 0.3:
                pts += [s["U_mean"], s["q90" if key == "allowance90" else "q80"]]
                ax.plot([s["U_mean"]], [s["q90" if key == "allowance90" else "q80"]], marker="o" if s["kind"] == "fixed" else "^",
                        color=C_FIXED if s["kind"] == "fixed" else C_BARRIER, ms=4, mec="white", mew=0.6, ls="none")
        d = digest(pts); DIGESTS[f"wear-margin/{lab} learned points"] = dict(before=d, after=d, changed=False)
        ax.set_yscale("log"); ax.set_xlabel("stopped utility $U$"); panel_label(ax, lab)
    axes[0].set_ylabel("breakdown probability $q_\\beta$")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", ls="none", color=C_FIXED, ms=4), Line2D([], [], marker="^", ls="none", color=C_BARRIER, ms=4)]
    labels += ["learned, fixed price", "learned, barrier"]
    legend_below(fig, handles, labels, ncol=3)
    save(fig, "wear-margin-v41", ROOT)


def fig_dual(summary, runs, protocol, inc=None):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.2), gridspec_kw=dict(wspace=0.35))
    ax = axes[0]
    arrs = []
    for arm, col, ls, lab in (("adaptive-a0.05-s5", C_ADAPT, "-", "tolerance $0.05$"), ("adaptive-a0.01-s5", PAL[6], "--", "tolerance $0.01$"), ("adaptive-a0-s5", PAL[7], "-.", "tolerance $0$")):
        if arm not in runs:
            continue
        tr = np.array([[t["multiplier"] for t in r["trace"]] for r in runs[arm]]) * (1 - BETA)
        steps = np.array([t["steps"] for t in runs[arm][0]["trace"]]) / 1e6
        arrs += [steps, tr.mean(0), tr.min(0), tr.max(0)]
        ax.plot(steps, tr.mean(0), color=col, ls=ls, label=lab)
        ax.fill_between(steps, tr.min(0), tr.max(0), color=col, alpha=0.18, lw=0)
    d = digest(*arrs); DIGESTS["wear-dual/a multiplier traces"] = dict(before=d, after=d, changed=False, note="y-label changed to (1-beta)nu")
    ax.set_yscale("log"); ax.set_xlabel("environment steps (millions)"); ax.set_ylabel("multiplier in price units, $(1-\\beta)\\nu$")
    legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax]); ax.set_ylim(2e-2, 8); panel_label(ax, "a")
    ax = axes[1]
    anchored = {a: s for a, s in summary.items() if s["kind"] == "anchored"}
    if anchored:
        inc = inc or incumbent_stats(summary)
        ref = dict(risk=inc["q"], U=inc["U"])
        A = np.logspace(-3, 2, 200)
        ub = [kl_upper(ref["risk"], a) for a in A]
        ax.plot(A, ub, color=TEXT2, label="upper bound from $\\mathcal{A}$")
        ax.axhline(ref["risk"], color=MUTED, ls=":", lw=0.8)
        ax.annotate("incumbent rule", (1e-3, ref["risk"]), textcoords="offset points", xytext=(2, -11), fontsize=FS, color=TEXT2)
        pts = []
        for arm, s in anchored.items():
            pts += [s["A"], s["q"]]
            ax.plot(s["A"], s["q"], marker="o", ls="none", color=C_ANCHOR, ms=3.5, mec="white", mew=0.5)
            ax.annotate(f"$\\eta={s['cfg']['eta']:g}$", (np.mean(s["A"]), np.mean(s["q"])), textcoords="offset points", xytext={1.0: (12, -3), 0.1: (10, -3), 0.01: (-48, -16)}[s["cfg"]["eta"]], fontsize=FS, color=TEXT2)
        d = digest(A, ub, ref["risk"], *pts); DIGESTS["wear-dual/b bound and anchored points"] = dict(before=d, after=d, changed=False)
        ax.plot([], [], marker="o", ls="none", color=C_ANCHOR, ms=3.5, label="action-regularized policies")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5e-3, 1.5); ax.set_xlabel("stopped action relative entropy $\\mathcal{A}$"); ax.set_ylabel("breakdown probability $q_\\beta$")
        legend_below(fig, *ax.get_legend_handles_labels(), ncol=1, axes=[ax]); summary["_incumbent"] = dict(q=ref["risk"], U=ref["U"])
    panel_label(ax, "b")
    save(fig, "wear-dual-v41", ROOT)


def main():
    bench, protocol, runs = load()
    summary, n_candidates = summarize(bench, protocol, runs)
    inc = incumbent_stats(summary)
    table(summary, inc)
    conv = conventions_table(bench)
    fig_frontier(bench, summary); fig_margin(bench, summary); fig_dual(summary, runs, protocol, inc)
    summary["_family"] = n_candidates
    summary["_benchmark_conventions"] = conv
    (ROOT / "wear-summary-v41.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    (ROOT / "wear-digests-v41.json").write_text(json.dumps(DIGESTS, indent=1) + "\n")
    print("gap conventions per configuration (percent; grid = revision 38, reeval = revision 41):")
    for arm, s in summary.items():
        if arm.startswith("_"):
            continue
        print(f"{arm:20s} n={s['seeds']} q={100*s['q_mean']:.2f}+-{100*s['q_sd']:.2f}% U={s['U_mean']:.3f}+-{s['U_sd']:.3f} I={s['I_mean']:.2f} "
              f"gap_grid={100*s['gap_grid_mean']:.1f} (max {100*s['gap_grid_max']:.1f})  gap_reeval={100*s['gap_reeval_mean']:.1f} (max {100*s['gap_reeval_max']:.1f}) "
              f"pass05={s['pass05']}/{s['seeds']} near05={100*s['near05']:.1f}% q90={100*s['q90']:.1f}% mean-action q={100*s['mean_action_q']:.2f}%"
              + (f" A={s['A_mean']:.3g}" if "A_mean" in s else ""))
    s = summary["fixed-m0.1"]
    print("per-seed q at m = 0.1:", ", ".join(f"{sid}: {100*q:.2f}%" for sid, q in zip(s["seed_ids"], s["q"])))
    print("benchmark discretization error: max |dU|/U = %.3f, max |dq|/q = %.3f" % (max(abs(c["dU_rel"]) for c in conv), max(abs(c["dq_rel"]) for c in conv)))
    print("incumbent (action reference):", summary.get("_incumbent"))
    print("price-risk law with the idle policy as comparison (bound / attained, mean over seeds; smallest per-seed factor):")
    for arm, s in summary.items():
        if not arm.startswith("_") and "bound_mean" in s:
            print(f"  {arm:12s} bound {100*s['bound_mean']:6.2f}%  attained {100*s['q_mean']:5.2f}%  factor {s['bound_factor_mean']:5.1f} (min {s['bound_factor_min']:5.1f})  "
                  f"delta {s['delta_mean']:.3f} (raw min {s['delta_raw_min']:+.3f})  I_m(idle) {s['bound_I_idle']:.3f}  sup U {s['bound_sup_U']:.3f}")


if __name__ == "__main__":
    main()
