"""Summaries, table and figures of the wear-allowance study (reads benchmark.json and campaign/)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
sys.path.insert(0, str(ROOT))
from plotstyle import PAL, TEXT, TEXT2, MUTED, TEXTW, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from env import Incumbent  # noqa: E402
from model import Model, simulate  # noqa: E402

STAGE = sys.argv[1] if len(sys.argv) > 1 else "campaign"
BETA = 0.95
C_FIXED, C_ADAPT, C_BARRIER, C_ANCHOR = PAL[0], PAL[1], PAL[2], PAL[3]


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
    bench = json.loads((ROOT / "benchmark.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    runs = {}
    for f in sorted((ROOT / STAGE).glob("*/result.json")):
        arm = f.parent.name.split("__seed")[0]
        runs.setdefault(arm, []).append(json.loads(f.read_text()))
    return bench, protocol, runs


def frontier(bench):
    rows = [r for r in bench["rows"] if r["kind"] == "failure"]
    q = np.array([r["grid"]["q"] for r in rows]); U = np.array([r["grid"]["U"] for r in rows])
    m = np.array([r["m"] for r in rows])
    order = np.argsort(q)
    return m, q, U, (lambda x: float(np.interp(np.log(max(x, q.min())), np.log(q[order]), U[order])))


def summarize(bench, protocol, runs):
    m_ex, q_ex, U_ex, U_star = frontier(bench)
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
        gap = np.array([(U_star(a) - b) / U_star(a) for a, b in zip(q, U)])
        s = dict(kind=cfg["kind"], cfg=cfg, seeds=len(rs), q_mean=q.mean(), q_sd=q.std(ddof=1), q_min=q.min(), q_max=q.max(),
                 U_mean=U.mean(), U_sd=U.std(ddof=1), gap_mean=gap.mean(), gap_max=gap.max(),
                 pass05=int(sum(u <= 0.05 for u in ups)), pass01=int(sum(u <= 0.01 for u in ups)),
                 q90=float(np.mean([r["robust90"]["risk"] for r in rs])), q80=float(np.mean([r["robust80"]["risk"] for r in rs])),
                 near05=float(np.mean([r["stochastic"]["near"]["0.05"] for r in rs])),
                 I_mean=float(np.mean([r["stochastic"]["I"] for r in rs])),
                 mean_action_q=float(np.mean([r["mean_action"]["risk"] for r in rs])),
                 mean_action_U=float(np.mean([r["mean_action"]["U"] for r in rs])),
                 multiplier=[r["final_multiplier"] for r in rs], q=q.tolist(), U=U.tolist())
        if cfg["kind"] == "anchored":
            s["A"] = [r["action_kl"]["A"] for r in rs]
        out[arm] = s
    return out, n_candidates


def table(summary):
    def price(c):
        if c["kind"] == "adaptive":
            return f"target ${c['alpha']:g}$" + (f", step ${c['dual_step']:g}$" if c["dual_step"] != 5 else "")
        if c["kind"] == "anchored":
            return f"$m=0.03$, $\\eta={c['eta']:g}$"
        return f"$m={c['m']:g}$"
    names = {"fixed": "Fixed price", "barrier": "Fixed price, barrier", "adaptive": "Adaptive multiplier", "anchored": "Anchored"}
    lines = []
    last = None
    for arm, s in summary.items():
        kind = s["kind"]
        label = names[kind] if kind != last else ""
        last = kind
        lines.append(f"{label} & {price(s['cfg'])} & ${100*s['q_mean']:.2f}\\pm{100*s['q_sd']:.2f}$ & ${s['U_mean']:.2f}\\pm{s['U_sd']:.2f}$ & "
                     f"${100*s['gap_mean']:.1f}$ & ${s['pass05']}/{s['seeds']}$ & ${100*s['near05']:.1f}$ & ${100*s['q90']:.1f}$ \\\\")
    (ROOT / "campaign-table.tex").write_text(
        "\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{llrrrrrr}\n\\toprule\nArm & Setting & $q_\\beta$ (\\%) & $U$ & gap (\\%) & pass & near (\\%) & at $0.9b$ (\\%) \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def fig_frontier(bench, summary):
    m_ex, q_ex, U_ex, _ = frontier(bench)
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.25), gridspec_kw=dict(wspace=0.42, width_ratios=[0.9, 1.25, 1.0]))
    # (a) exact optimal policies at the mean price
    ax = axes[0]
    z = np.array(bench["z_grid"]); mid = len(bench["s_grid"]) // 2
    for key, lab, col, ls in (("failure-m0", "unpriced", MUTED, "-"), ("failure-m1", "price $m=1$", C_FIXED, "-"), ("barrier-m1", "barrier, $m=1$", C_BARRIER, "-")):
        ax.plot(z, np.array(bench["policy_tables"][key])[mid], color=col, ls=ls, label=lab)
    ax.set_xlabel("remaining allowance $z/b$"); ax.set_ylabel("operating rate"); ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=6.2); panel_label(ax, "a")
    # (b) frontier
    ax = axes[1]
    ax.plot(q_ex, U_ex, color=TEXT2, lw=1.2, label="exact frontier", zorder=1)
    for mm in (0.1, 1, 10):
        i = int(np.argmin(np.abs(m_ex - mm)))
        ax.annotate(f"$m={mm:g}$", (q_ex[i], U_ex[i]), textcoords="offset points", xytext={0.1: (7, -12), 1: (-28, 3), 10: (-8, 7)}[mm], fontsize=6.5, color=TEXT2)
        ax.plot([q_ex[i]], [U_ex[i]], marker="|", color=TEXT2, ms=5)
    bar = [r for r in bench["rows"] if r["kind"] == "barrier" and r.get("theta") == 0.1]
    ax.plot([r["grid"]["q"] for r in bar], [r["grid"]["U"] for r in bar], color=C_BARRIER, lw=1.0, ls="--", label="exact, barrier", zorder=1)
    for kind, col, mk, lab in (("fixed", C_FIXED, "o", "fixed price"), ("adaptive", C_ADAPT, "s", "adaptive multiplier"), ("barrier", C_BARRIER, "^", "fixed price, barrier")):
        first = True
        for arm, s in summary.items():
            if s["kind"] != kind or (kind == "adaptive" and s["cfg"]["dual_step"] != 5):
                continue
            ax.errorbar(s["q_mean"], s["U_mean"], xerr=[[s["q_mean"] - s["q_min"]], [s["q_max"] - s["q_mean"]]], fmt=mk, color=col, ms=4,
                        mec="white", mew=0.6, elinewidth=0.8, label=lab if first else None, zorder=3)
            first = False
    ax.set_xscale("log"); ax.set_xlabel("breakdown probability $q_\\beta$"); ax.set_ylabel("stopped utility $U$")
    ax.axvline(0.05, color=MUTED, lw=0.7, ls=":"); ax.legend(loc="upper left", fontsize=6.0, borderaxespad=0.2, labelspacing=0.3); panel_label(ax, "b")
    # (c) price-risk law
    ax = axes[2]
    pos = m_ex > 0
    ax.plot(m_ex[pos], q_ex[pos], color=TEXT2, label="exact")
    ax.plot(m_ex[pos], np.minimum(1, (1 - BETA) * U_ex[m_ex == 0][0] / m_ex[pos]), color=MUTED, ls="--", label="upper bound")
    xs, ys, lo, hi = [], [], [], []
    for arm, s in summary.items():
        if s["kind"] == "fixed" and s["cfg"]["m"] > 0:
            xs.append(s["cfg"]["m"]); ys.append(s["q_mean"]); lo.append(s["q_mean"] - s["q_min"]); hi.append(s["q_max"] - s["q_mean"])
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=C_FIXED, ms=4, mec="white", mew=0.6, elinewidth=0.8, label="learned")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("price of failure $m$"); ax.set_ylabel("breakdown probability $q_\\beta$")
    ax.legend(loc="lower left", fontsize=6.2); panel_label(ax, "c")
    save(fig, "wear-frontier", ROOT)


def fig_margin(bench, summary):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, key, title, lab in ((axes[0], "allowance90", "true allowance $0.9\\,b$", "a"), (axes[1], "allowance80", "true allowance $0.8\\,b$", "b")):
        sets = (("failure", None, TEXT2, "-", "failure-only prices"), ("barrier", 0.1, C_BARRIER, "-", "interior barrier"),
                ("tightened", 0.9, C_ANCHOR, "--", "plan with $0.9\\,b$"), ("tightened", 0.8, PAL[6], "--", "plan with $0.8\\,b$"))
        for kind, par, col, ls, name in sets:
            rows = [r for r in bench["rows"] if r["kind"] == kind and (par is None or r.get("theta", r.get("planned_allowance")) == par) and r["m"] >= 0.3]
            U = [r["grid"]["U"] if r["grid"] else r["nominal"]["U"] for r in rows]
            ax.plot(U, [max(r[key]["risk"], 1e-4) for r in rows], color=col, ls=ls, marker=".", ms=3, label=name)
        for arm, s in summary.items():
            if s["kind"] in ("fixed", "barrier") and s["cfg"]["m"] >= 0.3:
                ax.plot([s["U_mean"]], [s["q90" if key == "allowance90" else "q80"]], marker="o" if s["kind"] == "fixed" else "^",
                        color=C_FIXED if s["kind"] == "fixed" else C_BARRIER, ms=4, mec="white", mew=0.6, ls="none")
        ax.set_yscale("log"); ax.set_xlabel("stopped utility $U$ at the nominal allowance"); ax.set_title(title); panel_label(ax, lab)
    axes[0].set_ylabel("breakdown probability")
    from matplotlib.lines import Line2D
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", ls="none", color=C_FIXED, ms=4), Line2D([], [], marker="^", ls="none", color=C_BARRIER, ms=4)]
    labels += ["learned, fixed price", "learned, barrier"]
    axes[1].legend(handles, labels, loc="lower right", fontsize=6.2)
    save(fig, "wear-margin", ROOT)


def fig_dual(summary, runs, protocol):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.1), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    for arm, col, lab in (("adaptive-a0.05-s5", C_ADAPT, "target $0.05$"), ("adaptive-a0.01-s5", PAL[6], "target $0.01$"), ("adaptive-a0-s5", PAL[7], "target $0$")):
        if arm not in runs:
            continue
        tr = np.array([[t["multiplier"] for t in r["trace"]] for r in runs[arm]]) * (1 - BETA)
        steps = np.array([t["steps"] for t in runs[arm][0]["trace"]]) / 1e6
        ax.plot(steps, tr.mean(0), color=col, label=lab)
        ax.fill_between(steps, tr.min(0), tr.max(0), color=col, alpha=0.18, lw=0)
    ax.set_yscale("log"); ax.set_xlabel("environment steps (millions)"); ax.set_ylabel("multiplier in price units, $(1-\\beta)\\lambda$")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.64)); ax.set_ylim(2e-2, 8); panel_label(ax, "a")
    ax = axes[1]
    anchored = {a: s for a, s in summary.items() if s["kind"] == "anchored"}
    if anchored:
        cfg = next(iter(anchored.values()))["cfg"]
        model = Model(**cfg["model"]); inc = Incumbent(model, **cfg["incumbent"])
        ref = simulate(model, inc.policy(seed=3), n=200_000, seed=77)
        A = np.logspace(-3, 2, 200)
        ax.plot(A, [kl_upper(ref["risk"], a) for a in A], color=TEXT2, label="upper bound from $\\mathcal{A}$")
        ax.axhline(ref["risk"], color=MUTED, ls=":", lw=0.8)
        ax.annotate("incumbent", (1e-3, ref["risk"]), textcoords="offset points", xytext=(2, -9), fontsize=6.5, color=TEXT2)
        for arm, s in anchored.items():
            ax.plot(s["A"], s["q"], marker="o", ls="none", color=C_ANCHOR, ms=3.5, mec="white", mew=0.5)
            ax.annotate(f"$\\eta={s['cfg']['eta']:g}$", (np.mean(s["A"]), np.mean(s["q"])), textcoords="offset points", xytext={1.0: (12, -2), 0.1: (10, -2), 0.01: (-42, 6)}[s["cfg"]["eta"]], fontsize=6.5, color=TEXT2)
        ax.plot([], [], marker="o", ls="none", color=C_ANCHOR, ms=3.5, label="anchored policies")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5e-3, 1.5); ax.set_xlabel("stopped action relative entropy $\\mathcal{A}$"); ax.set_ylabel("breakdown probability $q_\\beta$")
        ax.legend(loc="upper left"); summary["_incumbent"] = dict(q=ref["risk"], U=ref["U"])
    panel_label(ax, "b")
    save(fig, "wear-dual", ROOT)


def main():
    bench, protocol, runs = load()
    summary, n_candidates = summarize(bench, protocol, runs)
    table(summary)
    fig_frontier(bench, summary); fig_margin(bench, summary); fig_dual(summary, runs, protocol)
    summary["_family"] = n_candidates
    (ROOT / f"summary-{STAGE}.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    for arm, s in summary.items():
        if arm.startswith("_"):
            continue
        extra = f" lam*(1-b)={np.mean(s['multiplier'])*(1-BETA):.2f}" if s["kind"] == "adaptive" else (f" A={np.mean(s['A']):.3f}" if s["kind"] == "anchored" else "")
        print(f"{arm:22s} n={s['seeds']} q={100*s['q_mean']:.2f}+-{100*s['q_sd']:.2f}% U={s['U_mean']:.3f}+-{s['U_sd']:.3f} gap={100*s['gap_mean']:.1f}% pass05={s['pass05']}/{s['seeds']} near05={100*s['near05']:.1f}% q90={100*s['q90']:.1f}% q80={100*s['q80']:.1f}%{extra}")
    print("incumbent", summary.get("_incumbent"))


if __name__ == "__main__":
    main()
