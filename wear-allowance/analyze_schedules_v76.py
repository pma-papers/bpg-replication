"""Revision 76: figures and summary of the convergence check and the strength schedules (wear_schedules_v76.py).

Reads wear-schedules-v76/campaign, the records of the earlier campaigns, benchmark.json and soft-benchmark-v76.json.
Writes wear-schedules-v76.json (every group, seed means and standard deviations, convergence diagnostics) and the figures
  wear-convergence-v76      traces of the anchored runs under the recorded and a constant learning rate
  wear-strength-schedules-v76   anchor strength changed in steps during training
  wear-action-regularization-v76  action regularization against the benchmark kernel, fixed and lowered strength
  wear-noise-schedule-v76   the anchor with noise, and the noise switched off at half time
Runtime ~/.venvs/constrained-rl/bin/python.
"""
from __future__ import annotations
import glob, json, math, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from plotstyle import PAL, LS, MUTED, TEXT2, FS, TEXTW, panel_label, plt  # noqa: E402

BETA, M = 0.95, 0.03          # M is rebound per group in the summary loop and restored after it
N_PAR = 4418
runs = defaultdict(list)
for f in sorted(glob.glob(str(HERE / "wear-schedules-v76/campaign/*/result.json"))):
    r = json.load(open(f)); a = r["arm"]
    runs[(a["start"], a["optimizer"], a["lr_schedule"], round(r["steps"] / 1e6), a["kappa2"], a["kappa1"], a["eta"], a["m"], a.get("anchor_part", "all"))].append(r)
bench = json.load(open(HERE / "benchmark.json"))
exact = next(r["nominal"] for r in bench["rows"] if r["kind"] == "failure" and abs(r["m"] - M) < 1e-12)
soft = json.load(open(HERE / "soft-benchmark-v76.json"))["rows"] if (HERE / "soft-benchmark-v76.json").exists() else []
prior = dict(risk=0.0108, U=2.73)


def G(start="incumbent", opt="adam-coupled", lr="constant", steps=8, k2="0", k1="0", eta="0", m=0.03, part="all"):
    return runs.get((start, opt, lr, steps, k2, k1, eta, m, part), [])


def ev(r, key):
    return np.array([e["steps"] for e in r["evals"]]) / 1e6, np.array([e[key] for e in r["evals"]])


def tr(r, key, window=25):
    x = np.array([t["steps"] for t in r["trace"] if key in t]) / 1e6
    y = np.array([t[key] for t in r["trace"] if key in t], dtype=float)
    if window > 1 and len(y) > window:
        k = np.ones(window) / window
        y = np.convolve(y, k, mode="valid"); x = x[window - 1:]
    return x, y


def tail(r, key, frac=0.25):
    """Mean of an evaluated quantity over the last quarter of training."""
    x, y = ev(r, key); return float(y[x >= (1 - frac) * x[-1]].mean())


def drift(r, key):
    """Difference between the last quarter and the quarter before it."""
    x, y = ev(r, key); T = x[-1]
    return float(y[x >= 0.75 * T].mean() - y[(x >= 0.5 * T) & (x < 0.75 * T)].mean())


def J(r, eta=0.0):
    s = r["stochastic"]; return s["U"] - M * s["risk"] / (1 - BETA) - eta * r["action_kl"]["A"]


def lines(ax, rs, getter, color, ls="-", label=None, mean=True, alpha=0.35, lw=0.6):
    curves = [getter(r) for r in rs]
    for x, y in curves:
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, ls="-")
    if mean and curves:
        n = min(len(y) for _, y in curves)
        ax.plot(curves[0][0][:n], np.mean([y[:n] for _, y in curves], axis=0), color=color, lw=1.5, ls=ls, label=label)
    elif curves and label:
        ax.plot([], [], color=color, lw=1.2, label=label)


def href(ax, y, text, ls=":", color=TEXT2):
    ax.axhline(y, color=color, lw=0.8, ls=ls, zorder=0)
    ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom", fontsize=FS - 1.5, color=color)


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf"); fig.savefig(HERE / f"{name}.png"); plt.close(fig)


summary = dict(groups=[], notes={})
for key, rs in sorted(runs.items()):
    q = np.array([r["stochastic"]["risk"] for r in rs]) * 100; U = np.array([r["stochastic"]["U"] for r in rs]); A = np.array([r["action_kl"]["A"] for r in rs])
    M = key[7]
    row = dict(start=key[0], optimizer=key[1], lr_schedule=key[2], steps_M=key[3], kappa2=key[4], kappa1=key[5], eta=key[6], m=key[7], anchor_part=key[8], seeds=[r["seed"] for r in rs],
               q_mean=q.mean(), q_sd=q.std(ddof=1) if len(q) > 1 else None, q_seeds=q.tolist(), U_mean=U.mean(), U_sd=U.std(ddof=1) if len(U) > 1 else None, U_seeds=U.tolist(),
               A_mean=A.mean(), J_own_price_mean=float(np.mean(U - M * q / 100 / (1 - BETA))), sigma_mean=float(np.mean([math.exp(r["log_std"]) for r in rs])), flow_time=rs[0]["lr_sum_flow_time"],
               q_tail_mean=float(np.mean([tail(r, "risk") for r in rs]) * 100), U_tail_mean=float(np.mean([tail(r, "U") for r in rs])),
               q_drift_last_quarter=float(np.mean([drift(r, "risk") for r in rs]) * 100), U_drift_last_quarter=float(np.mean([drift(r, "U") for r in rs])))
    if "distance" in rs[0]:
        d = np.array([r["distance"] for r in rs]); row.update(distance_mean=d.mean(), distance_seeds=d.tolist())
        dd = []
        for r in rs:
            x, y = tr(r, "distance", 1); dd.append(y[x >= 0.75 * x[-1]].mean() / y[(x >= 0.5 * x[-1]) & (x < 0.75 * x[-1])].mean())
        row["distance_ratio_last_quarter"] = float(np.mean(dd))
        res = []
        for r in rs:
            x, g = tr(r, "g_mean_norm", 1); _, p = tr(r, "pull_mean_norm", 1); _, e = tr(r, "residual_norm", 1)
            sel = (x >= 0.25 * x[-1]) & (x <= 0.75 * x[-1])
            res.append((g[sel].mean(), p[sel].mean(), e[sel].mean()))
        res = np.mean(res, axis=0); row.update(g_mean_norm=res[0], pull_mean_norm=res[1], residual_norm=res[2])
    summary["groups"].append(row)
    print(f"{'|'.join(str(k) for k in key):75s} n={len(rs)} q={row['q_mean']:.2f} ({row['q_sd'] or 0:.2f}) U={row['U_mean']:.3f} A={row['A_mean']:.3g} "
          f"dist={row.get('distance_mean', float('nan')):.4f} sig={row['sigma_mean']:.3f} | drift q {row['q_drift_last_quarter']:+.2f} U {row['U_drift_last_quarter']:+.3f} "
          f"dist ratio {row.get('distance_ratio_last_quarter', float('nan')):.3f} | g,pull,res {row.get('g_mean_norm', float('nan')):.3f},{row.get('pull_mean_norm', float('nan')):.3f},{row.get('residual_norm', float('nan')):.3f}")

M = 0.03
# ------------------------------------------------------------------ figure 1: convergence of the anchored runs
C_LIN, C_CON, C_LONG, C_SGD = PAL[0], PAL[1], PAL[6], PAL[2]
fig, axes = plt.subplots(4, 3, figsize=(TEXTW * 1.3, TEXTW * 1.25), sharex="col")
for j, k in enumerate(["0", "1", "10"]):
    sets = [(G(lr="linear", k2=k), C_LIN, "-", "learning rate decayed to zero (the runs of the paper)"), (G(lr="constant", k2=k), C_CON, "--", "constant learning rate")]
    if k == "0":
        sets += [(G(lr="constant", steps=24, k2=k), C_LONG, "-.", "constant learning rate, three times longer"), (G(opt="sgd", lr="linear", k2=k), C_SGD, ":", "plain gradient steps"),
                 (G(lr="constant", k2="10", part="dispersion"), PAL[3], (0, (5, 1, 1, 1)), "constant learning rate, dispersion alone anchored")]
    for rs, c, ls, lab in sets:
        if not rs:
            continue
        lines(axes[0, j], rs, lambda r: (ev(r, "risk")[0], 100 * ev(r, "risk")[1]), c, ls, lab)
        lines(axes[1, j], rs, lambda r: ev(r, "U"), c, ls)
        lines(axes[2, j], rs, lambda r: tr(r, "distance", 1), c, ls)
        lines(axes[3, j], rs, lambda r: (tr(r, "log_std", 1)[0], np.exp(tr(r, "log_std", 1)[1])), c, ls)
    axes[0, j].set_title(rf"$\kappa_p={k}$" + (" (no anchor)" if k == "0" else ""), fontsize=FS + 0.5)
    href(axes[0, j], 100 * prior["risk"], "incumbent"); href(axes[1, j], prior["U"], "incumbent")
    if k == "0":
        href(axes[0, j], 100 * exact["risk"], "exact optimum", "--"); href(axes[1, j], exact["U"], "exact optimum", "--")
        axes[3, j].set_xlim(0, 24)
    axes[0, j].set_ylim(bottom=0); axes[2, j].set_ylim(bottom=0)
    axes[3, j].set_yscale("log"); axes[3, j].set_ylim(1e-3, 0.3); axes[3, j].set_xlabel("training steps (millions)")
for i, lab in enumerate(["breakdown probability (%)", "stopped utility $U$", r"distance $\|x-x_0\|$", r"dispersion $\sigma_a$"]):
    axes[i, 0].set_ylabel(lab)
for i, s_ in enumerate("abcd"):
    panel_label(axes[i, 0], s_, x=-0.32)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.09))
fig.tight_layout(h_pad=0.6, w_pad=0.8)
save(fig, "wear-convergence-v76")

# ------------------------------------------------------------------ figure 2: schedules of the anchor strength
SCHEDS = [("100@0.25,10@0.5,1@1", "lowered: 100, 10, 1"), ("1@0.5,10@1", "raised: 1, 10"), ("0@0.5,10@1", "anchor added: 0, 10"), ("100@0.2,10@0.4,1@0.6,0@1", "lowered to zero")]
fixed = {k: G(lr="constant", k2=k) for k in ["0", "1", "10"]}
FC = {"0": PAL[7], "1": PAL[2], "10": PAL[6]}
fig, axes = plt.subplots(4, 4, figsize=(TEXTW * 1.45, TEXTW * 1.3), sharex=True, gridspec_kw=dict(height_ratios=[0.45, 1, 1, 1]))
for j, (sch, title) in enumerate(SCHEDS):
    rs = G(lr="constant", k2=sch)
    stages = [(float(p.split("@")[1]), float(p.split("@")[0])) for p in sch.split(",")]
    xs, ys, t0 = [], [], 0.0
    for end, v in stages:
        xs += [8 * t0, 8 * end]; ys += [v, v]; t0 = end
    axes[0, j].plot(xs, ys, color=TEXT2, lw=1.4); axes[0, j].set_yscale("symlog", linthresh=1); axes[0, j].set_ylim(-0.3, 300); axes[0, j].set_yticks([0, 1, 10, 100])
    axes[0, j].set_yticklabels(["0", "1", "10", "100"]); axes[0, j].set_title(title, fontsize=FS + 0.5)
    used = {("%g" % v) for _, v in stages}
    for k, frs in fixed.items():
        if k in used and frs:
            lines(axes[1, j], frs, lambda r: (ev(r, "risk")[0], 100 * ev(r, "risk")[1]), FC[k], "--", rf"fixed $\kappa_p={k}$", alpha=0.0)
            lines(axes[2, j], frs, lambda r: ev(r, "U"), FC[k], "--", alpha=0.0)
            lines(axes[3, j], frs, lambda r: tr(r, "distance", 1), FC[k], "--", alpha=0.0)
    if rs:
        lines(axes[1, j], rs, lambda r: (ev(r, "risk")[0], 100 * ev(r, "risk")[1]), PAL[0], "-", "strength changed in steps")
        lines(axes[2, j], rs, lambda r: ev(r, "U"), PAL[0], "-")
        lines(axes[3, j], rs, lambda r: tr(r, "distance", 1), PAL[0], "-")
    for i in range(4):
        for end, _ in stages[:-1]:
            axes[i, j].axvline(8 * end, color=MUTED, lw=0.6, ls=":")
    axes[3, j].set_yscale("log"); axes[3, j].set_xlabel("training steps (millions)")
    href(axes[1, j], 100 * prior["risk"], "incumbent"); href(axes[2, j], prior["U"], "incumbent")
for i, lab in enumerate([r"$\kappa_p$", "breakdown probability (%)", "stopped utility $U$", r"distance $\|x-x_0\|$"]):
    axes[i, 0].set_ylabel(lab)
for i, s in enumerate("abcd"):
    panel_label(axes[i, 0], s, x=-0.35)
hs, ls_ = [], []
for ax in axes[1]:
    for h_, l_ in zip(*ax.get_legend_handles_labels()):
        if l_ not in ls_:
            hs.append(h_); ls_.append(l_)
fig.legend(hs, ls_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(h_pad=0.5, w_pad=0.6)
save(fig, "wear-strength-schedules-v76")

# ------------------------------------------------------------------ figure 3: action regularization
rec = defaultdict(list)
for f in sorted(glob.glob(str(HERE / "campaign/anchored-e*/result.json"))):
    r = json.load(open(f)); rec[r["arm"]["eta"]].append(r)
fig, axes = plt.subplots(2, 3, figsize=(TEXTW * 1.45, TEXTW * 0.95))
if soft:
    e = np.array([s["eta"] for s in soft])
    for ax, key, sc in zip(axes[0], ["risk", "U", "A"], [100, 1, 1]):
        ax.plot(e, sc * np.array([s[key] for s in soft]), color=TEXT2, lw=1.3, marker="o", ms=3.5, label="kernel from dynamic programming")
for ax, key, sc in zip(axes[0], ["risk", "U", "A"], [100, 1, 1]):
    for i, (eta, rs) in enumerate(sorted(rec.items())):
        y = [sc * (r["stochastic"][key] if key != "A" else r["action_kl"]["A"]) for r in rs]
        ax.plot([eta] * len(y), y, ls="", marker="D", ms=3.5, mfc="none", color=PAL[1], label="learned policies of the paper (eight seeds)" if i == 0 else None)
    sch = G(start="default", lr="constant", eta="1@0.25,0.1@0.5,0.01@1")
    if sch:
        y = [sc * (r["stochastic"][key] if key != "A" else r["action_kl"]["A"]) for r in sch]
        ax.plot([0.0135] * len(y), y, ls="", marker="s", ms=3.5, color=PAL[0], label="learned, strength lowered to 0.01")
    con = G(start="default", lr="constant", eta="0.01")
    if con:
        y = [sc * (r["stochastic"][key] if key != "A" else r["action_kl"]["A"]) for r in con]
        ax.plot([0.0075] * len(y), y, ls="", marker="^", ms=3.5, color=PAL[2], label="learned, constant learning rate")
    ax.set_xscale("log"); ax.set_xlabel(r"strength $\eta$")
axes[0, 0].set_ylabel("breakdown probability (%)"); axes[0, 1].set_ylabel("stopped utility $U$"); axes[0, 2].set_ylabel(r"action relative entropy $\mathcal{A}$"); axes[0, 2].set_yscale("log")
b01 = next((s for s in soft if abs(s["eta"] - 0.01) < 1e-12), None)
for ax, key, sc in zip(axes[1], ["risk", "U", "A"], [100, 1, 1]):
    for rs, c, ls, lab in [(G(start="default", lr="linear", eta="0.01"), C_LIN, "-", r"$\eta=0.01$, learning rate decayed to zero (as in the paper)"), (G(start="default", lr="constant", eta="0.01"), C_CON, "--", r"$\eta=0.01$, constant learning rate"),
                           (G(start="default", lr="constant", eta="1@0.25,0.1@0.5,0.01@1"), C_SGD, "-.", r"$\eta$ lowered: 1, 0.1, 0.01 (constant learning rate)")]:
        if rs:
            lines(ax, rs, lambda r: (ev(r, key)[0], sc * ev(r, key)[1]), c, ls, lab)
    if b01:
        href(ax, sc * b01[key], "kernel from dynamic programming", "--")
    ax.set_xlabel("training steps (millions)")
    for end in (2, 4):
        ax.axvline(end, color=MUTED, lw=0.6, ls=":")
axes[1, 0].set_ylabel("breakdown probability (%)"); axes[1, 1].set_ylabel("stopped utility $U$"); axes[1, 2].set_ylabel(r"$\mathcal{A}$"); axes[1, 2].set_yscale("log"); axes[1, 0].set_ylim(0, 20)
panel_label(axes[0, 0], "a", x=-0.2); panel_label(axes[1, 0], "b", x=-0.2)
h0, l0 = axes[0, 0].get_legend_handles_labels(); h1, l1 = axes[1, 0].get_legend_handles_labels()
fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.13))
fig.tight_layout(h_pad=0.8, w_pad=0.8)
save(fig, "wear-action-regularization-v76")

# ------------------------------------------------------------------ figure 4: noise with the anchor
fig, axes = plt.subplots(2, 4, figsize=(TEXTW * 1.45, TEXTW * 0.8), sharex=True)
COLS = [(G(lr="linear", k2="10", k1="0.009"), r"$\kappa_1=0.009$", 0.009, None), (G(lr="linear", k2="10", k1="0.101"), r"$\kappa_1=0.101$", 0.101, None),
        (G(lr="constant", k2="10", k1="0.101@0.5,0@1"), "noise removed, Adam", 0.101, 4), (G(opt="sgd", lr="constant", k2="10", k1="0.101@0.5,0@1"), "noise removed, gradient steps", 0.101, 4)]
for j, (rs, title, k1, off) in enumerate(COLS):
    for i, r in enumerate(rs):
        x, y = ev(r, "risk"); axes[0, j].plot(x, 100 * y, color=PAL[i % len(PAL)], lw=0.9)
        x, y = tr(r, "distance", 1); axes[1, j].plot(x, y / math.sqrt(N_PAR), color=PAL[i % len(PAL)], lw=0.9)
    axes[0, j].set_title(title, fontsize=FS)
    axes[1, j].axhline(math.sqrt(k1 / 10), color=TEXT2, lw=0.8, ls=":"); axes[1, j].axhline(math.sqrt(2) * k1, color=TEXT2, lw=0.8, ls="--")
    axes[1, j].set_yscale("log"); axes[1, j].set_xlabel("training steps (millions)")
    if off:
        for ax in axes[:, j]:
            ax.axvline(off, color=MUTED, lw=0.6, ls=":")
    f10 = fixed["10"]
    if f10:
        href(axes[0, j], float(np.mean([tail(r, "risk") for r in f10]) * 100), "anchor alone" if j == 0 else "", "-.")
axes[0, 0].set_ylabel("breakdown probability (%)"); axes[1, 0].set_ylabel(r"spread $\|x-x_0\|/\sqrt{d}$")
panel_label(axes[0, 0], "a", x=-0.3); panel_label(axes[1, 0], "b", x=-0.3)
fig.tight_layout(h_pad=0.6, w_pad=0.6)
save(fig, "wear-noise-schedule-v76")

# ------------------------------------------------------------------ action regularization against the benchmark
if soft:
    for s in soft:
        rs = rec.get(s["eta"], [])
        if rs:
            Jl = np.array([J(r, s["eta"]) for r in rs]); ql = np.array([r["stochastic"]["risk"] for r in rs]) * 100
            summary["notes"][f"eta={s['eta']:g}"] = dict(benchmark_J=s["J"], benchmark_q=100 * s["risk"], benchmark_U=s["U"], benchmark_A=s["A"], learned_J_mean=Jl.mean(), learned_J_min=Jl.min(), learned_q=ql.tolist())
            print(f"eta={s['eta']:g}: benchmark J={s['J']:.3f} q={100 * s['risk']:.2f}% U={s['U']:.3f} A={s['A']:.3g} | recorded runs J={Jl.mean():.3f} (min {Jl.min():.3f}) q={ql.mean():.2f}% [{ql.min():.1f}, {ql.max():.1f}]")
(HERE / "wear-schedules-v76.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
