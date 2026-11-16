"""Revision 76: the figures and tables of the paper and of Online Resource 1 on the regularized runs of the
wear-allowance example (convergence, strengths changed during training, three optimizers).

Reads the records of the earlier runs (campaign, wear-anchor/campaign-matched, wear-noise/campaign,
wear-noise-decoupled-v68/campaign, wear-optimizers-v68/campaign), the runs of wear_schedules_v76.py
(wear-schedules-v76/campaign), benchmark.json and soft-benchmark-v76.json. No new runs. All figures are drawn at the
journal's print width. Output, in this folder:
  wear-optimizers-v76.pdf            paper, noise and the anchor under three optimizers
  wear-anchor-traces-v76.pdf         Online Resource, the anchored runs during training
  wear-anchor-schedules-v76.pdf      Online Resource, the anchor strength changed in steps
  wear-noise-traces-v76.pdf          Online Resource, the breakdown probability during training with noise
  wear-action-kernel-v76.pdf         Online Resource, action regularization against the kernel from dynamic programming
  wear-price-traces-v76.pdf          Online Resource, breakdown frequency during training in the first study
  wear-optimizers-table-v76.tex, wear-anchor-schedules-table-v76.tex, wear-own-price-table-v76.tex
  regularization-figures-v76.json    the numbers of the tables
Runtime ~/.venvs/constrained-rl/bin/python.
"""
from __future__ import annotations
import glob, json, math, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from plotstyle import PAL, MK, MUTED, TEXT2, FS, TEXTW, legend_below, panel_label, plt  # noqa: E402

BETA, M0, D = 0.95, 0.03, 4418
bench = json.load(open(HERE / "benchmark.json"))
EX = {(r["kind"], r["m"]): r["nominal"] for r in bench["rows"]}
exact = EX[("failure", M0)]
soft = json.load(open(HERE / "soft-benchmark-v76.json"))["rows"]
SOFT = {s["eta"]: s for s in soft}
PRIOR = dict(risk=0.0108, U=2.73)
out = {}

runs = defaultdict(list)
for f in sorted(glob.glob(str(HERE / "wear-schedules-v76/campaign/*/result.json"))):
    r = json.load(open(f)); a = r["arm"]
    runs[(a["start"], a["optimizer"], a["lr_schedule"], round(r["steps"] / 1e6), a["kappa2"], a["kappa1"], a["eta"], a["m"], a.get("anchor_part", "all"))].append(r)


def G(start="incumbent", opt="adam-coupled", lr="constant", steps=8, k2="0", k1="0", eta="0", m=0.03, part="all"):
    rs = runs.get((start, opt, lr, steps, k2, k1, eta, m, part), [])
    assert rs, (start, opt, lr, steps, k2, k1, eta, m, part)
    return rs


def ev(r, key, scale=1.0):
    return np.array([e["steps"] for e in r["evals"]]) / 1e6, scale * np.array([e[key] for e in r["evals"]])


def tr(r, key, fn=None):
    x = np.array([t["steps"] for t in r["trace"] if key in t]) / 1e6
    y = np.array([t[key] for t in r["trace"] if key in t], dtype=float)
    return x, (fn(y) if fn else y)


def lines(ax, rs, getter, color, ls="-", label=None, seeds=True):
    curves = [getter(r) for r in rs]
    if seeds:
        for x, y in curves:
            ax.plot(x, y, color=color, lw=0.5, alpha=0.3)
    n = min(len(y) for _, y in curves)
    ax.plot(curves[0][0][:n], np.mean([y[:n] for _, y in curves], axis=0), color=color, lw=1.3, ls=ls, label=label)


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf"); fig.savefig(HERE / f"{name}.png"); plt.close(fig)


def pm(x, d=2):
    x = np.asarray(x, dtype=float)
    return f"${x.mean():.{d}f}\\pm{x.std(ddof=1):.{d}f}$" if len(x) > 1 else f"${x.mean():.{d}f}$"


def sci(x):
    if x == 0:
        return "$0$"
    if x >= 0.00995:
        return f"${x:.3f}$" if x < 1 else f"${x:.2f}$"
    e = math.floor(math.log10(x)); return f"${x / 10 ** e:.1f}\\times10^{{{e}}}$"


def final(rs, key):
    return np.array([(100 * r["stochastic"]["risk"] if key == "q" else r["stochastic"]["U"] if key == "U" else r["distance"] if key == "d" else math.exp(r["log_std"])) for r in rs])


# ------------------------------------------------------------------ the three optimizers (records of revisions 41, 60, 68)
ARMS = [("sgd", "plain gradient steps", PAL[2], MK[0]), ("adam-coupled", "Adam, coupled anchor", PAL[1], MK[1]), ("adam-decoupled", "Adam, decoupled anchor", PAL[0], MK[3])]
opt = defaultdict(list)       # (arm, k1, k2) -> list of (spread, risk %, U)
seen = set()                  # a configuration run twice (in the anchor study and again with the three optimizers) is taken once


def take(pattern, arm, k1=None):
    for f in sorted(glob.glob(str(HERE / pattern))):
        r = json.load(open(f)); a = r["arm"]
        key = (arm, float(a.get("kappa_1", 0.0) if k1 is None else k1), float(a["kappa_p"]), r["seed"])
        if key in seen:
            continue
        seen.add(key)
        opt[key[:3]].append((r["distance"] / math.sqrt(D), 100 * r["stochastic"]["risk"], r["stochastic"]["U"]))


for name in glob.glob(str(HERE / "wear-optimizers-v68/campaign/*__seed94001")):
    base = Path(name).name.rsplit("__seed", 1)[0]
    if "zerograd" not in base:
        take(f"wear-optimizers-v68/campaign/{base}__seed*/result.json", base.split("__")[0])
take("wear-noise/campaign/*/result.json", "adam-coupled")
take("wear-anchor/campaign-matched/kappa*/result.json", "adam-coupled", k1=0.0)
take("wear-noise-decoupled-v68/campaign/*/result.json", "adam-decoupled")
opt[("adam-decoupled", 0.0, 0.0)] = opt[("adam-coupled", 0.0, 0.0)]          # without the anchor the two forms are the same runs
assert all(len(v) == 8 for v in opt.values()), {k: len(v) for k, v in opt.items() if len(v) != 8}

fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.25), gridspec_kw=dict(wspace=0.62, width_ratios=[1.25, 1, 1]))
ax = axes[0]
k = np.geomspace(6e-4, 1.6, 50)
ax.plot(k, np.sqrt(k / 10), color=TEXT2, lw=0.8, ls=":"); ax.plot(k, np.sqrt(k), color=TEXT2, lw=0.8, ls=":"); ax.plot(k, math.sqrt(2) * k, color=TEXT2, lw=0.8, ls="--")
for arm, lab, c, mk in ARMS:
    for (a, k1, k2), v in opt.items():
        if a == arm and k1 > 0 and k2 > 0:
            ax.plot(k1, np.mean([x[0] for x in v]), ls="", marker=mk, ms=4.5, color=c, mfc=(c if k2 == 10 else "none"), mew=1.0, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1e-3, 1e-2, 1e-1, 1]); ax.set_xlabel(r"noise strength $\kappa_1$"); ax.set_ylabel(r"spread $\|x-x_0\|/\sqrt{d}$")
ax = axes[1]
for arm, lab, c, mk in ARMS:
    first = True
    for (a, k1, k2), v in sorted(opt.items()):
        if a == arm and k1 > 0 and k2 > 0:
            ax.plot([x[0] for x in v], [x[1] for x in v], ls="", marker=mk, ms=3.2, color=c, mfc=(c if k2 == 10 else "none"), mew=0.8, alpha=0.85, label=lab if first else None); first = False
ax.axvline(0.1, color=MUTED, lw=0.8, ls=":"); ax.axhline(100 * PRIOR["risk"], color=TEXT2, lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xticks([1e-3, 1e-2, 1e-1, 1]); ax.set_xlabel(r"spread $\|x-x_0\|/\sqrt{d}$"); ax.set_ylabel("breakdown probability (%)"); ax.set_ylim(-2, 66)
ax = axes[2]
for i, (arm, lab, c, mk) in enumerate(ARMS):
    for k2, x0 in {0.0: 0, 1.0: 1, 10.0: 2}.items():
        v = opt[(arm, 0.0, k2)]
        ax.plot(x0 + 0.24 * (i - 1) + np.zeros(len(v)), [x[1] for x in v], ls="", marker=mk, ms=3.2, color=c, alpha=0.85)
ax.axhline(100 * PRIOR["risk"], color=TEXT2, lw=0.8, ls=":"); ax.axhline(100 * exact["risk"], color=TEXT2, lw=0.8, ls="--")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["0", "1", "10"]); ax.set_xlim(-0.5, 2.5); ax.set_xlabel(r"anchor strength $\kappa_2$"); ax.set_ylabel("breakdown probability (%)"); ax.set_ylim(-2, 66)
for a_, s in zip(axes, "abc"):
    panel_label(a_, s, x=-0.2)
h, l = axes[1].get_legend_handles_labels()
legend_below(fig, h, l, ncol=3, columnspacing=1.0, handletextpad=0.3)
save(fig, "wear-optimizers-v76")

ROWS = [("Anchor alone", 0.0, 10.0), ("", 0.0, 1.0), ("Anchor with noise", 0.001, 10.0), ("", 0.009, 10.0), ("", 0.101, 10.0), ("", 0.989, 10.0), ("", 0.101, 1.0), ("Noise alone", 0.101, 0.0), ("Neither", 0.0, 0.0)]
tex = ["\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lrrrrrrrrrrrr}", "\\toprule",
       " & & & & \\multicolumn{3}{c}{Plain gradient steps} & \\multicolumn{3}{c}{Adam, coupled anchor} & \\multicolumn{3}{c}{Adam, decoupled anchor} \\\\",
       "\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}",
       "Configuration & $\\kappa_1$ & $\\kappa_2$ & $\\sqrt{\\kappa_1/\\kappa_2}$ & spread & $q_\\beta$ (\\%) & $U$ & spread & $q_\\beta$ (\\%) & $U$ & spread & $q_\\beta$ (\\%) & $U$ \\\\", "\\midrule"]
out["optimizers"] = {}
for name, k1, k2 in ROWS:
    cells = [name, f"${k1:g}$", f"${k2:g}$", (f"${math.sqrt(k1 / k2):.3f}$" if k1 > 0 and k2 > 0 else "--")]
    for arm, *_ in ARMS:
        v = np.array(opt.get((arm, k1, k2), []))
        if len(v) == 0:
            cells += ["--", "--", "--"]; continue
        cells += [sci(v[:, 0].mean()), pm(v[:, 1], 1), pm(v[:, 2])]
        out["optimizers"][f"{arm}|{k1:g}|{k2:g}"] = dict(spread=v[:, 0].mean(), q_mean=v[:, 1].mean(), q_sd=v[:, 1].std(ddof=1), q_min=v[:, 1].min(), q_max=v[:, 1].max(), U_mean=v[:, 2].mean())
    tex.append(" & ".join(cells) + " \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
(HERE / "wear-optimizers-table-v76.tex").write_text("\n".join(tex) + "\n")

# ------------------------------------------------------------------ the anchored runs during training
C_LIN, C_CON, C_LONG, C_SGD, C_DISP = PAL[0], PAL[1], PAL[6], PAL[2], PAL[3]
fig, axes = plt.subplots(4, 3, figsize=(TEXTW, TEXTW * 1.22), sharex="col")
for j, k in enumerate(["0", "1", "10"]):
    sets = [(G(lr="linear", k2=k), C_LIN, "-", "learning rate decayed to zero"), (G(lr="constant", k2=k), C_CON, "--", "constant learning rate")]
    if k == "0":
        sets += [(G(lr="constant", steps=24, k2=k), C_LONG, "-.", "constant, three times longer"), (G(opt="sgd", lr="linear", k2=k), C_SGD, ":", "plain gradient steps"),
                 (G(lr="constant", k2="10", part="dispersion"), C_DISP, (0, (5, 1, 1, 1)), "constant, dispersion alone anchored")]
    for rs, c, ls, lab in sets:
        lines(axes[0, j], rs, lambda r: ev(r, "risk", 100), c, ls, lab)
        lines(axes[1, j], rs, lambda r: ev(r, "U"), c, ls)
        lines(axes[2, j], rs, lambda r: tr(r, "distance"), c, ls)
        lines(axes[3, j], rs, lambda r: tr(r, "log_std", np.exp), c, ls)
    axes[0, j].set_title(rf"$\kappa_p={k}$", fontsize=FS + 0.5)
    axes[0, j].axhline(100 * PRIOR["risk"], color=TEXT2, lw=0.8, ls=":"); axes[1, j].axhline(PRIOR["U"], color=TEXT2, lw=0.8, ls=":")
    if k == "0":
        axes[0, j].axhline(100 * exact["risk"], color=TEXT2, lw=0.8, ls="--"); axes[1, j].axhline(exact["U"], color=TEXT2, lw=0.8, ls="--")
        axes[3, j].set_xlim(0, 24)
    axes[0, j].set_ylim(bottom=0); axes[2, j].set_ylim(bottom=0)
    axes[3, j].set_yscale("log"); axes[3, j].set_ylim(1e-3, 0.3); axes[3, j].set_xlabel("steps (millions)")
for i, lab in enumerate(["breakdown prob. (%)", "stopped utility $U$", r"distance $\|x-x_0\|$", r"dispersion $\sigma_a$"]):
    axes[i, 0].set_ylabel(lab)
for i, s in enumerate("abcd"):
    panel_label(axes[i, 0], s, x=-0.42)
h, l = axes[0, 0].get_legend_handles_labels()
fig.tight_layout(h_pad=0.5, w_pad=0.5)
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.085))
save(fig, "wear-anchor-traces-v76")
out["anchor"] = {}
for k in ["0", "1", "10", "100"]:
    for lr in ["linear", "constant"]:
        if k == "100" and lr == "constant":
            continue
        rs = G(lr=lr, k2=k)
        out["anchor"][f"{k}|{lr}"] = dict(n=len(rs), q=final(rs, "q").tolist(), U=final(rs, "U").tolist(), distance=final(rs, "d").tolist(), sigma=final(rs, "s").tolist())
for key, rs in [("0|constant-24M", G(lr="constant", steps=24, k2="0")), ("0|sgd", G(opt="sgd", lr="linear", k2="0")), ("10|dispersion-alone", G(lr="constant", k2="10", part="dispersion")),
                ("lowered-to-zero|constant-24M", G(lr="constant", steps=24, k2="100@0.0667,10@0.1333,1@0.2,0@1"))]:
    out["anchor"][key] = dict(n=len(rs), q=final(rs, "q").tolist(), U=final(rs, "U").tolist(), distance=final(rs, "d").tolist(), sigma=final(rs, "s").tolist())

# ------------------------------------------------------------------ the anchor strength changed in steps
SCHEDS = [("100@0.25,10@0.5,1@1", "lowered", "$100,\\,10,\\,1$"), ("0@0.5,10@1", "added", "$0,\\,10$"), ("100@0.2,10@0.4,1@0.6,0@1", "lowered to zero", "$100,\\,10,\\,1,\\,0$")]
fixed = {k: G(lr="constant", k2=k) for k in ["0", "1", "10"]}
FC, FL = {"0": PAL[7], "1": PAL[2], "10": PAL[6]}, {"0": ":", "1": "--", "10": "-."}
fig, axes = plt.subplots(4, 3, figsize=(TEXTW, TEXTW * 1.12), sharex=True, gridspec_kw=dict(height_ratios=[0.5, 1, 1, 1]))
for j, (sch, title, _) in enumerate(SCHEDS):
    rs = G(lr="constant", k2=sch)
    stages = [(float(p.split("@")[1]), float(p.split("@")[0])) for p in sch.split(",")]
    xs, ys, t0 = [], [], 0.0
    for end, v in stages:
        xs += [8 * t0, 8 * end]; ys += [v, v]; t0 = end
    axes[0, j].plot(xs, ys, color=TEXT2, lw=1.3); axes[0, j].set_yscale("symlog", linthresh=1); axes[0, j].set_ylim(-0.4, 300); axes[0, j].set_yticks([0, 1, 10, 100])
    axes[0, j].set_yticklabels(["0", "1", "10", "100"] if j == 0 else []); axes[0, j].set_title(title, fontsize=FS + 0.5)
    for k_ in ["0", "1", "10"]:
        if any(abs(v - float(k_)) < 1e-12 for _, v in stages):
            lines(axes[1, j], fixed[k_], lambda r: ev(r, "risk", 100), FC[k_], FL[k_], rf"$\kappa_p={k_}$ held fixed", seeds=False)
            lines(axes[2, j], fixed[k_], lambda r: ev(r, "U"), FC[k_], FL[k_], seeds=False)
            lines(axes[3, j], fixed[k_], lambda r: tr(r, "distance"), FC[k_], FL[k_], seeds=False)
    lines(axes[1, j], rs, lambda r: ev(r, "risk", 100), PAL[0], "-", r"$\kappa_p$ changed in steps")
    lines(axes[2, j], rs, lambda r: ev(r, "U"), PAL[0], "-")
    lines(axes[3, j], rs, lambda r: tr(r, "distance"), PAL[0], "-")
    for i in range(4):
        for end, _ in stages[:-1]:
            axes[i, j].axvline(8 * end, color=MUTED, lw=0.6, ls=":")
    axes[1, j].set_ylim(0, 42); axes[2, j].set_ylim(2.6, 4.7); axes[3, j].set_yscale("log"); axes[3, j].set_ylim(1.5e-3, 8); axes[3, j].set_xlabel("steps (millions)")
    if j:
        for i in (1, 2, 3):
            axes[i, j].set_yticklabels([])
for i, lab in enumerate([r"$\kappa_p$", "breakdown prob. (%)", "stopped utility $U$", r"distance $\|x-x_0\|$"]):
    axes[i, 0].set_ylabel(lab)
for i, s in enumerate("abcd"):
    panel_label(axes[i, 0], s, x=-0.36)
hs, ls_ = [], []
for ax in axes[1]:
    for h_, l_ in zip(*ax.get_legend_handles_labels()):
        if l_ not in ls_:
            hs.append(h_); ls_.append(l_)
order = np.argsort([0 if "changed" in l_ else 1 + ["0", "1", "10"].index(l_.split("=")[1].split("$")[0]) for l_ in ls_])
fig.tight_layout(h_pad=0.4, w_pad=0.4)
fig.legend([hs[i] for i in order], [ls_[i] for i in order], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.045), columnspacing=1.0)
save(fig, "wear-anchor-schedules-v76")

TROWS = [("Lowered", "$100,\\,10,\\,1$", G(lr="constant", k2="100@0.25,10@0.5,1@1")), ("Raised", "$1,\\,10$", G(lr="constant", k2="1@0.5,10@1")), ("Added", "$0,\\,10$", G(lr="constant", k2="0@0.5,10@1")),
         ("Lowered to zero", "$100,\\,10,\\,1,\\,0$", G(lr="constant", k2="100@0.2,10@0.4,1@0.6,0@1")), None,
         ("Held fixed", "$10$", fixed["10"]), ("", "$1$", fixed["1"]), ("", "$0$", fixed["0"])]
tex = ["\\begin{tabular}{llrrrr}", "\\toprule", "Anchor strength & Values of $\\kappa_p$ & $q_\\beta$ (\\%) & $U$ & $\\|x-x_0\\|$ & $\\sigma_a$ \\\\", "\\midrule"]
out["schedules"] = {}
for row in TROWS:
    if row is None:
        tex.append("\\addlinespace"); continue
    name, vals, rs = row
    tex.append(f"{name} & {vals} & {pm(final(rs, 'q'))} & {pm(final(rs, 'U'))} & {sci(final(rs, 'd').mean())} & ${final(rs, 's').mean():.3f}$ \\\\")
    out["schedules"][f"{name or 'Held fixed'} {vals}"] = dict(n=len(rs), q=final(rs, "q").tolist(), U=final(rs, "U").tolist(), distance=final(rs, "d").tolist(), sigma=final(rs, "s").tolist())
tex += ["\\bottomrule", "\\end{tabular}"]
(HERE / "wear-anchor-schedules-table-v76.tex").write_text("\n".join(tex) + "\n")

# ------------------------------------------------------------------ noise with the anchor during training
fig, axes = plt.subplots(2, 2, figsize=(TEXTW, TEXTW * 0.72), sharex=True)
anchor_alone = float(np.mean([np.mean(ev(r, "risk", 100)[1][ev(r, "risk")[0] > 2]) for r in fixed["10"]]))
COLS = [(G(lr="linear", k2="10", k1="0.009"), None), (G(lr="linear", k2="10", k1="0.101"), None),
        (G(lr="constant", k2="10", k1="0.101@0.5,0@1"), 4), (G(opt="sgd", lr="constant", k2="10", k1="0.101@0.5,0@1"), 4)]
out["noise"] = {}
for ax, (rs, off), s, key in zip(axes.ravel(), COLS, "abcd", ["0.009", "0.101", "removed-adam", "removed-gradient-steps"]):
    for i, r in enumerate(rs):
        x, y = ev(r, "risk", 100); ax.plot(x, y, color=PAL[i % len(PAL)], lw=0.9, ls=["-", "--", "-.", ":"][i % 4])
    ax.axhline(anchor_alone, color=TEXT2, lw=0.8, ls=":")
    if off:
        ax.axvline(off, color=MUTED, lw=0.6, ls=":")
    ax.set_ylim(bottom=-1 if key != "0.009" else -0.2); panel_label(ax, s, x=-0.14)
    w = [ev(r, "risk", 100)[1][ev(r, "risk")[0] > 1] for r in rs]
    out["noise"][key] = dict(n=len(rs), within_run_min=[float(v.min()) for v in w], within_run_max=[float(v.max()) for v in w], final_q=final(rs, "q").tolist(), final_U=final(rs, "U").tolist(), final_distance=final(rs, "d").tolist())
for ax in axes[1]:
    ax.set_xlabel("steps (millions)")
for ax in axes[:, 0]:
    ax.set_ylabel("breakdown prob. (%)")
w = [ev(r, "risk", 100)[1][ev(r, "risk")[0] > 1] for r in G(lr="linear", k2="10")]
out["noise"]["anchor-alone"] = dict(level=anchor_alone, within_run_min=[float(v.min()) for v in w], within_run_max=[float(v.max()) for v in w])
fig.tight_layout(h_pad=0.6, w_pad=0.8)
save(fig, "wear-noise-traces-v76")

# ------------------------------------------------------------------ action regularization against the kernel from dynamic programming
rec = defaultdict(list)
for f in sorted(glob.glob(str(HERE / "campaign/anchored-e*/result.json"))):
    r = json.load(open(f)); rec[r["arm"]["eta"]].append(r)
fig, axes = plt.subplots(2, 3, figsize=(TEXTW, TEXTW * 0.82))
e = np.array([s["eta"] for s in soft])
val = lambda r, key: (100 * r["stochastic"]["risk"] if key == "risk" else r["stochastic"]["U"] if key == "U" else r["action_kl"]["A"])
SCH = "1@0.25,0.1@0.5,0.01@1"
for ax, key, sc in zip(axes[0], ["risk", "U", "A"], [100, 1, 1]):
    ax.plot(e, sc * np.array([s[key] for s in soft]), color=TEXT2, lw=1.2, marker="o", ms=3, label="kernel from dynamic programming")
    for i, (eta, rs) in enumerate(sorted(rec.items())):
        ax.plot([eta] * len(rs), [val(r, key) for r in rs], ls="", marker="D", ms=3.2, mfc="none", color=PAL[1], label="learned, strength held fixed" if i == 0 else None)
    rs = G(start="default", lr="constant", eta=SCH)
    ax.plot([0.016] * len(rs), [val(r, key) for r in rs], ls="", marker="s", ms=3.2, color=PAL[2], label="learned, strength lowered")
    ax.set_xscale("log"); ax.set_xticks([1e-3, 1e-2, 1e-1, 1]); ax.set_xlabel(r"strength $\eta$")
axes[0, 2].set_yscale("log")
b01 = SOFT[0.01]
for ax, key, sc in zip(axes[1], ["risk", "U", "A"], [100, 1, 1]):
    for rs, c, ls, lab in [(G(start="default", lr="linear", eta="0.01"), C_LIN, "-", r"$\eta=0.01$, learning rate decayed to zero"), (G(start="default", lr="constant", eta="0.01"), C_CON, "--", r"$\eta=0.01$, constant learning rate"),
                           (G(start="default", lr="constant", eta=SCH), PAL[2], "-.", r"$\eta$ lowered through $1$, $0.1$, $0.01$")]:
        lines(ax, rs, lambda r: ev(r, key, sc), c, ls, lab)
    ax.axhline(sc * b01[key], color=TEXT2, lw=0.8, ls=":"); ax.set_xlabel("steps (millions)")
    for end in (2, 4):
        ax.axvline(end, color=MUTED, lw=0.6, ls=":")
for i in (0, 1):
    axes[i, 0].set_ylabel("breakdown prob. (%)"); axes[i, 1].set_ylabel("stopped utility $U$"); axes[i, 2].set_ylabel(r"$\mathcal{A}$")
axes[1, 2].set_yscale("log"); axes[1, 0].set_ylim(0, 20); axes[1, 1].set_ylim(2.6, 4.0)
panel_label(axes[0, 0], "a", x=-0.3); panel_label(axes[1, 0], "b", x=-0.3)
h0, l0 = axes[0, 0].get_legend_handles_labels(); h1, l1 = axes[1, 0].get_legend_handles_labels()
fig.tight_layout(h_pad=0.8, w_pad=0.6)
fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.17), columnspacing=1.0)
save(fig, "wear-action-kernel-v76")
out["action"] = {}
for name, rs in [("lowered|constant", G(start="default", lr="constant", eta=SCH)), ("0.01|constant", G(start="default", lr="constant", eta="0.01"))]:
    out["action"][name] = dict(n=len(rs), q=final(rs, "q").tolist(), U=final(rs, "U").tolist(), A=[r["action_kl"]["A"] for r in rs])

# ------------------------------------------------------------------ the first study during training, and the objective at the policy's own price
g = defaultdict(list)
for f in sorted(glob.glob(str(HERE / "campaign/*/result.json"))):
    g[os.path.basename(os.path.dirname(f)).split("__seed")[0]].append(json.load(open(f)))
PANELS = [("fixed-m0", "$m=0$"), ("fixed-m0.1", "$m=0.1$"), ("fixed-m0.3", "$m=0.3$"), ("fixed-m1", "$m=1$"), ("fixed-m3", "$m=3$"), ("fixed-m10", "$m=10$"),
          ("anchored-e0.01", r"$\eta=0.01$"), ("anchored-e0.1", r"$\eta=0.1$"), ("anchored-e1", r"$\eta=1$")]


def smooth(y, w=50):
    y = np.asarray(y, dtype=float); y = np.where(np.isnan(y), 0.0, y)
    return np.convolve(y, np.ones(w) / w, mode="valid")


fig, axes = plt.subplots(3, 3, figsize=(TEXTW, TEXTW * 0.95), sharex=True)
tex = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Setting & \\multicolumn{2}{c}{$q_\\beta$ (\\%)} & \\multicolumn{3}{c}{Objective} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
       " & computed & learned & computed & learned & largest shortfall \\\\", "\\midrule"]
out["own_price"] = {}
for ax, (key, title) in zip(axes.ravel(), PANELS):
    rs = g[key]; arm = rs[0]["arm"]; m = arm.get("m", 0.0); eta = arm.get("eta", 0.0)
    if arm["kind"] == "fixed":
        ref = EX[("failure", m)]; ref_q = 100 * ref["risk"]; Jex = ref["U"] - m * ref["risk"] / (1 - BETA)
    else:
        ref = SOFT[eta]; ref_q = 100 * ref["risk"]; Jex = ref["J"]
    top = ref_q
    for i, r in enumerate(rs):
        fr = np.array([t["failure_rate"] for t in r["trace"]], dtype=float); st = np.array([t["steps"] for t in r["trace"]]) / 1e6
        ax.plot(st[49:], 100 * smooth(fr), color=PAL[i % len(PAL)], lw=0.6)
        top = max(top, 100 * smooth(fr)[st[49:] > 1.0].max())
    ax.axhline(ref_q, color=TEXT2, lw=1.0, ls=":")
    ax.set_title(title, fontsize=FS + 0.5, loc="left"); ax.set_ylim(0, 1.2 * top)
    J = np.array([r["stochastic"]["U"] - m * r["stochastic"]["risk"] / (1 - BETA) - (eta * r["action_kl"]["A"] if eta else 0.0) for r in rs])
    q = final(rs, "q")
    if key == "anchored-e0.01":
        tex.append("\\addlinespace")
    tex.append(f"{title} & ${ref_q:.2f}$ & {pm(q)} & ${Jex:.2f}$ & {pm(J)} & ${Jex - J.min():.2f}$ \\\\")
    out["own_price"][key] = dict(reference_q=ref_q, q=q.tolist(), J_reference=Jex, J=J.tolist(), shortfall_mean=float(Jex - J.mean()), shortfall_max=float(Jex - J.min()))
tex += ["\\bottomrule", "\\end{tabular}"]
(HERE / "wear-own-price-table-v76.tex").write_text("\n".join(tex) + "\n")
for ax in axes[-1]:
    ax.set_xlabel("steps (millions)")
for ax in axes[:, 0]:
    ax.set_ylabel("breakdowns (%)")
fig.tight_layout(h_pad=0.6, w_pad=0.6)
save(fig, "wear-price-traces-v76")
for mm in (0.1, 1.0, 10.0):
    for lr in ("linear", "constant"):
        rs = G(start="default", lr=lr, m=mm)
        out["own_price"][f"rerun m={mm:g} {lr}"] = dict(n=len(rs), q=final(rs, "q").tolist(), U=final(rs, "U").tolist())

(HERE / "regularization-figures-v76.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print(open(HERE / "wear-optimizers-table-v76.tex").read()); print(open(HERE / "wear-anchor-schedules-table-v76.tex").read()); print(open(HERE / "wear-own-price-table-v76.tex").read())
