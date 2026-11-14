"""Revision 76: the analytical example under relative entropy with a strength that is lowered in steps.

The particle system of the gradient flow,
    dx = V_m'(c) phi'(x) dl - (kappa/sigma^2) x dl + sqrt(2 kappa) dB,    c = mean of phi over the units,
is simulated by the Euler-Maruyama scheme for the priors N(0, sigma^2), sigma in {1/2, 1, 2}, from the initial law of
the paper (N(0,1) conditioned on [-1/2, 1/2]). The clipped value is that of the tilt c, read from the value curve of
regularizer-flow.json. The strength takes the values 10, 3, 1, 0.3, 0.1, 0.03 in turn, at which the minimizers are
known from the finite-volume runs (prior-path.json, with the rerun of the one run that had not converged). The
configurations differ in when the strength is lowered:
  fixed         the strength 0.03 from the start;
  equal         every stage has the length 2;
  reciprocal    the stage at the strength kappa has the length c/kappa, for c = 1 and c = 3;
  rule          a stage ends when the mean of the clipped value over a window of length 0.5/kappa differs from the mean
                over the window before it by less than 0.002.
Reported: the clipped value against the learning time (means over the seeds), the learning time at which the clipped
value first comes within 0.02 of the value of the minimizer at the strength 0.03, and the clipped value at the end of
the stage at the strength 0.1 against the value of the minimizer at that strength.
Outputs: decreasing-strength-v76 (pdf/png), decreasing-strength-v76.json, decreasing-strength-table-v76.tex.
Run with ~/.venvs/constrained-rl/bin/python decreasing_strength_v76.py (about five minutes on fourteen cores).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))

D = json.loads((ROOT / "regularizer-flow.json").read_text())
PATH = json.loads((ROOT / "prior-path.json").read_text())
C, V = np.array(D["curve"]["c"]), np.array(D["curve"]["V_m"])
DV = np.gradient(V, C)
N, T, SEEDS = 10000, 400.0, tuple(range(1, 9))
STAGES = (10.0, 3.0, 1.0, 0.3, 0.1, 0.03)
EQUAL_LENGTH, WINDOW, STEADY, WITHIN = 2.0, 0.5, 2e-3, 0.02
MARKS = np.geomspace(0.05, T, 200)
SIGMAS = (0.5, 1.0, 2.0)
CONFIGS = (("fixed", None), ("equal", None), ("reciprocal", 1.0), ("reciprocal", 3.0), ("rule", None))


def phi(x):
    y = np.maximum(np.abs(x) - 1, 1e-12)
    return np.where(np.abs(x) > 1, np.exp(-y ** -2), 0.0)


def dphi(x):
    y = np.maximum(np.abs(x) - 1, 1e-12)
    return np.where(np.abs(x) > 1, np.sign(x) * 2 * y ** -3 * np.exp(-y ** -2), 0.0)


RERUN = json.loads((ROOT / "prior-path-rerun.json").read_text())      # the one path run that had not converged
PATH["runs"][RERUN["name"]] = RERUN


def minimizer_values(sigma):
    return {r["kappa"]: r["final"]["V_m"] for r in PATH["runs"].values() if r["x0"] == 0.0 and r["sigma"] == sigma}


def run(args):
    kind, c, sigma, seed = args
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(12 * N)
    x = x[np.abs(x) <= 0.5][:N]
    target = minimizer_values(sigma)[STAGES[-1]] - WITHIN
    stage = len(STAGES) - 1 if kind == "fixed" else 0
    l, start, i = 0.0, 0.0, 0
    trace, ends, reached = [], {}, None
    acc, steps, previous = 0.0, 0, None                      # the window means of the rule
    while i < len(MARKS):
        k = STAGES[stage]
        dt = min(0.01, 0.02 / max(k / sigma ** 2, 1.0))
        tilt = phi(x).mean()
        x = x + dt * (np.interp(tilt, C, DV) * dphi(x) - k * x / sigma ** 2) + np.sqrt(2 * k * dt) * rng.standard_normal(N)
        l += dt
        v = float(np.interp(phi(x).mean(), C, V))
        if reached is None and v >= target:
            reached = l
        if stage < len(STAGES) - 1:
            leave = False
            if kind == "equal":
                leave = l - start >= EQUAL_LENGTH
            elif kind == "reciprocal":
                leave = l - start >= c / k
            elif kind == "rule":
                acc += v; steps += 1
                if steps * dt >= WINDOW / k:
                    mean = acc / steps
                    leave = previous is not None and abs(mean - previous) < STEADY
                    previous, acc, steps = mean, 0.0, 0
            if leave:
                ends[k] = (l, v)
                stage += 1; start = l; previous, acc, steps = None, 0.0, 0
        if l >= MARKS[i]:
            trace.append(v); i += 1
    return dict(kind=kind, c=c, sigma=sigma, seed=seed, trace=trace, ends=ends, reached=reached)


def main():
    from plotstyle import PAL, TEXT2, MUTED, TEXTW, FS, panel_label, legend_below, save
    import matplotlib.pyplot as plt
    jobs = [(kind, c, sigma, seed) for sigma in SIGMAS for kind, c in CONFIGS for seed in SEEDS]
    with Pool() as pool:
        results = pool.map(run, jobs, chunksize=1)
    key = lambda kind, c: kind if c is None else f"{kind}, c = {c:g}"
    label = {"fixed": "fixed $\\kappa=0.03$", "equal": "stages of length $2$", "reciprocal, c = 1": "stages of length $1/\\kappa$",
             "reciprocal, c = 3": "stages of length $3/\\kappa$", "rule": "rule"}
    style = {"fixed": (PAL[0], "-"), "equal": (PAL[1], "--"), "reciprocal, c = 1": (PAL[2], "-."),
             "reciprocal, c = 3": (PAL[3], (0, (5, 1, 1, 1))), "rule": (PAL[6], ":")}
    name = {0.5: "$N(0,1/4)$", 1.0: "$N(0,1)$", 2.0: "$N(0,4)$"}
    out = dict(units=N, seeds=list(SEEDS), stages=list(STAGES), learning_time=T, times=[round(float(a), 4) for a in MARKS], priors={})
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.3), sharey=True, gridspec_kw=dict(wspace=0.1))
    rows = {}
    for ax, sigma, lab in zip(axes, SIGMAS, "abc"):
        star = minimizer_values(sigma)
        block = dict(minimizer_values={f"{k:g}": round(star[k], 4) for k in STAGES}, configurations={})
        for kind, c in CONFIGS:
            rs = [r for r in results if r["sigma"] == sigma and r["kind"] == kind and r["c"] == c]
            tr = np.array([r["trace"] for r in rs])
            reached = [r["reached"] for r in rs]
            ends = {f"{k:g}": dict(time=round(float(np.mean([r["ends"][k][0] for r in rs])), 2), value=round(float(np.mean([r["ends"][k][1] for r in rs])), 4),
                                    minimizer=round(star[k], 4)) for k in STAGES[:-1]} if kind != "fixed" else {}
            entry = dict(mean=[round(float(a), 4) for a in tr.mean(0)], largest_sd=round(float(tr.std(0, ddof=1).max()), 4),
                         final_value=round(float(tr[:, -1].mean()), 4), seeds_within=int(sum(t is not None for t in reached)),
                         time_within=None if any(t is None for t in reached) else dict(mean=round(float(np.mean(reached)), 1), min=round(float(np.min(reached)), 1), max=round(float(np.max(reached)), 1)),
                         stage_ends=ends)
            block["configurations"][key(kind, c)] = entry
            rows.setdefault(key(kind, c), {})[sigma] = entry
            col, ls = style[key(kind, c)]
            ax.plot(MARKS, tr.mean(0), color=col, ls=ls, lw=1.2, label=label[key(kind, c)])
            print(name[sigma], key(kind, c), "final", entry["final_value"], "time", entry["time_within"], "largest sd", entry["largest_sd"],
                  "stage 0.1:", ends.get("0.1"))
        out["priors"][f"sigma = {sigma:g}"] = block
        ax.axhline(V[0], color=MUTED, lw=0.7); ax.axhline(V.max(), color=MUTED, lw=0.7); ax.axhline(star[STAGES[-1]], color=MUTED, lw=0.7, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("learning time $\\ell$"); panel_label(ax, lab)
        ax.text(0.04, 0.88, "prior " + name[sigma], transform=ax.transAxes, fontsize=FS, color=TEXT2)
    axes[0].set_ylabel("clipped value $V_m$")
    legend_below(fig, *axes[0].get_legend_handles_labels(), ncol=3)
    save(fig, "decreasing-strength-v76", ROOT)
    (ROOT / "decreasing-strength-v76.json").write_text(json.dumps(out, indent=1) + "\n")

    def time_cell(e):
        t = e["time_within"]
        return "--" if t is None else f"${t['mean']:.0f}$ (${t['min']:.0f}$ to ${t['max']:.0f}$)"

    def gap_cell(e):
        s = e["stage_ends"].get("0.1")
        return "--" if s is None else f"${s['minimizer'] - s['value']:.3f}$"

    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", " & \\multicolumn{2}{c}{prior $N(0,1)$} & \\multicolumn{2}{c}{prior $N(0,4)$} \\\\",
             "Lowering of the strength & time & shortfall & time & shortfall \\\\", "\\midrule"]
    for kind, c in CONFIGS:
        r = rows[key(kind, c)]
        lines.append(f"{label[key(kind, c)]} & {time_cell(r[1.0])} & {gap_cell(r[1.0])} & {time_cell(r[2.0])} & {gap_cell(r[2.0])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "decreasing-strength-table-v76.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
