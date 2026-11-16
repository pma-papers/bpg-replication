"""Revision 76: convergence check of the first wear-allowance campaign from its records.

For each of the fifteen configurations (eight seeds, campaign/*/result.json) the script draws the breakdown frequency of
the episodes completed in each rollout during training (moving mean over 50 rollouts), one line per seed, with the
breakdown probability of the exact optimum at the configuration's own price, of the tolerance, or of the benchmark
kernel (soft-benchmark-v76.json). It prints, per configuration, the shortfall of the objective at the policy's own price
from the exact optimum (fixed prices: J = U - m q / (1 - beta); barrier: J = U - I_m - m q / (1 - beta)), and the change
of the breakdown frequency between the third and the fourth quarter of training. Where reruns with a constant
learning rate exist (wear-schedules-v76/campaign) their evaluated breakdown probabilities are drawn as dashed lines.
The learning rate of the campaign decays linearly to zero, so a flat end of a trace is not by itself evidence of
convergence; the shortfall and the reruns are.
Runtime ~/.venvs/constrained-rl/bin/python. Output wear-campaign-convergence-v76.pdf/png and .json.
"""
from __future__ import annotations
import glob, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from plotstyle import PAL, TEXT2, FS, TEXTW, plt  # noqa: E402

BETA = 0.95
bench = json.load(open(HERE / "benchmark.json"))
ex = {(r["kind"], r["m"]): r["nominal"] for r in bench["rows"] if r["kind"] == "failure" or (r["kind"] == "barrier" and r["theta"] == 0.1)}   # the barrier of the study has the scale 0.1
soft = {s["eta"]: s for s in json.load(open(HERE / "soft-benchmark-v76.json"))["rows"]} if (HERE / "soft-benchmark-v76.json").exists() else {}
g = defaultdict(list)
for f in sorted(glob.glob(str(HERE / "campaign/*/result.json"))):
    g[os.path.basename(os.path.dirname(f)).split("__seed")[0]].append(json.load(open(f)))
rer = defaultdict(list)
for f in sorted(glob.glob(str(HERE / "wear-schedules-v76/campaign/default__*/result.json"))):
    r = json.load(open(f)); a = r["arm"]
    if a["lr_schedule"] == "constant" and "@" not in a["eta"]:
        rer[(a["m"], float(a["eta"]))].append(r)
ORDER = [("fixed-m0", "price 0"), ("fixed-m0.1", "price 0.1"), ("fixed-m0.3", "price 0.3"), ("fixed-m1", "price 1"), ("fixed-m3", "price 3"), ("fixed-m10", "price 10"),
         ("barrier-m1", "barrier, price 1"), ("barrier-m3", "barrier, price 3"), ("adaptive-a0.05-s5", "tolerance 0.05"), ("adaptive-a0.05-s1", "tolerance 0.05, dual step 1"),
         ("adaptive-a0.01-s5", "tolerance 0.01"), ("adaptive-a0-s5", "tolerance 0"), ("anchored-e0.01", r"$\eta=0.01$"), ("anchored-e0.1", r"$\eta=0.1$"), ("anchored-e1", r"$\eta=1$")]


def smooth(y, w=50):
    y = np.asarray(y, dtype=float); y = np.where(np.isnan(y), 0.0, y)
    return np.convolve(y, np.ones(w) / w, mode="valid")


out = {}
fig, axes = plt.subplots(5, 3, figsize=(TEXTW * 1.3, TEXTW * 1.35), sharex=True)
for ax, (key, title) in zip(axes.ravel(), ORDER):
    rs = g[key]; arm = rs[0]["arm"]; m = arm.get("m", 0.0)
    ref, ref_label, Jex = None, None, None
    if arm["kind"] in ("fixed", "barrier"):
        e = ex[("failure" if arm["kind"] == "fixed" else "barrier", m)]
        ref, ref_label = 100 * e["risk"], "exact optimum"
        Jex = e["U"] - m * e["risk"] / (1 - BETA) - (e["I"] if arm["kind"] == "barrier" else 0.0)
    elif arm["kind"] == "adaptive":
        ref, ref_label = 100 * arm["alpha"], "tolerance"
    elif arm["eta"] in soft:
        ref, ref_label = 100 * soft[arm["eta"]]["risk"], "benchmark kernel"; Jex = soft[arm["eta"]]["J"]
    q34, top = [], 0.0
    for i, r in enumerate(rs):
        fr = np.array([t["failure_rate"] for t in r["trace"]], dtype=float); st = np.array([t["steps"] for t in r["trace"]]) / 1e6
        ax.plot(st[49:], 100 * smooth(fr), color=PAL[i % len(PAL)], lw=0.7)
        top = max(top, 100 * smooth(fr)[st[49:] > 1.0].max())
        n = len(fr); q34.append((np.nanmean(fr[n // 2:3 * n // 4]), np.nanmean(fr[3 * n // 4:])))
    for r in (rer.get((m, arm.get("eta", 0.0)), []) if arm["kind"] in ("fixed", "anchored") else []):
        ax.plot([e_["steps"] / 1e6 for e_ in r["evals"]], [100 * e_["risk"] for e_ in r["evals"]], color=TEXT2, lw=0.9, ls="--")
    if ref is not None:
        ax.axhline(ref, color=TEXT2, lw=1.0, ls=":")
        top = max(top, ref)
    ax.set_title(title, fontsize=FS, loc="left"); ax.set_ylim(0, 1.25 * top)
    q34 = 100 * np.array(q34)
    row = dict(q_final_percent=[100 * r["stochastic"]["risk"] for r in rs], change_quarter3_to_quarter4_points=(q34[:, 1] - q34[:, 0]).tolist())
    if Jex is not None:
        eta = arm.get("eta", 0.0)
        J = np.array([r["stochastic"]["U"] - m * r["stochastic"]["risk"] / (1 - BETA) - (r["stochastic"]["I"] if arm["kind"] == "barrier" else 0.0) - (eta * r["action_kl"]["A"] if eta else 0.0) for r in rs])
        row.update(J_exact=Jex, J_learned=J.tolist(), shortfall_mean=float(Jex - J.mean()), shortfall_max=float(Jex - J.min()), reference_q_percent=ref)
    out[key] = row
    print(f"{key:20s} q final {np.mean(row['q_final_percent']):6.2f}% [{min(row['q_final_percent']):.2f}, {max(row['q_final_percent']):.2f}]  ref {ref if ref is not None else float('nan'):6.2f}%  "
          f"change Q3->Q4 max |{np.abs(row['change_quarter3_to_quarter4_points']).max():.2f}| points  " + (f"shortfall of J: mean {row['shortfall_mean']:.3f}, max {row['shortfall_max']:.3f} (J exact {Jex:.3f})" if Jex is not None else ""))
for ax in axes[-1]:
    ax.set_xlabel("training steps (millions)")
for ax in axes[:, 0]:
    ax.set_ylabel("breakdowns (%)")
fig.tight_layout(h_pad=0.8, w_pad=0.8)
fig.savefig(HERE / "wear-campaign-convergence-v76.pdf"); fig.savefig(HERE / "wear-campaign-convergence-v76.png")
(HERE / "wear-campaign-convergence-v76.json").write_text(json.dumps(out, indent=1) + "\n")
