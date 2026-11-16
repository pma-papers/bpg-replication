"""The adaptive multiplier in the final design of the lander study (2026-11-16).

In the adaptive configurations a failure is charged the multiplier, a flat amount at the failing frame. After every
rollout the multiplier changes by the dual step times the difference between the failure rate of the episodes that
ended in the rollout and the tolerance, and it is kept nonnegative. A fixed price m charges
m (1 - beta^(H - t)) / (1 - beta) at the failing frame t, so a multiplier corresponds to the price at which this
charge, at the median failing frame of the fixed-price policies, equals it.

Usage:  python lunar_adaptive_fuelprice.py ADAPTIVE-STAGE FIXED-STAGE [FINAL-PHASE-STAGE]
        (for the final design: fuelprice-20-adaptive fuelprice-20-anneal fuelprice-20-adaptive-anneal; the fixed
        stage needs priced-risk.json)
With a final-phase stage, the final policies (failure probability, fuel, utility) are those of that stage, in which
the runs of the adaptive stage are continued with the multiplier held and the learning rate decayed to zero. The
multiplier and the failure rate of training are those of the adaptive stage in both cases.
Outputs (this directory): lunar-ADAPTIVE-STAGE.json, lunar-ADAPTIVE-STAGE-table.tex and lunar-ADAPTIVE-STAGE
(pdf/png) with two panels: (a) the multiplier during training, with the charges of the fixed prices 0.5 and 2 as
dotted lines, (b) the failure rate in training, with the tolerances as dotted lines; both as the mean (line) and the
range (band) over the seeds, each point a mean over about 2 percent of the training.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, LS, TEXT2, MUTED, TEXTW, FS, panel_label, legend_below, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

stage, fixed = sys.argv[1:3]
final = sys.argv[3] if len(sys.argv) > 3 else None
BETA, H = 0.999, 1000
runs = {}
for f in sorted((ROOT / "lunar" / stage).glob("*/result.json")):
    r = json.loads(f.read_text())
    if final is not None:                                       # the final policy is that of the same run in the final-phase stage
        g = json.loads((ROOT / "lunar" / final / f.parent.name / "result.json").read_text())
        assert abs(g["final_multiplier"] - r["final_multiplier"]) < 1e-9, f.parent.name      # the multiplier was held
        r["policy"] = g["stochastic"]
    else:
        r["policy"] = r["stochastic"]
    runs.setdefault((r["arm"]["alpha"], r["arm"]["dual_step"]), {})[r["seed"]] = r
seeds = sorted(next(iter(runs.values())))
configs = sorted(k for k in runs if sorted(runs[k]) == seeds)

frames = []                                                     # failing frames of the fixed-price policies with a price
for f in sorted((ROOT / "lunar" / fixed).glob("*/priced-risk.json")):
    res = json.loads((f.parent / "result.json").read_text())
    if res["arm"]["m"] > 0:
        frames += json.loads(f.read_text())["exhaustion_frames"]            # the frames of all failures of the stage
t_med = float(np.median(frames))
per_price = (1 - BETA ** (H - t_med)) / (1 - BETA)              # the charge at the median failing frame per unit of price

out = dict(stage=stage, fixed_stage=fixed, final_phase_stage=final, seeds=seeds, median_failing_frame=t_med, charge_per_unit_price=per_price, configurations={})
rows = []
for a, s in configs:
    R = [runs[(a, s)][x] for x in seeds]
    n = len(R[0]["trace"])
    lam = np.array([[x["multiplier"] for x in r["trace"][:n]] for r in R]); fr = np.array([[x["failure_rate"] for x in r["trace"][:n]] for r in R], float)
    last = slice(n * 9 // 10, n)
    c = dict(alpha=a, dual_step=s, steps=R[0]["steps"], risk=[r["policy"]["risk"] for r in R], U=[r["policy"]["U"] for r in R],
             fuel_median=[r["policy"]["fuel_quantiles"][2] for r in R], final_multiplier=[r["final_multiplier"] for r in R],
             largest_multiplier=[float(x.max()) for x in lam], training_failure_rate_last_tenth=[float(np.nanmean(x[last])) for x in fr],
             multiplier_change_last_tenth=[float(x[-1] - x[last][0]) for x in lam])
    c.update(risk_mean=float(np.mean(c["risk"])), risk_sd=float(np.std(c["risk"], ddof=1)), U_mean=float(np.mean(c["U"])), fuel_mean=float(np.mean(c["fuel_median"])),
             training_rate_mean=float(np.mean(c["training_failure_rate_last_tenth"])), multiplier_mean=float(np.mean(c["final_multiplier"])),
             price_equivalent=[x / per_price for x in c["final_multiplier"]])
    out["configurations"][f"a{a:g}-s{s:g}"] = c
    rows.append(f"${a:g}$ & ${s:g}$ & ${100 * c['training_rate_mean']:.1f}$ & ${100 * c['risk_mean']:.1f}\\pm{100 * c['risk_sd']:.1f}$ & ${min(c['final_multiplier']):.0f}$ to ${max(c['final_multiplier']):.0f}$ & "
                f"${min(c['price_equivalent']):.2f}$ to ${max(c['price_equivalent']):.2f}$ & ${np.mean(c['multiplier_change_last_tenth']):+.0f}$ & ${c['fuel_mean']:.1f}$ & ${c['U_mean']:.0f}$ \\\\")
(ROOT / f"lunar-{stage}.json").write_text(json.dumps(out, indent=1) + "\n")
(ROOT / f"lunar-{stage}-table.tex").write_text(
    "\\begin{tabular}{rrrrrrrrr}\n\\toprule\nTolerance & Dual step & Training (\\%) & Failed (\\%) & Multiplier & As a price & Change & Fuel & $U_H$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTW, 2.35), gridspec_kw=dict(wspace=0.45))
for i, (a, s) in enumerate(configs):
    R = [runs[(a, s)][x] for x in seeds]
    n = min(len(r["trace"]) for r in R); k = max(1, n // 50); nb = n // k
    st = np.array([x["steps"] for x in R[0]["trace"][:nb * k]], float).reshape(nb, k).mean(1) / 1e6
    lam = np.array([[x["multiplier"] for x in r["trace"][:nb * k]] for r in R]).reshape(len(seeds), nb, k).mean(2)
    fr = np.nanmean(np.array([[x["failure_rate"] for x in r["trace"][:nb * k]] for r in R], float).reshape(len(seeds), nb, k), axis=2)
    lab = f"tolerance ${a:g}$, dual step ${s:g}$"
    ax1.plot(st, lam.mean(0), color=PAL[i], ls=LS[i], lw=1.2, label=lab); ax1.fill_between(st, lam.min(0), lam.max(0), color=PAL[i], alpha=0.13, lw=0)
    ax2.plot(st, np.maximum(fr.mean(0), 1e-3), color=PAL[i], ls=LS[i], lw=1.2); ax2.fill_between(st, np.maximum(fr.min(0), 1e-3), np.maximum(fr.max(0), 1e-3), color=PAL[i], alpha=0.13, lw=0)
ax1.set_yscale("log"); ax1.set_ylim(50, 12000); ax1.set_ylabel("multiplier"); ax1.set_xlabel("training steps ($10^6$)")
for m in (0.5, 2):
    ax1.axhline(m * per_price, color=MUTED, lw=0.7, ls=":")           # the charge of the fixed price m at the median failing frame
ax2.set_yscale("log"); ax2.set_ylim(0.008, 1.2); ax2.set_yticks([0.01, 0.02, 0.05, 0.1, 0.3, 1.0]); ax2.set_yticklabels(["1%", "2%", "5%", "10%", "30%", "100%"]); ax2.minorticks_off()
for a in sorted({a for a, _ in configs}):
    ax2.axhline(a, color=MUTED, lw=0.7, ls=":")
ax2.set_ylabel("failure rate in training"); ax2.set_xlabel("training steps ($10^6$)")
panel_label(ax1, "a"); panel_label(ax2, "b")
legend_below(fig, *ax1.get_legend_handles_labels(), ncol=2)
save(fig, f"lunar-{stage}", ROOT)
print(f"median failing frame {t_med:.0f}, charge per unit of price {per_price:.0f}")
for key, c in out["configurations"].items():
    print(f"{key:12s} training rate, last tenth {100 * c['training_rate_mean']:.1f} {[round(100 * x, 1) for x in c['training_failure_rate_last_tenth']]}  failed {100 * c['risk_mean']:.1f} {[round(100 * x, 1) for x in c['risk']]}"
          f"  multiplier {[round(x) for x in c['final_multiplier']]} as a price {[round(x, 2) for x in c['price_equivalent']]} change in the last tenth {[round(x) for x in c['multiplier_change_last_tenth']]}  fuel {c['fuel_mean']:.1f}  U {c['U_mean']:.1f}")
