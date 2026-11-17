"""The pilot across formulations of the lander problem (2026-11-17, run on Roihu; stage formulations-anneal).

Two families. ledger-R*: the utility is the revenue R, paid once when the lander comes to rest on the pad, less fuel
at 1 per unit. bench-c*: the benchmark's reward with fuel at c per unit. Tanks of 18 and 24 units. The price is set so
that a failure near frame 100 is charged k times what a landing earns: the lost vehicle is worth k landings.

For every formulation, tank and k this script gives the means over the seeds of the frequencies of the ends of an
episode (failed, of which lander lost and tank empty before touchdown; completed on the pad), the fuel used and the
stopped utility, and it marks three properties of a formulation:
    lands without a price   the policies trained at k = 0 complete a landing in most episodes
    worth flying            the utility of the policies at k = 0 is positive (in the ledger a lander that is never flown has utility zero)
    price orders            the failure probability does not rise and the utility does not rise from one k to the next (means)

Usage:  python lunar_formulations_pilot.py [STAGE]      (default formulations-anneal; runtime ~/.venvs/constrained-rl)
Outputs: lunar-formulations-pilot.json, lunar-formulations-pilot-table.tex, lunar-formulations-pilot (pdf/png).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "lib"))
from plotstyle import PAL, MUTED, TEXT2, FS, TEXTW, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

stage = sys.argv[1] if len(sys.argv) > 1 else "formulations-anneal"
runs = defaultdict(dict)
for f in sorted((ROOT / "lunar" / stage).glob("*/result.json")):
    r = json.loads(f.read_text()); a = r["arm"]
    fam = ("ledger", a["revenue"]) if a.get("revenue") is not None else ("bench", a["fuel_price"])
    runs[(fam[0], float(fam[1]), a["budget"], float(a["vehicle_worth_in_landings"]))][r["seed"]] = r
KS = sorted({k[3] for k in runs})
FORMS = sorted({k[:3] for k in runs})
EARNS = {"ledger": lambda p: p, "bench": lambda p: 290.0}                  # what a landing earns, in the units of the utility


def cell(key):
    R = runs.get(key, {})
    S = [R[s]["stochastic"] for s in sorted(R)]
    if not S:
        return None
    g = lambda name: [x[name] for x in S]
    return dict(n=len(S), seeds=sorted(R), risk=g("risk"), lost=g("lost"), exhausted=g("exhausted"), landed=g("landed"), completed=g("completed"),
                fuel=g("fuel_mean"), U=g("U"), m=next(iter(R.values()))["arm"]["m"])


out = dict(stage=stage, vehicle_worth=KS, formulations={})
rows = []
for fam, par, tank in FORMS:
    C = {k: cell((fam, par, tank, k)) for k in KS}
    if any(c is None for c in C.values()):
        continue
    earns = EARNS[fam](par)
    mean = lambda k, name: float(np.mean(C[k][name]))
    risk = [mean(k, "risk") for k in KS]; U = [mean(k, "U") for k in KS]
    flags = dict(lands_without_a_price=bool(mean(0.0, "completed") > 0.5), worth_flying=bool(mean(0.0, "U") > 0.0),
                 failure_does_not_rise=bool(all(b <= a + 0.003 for a, b in zip(risk, risk[1:]))), utility_does_not_rise=bool(all(b <= a + 0.01 * earns for a, b in zip(U, U[1:]))),
                 fell_in_seeds={f"{k:g}": int(sum(x < y for x, y in zip(C[k]["risk"], C[0.0]["risk"]))) for k in KS[1:]})
    name = (f"ledger, revenue {par:g}" if fam == "ledger" else f"benchmark reward, fuel at {par:g}") + f", tank {tank:g}"
    out["formulations"][name] = dict(family=fam, parameter=par, tank=tank, a_landing_earns=earns, cells={f"{k:g}": C[k] for k in KS}, **flags)
    for k in KS:
        c = C[k]
        rows.append((name if k == KS[0] else "") + f" & ${k:g}$ & ${100 * np.mean(c['risk']):.1f}\\pm{100 * np.std(c['risk'], ddof=1):.1f}$ & ${100 * np.mean(c['lost']):.1f}$ & ${100 * np.mean(c['exhausted']):.1f}$"
                    f" & ${100 * np.mean(c['completed']):.1f}$ & ${np.mean(c['fuel']):.1f}$ & ${np.mean(c['U']):.1f}$ & ${np.mean(c['U']) / earns:.2f}$ \\\\")
    rows.append("\\addlinespace")
(ROOT / "lunar-formulations-pilot.json").write_text(json.dumps(out, indent=1) + "\n")
(ROOT / "lunar-formulations-pilot-table.tex").write_text(
    "\\begin{tabular}{lrrrrrrrr}\n\\toprule\nFormulation & Vehicle worth $k$ & Failed (\\%) & Lost (\\%) & Tank empty (\\%) & Completed (\\%) & Fuel & $U$ & $U$ per landing's revenue \\\\\n\\midrule\n"
    + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

print(f"{'formulation':42s} {'k':>4s} {'n':>2s} {'failed':>7s} {'lost':>6s} {'empty':>6s} {'compl':>6s} {'fuel':>5s} {'U':>8s} {'U/earn':>7s}  per seed failed")
for name, F in out["formulations"].items():
    for k in KS:
        c = F["cells"][f"{k:g}"]
        print(f"{name if k == KS[0] else '':42s} {k:4g} {c['n']:2d} {100 * np.mean(c['risk']):6.1f}% {100 * np.mean(c['lost']):5.1f}% {100 * np.mean(c['exhausted']):5.1f}% {100 * np.mean(c['completed']):5.1f}% {np.mean(c['fuel']):5.1f} {np.mean(c['U']):8.1f} {np.mean(c['U']) / F['a_landing_earns']:7.2f}  {[round(100 * x, 1) for x in c['risk']]}")
    print(f"{'':42s} lands without a price: {F['lands_without_a_price']}; worth flying: {F['worth_flying']}; failure does not rise: {F['failure_does_not_rise']}; utility does not rise: {F['utility_does_not_rise']}; fell in seeds {F['fell_in_seeds']}")

if out["formulations"]:
    fams = [("ledger", "revenue"), ("bench", "fuel at")]
    fig, axes = plt.subplots(3, 4, figsize=(TEXTW, 4.6), sharex=True, gridspec_kw=dict(wspace=0.42, hspace=0.22))
    x = np.arange(len(KS))
    for j, (fam, tank) in enumerate([(f, t) for f, _ in fams for t in sorted({k[2] for k in FORMS})]):
        pars = sorted({k[1] for k in FORMS if k[0] == fam and k[2] == tank})
        for i, par in enumerate(pars):
            name = (f"ledger, revenue {par:g}" if fam == "ledger" else f"benchmark reward, fuel at {par:g}") + f", tank {tank:g}"
            if name not in out["formulations"]:
                continue
            F = out["formulations"][name]; C = [F["cells"][f"{k:g}"] for k in KS]
            lab = (f"revenue ${par:g}$" if fam == "ledger" else f"fuel at ${par:g}$")
            axes[0, j].plot(x, [max(100 * np.mean(c["risk"]), 0.3) for c in C], color=PAL[i], marker="o", ms=3, lw=1.1, label=lab)
            axes[1, j].plot(x, [np.mean(c["fuel"]) for c in C], color=PAL[i], marker="o", ms=3, lw=1.1)
            axes[2, j].plot(x, [np.mean(c["U"]) / F["a_landing_earns"] for c in C], color=PAL[i], marker="o", ms=3, lw=1.1)
        axes[0, j].set_yscale("log"); axes[0, j].set_ylim(0.3, 110); axes[0, j].set_yticks([1, 3, 10, 30, 100]); axes[0, j].set_yticklabels(["1", "3", "10", "30", "100"]); axes[0, j].minorticks_off()
        axes[0, j].set_title(("ledger" if fam == "ledger" else "benchmark reward") + f",\ntank ${tank:g}$", fontsize=FS - 1.5, pad=8)
        axes[0, j].legend(frameon=False, fontsize=FS - 3, loc="upper right", handlelength=1.2)
        axes[2, j].axhline(0, color=MUTED, lw=0.6); axes[1, j].axhline(tank, color=MUTED, lw=0.6, ls=":")
        axes[2, j].set_xticks(x); axes[2, j].set_xticklabels([f"{k:g}" for k in KS]); axes[2, j].set_xlabel("vehicle worth $k$")
        for i in range(3):
            panel_label(axes[i, j], "abcdefghijkl"[i * 4 + j])
    axes[0, 0].set_ylabel("failed (%)"); axes[1, 0].set_ylabel("fuel used"); axes[2, 0].set_ylabel("utility per\nlanding's revenue")
    save(fig, "lunar-formulations-pilot", ROOT)
