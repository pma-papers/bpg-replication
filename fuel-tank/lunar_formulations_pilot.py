"""The pilot across formulations of the lander problem (2026-11-17, run on Roihu; stage formulations-anneal).

Two families. ledger-R*: the utility is the revenue R, paid once when the lander comes to rest on the pad, less fuel
at 1 per unit. bench-c*: the benchmark's reward with fuel at c per unit. Tanks of 18, 20, 22 and 24 units (the author, 18 November: only tanks of at least 20 units are worth
considering, so the tables and figures show 20, 22 and 24 unless --tanks says otherwise). The price is set so
that a failure near frame 100 is charged k times what a landing earns: the lost vehicle is worth k landings.

For every formulation, tank and k this script gives the means over the seeds of the frequencies of the ends of an
episode (failed, of which lander lost and tank empty before touchdown; completed on the pad), the fuel used and the
stopped utility, and it marks three properties of a formulation:
    lands without a price   the policies trained at k = 0 complete a landing in most episodes
    worth flying            the utility of the policies at k = 0 is positive (in the ledger a lander that is never flown has utility zero)
    price orders            the failure probability does not rise and the utility does not rise from one k to the next (means)

Usage:  python lunar_formulations_pilot.py [STAGE] [--tanks 20,22,24]      (default formulations-anneal; runtime ~/.venvs/constrained-rl)
Outputs: lunar-formulations-pilot.json, lunar-formulations-pilot-table.tex, lunar-formulations-pilot-ledger and -bench (pdf/png).
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

args = sys.argv[1:]
TANKS = [20.0, 22.0, 24.0]
if "--tanks" in args:
    i = args.index("--tanks"); TANKS = [float(t) for t in args[i + 1].split(",")]; del args[i:i + 2]
stage = args[0] if args else "formulations-anneal"
runs = defaultdict(dict)
for f in sorted((ROOT / "lunar" / stage).glob("*/result.json")):
    r = json.loads(f.read_text()); a = r["arm"]
    if float(a["budget"]) not in TANKS:
        continue
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
        print(f"{name if k == KS[0] else '':42s} {k:4g} {c['n']:2d} {100 * np.mean(c['risk']):6.1f}% {100 * np.mean(c['lost']):5.1f}% {100 * np.mean(c['exhausted']):5.1f}% {100 * np.mean(c['completed']):5.1f}% {np.mean(c['fuel']):5.1f} {np.mean(c['U']):8.1f} {np.mean(c['U']) / F['a_landing_earns']:7.2f}  {[round(100 * x, 1) for x in c['risk']]}  median {100 * np.median(c['risk']):.1f}%")
    print(f"{'':42s} lands without a price: {F['lands_without_a_price']}; worth flying: {F['worth_flying']}; failure does not rise: {F['failure_does_not_rise']}; utility does not rise: {F['utility_does_not_rise']}; fell in seeds {F['fell_in_seeds']}")

tanks = sorted({k[2] for k in FORMS})
for fam, word in (("ledger", "revenue"), ("bench", "fuel at")):
    if not any(F["family"] == fam for F in out["formulations"].values()):
        continue
    fig, axes = plt.subplots(4, len(tanks), figsize=(TEXTW, 5.6), sharex=True, sharey="row", squeeze=False, gridspec_kw=dict(wspace=0.12, hspace=0.2))
    x = np.arange(len(KS))
    for j, tank in enumerate(tanks):
        pars = sorted({k[1] for k in FORMS if k[0] == fam and k[2] == tank})
        for i, par in enumerate(pars):
            name = (f"ledger, revenue {par:g}" if fam == "ledger" else f"benchmark reward, fuel at {par:g}") + f", tank {tank:g}"
            if name not in out["formulations"]:
                continue
            F = out["formulations"][name]; C = [F["cells"][f"{k:g}"] for k in KS]
            kw = dict(color=PAL[i], marker="os^"[i % 3], ms=3, lw=1.1)
            off = 0.16 * (i - (len(pars) - 1) / 2)
            for row, (key, scale, floor) in enumerate((("risk", 100.0, 0.3), ("completed", 100.0, None), ("fuel", 1.0, None), ("U", 1.0 / F["a_landing_earns"], None))):
                med = [scale * np.median(c[key]) for c in C]
                axes[row, j].plot(x + off, med if floor is None else np.maximum(med, floor), label=f"{word} ${par:g}$" if row == 0 else None, **kw)
                for xi, c in zip(x, C):
                    v = scale * np.array(c[key]); axes[row, j].plot([xi + off] * len(v), v if floor is None else np.maximum(v, floor), ls="", marker=".", ms=2.5, color=PAL[i], alpha=0.45)
        axes[0, j].set_yscale("log"); axes[0, j].set_ylim(0.3, 130); axes[0, j].set_yticks([1, 3, 10, 30, 100]); axes[0, j].set_yticklabels(["1", "3", "10", "30", "100"]); axes[0, j].minorticks_off()
        axes[0, j].set_title(f"tank ${tank:g}$", fontsize=FS - 1, pad=6)
        axes[1, j].set_ylim(-4, 104)
        axes[2, j].set_ylim(13, 25)
        axes[3, j].axhline(0, color=MUTED, lw=0.6); axes[2, j].axhline(tank, color=MUTED, lw=0.6, ls=":")
        axes[3, j].set_xticks(x); axes[3, j].set_xticklabels([f"{k:g}" for k in KS])
        for i in range(4):
            panel_label(axes[i, j], "abcdefghijklmnop"[i * len(tanks) + j])
    axes[3, len(tanks) // 2].set_xlabel("worth of the vehicle in landings, $k$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=FS - 2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965), handlelength=1.6)
    axes[0, 0].set_ylabel("failed (%)"); axes[1, 0].set_ylabel("completed\non the pad (%)"); axes[2, 0].set_ylabel("fuel used"); axes[3, 0].set_ylabel("utility per\nlanding's revenue")
    save(fig, f"lunar-formulations-pilot-{fam}", ROOT)

# the ledger by the amount at stake in a failure, (1 + k) R: the forgone revenue and the charge
led = [F for F in out["formulations"].values() if F["family"] == "ledger"]
if led:
    fig, axes = plt.subplots(2, len(tanks), figsize=(TEXTW, 3.3), sharex=True, sharey="row", squeeze=False, gridspec_kw=dict(wspace=0.12, hspace=0.16))
    revs = sorted({F["parameter"] for F in led})
    for j, tank in enumerate(tanks):
        for i, R in enumerate(revs):
            F = next((F for F in led if F["parameter"] == R and F["tank"] == tank), None)
            if F is None:
                continue
            xs = [(1 + k) * R for k in KS]; C = [F["cells"][f"{k:g}"] for k in KS]
            for row, (key, scale, floor) in enumerate((("risk", 100.0, 0.3), ("fuel", 1.0, None))):
                med = [scale * np.median(c[key]) for c in C]
                axes[row, j].plot(xs, med if floor is None else np.maximum(med, floor), color=PAL[i], marker="os^"[i % 3], ms=3, lw=0.9, label=f"revenue ${R:g}$" if row == 0 else None)
                for xi, c in zip(xs, C):
                    v = scale * np.array(c[key]); axes[row, j].plot([xi] * len(v), v if floor is None else np.maximum(v, floor), ls="", marker=".", ms=2.5, color=PAL[i], alpha=0.45)
        axes[0, j].set_xscale("log"); axes[0, j].set_yscale("log"); axes[0, j].set_ylim(0.3, 130); axes[0, j].set_yticks([1, 3, 10, 30, 100]); axes[0, j].set_yticklabels(["1", "3", "10", "30", "100"])
        axes[0, j].set_title(f"tank ${tank:g}$", fontsize=FS - 1, pad=6); axes[1, j].axhline(tank, color=MUTED, lw=0.6, ls=":")
        axes[1, j].set_ylim(13, 25)
        axes[1, j].set_xticks([30, 100, 300]); axes[1, j].set_xticklabels(["30", "100", "300"]); axes[0, j].minorticks_off(); axes[1, j].minorticks_off()
        panel_label(axes[0, j], "abc"[j]); panel_label(axes[1, j], "def"[j])
    axes[1, len(tanks) // 2].set_xlabel("amount at stake in a failure, $(1+k)R$ (fuel units)")
    axes[0, 0].set_ylabel("failed (%)"); axes[1, 0].set_ylabel("fuel used")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=FS - 2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.6)
    save(fig, "lunar-formulations-pilot-stake", ROOT)
