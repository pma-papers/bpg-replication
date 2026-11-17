"""The parts of the utility in the lander study (2026-11-17).

The utility of a frame is the task's reward with fuel charged at the value of the stage. The task's reward is the
change of a shaping potential (distance to the pad, speed, tilt, leg contact), 100 when the lander comes to rest, and
the fuel charge. The failing frame earns no utility, so the task's penalty of 100 for a crash is not part of it.
For the final policy of every run of the named configurations this script evaluates the same episodes as the result
of the run and splits the stopped utility of each episode into the three parts, discounted as the utility is:

    approach   the reward of the task without its fuel charge and without the 100 for coming to rest
    landing    the 100 for coming to rest
    fuel       the fuel used times its value (entered with a minus sign)

Usage:  python lunar_utility_parts.py STAGE ARM [ARM ...]   (runtime: the constrained-rl-numerics environment)
        python lunar_utility_parts.py --table STAGE         (the table again from the stored json, no evaluation)
Output: lunar-utility-parts-STAGE.json and a printed table (means over the episodes of each seed, then over seeds;
separately for the episodes that landed and those that failed).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import lunar  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

TABLE_ONLY = "--table" in sys.argv
argv = [a for a in sys.argv[1:] if a != "--table"]
stage, arms = argv[0], argv[1:]
out = json.loads((ROOT / f"lunar-utility-parts-{stage}.json").read_text()) if TABLE_ONLY else {}
for arm in ([] if TABLE_ONLY else arms):
    rows = []
    for d in sorted((ROOT / "lunar" / stage).glob(f"{arm}__seed*")):
        r = json.loads((d / "result.json").read_text()); a = r["arm"]; s = r["stochastic"]
        beta, price = 0.999, float(a.get("fuel_price", 1.0))
        model = PPO.load(d / "final_model.zip", device="cpu")
        envs = DummyVecEnv([lambda: lunar.FuelLander(a["budget"], "failure", 0.0, beta, crash_fails=a.get("crash_fails", False),
                                                     tank_until_touchdown=a.get("tank_until_touchdown", False), fuel_price=price) for _ in range(32)])
        envs.seed(s["seed"]); torch.manual_seed(s["seed"])
        obs = envs.reset(); done = 0; rec = []
        acc = [dict(U=0.0, approach=0.0, landing=0.0, fuel=0.0, units=0.0, t=0) for _ in range(32)]
        while done < s["n"]:
            act, _ = model.predict(obs, deterministic=False)
            obs, rew, dones, infos = envs.step(act)
            for i in range(32):
                c, info = acc[i], infos[i]; w = beta ** c["t"]
                if not info["failure"]:                               # the failing frame earns no utility
                    bonus = 100.0 if info["landed"] else 0.0
                    c["U"] += w * info["utility"]; c["landing"] += w * bonus; c["fuel"] -= w * price * info["fuel"]
                    c["approach"] += w * (info["utility"] + price * info["fuel"] - bonus)
                c["units"] += info["fuel"]; c["t"] += 1
                if dones[i]:
                    c["failed"] = bool(info["failure"]); c["landed"] = bool(info["landed"])
                    if done < s["n"]:
                        rec.append(c)
                    done += 1
                    acc[i] = dict(U=0.0, approach=0.0, landing=0.0, fuel=0.0, units=0.0, t=0)
        envs.close()
        U = np.mean([c["U"] for c in rec]); risk = np.mean([c["failed"] for c in rec])
        assert abs(U - s["U"]) < 1e-6 and abs(risk - s["risk"]) < 1e-9, (arm, r["seed"], U, s["U"])      # the same episodes as the result of the run

        def part(sel):
            S = [c for c in rec if sel(c)]
            return dict(n=len(S), **{k: (float(np.mean([c[k] for c in S])) if S else None) for k in ("U", "approach", "landing", "fuel", "units", "t")})
        rows.append(dict(seed=r["seed"], risk=float(risk), all=part(lambda c: True), landed=part(lambda c: c["landed"]), failed=part(lambda c: c["failed"])))
        print(f"{arm} seed {r['seed']}: failed {100 * risk:5.1f}%  all {rows[-1]['all']}", flush=True)
    out[arm] = rows
if not TABLE_ONLY:
    (ROOT / f"lunar-utility-parts-{stage}.json").write_text(json.dumps(out, indent=1) + "\n")
print(f"\n{'configuration':18s} {'episodes':9s} {'share':>6s} {'approach':>9s} {'landing':>8s} {'fuel':>8s} {'utility':>8s} {'units':>6s} {'frames':>7s}")
for arm, rows in out.items():
    for kind in ("all", "landed", "failed"):
        R = [x[kind] for x in rows if x[kind]["n"] > 0]
        if not R:
            continue
        w = np.array([x["n"] for x in R], float)
        m = {k: float(np.average([x[k] for x in R], weights=w)) for k in ("U", "approach", "landing", "fuel", "units", "t")}
        share = w.sum() / sum(x["all"]["n"] for x in rows)
        print(f"{arm:18s} {kind:9s} {100 * share:5.1f}% {m['approach']:9.1f} {m['landing']:8.1f} {m['fuel']:8.1f} {m['U']:8.1f} {m['units']:6.1f} {m['t']:7.0f}")

# table for the Online Resource: one row per configuration and kind of episode
names = {"all": "all", "landed": "landed", "failed": "failed"}
lines = []
for arm, rows in out.items():
    m = json.loads(next((ROOT / "lunar" / stage).glob(f"{arm}__seed*/result.json")).read_text())["arm"]["m"]
    first = True
    for kind in ("landed", "failed", "all"):
        R = [x[kind] for x in rows if x[kind]["n"] > 0]
        if not R:
            continue
        w = np.array([x["n"] for x in R], float); share = w.sum() / sum(x["all"]["n"] for x in rows)
        v = {k: float(np.average([x[k] for x in R], weights=w)) for k in ("U", "approach", "landing", "fuel", "units")}
        lines.append((("no price" if m == 0 else f"${m:g}$") if first else "") + f" & {names[kind]} & ${100 * share:.0f}$ & ${v['approach']:.0f}$ & ${v['landing']:.0f}$ & ${v['fuel']:.0f}$ & ${v['U']:.0f}$ & ${v['units']:.1f}$ \\\\")
        first = False
    lines.append("\\addlinespace")
(ROOT / f"lunar-utility-parts-{stage}-table.tex").write_text(
    "\\begin{tabular}{llrrrrrr}\n\\toprule\nPrice $m$ & Episodes & Share (\\%) & Approach & Landing & Fuel cost & Utility & Fuel used \\\\\n\\midrule\n" + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
