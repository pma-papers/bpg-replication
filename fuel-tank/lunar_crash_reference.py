"""The controllers trained without a fuel limit, evaluated under the failure definition of the stage crash-campaign
(2026-11-14): an empty tank before touchdown or the loss of the lander. The controllers are those of the stage campaign, trained on
the native task, and are not trained again. For each seed the two evaluations of lunar.py (the stochastic policy and
its mean action, with the evaluation seeds of a run) are written to lunar/crash-campaign/lunar-unlimited__seed*/result.json,
so that the analysis reads them like the runs of the stage.

Usage:  python lunar_crash_reference.py        (runtime: the constrained-rl-numerics environment)
"""
import json
from pathlib import Path

from stable_baselines3 import PPO

import lunar

ROOT = Path(__file__).resolve().parent
protocol = json.loads((ROOT / "lunar-protocol.json").read_text())
stage = protocol["stages"]["crash-campaign"]
arm = {**protocol["arms"]["lunar-unlimited"], **stage["arm_overrides"]}
options = dict(crash_fails=arm["crash_fails"], tank_until_touchdown=arm["tank_until_touchdown"])
for seed in stage["seeds"]:
    src = ROOT / "lunar" / stage["warm_stage"] / f"lunar-unlimited__seed{seed}"
    out = ROOT / "lunar" / "crash-campaign" / f"lunar-unlimited__seed{seed}"
    if (out / "result.json").exists():
        continue
    old = json.loads((src / "result.json").read_text())
    model = PPO.load(src / "final_model.zip", device="cpu")
    r = dict(arm=arm, seed=seed, steps=old["steps"], trace=[], final_multiplier=0.0, model=str((src / "final_model.zip").relative_to(ROOT)),
             note="evaluation of the controller of the stage campaign; not trained in this stage",
             stochastic=lunar.evaluate(model, arm["budget"], 0.999, n=stage["eval_n"], seed=10_000 + seed, **options),
             mean_action=lunar.evaluate(model, arm["budget"], 0.999, n=stage["eval_n"], seed=20_000 + seed, deterministic=True, **options))
    assert r["stochastic"]["seed"] == old["stochastic"]["seed"] and abs(r["stochastic"]["landed"] - old["stochastic"]["landed"]) < 1e-12   # the same thousand episodes
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(r, indent=1) + "\n")
    s = r["stochastic"]
    print(f"{out.name}: failure {s['risk']:.3f} (exhausted {s['exhausted']:.3f}, lost {s['lost']:.3f})  landed {s['landed']:.3f}  U {s['U']:.1f}", flush=True)
