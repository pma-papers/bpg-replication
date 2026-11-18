"""Where the fuel of the lander goes: before and after the first touchdown.

For every saved policy of a stage, episodes are flown with the stochastic policy and the fuel used is split at the frame
in which a leg first touches the ground. Reported per policy: the failure share, the fuel before and after the first
touchdown, the frames before and after it, and the share of episodes that end with the tank empty (after touchdown an
empty tank is no failure and the engines are off).

Usage:  python lunar_touchdown_split.py STAGE [PREFIX [N]]      (default PREFIX ledger-R36-b24, N = 500 episodes; runtime ~/.venvs/constrained-rl-numerics)
Output: lunar-touchdown-split-STAGE.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import lunar  # noqa: E402

STAGE = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "ledger-R36-b24"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 500


def split(model, arm, n, seed, n_envs=16):
    cf = dict(crash_fails=bool(arm.get("crash_fails", False)), tank_until_touchdown=bool(arm.get("tank_until_touchdown", False)), fuel_price=float(arm.get("fuel_price", 1.0)))
    if arm.get("revenue") is not None:
        cf.update(revenue=float(arm["revenue"]), progress=arm.get("progress", "aid"), aid_scale=float(arm.get("aid_scale", 1.0 / 12.0)))
    envs = DummyVecEnv([lambda: lunar.FuelLander(arm["budget"], "failure", 0.0, arm.get("beta", 0.999), **cf) for _ in range(n_envs)])
    envs.seed(seed); torch.manual_seed(seed)
    obs = envs.reset()
    recs, acc = [], [dict(t=0, t_touch=None, f_touch=None) for _ in range(n_envs)]
    while len(recs) < n:
        act, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = envs.step(act)
        for i in range(n_envs):
            a = acc[i]; a["t"] += 1
            if a["t_touch"] is None and infos[i]["touched"]:
                a["t_touch"], a["f_touch"] = a["t"], infos[i]["used"]
            if dones[i]:
                recs.append(dict(a, used=infos[i]["used"], failed=infos[i]["failure"], completed=infos[i]["completed"], empty=infos[i]["empty"]))
                acc[i] = dict(t=0, t_touch=None, f_touch=None)
    envs.close()
    recs = recs[:n]
    T = [r for r in recs if r["t_touch"] is not None]
    mean = lambda v: float(np.mean(v)) if len(v) else float("nan")
    return dict(n=len(recs), failed=mean([r["failed"] for r in recs]), completed=mean([r["completed"] for r in recs]), touched=len(T) / len(recs),
                fuel=mean([r["used"] for r in recs]), fuel_before_touchdown=mean([r["f_touch"] for r in T]), fuel_after_touchdown=mean([r["used"] - r["f_touch"] for r in T]),
                frames_before_touchdown=mean([r["t_touch"] for r in T]), frames_after_touchdown=mean([r["t"] - r["t_touch"] for r in T]),
                ends_with_tank_empty=mean([r["empty"] for r in recs]))


out = {}
for f in sorted((ROOT / "lunar" / STAGE).glob(f"{PREFIX}-k*__seed*/result.json")):
    r = json.loads(f.read_text())
    d = split(PPO.load(f.parent / "final_model.zip", device="cpu"), r["arm"], N, 30_000 + r["seed"])
    out[f.parent.name] = dict(worth=r["arm"]["vehicle_worth_in_landings"], seed=r["seed"], **d)
    print(f"{f.parent.name:34s} failed {100 * d['failed']:5.1f} %  fuel {d['fuel']:5.2f} = {d['fuel_before_touchdown']:5.2f} before touchdown + {d['fuel_after_touchdown']:5.2f} after"
          f"   frames {d['frames_before_touchdown']:4.0f} + {d['frames_after_touchdown']:4.0f}   ends with the tank empty {100 * d['ends_with_tank_empty']:5.1f} %", flush=True)
print("\nmeans over the seeds, by worth of the vehicle")
for k in sorted({v["worth"] for v in out.values()}):
    c = [v for v in out.values() if v["worth"] == k]
    print(f"  k={k:5g} n={len(c):2d}  fuel before touchdown {np.mean([v['fuel_before_touchdown'] for v in c]):5.2f}  after {np.mean([v['fuel_after_touchdown'] for v in c]):5.2f}"
          f"  frames before {np.mean([v['frames_before_touchdown'] for v in c]):4.0f}  policies that burn the tank out on the ground (over half of episodes) {sum(v['ends_with_tank_empty'] > 0.5 for v in c)}")
(ROOT / f"lunar-touchdown-split-{STAGE}.json").write_text(json.dumps(out, indent=1) + "\n")
