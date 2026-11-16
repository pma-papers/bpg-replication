"""Flight paths of stored fuel-tank policies, for the figure of the fuel-tank problem (2026-11-14, revision 76).

For each configuration named on the command line and each of its five seeds, the stored final policy is evaluated
again with the evaluation seed and the number of episodes of its result.json, exactly as in lunar_setup_eval.py.
For the first episode of each of the 32 parallel environments the position of the lander (the first two
coordinates of the observation: the pad is at height zero between -0.2 and 0.2), its vertical velocity and its
angle (coordinates four and five of the observation) are recorded frame by frame, with how the episode ended.
The run stops if the exhaustion probability of the new evaluation differs from the stored one, so the paths belong
to the same thousand episodes as the results of the study.

Usage:  python lunar_problem_eval_v76.py [--stage STAGE] lunar-unlimited lunar-b20-m0 lunar-b20-m0.5 lunar-b20-m2
        (runtime: the constrained-rl-numerics environment; about ten seconds per policy)
Output: lunar-problem-paths-v76.json for the stage campaign, lunar-paths-STAGE.json for another stage (one entry per
configuration and seed; existing entries are kept). For another stage the task has the options of the run (the
failure rule and the value of fuel), and an episode ends as landed, lost (the lander crashed or left the frame),
empty (the tank ran out before touchdown) or other.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import lunar  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

BETA, N_ENVS = 0.999, 32


def evaluate(model, budget, n, seed, **options):
    revised = bool(options.get("crash_fails"))
    envs = DummyVecEnv([lambda: lunar.FuelLander(budget, "failure", 0.0, BETA, **options) for _ in range(N_ENVS)])
    envs.seed(seed)
    torch.manual_seed(seed)
    obs = envs.reset()
    done, ends = 0, []
    state = lambda o: tuple(float(o[j]) for j in (0, 1, 3, 4))
    track = [[state(obs[i])] for i in range(N_ENVS)]
    paths = [None] * N_ENVS
    while done < n:
        act, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = envs.step(act)
        for i in range(N_ENVS):
            last = infos[i]["terminal_observation"] if dones[i] else obs[i]
            if paths[i] is None:
                track[i].append(state(last))
            if dones[i]:
                if revised:
                    end = "empty" if infos[i]["exhausted"] else ("lost" if infos[i]["lost"] else ("landed" if infos[i]["landed"] else "other"))
                else:
                    end = "exhausted" if infos[i]["failure"] else ("landed" if infos[i]["landed"] else ("crashed" if infos[i]["crashed"] else "other"))
                if done < n:
                    ends.append(end)
                if paths[i] is None:
                    cols = list(zip(*track[i]))
                    paths[i] = dict(end=end, used=round(float(infos[i]["used"]), 4), x=[round(v, 4) for v in cols[0]], y=[round(v, 4) for v in cols[1]],
                                    vy=[round(v, 4) for v in cols[2]], angle=[round(v, 4) for v in cols[3]])
                done += 1
    envs.close()
    return ends, paths


def main():
    args = sys.argv[1:]
    stage = "campaign"
    if args and args[0] == "--stage":
        stage, args = args[1], args[2:]
    OUT = ROOT / ("lunar-problem-paths-v76.json" if stage == "campaign" else f"lunar-paths-{stage}.json")
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for arm in args:
        for d in sorted((ROOT / "lunar" / stage).glob(f"{arm}__seed*")):
            if d.name in out:
                continue
            r = json.loads((d / "result.json").read_text())
            st = r["stochastic"]
            a = r["arm"]
            options = {k: a[k] for k in ("crash_fails", "tank_until_touchdown", "fuel_price") if k in a}
            loader = SAC if (d / "final_model.zip").exists() and "sac" in stage else PPO
            model = d / "final_model.zip" if (d / "final_model.zip").exists() else ROOT / r["model"]
            ends, paths = evaluate(loader.load(model, device="cpu"), a["budget"], st["n"], st["seed"], **options)
            if options.get("crash_fails"):
                freq = {e: float(np.mean([x == e for x in ends])) for e in ("landed", "lost", "empty", "other")}
                same = abs(freq["empty"] - st["exhausted"]) < 1e-12 and abs(freq["lost"] - st["lost"]) < 1e-12 and abs(freq["landed"] - st["landed"]) < 1e-12
            else:
                freq = {e: float(np.mean([x == e for x in ends])) for e in ("landed", "crashed", "exhausted", "other")}
                same = abs(freq["exhausted"] - st["risk"]) < 1e-12 and abs(freq["landed"] - st["landed"]) < 1e-12 and abs(freq["crashed"] - st["crashed"]) < 1e-12
            if not same:
                raise SystemExit(f"{d.name}: the new evaluation gives {freq}, the record differs")
            out[d.name] = dict(arm=arm, seed=r["seed"], budget=r["arm"]["budget"], m=r["arm"]["m"], n=len(ends), frequencies=freq, paths=paths,
                               U=st["U"], fuel_median=st["fuel_quantiles"][2])
            OUT.write_text(json.dumps(out) + "\n")
            print(f"{d.name}: {freq}", flush=True)


if __name__ == "__main__":
    main()
