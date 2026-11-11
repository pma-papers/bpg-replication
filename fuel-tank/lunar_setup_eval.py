"""Fuel use of stored fuel-tank policies, for the setup figure of the study (added on 2026-11-11, revision 76).

For each configuration named on the command line and each of its five seeds, the stored final policy is evaluated
again with the evaluation seed and the number of episodes of its result.json. Recorded per episode: the fuel used,
the number of frames and how the episode ended. For the first episode of the first ten environments the fuel used
and the position of the lander (the first two coordinates of the observation: the pad is at height zero between
-0.2 and 0.2) are recorded frame by frame. The run stops if the exhaustion probability of the new evaluation differs from the
stored one, so the record belongs to the same thousand episodes.

Usage:  python lunar_setup_eval.py lunar-unlimited lunar-b20-m0 lunar-b20-m0.5
        (runtime: the constrained-rl-numerics environment; about ten seconds per policy)
Output: lunar-setup-v76.json (one entry per configuration and seed; existing entries are kept).
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
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

BETA, N_ENVS, N_PATHS = 0.999, 32, 10
OUT = ROOT / "lunar-setup-v76.json"


def evaluate(model, budget, n, seed):
    envs = DummyVecEnv([lambda: lunar.FuelLander(budget, "failure", 0.0, BETA) for _ in range(N_ENVS)])
    envs.seed(seed)
    torch.manual_seed(seed)
    obs = envs.reset()
    done, episodes = 0, []
    used = [[] for _ in range(N_ENVS)]
    xy = [[(float(obs[i][0]), float(obs[i][1]))] for i in range(N_ENVS)]
    paths = [None] * N_PATHS
    while done < n:
        act, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = envs.step(act)
        for i in range(N_ENVS):
            used[i].append(float(infos[i]["used"]))
            last = infos[i]["terminal_observation"] if dones[i] else obs[i]
            xy[i].append((float(last[0]), float(last[1])))
            if dones[i]:
                end = "exhausted" if infos[i]["failure"] else ("landed" if infos[i]["landed"] else ("crashed" if infos[i]["crashed"] else "other"))
                if done < n:
                    episodes.append(dict(used=used[i][-1], frames=len(used[i]), end=end))
                if i < N_PATHS and paths[i] is None:
                    paths[i] = dict(used=[round(x, 4) for x in used[i]], end=end, x=[round(p[0], 4) for p in xy[i]], y=[round(p[1], 4) for p in xy[i]])
                done += 1
                used[i] = []
                xy[i] = [(float(obs[i][0]), float(obs[i][1]))]
    envs.close()
    return episodes, paths


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for arm in sys.argv[1:]:
        for d in sorted((ROOT / "lunar" / "campaign").glob(f"{arm}__seed*")):
            if d.name in out:
                continue
            r = json.loads((d / "result.json").read_text())
            st = r["stochastic"]
            episodes, paths = evaluate(PPO.load(d / "final_model.zip", device="cpu"), r["arm"]["budget"], st["n"], st["seed"])
            risk = float(np.mean([e["end"] == "exhausted" for e in episodes]))
            if abs(risk - st["risk"]) > 1e-12:
                raise SystemExit(f"{d.name}: the new evaluation gives the exhaustion probability {risk}, the record has {st['risk']}")
            out[d.name] = dict(arm=arm, seed=r["seed"], budget=r["arm"]["budget"], m=r["arm"]["m"], n=len(episodes), risk=risk,
                               used=[round(e["used"], 4) for e in episodes], frames=[e["frames"] for e in episodes],
                               end=[e["end"] for e in episodes], paths=paths)
            OUT.write_text(json.dumps(out) + "\n")
            print(f"{d.name}: exhausted {risk:.3f}  median fuel {np.median([e['used'] for e in episodes]):.1f}", flush=True)


if __name__ == "__main__":
    main()
