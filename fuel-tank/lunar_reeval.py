"""Priced risk of the stored fuel-tank policies (added on 2026-11-10).

A run's result.json stores the exhaustion probability and the stopped utility of the final stochastic policy but
not the frames at which the tank ran out, and the price acts on the priced risk
    q_{beta,H} = E[ beta^tau (1 - beta^(H - tau)) 1{tau < H} ],
which needs them. This script evaluates every stored fixed-price policy again, with the evaluation seed and the
number of episodes of its result.json, and writes priced-risk.json next to it: the exhaustion frames, the priced
risk, and the exhaustion probability and stopped utility of the new evaluation. The run stops if these two do not
equal the stored values, so the priced risk belongs to the same thousand episodes as the stored record.

Usage:  python lunar_reeval.py [--shard i/n] [--force]        (runtime: the constrained-rl-numerics environment)
About six seconds per policy. Shards split the runs for parallel processes, for example
    for i in 0 1 2 3 4; do python lunar_reeval.py --shard $i/5 & done; wait
"""
from __future__ import annotations

import argparse
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

BETA, H, N_ENVS = 0.999, lunar.H, 32


def evaluate(model, budget, n, seed):
    """The episodes of lunar.py's evaluation of a stochastic policy, with the frame at which each one ended."""
    envs = DummyVecEnv([lambda: lunar.FuelLander(budget, "failure", 0.0, BETA) for _ in range(N_ENVS)])
    envs.seed(seed)
    torch.manual_seed(seed)
    obs = envs.reset()
    done, rec = 0, []
    acc = [dict(U=0.0, t=0) for _ in range(N_ENVS)]
    while done < n:
        act, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = envs.step(act)
        for i in range(N_ENVS):
            a = acc[i]
            a["U"] += BETA ** a["t"] * infos[i]["utility"]
            a["t"] += 1
            if dones[i]:
                if done < n:
                    rec.append((a["U"], a["t"], bool(infos[i]["failure"]), bool(infos[i]["landed"]), bool(infos[i]["crashed"])))
                done += 1
                acc[i] = dict(U=0.0, t=0)
    envs.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    i, n = (int(x) for x in args.shard.split("/"))
    folders = sorted(p.parent for p in (ROOT / "lunar" / "campaign").glob("*/result.json"))
    fixed = [d for d in folders if json.loads((d / "result.json").read_text())["arm"]["kind"] == "fixed"]
    for d in fixed[i::n]:
        out = d / "priced-risk.json"
        if out.exists() and not args.force:
            continue
        r = json.loads((d / "result.json").read_text())
        st = r["stochastic"]
        rec = evaluate(PPO.load(d / "final_model.zip", device="cpu"), r["arm"]["budget"], st["n"], st["seed"])
        U = np.array([x[0] for x in rec])
        failed = np.array([x[2] for x in rec])
        frames = np.array([x[1] - 1 for x in rec])[failed]          # index of the exhausting frame
        risk, Umean = float(failed.mean()), float(U.mean())
        if abs(risk - st["risk"]) > 1e-12 or abs(Umean - st["U"]) > 1e-6:
            raise SystemExit(f"{d.name}: the new evaluation gives risk {risk} and U {Umean}, the record has {st['risk']} and {st['U']}")
        q = float((BETA ** frames * (1 - BETA ** (H - frames))).sum() / len(rec))
        out.write_text(json.dumps(dict(run=d.name, n=len(rec), evaluation_seed=st["seed"], beta=BETA, H=H, risk=risk, U=Umean,
                                       priced_risk=q, exhaustion_frames=[int(t) for t in frames],
                                       landed=float(np.mean([x[3] for x in rec])), crashed=float(np.mean([x[4] for x in rec]))), indent=1) + "\n")
        print(f"{d.name}: risk {risk:.3f}  U {Umean:.2f}  priced risk {q:.4f}", flush=True)


if __name__ == "__main__":
    main()
