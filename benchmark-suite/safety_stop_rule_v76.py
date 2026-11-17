"""Benchmark-suite study, design of November 2026: a policy at hand for the check that precedes every result.

The check compares each trained policy, at its own price, with policies at hand. This script evaluates one more
such policy, made by hand from a stored controller and not trained: the controller acts as stored until the
remaining budget falls to a given share of the budget; from then on it still acts as stored while the robot is
inside a hazard (so that it leaves it), and from the first step outside a hazard the action is zero for the rest
of the episode. The rule uses the remaining budget, which the policy observes, and whether the last step added
cost, which the hazard sensors show. Share 0 is the stored controller itself.

No training. Episodes of the stopped task without a charge, the same number in each of eight environments and
the same environment seeds for every share. The value at a price m is U - m/(1-beta) q, as in the runner.

Usage:
    safety_stop_rule_v76.py --stage pilot-no-price --config no-price-constant --seed 95001 --steps 10000000
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from safety_study_v76 import BETA, BUDGET, StudyGoal, summarize  # noqa: E402

torch.set_num_threads(1)
SHARES = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
PRICES = [0.0, 0.04, 0.08, 0.16]


def episodes(model, norm, share, n, key, n_envs=8, mean_action=False):
    envs = [StudyGoal(BUDGET, "failure", 0.0, BETA) for _ in range(n_envs)]
    per = max(n // n_envs, 1)
    seeds = [int(s) for s in np.random.SeedSequence(key).generate_state(n_envs + 1)]
    torch.manual_seed(seeds[-1] % (2 ** 31))
    obs = np.stack([e.reset(seed=s % (2 ** 31))[0] for e, s in zip(envs, seeds)])
    done = [0] * n_envs; eps = []
    ret = np.zeros(n_envs); util = np.zeros(n_envs); cost = np.zeros(n_envs); t = np.zeros(n_envs, dtype=int)
    stopped = np.zeros(n_envs, dtype=bool); last_cost = np.zeros(n_envs); stop_step = [None] * n_envs
    while min(done) < per:
        act, _ = model.policy.predict(norm.normalize_obs(obs), deterministic=mean_action)
        for i, e in enumerate(envs):
            if done[i] >= per:
                continue
            if not stopped[i] and share > 0 and e.reserve <= share * BUDGET and last_cost[i] == 0.0:
                stopped[i] = True; stop_step[i] = int(t[i])
            a = np.zeros_like(act[i]) if stopped[i] else act[i]
            o, _, terminated, truncated, info = e.step(a)
            ret[i] += info["utility"]; util[i] += BETA ** t[i] * info["utility"]; cost[i] += info["cost"]; last_cost[i] = info["cost"]
            if terminated or truncated:
                failed = bool(info["episode_failed"])
                eps.append(dict(failed=failed, tau=int(t[i]) if failed else None, ret=float(ret[i]), U=float(util[i]), cost=float(cost[i]),
                                goals=int(info["episode_goals"]), path=float(info["episode_path"]), steps=int(info["episode_steps"]),
                                stop_step=stop_step[i]))
                done[i] += 1; ret[i] = util[i] = cost[i] = last_cost[i] = 0.0; t[i] = 0; stopped[i] = False; stop_step[i] = None
                o, _ = e.reset()
            else:
                t[i] += 1
            obs[i] = o
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--mean-action", action="store_true", help="the mean action in place of a sampled one")
    ap.add_argument("--shares", type=float, nargs="+", default=SHARES)
    a = ap.parse_args()
    folder = HERE / "safety" / a.stage / f"{a.config}__seed{a.seed}"
    tag = "" if a.steps is None else f"-{a.steps // 1_000_000}M"
    sub = folder if a.steps is None else folder / "copies"
    model = PPO.load(sub / f"model{tag}.zip", device="cpu")
    with open(sub / f"vecnormalize{tag}.pkl", "rb") as f:
        norm = pickle.load(f)
    out = dict(source=dict(stage=a.stage, config=a.config, seed=a.seed, steps=a.steps), n=a.n, mean_action=a.mean_action, shares={})
    for share in a.shares:
        eps = episodes(model, norm, share, a.n, [a.seed, 13], mean_action=a.mean_action)
        rows = {str(m): summarize(eps, m) for m in PRICES}
        stops = [e["stop_step"] for e in eps if e["stop_step"] is not None]
        out["shares"][str(share)] = dict(by_price=rows, share_stopped=len(stops) / len(eps), mean_stop_step=float(np.mean(stops)) if stops else None, episodes=eps)
        s = rows["0.0"]
        print("share %.1f: exhausted %.3f U %.2f (%.2f) goals %.2f stopped %.2f | value at prices " % (share, s["exhausted"], s["U"], s["U_se"], s["mean_goals"], len(stops) / len(eps))
              + "  ".join("%s: %.2f (%.2f)" % (m, rows[str(m)]["objective"], rows[str(m)]["objective_se"]) for m in PRICES), flush=True)
    dest = HERE / "safety" / "stop-rule" / f"{a.stage}__{a.config}__seed{a.seed}{tag}{'-mean-action' if a.mean_action else ''}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
