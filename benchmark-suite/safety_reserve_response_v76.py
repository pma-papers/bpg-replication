"""Benchmark-suite study, design of November 2026: does a stored learner respond to the remaining budget?

A diagnostic without training. The stored policy is run on fresh episodes of the stopped task (no charge), and
the states it visits are collected. On these states the remaining budget, which is the last component of the
observation, is replaced by a low and by a high value, and two differences are recorded:

  actor   the change of the mean action between the two inputs, in units of the action standard deviation
          (root mean square over the states and the two action components);
  critic  the change of the value estimate between the two inputs, in units of the reward (the estimate times
          the standard deviation the learner's reward normalization ended with).

Also the value estimate on the visited states by the true remaining budget. A learner that has taken up the
charge shows a value that falls as the budget runs out; a policy that protects the budget shows an action that
depends on it.

Usage:
    safety_reserve_response_v76.py --stage pilot-price --config price-0.16 --seed 95001 [--steps 10000000]
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
from safety_study_v76 import BETA, BUDGET, StudyGoal  # noqa: E402

torch.set_num_threads(1)
LOW, HIGH = 0.1, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n_steps", type=int, default=40000)
    a = ap.parse_args()
    folder = HERE / "safety" / a.stage / f"{a.config}__seed{a.seed}"
    tag = "" if a.steps is None else f"-{a.steps // 1_000_000}M"
    sub = folder if a.steps is None else folder / "copies"
    model = PPO.load(sub / f"model{tag}.zip", device="cpu")
    with open(sub / f"vecnormalize{tag}.pkl", "rb") as f:
        norm = pickle.load(f)
    scale = float(np.sqrt(norm.ret_rms.var + norm.epsilon))
    envs = [StudyGoal(BUDGET, "failure", 0.0, BETA) for _ in range(8)]
    seeds = [int(s) for s in np.random.SeedSequence([a.seed, 17]).generate_state(9)]
    torch.manual_seed(seeds[-1] % (2 ** 31))
    obs = np.stack([e.reset(seed=s % (2 ** 31))[0] for e, s in zip(envs, seeds)])
    states = []
    while len(states) * 8 < a.n_steps:
        states.append(obs.copy())
        act, _ = model.policy.predict(norm.normalize_obs(obs), deterministic=False)
        for i, e in enumerate(envs):
            o, _, terminated, truncated, _ = e.step(act[i])
            if terminated or truncated:
                o, _ = e.reset()
            obs[i] = o
    S = np.concatenate(states)

    def heads(raw):
        with torch.no_grad():
            t = torch.as_tensor(norm.normalize_obs(raw), dtype=torch.float32)
            dist = model.policy.get_distribution(t)
            return dist.distribution.mean.numpy(), dist.distribution.stddev.numpy(), model.policy.predict_values(t).numpy()[:, 0] * scale

    lo, hi = S.copy(), S.copy()
    lo[:, -1], hi[:, -1] = LOW, HIGH
    mu, sd, v = heads(S); mu_lo, _, v_lo = heads(lo); mu_hi, _, v_hi = heads(hi)
    actor = float(np.sqrt(np.mean(((mu_lo - mu_hi) / sd) ** 2)))
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    by = {f"{b[0]:.1f}-{min(b[1], 1.0):.1f}": (float(v[(S[:, -1] >= b[0]) & (S[:, -1] < b[1])].mean()), int(((S[:, -1] >= b[0]) & (S[:, -1] < b[1])).sum())) for b in bins}
    out = dict(source=dict(stage=a.stage, config=a.config, seed=a.seed, steps=a.steps), states=int(len(S)), low=LOW, high=HIGH, reward_scale=scale,
               action_std=[float(x) for x in sd.mean(axis=0)], actor_change_in_std=actor,
               critic_change=float(np.mean(v_lo - v_hi)), value_by_remaining_budget=by)
    dest = HERE / "safety" / "reserve-response" / f"{a.stage}__{a.config}__seed{a.seed}{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    print("%s %s seed %d%s: actor change %.2f std | critic change (low minus high budget) %.1f | value by remaining budget %s | reward scale %.2f"
          % (a.stage, a.config, a.seed, tag, actor, out["critic_change"], "  ".join("%s: %.1f (n=%d)" % (k, x[0], x[1]) for k, x in by.items()), scale), flush=True)


if __name__ == "__main__":
    main()
