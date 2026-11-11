"""Episodes of the benchmark task for the figure of its problem (added on 2026-11-11, revision 76).

For the stored policies of one seed without a price and at one price, a few episodes are run from fixed reset
seeds, so that both policies meet the same layouts. Recorded per episode: the layout (hazards, vase), the path
of the robot, the goals it was given, the cumulative cost at each step and how the episode ended.

Usage:  python safety_problem_eval_v76.py        (runtime: the constrained-rl-safety310 environment)
Output: safety-problem-v76.json
"""
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import safety_pilot_v63 as sp

HERE = Path(__file__).resolve().parent
SEED, RESETS, ARMS = 95001, (1, 2, 3, 4, 5, 6), ("fixed_m0_b25", "fixed_m0.03_b25")
out = {}
for arm in ARMS:
    d = HERE / "safety" / "campaign" / f"{arm}__seed{SEED}"
    r = json.loads((d / "result.json").read_text())
    venv = DummyVecEnv([lambda: sp.BudgetGoal(r["arm"]["budget"], "failure", 0.0, r["arm"]["beta"])])
    venv = VecNormalize.load(str(d / "vecnormalize.pkl"), venv)
    venv.training = False; venv.norm_reward = False
    model = PPO.load(d / "final_model.zip", device="cpu")
    env = venv.venv.envs[0]
    eps = []
    for rs in RESETS:
        venv.seed(rs); torch.manual_seed(rs)
        obs = venv.reset()
        task = env.inner.unwrapped.task
        layout = dict(hazards=np.array(task.hazards.pos)[:, :2].round(4).tolist(), hazard_radius=float(task.hazards.size),
                      vases=np.array(task.vases.pos)[:, :2].round(4).tolist(), goal_radius=float(task.goal.size))
        path, cost, goals = [np.array(task.agent.pos)[:2].round(4).tolist()], [0.0], [(0, *np.array(task.goal.pos)[:2].round(4).tolist())]
        ret, end = 0.0, "horizon"
        for t in range(sp.H):
            act, _ = model.predict(obs, deterministic=False)
            pos, goal = np.array(task.agent.pos)[:2], np.array(task.goal.pos)[:2]
            obs, _, dones, infos = venv.step(act)
            info = infos[0]; ret += info["utility"]; cost.append(info["used"])
            if dones[0]:
                end = "exhausted" if info["failure"] else "horizon"
                break
            path.append(np.array(task.agent.pos)[:2].round(4).tolist())
            g = np.array(task.goal.pos)[:2]
            if np.abs(g - goal).max() > 1e-6:
                goals.append((t + 1, *g.round(4).tolist()))
        eps.append(dict(reset_seed=rs, layout=layout, path=path, cost=cost, goals=goals, end=end, steps=len(cost) - 1, utility=ret))
        print(arm, rs, end, len(cost) - 1, "cost", cost[-1], "goals", len(goals) - 1, "return %.1f" % ret, flush=True)
    out[arm] = dict(m=r["arm"]["m"], budget=r["arm"]["budget"], stored=r["stochastic"], episodes=eps)
    venv.close()
(HERE / "safety-problem-v76.json").write_text(json.dumps(out) + "\n")
