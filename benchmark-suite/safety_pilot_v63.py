"""Pilot for a benchmark-suite study (revision 63 candidate): the paper's class on a Safety-Gymnasium task.

Task: SafetyPointGoal1-v0 (Safety-Gymnasium 1.0.0), episode length H = 1000, per-step cost c_t in {0, 1}
(hazard contact). Reading in the paper's class: the resource is a cost budget b per episode (the suite's
customary cost limit is 25), the consumption is c_t, failure is the first step after which the cumulative
cost exceeds b, and the requirement is a tolerated probability of exhaustion within the episode. This is
the Saute/Simmer reading of the suite, as opposed to the CMDP reading E[sum c_t] <= b.

Wrapper (mirrors fuel-tank/lunar.py): observation augmented with the remaining fraction of the budget;
reward = native reward before failure; at the exhausting step the price's tail to the physical horizon,
-m (1 - beta^(H-t)) / (1 - beta), and termination (the "failure" objective), or -nu for the adaptive
multiplier (the "chance" objective, dual ascent on the exhaustion frequency); utility after failure is zero.
The unpriced learner (m = 0) is the control; a policy trained with the CMDP reading is not run here.

Pilot purpose: measure wall time per environment step and per PPO update, and record the exhaustion
frequency and return of the unpriced and of one priced learner after a short run, to size a campaign.
Usage: .venv310/bin/python safety_pilot_v63.py --m 0 --seed 1 --steps 100000 --stage pilot
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import safety_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

HERE = Path(__file__).resolve().parent
TASK, H = "SafetyPointGoal1-v0", 1000


class BudgetGoal(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, budget, objective="failure", m=0.0, beta=0.999, multiplier=0.0):
        super().__init__()
        self.inner = safety_gymnasium.make(TASK)
        self.budget, self.objective, self.m, self.beta = float(budget), objective, float(m), beta
        self.multiplier = float(multiplier)
        self.action_space = self.inner.action_space
        low = np.append(self.inner.observation_space.low, 0.0).astype(np.float32)
        high = np.append(self.inner.observation_space.high, 1.0).astype(np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def set_multiplier(self, value):
        self.multiplier = float(value)

    def _obs(self, o):
        return np.append(o, max(self.reserve, 0.0) / self.budget).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        o, info = self.inner.reset(seed=seed)
        self.reserve, self.t, self.used, self.failed_flag = self.budget, 0, 0.0, False
        return self._obs(o), info

    def step(self, action):
        o, r, c, terminated, truncated, info = self.inner.step(action)
        utility = float(r)
        self.reserve -= float(c)
        self.used += float(c)
        failed = self.reserve < 0.0
        if self.objective == "cmdp":
            # the expected-cost (CMDP) reading of the same task: a Lagrangian penalty on every unit of cost, no termination;
            # 'failed' is recorded only for evaluation
            reward = utility - self.multiplier * float(c)
            failed_now = failed and not self.failed_flag
            self.failed_flag = self.failed_flag or failed
            failed = failed_now
        elif failed:
            reward = -self.multiplier if self.objective == "chance" else -self.m * (1 - self.beta ** (H - self.t)) / (1 - self.beta)
            utility = 0.0
            terminated = True
        else:
            reward = utility
        self.t += 1
        info = dict(cost=float(c), failure=bool(failed), utility=utility, used=self.used)
        if terminated or truncated:
            info["episode_failed"] = bool(self.failed_flag) if self.objective == "cmdp" else bool(failed)
            info["episode_cost"] = self.used
        return self._obs(o), float(reward), bool(terminated), bool(truncated), info


class Control(BaseCallback):
    def __init__(self, arm):
        super().__init__(verbose=0)
        self.arm = arm
        self.multiplier = float(arm.get("multiplier0", 0.0))
        self.episodes = self.failures = 0
        self.trace = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode_failed" in info:
                self.episodes += 1
                self.failures += int(info["episode_failed"])
                self.cost_sum = getattr(self, "cost_sum", 0.0) + float(info.get("episode_cost", 0.0))
        return True

    def _on_rollout_end(self):
        rate = self.failures / self.episodes if self.episodes else float("nan")
        mean_cost = getattr(self, "cost_sum", 0.0) / self.episodes if self.episodes else float("nan")
        if self.arm["kind"] == "adaptive" and self.episodes:
            self.multiplier = max(0.0, self.multiplier + self.arm["dual_step"] * (rate - self.arm["alpha"]))
            self.training_env.env_method("set_multiplier", self.multiplier)
        if self.arm["kind"] == "cmdp" and self.episodes:
            self.multiplier = max(0.0, self.multiplier + self.arm["dual_step"] * (mean_cost - self.arm["cost_limit"]))
            self.training_env.env_method("set_multiplier", self.multiplier)
        self.cost_sum = 0.0
        self.trace.append(dict(steps=int(self.num_timesteps), multiplier=self.multiplier, episodes=self.episodes, failure_rate=rate, mean_cost=mean_cost))
        self.episodes = self.failures = 0


def evaluate(sb3, budget, beta, *, n, seed, n_envs=8, deterministic=False):
    """n independent episodes: exhaustion frequency, undiscounted return, cost used, discounted stopped utility."""
    envs = DummyVecEnv([lambda i=i: BudgetGoal(budget, "failure", 0.0, beta) for i in range(n_envs)])
    envs.seed(seed)
    torch.manual_seed(seed)
    obs = envs.reset()
    done_count, fails, returns, costs, utils = 0, 0, [], [], []
    ep_ret = np.zeros(n_envs); ep_cost = np.zeros(n_envs); ep_util = np.zeros(n_envs); ep_t = np.zeros(n_envs, dtype=int)
    while done_count < n:
        act, _ = sb3.predict(obs, deterministic=deterministic)
        obs, rew, dones, infos = envs.step(act)
        for i, info in enumerate(infos):
            ep_ret[i] += info["utility"]; ep_cost[i] += info["cost"]; ep_util[i] += beta ** ep_t[i] * info["utility"]; ep_t[i] += 1
            if dones[i]:
                done_count += 1; fails += int(info["episode_failed"]); returns.append(ep_ret[i]); costs.append(ep_cost[i]); utils.append(ep_util[i])
                ep_ret[i] = ep_cost[i] = ep_util[i] = 0.0; ep_t[i] = 0
                if done_count >= n:
                    break
    return dict(n=int(done_count), exhausted=fails / done_count, mean_return=float(np.mean(returns)), mean_cost=float(np.mean(costs)),
                stopped_utility=float(np.mean(utils)), cost_q90=float(np.quantile(costs, 0.9)))


def run(arm, seed, steps, out, eval_n, hp):
    budget, beta = arm["budget"], arm.get("beta", 0.999)
    objective = {"fixed": "failure", "adaptive": "chance", "cmdp": "cmdp"}[arm["kind"]]
    n_envs = hp.get("n_envs", 8)
    make = lambda: BudgetGoal(budget, objective, arm.get("m", 0.0), beta, arm.get("multiplier0", 0.0))
    venv = SubprocVecEnv([make for _ in range(n_envs)]) if hp.get("subproc", False) else DummyVecEnv([make for _ in range(n_envs)])
    venv.seed(seed)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_reward=1e9, gamma=beta)
    sb3 = PPO("MlpPolicy", venv, seed=seed, n_steps=hp.get("n_steps", 1024), batch_size=hp.get("batch_size", 64),
              n_epochs=hp.get("n_epochs", 4), learning_rate=(lambda p: 3e-4 * p), gamma=beta, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    control = Control(arm)
    t0 = time.monotonic()
    sb3.learn(total_timesteps=steps, callback=control)
    train_wall = time.monotonic() - t0
    out.mkdir(parents=True, exist_ok=True)
    sb3.save(out / "final_model.zip")
    venv.save(out / "vecnormalize.pkl")
    t1 = time.monotonic()
    res = dict(arm=arm, seed=seed, steps=int(sb3.num_timesteps), train_wall=train_wall, steps_per_second=sb3.num_timesteps / train_wall,
               trace=control.trace, final_multiplier=control.multiplier)
    # evaluation uses the raw (unnormalized) policy inputs: wrap evaluation envs with the saved statistics
    eval_envs = VecNormalize.load(out / "vecnormalize.pkl", DummyVecEnv([lambda: BudgetGoal(budget, "failure", 0.0, beta)]))
    eval_envs.training = False; eval_envs.norm_reward = False
    res["stochastic"] = evaluate_norm(sb3, eval_envs, budget, beta, n=eval_n, seed=10_000 + seed)
    res["eval_wall"] = time.monotonic() - t1
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def evaluate_norm(sb3, envs, budget, beta, *, n, seed, deterministic=False):
    envs.seed(seed); torch.manual_seed(seed)
    obs = envs.reset()
    done_count, fails, returns, costs, utils = 0, 0, [], [], []
    ep_ret = ep_cost = ep_util = 0.0; ep_t = 0
    while done_count < n:
        act, _ = sb3.predict(obs, deterministic=deterministic)
        obs, rew, dones, infos = envs.step(act)
        info = infos[0]
        ep_ret += info["utility"]; ep_cost += info["cost"]; ep_util += beta ** ep_t * info["utility"]; ep_t += 1
        if dones[0]:
            done_count += 1; fails += int(info["episode_failed"]); returns.append(ep_ret); costs.append(ep_cost); utils.append(ep_util)
            ep_ret = ep_cost = ep_util = 0.0; ep_t = 0
    return dict(n=int(done_count), exhausted=fails / done_count, mean_return=float(np.mean(returns)), mean_cost=float(np.mean(costs)),
                stopped_utility=float(np.mean(utils)), cost_q90=float(np.quantile(costs, 0.9)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, default=0.0); ap.add_argument("--kind", default="fixed"); ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--dual-step", type=float, default=1.0); ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--cost-limit", type=float, default=25.0, help="CMDP reading: expected episode cost limit d")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--steps", type=int, default=100_000); ap.add_argument("--eval", type=int, default=50)
    ap.add_argument("--stage", default="pilot"); ap.add_argument("--n-envs", type=int, default=8); ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="evaluate the saved model of this configuration")
    a = ap.parse_args()
    arm = dict(kind=a.kind, m=a.m, budget=a.budget, beta=0.999, alpha=a.alpha, dual_step=a.dual_step, multiplier0=0.0, task=TASK, H=H, cost_limit=a.cost_limit)
    out = HERE / "safety" / a.stage / f"{a.kind}_m{a.m:g}_b{a.budget:g}__seed{a.seed}"
    if a.eval_only:
        eval_envs = VecNormalize.load(out / "vecnormalize.pkl", DummyVecEnv([lambda: BudgetGoal(a.budget, "failure", 0.0, 0.999)]))
        eval_envs.training = False; eval_envs.norm_reward = False
        sb3 = PPO.load(out / "final_model.zip", env=eval_envs, device="cpu")
        r = dict(arm=arm, seed=a.seed, stochastic=evaluate_norm(sb3, eval_envs, a.budget, 0.999, n=a.eval, seed=10_000 + a.seed))
        (out / "result.json").write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps(r["stochastic"], indent=1)); raise SystemExit
    r = run(arm, a.seed, a.steps, out, a.eval, dict(n_envs=a.n_envs, subproc=a.subproc))
    print(json.dumps({k: r[k] for k in ("steps", "train_wall", "steps_per_second", "stochastic", "eval_wall")}, indent=1))
