"""Lunar landing with a fuel tank: a standard continuous-control benchmark under a cumulative
resource budget (revision 38, second example; runtime ~/.venvs/constrained-rl-numerics/bin/python with Box2D).

The native LunarLanderContinuous-v3 reward charges fuel per frame, 0.30 per unit of main
engine power and 0.03 per unit of side engine power. Here that consumption is the resource
instead: the lander starts with a tank b, each frame consumes g = 0.30 m_power + 0.03 s_power,
and running out of fuel before touchdown is the failure. The utility is the native reward itself
(shaping, the fuel charge, and the terminal +100 for landing and -100 for crashing); the
tank is an additional limit on the same consumption.
Failure-only price m: the failing frame receives -m (1 - beta^(H - t)) / (1 - beta), the
discounted tail of a per-frame penalty m until the physical horizon H = 1000.
Adaptive arm: the failing frame receives -lambda with lambda updated by dual ascent.
Observation: the eight native coordinates and the remaining fuel fraction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parent

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

torch.set_num_threads(1)
H = 1000


def fuel_use(action):
    a0, a1 = float(action[0]), float(action[1])
    m_power = (min(max(a0, 0.0), 1.0) + 1.0) * 0.5 if a0 > 0.0 else 0.0
    s_power = min(max(abs(a1), 0.5), 1.0) if abs(a1) > 0.5 else 0.0
    return 0.30 * m_power + 0.03 * s_power


class FuelLander(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, budget, objective="failure", m=0.0, beta=0.999, multiplier=0.0):
        super().__init__()
        self.inner = gym.make("LunarLanderContinuous-v3")
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
        self.reserve, self.t, self.used = self.budget, 0, 0.0
        return self._obs(o), info

    def step(self, action):
        o, r, terminated, truncated, info = self.inner.step(action)
        g = fuel_use(action)
        utility = float(r)                          # native reward, fuel charge included
        self.reserve -= g
        self.used += g
        failed = self.reserve < 0.0
        if failed:
            if self.objective == "chance":
                reward = -self.multiplier
            else:
                reward = -self.m * (1 - self.beta ** (H - self.t)) / (1 - self.beta)
            utility = 0.0
            terminated = True
        else:
            reward = utility
        self.t += 1
        core = self.inner.unwrapped
        landed = bool(terminated and not failed and not core.game_over and not core.lander.awake)
        crashed = bool(terminated and not failed and core.game_over)
        info = dict(fuel=g, failure=bool(failed), utility=utility, landed=landed, crashed=crashed, used=self.used)
        if terminated or truncated:
            info["episode_failed"] = bool(failed)
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
        return True

    def _on_rollout_end(self):
        rate = self.failures / self.episodes if self.episodes else float("nan")
        if self.arm["kind"] == "adaptive" and self.episodes:
            self.multiplier = max(0.0, self.multiplier + self.arm["dual_step"] * (rate - self.arm["alpha"]))
            self.training_env.env_method("set_multiplier", self.multiplier)
        self.trace.append(dict(steps=int(self.num_timesteps), multiplier=self.multiplier, episodes=self.episodes, failure_rate=rate))
        self.episodes = self.failures = 0


def evaluate(sb3, budget, beta, *, n, seed, n_envs=32, deterministic=False):
    """n independent episodes; failure = fuel exhausted; utility discounted by beta."""
    envs = DummyVecEnv([lambda i=i: FuelLander(budget, "failure", 0.0, beta) for i in range(n_envs)])
    envs.seed(seed)
    torch.manual_seed(seed)
    obs = envs.reset()
    done_count = 0
    records = []
    acc = [dict(U=0.0, t=0, failed=False, landed=False, used=0.0, crashed=False) for _ in range(n_envs)]
    while done_count < n:
        act, _ = sb3.predict(obs, deterministic=deterministic)
        obs, rew, dones, infos = envs.step(act)
        for i in range(n_envs):
            a = acc[i]
            a["U"] += beta ** a["t"] * infos[i]["utility"]
            a["t"] += 1
            if dones[i]:
                a["failed"] = infos[i]["failure"]; a["landed"] = infos[i]["landed"]; a["used"] = infos[i]["used"]
                a["crashed"] = infos[i]["crashed"]
                if done_count < n:
                    records.append(a)
                done_count += 1
                acc[i] = dict(U=0.0, t=0, failed=False, landed=False, used=0.0, crashed=False)
    envs.close()
    fail = np.array([r["failed"] for r in records]); land = np.array([r["landed"] for r in records])
    used = np.array([r["used"] for r in records]); U = np.array([r["U"] for r in records]); T = np.array([r["t"] for r in records])
    return dict(n=n, seed=seed, deterministic=deterministic, risk=float(fail.mean()), failures=int(fail.sum()),
                landed=float(land.mean()), crashed=float(np.mean([r["crashed"] for r in records])),
                U=float(U.mean()), U_se=float(U.std(ddof=1) / np.sqrt(n)), length=float(T.mean()),
                fuel_quantiles=np.quantile(used, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist())


def run(arm, seed, steps, out, eval_n, hp, init=None):
    """init: path of a saved model to continue from (the unconstrained lander of the same seed);
    the learning-rate schedule restarts, the tank, price and objective are those of `arm`."""
    budget, beta = arm["budget"], arm.get("beta", 0.999)
    objective = {"fixed": "failure", "adaptive": "chance"}[arm["kind"]]
    n_envs = hp.get("n_envs", 16)
    venv = SubprocVecEnv([lambda: FuelLander(budget, objective, arm.get("m", 0.0), beta, arm.get("multiplier0", 0.0)) for _ in range(n_envs)]) \
        if hp.get("subproc", False) else DummyVecEnv([lambda: FuelLander(budget, objective, arm.get("m", 0.0), beta, arm.get("multiplier0", 0.0)) for _ in range(n_envs)])
    venv.seed(seed)
    if hp.get("normalize", True):
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=1e9, gamma=beta)
    lr = (lambda p: 3e-4 * p) if hp.get("lr_linear", True) else 3e-4
    if init is None:
        sb3 = PPO("MlpPolicy", venv, seed=seed, n_steps=hp.get("n_steps", 1024), batch_size=hp.get("batch_size", 64),
                  n_epochs=hp.get("n_epochs", 4), learning_rate=lr,
                  gamma=beta, gae_lambda=hp.get("gae_lambda", 0.98), clip_range=0.2, ent_coef=hp.get("ent_coef", 0.01),
                  max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    else:
        sb3 = PPO.load(init, env=venv, device="cpu", seed=seed, custom_objects=dict(learning_rate=lr))
    control = Control(arm)
    t0 = time.monotonic()
    sb3.learn(total_timesteps=steps, callback=control, reset_num_timesteps=True)
    train_wall = time.monotonic() - t0
    out.mkdir(parents=True, exist_ok=True)
    sb3.save(out / "final_model.zip")
    t1 = time.monotonic()
    result = dict(arm=arm, seed=seed, steps=int(sb3.num_timesteps), train_wall=train_wall, trace=control.trace,
                  final_multiplier=control.multiplier,
                  stochastic=evaluate(sb3, budget, beta, n=eval_n, seed=10_000 + seed),
                  mean_action=evaluate(sb3, budget, beta, n=eval_n, seed=20_000 + seed, deterministic=True))
    result["eval_wall"] = time.monotonic() - t1
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--stage", default="campaign")
    ap.add_argument("--init", default=None, help="model to continue from; default from the stage's warm_from entry")
    args = ap.parse_args()
    protocol = json.loads((ROOT / "lunar-protocol.json").read_text())
    arm, stage = protocol["arms"][args.arm], protocol["stages"][args.stage]
    out = ROOT / "lunar" / args.stage / f"{args.arm}__seed{args.seed}"
    if (out / "result.json").exists():
        print("exists", out.name); sys.exit(0)
    init = args.init
    if init is None and stage.get("warm_from") and arm.get("warm", True) and args.arm != stage["warm_from"]:
        init = str(ROOT / "lunar" / args.stage / f"{stage['warm_from']}__seed{args.seed}" / "final_model.zip")
    steps = stage["warm_steps"] if init else stage["steps"]
    r = run(arm, args.seed, steps, out, stage["eval_n"], stage.get("ppo", {}), init=init)
    r["init"] = init
    (out / "result.json").write_text(json.dumps(r, indent=1) + "\n")
    s = r["stochastic"]
    print(f"{args.arm} seed {args.seed}: risk={s['risk']:.3f} landed={s['landed']:.3f} crashed={s['crashed']:.3f} U={s['U']:.1f} fuel q={[round(x,1) for x in s['fuel_quantiles']]} train={r['train_wall']:.0f}s eval={r['eval_wall']:.0f}s init={init}")
