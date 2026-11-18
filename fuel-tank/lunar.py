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

Option crash_fails (added on 2026-11-14, stage crash-campaign; off in all earlier stages): the loss of the lander
is a failure as well. The lander is lost when it crashes or leaves the frame, the two events that end a native
episode with the reward -100. The frame of the loss then receives the same charge as the frame at which the tank
runs out, in place of the native reward, and its utility is zero, so the two failures are treated alike. Without
the option a price above about 0.17 makes an empty tank cost more than a crash, and the learner trades the one
for the other (see the note of the stage crash-campaign in lunar-protocol.json).

Option tank_until_touchdown (added on 2026-11-14, stage crash-campaign; off in all earlier stages): the tank can
fail only before touchdown, the first frame at which a leg of the lander touches the ground. From then on an empty
tank is not a failure: the engines are off once the tank is empty, and the episode continues to its native end,
a landing or the loss of the lander. Without the option most failures of the stage campaign are landers that have
touched down and are still firing to come to rest.

Option fuel_price (added on 2026-11-14, pilot stage fuelprice-pilot; 1 in all other stages): the utility charges
fuel_price per unit of fuel in place of the native 1. Reason: under the revised failure the native utility and
safety point the same way (a failed episode forgoes the landing bonus, which is worth far more than the fuel a
risky descent saves), so a price on failure has nothing to trade against. A higher value of fuel gives the
utility a reason to take risk, to brake late and land with thin reserves, which the price then counters.

Learner (added on 2026-11-14): a stage may name another learner in its entry "learner". With algo "sac" the policy
is trained from scratch by the soft actor-critic of Stable-Baselines3, with the discount of the study. The price
enters through the reward alone, so the task, the failure and the evaluation are the same for every learner.
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
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

torch.set_num_threads(1)
H = 1000


def fuel_use(action):
    a0, a1 = float(action[0]), float(action[1])
    m_power = (min(max(a0, 0.0), 1.0) + 1.0) * 0.5 if a0 > 0.0 else 0.0
    s_power = min(max(abs(a1), 0.5), 1.0) if abs(a1) > 0.5 else 0.0
    return 0.30 * m_power + 0.03 * s_power


def potential(o):
    """The potential of the benchmark's reward, from the first eight coordinates of the observation."""
    return float(-100.0 * np.hypot(o[0], o[1]) - 100.0 * np.hypot(o[2], o[3]) - 100.0 * abs(o[4]) + 10.0 * o[6] + 10.0 * o[7])


class FuelLander(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, budget, objective="failure", m=0.0, beta=0.999, multiplier=0.0, crash_fails=False, tank_until_touchdown=False, fuel_price=1.0,
                 revenue=None, progress="benchmark", aid_scale=1.0 / 12.0):
        super().__init__()
        self.crash_fails, self.tank_until_touchdown, self.fuel_price = bool(crash_fails), bool(tank_until_touchdown), float(fuel_price)
        # Formulations added on 2026-11-17 (stages formulations-*). With `revenue` set, the utility is a ledger: the revenue is
        # paid once, when the lander comes to rest on the pad, and fuel costs fuel_price per unit; nothing else enters it.
        # With progress == "aid" the benchmark's potential enters the training reward as beta * Phi(s') - Phi(s), with
        # Phi = 0 at the end of an episode, so that its discounted sum is -Phi(s_0) on every path and the objective is unchanged.
        self.revenue = None if revenue is None else float(revenue)
        self.progress, self.aid_scale = progress, float(aid_scale)
        assert progress in ("benchmark", "aid") and (self.revenue is not None or progress == "benchmark")
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
        self.episode_utility = 0.0                   # the discounted utility of the episode so far
        self.touched = False                         # whether a leg has touched the ground in an earlier frame
        self.prev_phi = potential(o)
        return self._obs(o), info

    def step(self, action):
        after_touchdown = self.tank_until_touchdown and self.touched
        if after_touchdown and self.reserve <= 0.0:
            action = np.array([-1.0, 0.0], dtype=np.float32)      # the tank is empty: both engines are off
        o, r, terminated, truncated, info = self.inner.step(action)
        g = fuel_use(action)
        if after_touchdown:
            g = min(g, max(self.reserve, 0.0))      # the last frame of thrust uses what is left
        if self.revenue is None:
            utility = float(r) - (self.fuel_price - 1.0) * g  # native reward, with fuel charged at fuel_price per unit (native: 1)
        else:
            utility = -self.fuel_price * g                    # the ledger: fuel now, the revenue at completion (below)
        self.reserve -= g
        self.used += g
        exhausted = self.reserve < 0.0 and not after_touchdown
        self.touched = self.touched or bool(o[6] > 0.5 or o[7] > 0.5)
        core = self.inner.unwrapped
        crashed = bool(terminated and not exhausted and core.game_over)
        landed = bool(terminated and not exhausted and not core.game_over and not core.lander.awake)
        lost = bool(terminated and not exhausted and not landed)          # crashed, or left the frame
        failed = exhausted or (self.crash_fails and lost)
        on_pad = bool(abs(float(o[0])) <= 0.2)                 # the pad spans the horizontal coordinates -0.2 to 0.2
        completed = bool(landed and on_pad)
        if self.revenue is not None and completed:
            utility += self.revenue
        if failed:
            if self.objective == "chance":
                reward = -self.multiplier
            else:
                reward = -self.m * (1 - self.beta ** (H - self.t)) / (1 - self.beta)
            utility = 0.0
            terminated = True
        else:
            reward = utility
        if self.progress == "aid":
            phi = 0.0 if (terminated or truncated) else potential(o)
            reward += self.aid_scale * (self.beta * phi - self.prev_phi)
            self.prev_phi = phi
        self.episode_utility += self.beta ** self.t * utility
        self.t += 1
        info = dict(fuel=g, failure=bool(failed), utility=utility, landed=landed, crashed=crashed, used=self.used, episode_utility=self.episode_utility,
                    exhausted=bool(exhausted), lost=lost, touched=self.touched, empty=bool(self.reserve <= 0.0), completed=completed)
        if terminated or truncated:
            info["episode_failed"] = bool(failed)
        return self._obs(o), float(reward), bool(terminated), bool(truncated), info


class Control(BaseCallback):
    def __init__(self, arm, every=None, log=None):
        super().__init__(verbose=0)
        self.arm = arm
        self.every, self.last = every, 0             # a learner with short rollouts records and updates every `every` steps
        self.log = log                               # file that receives each record of the trace as it is made
        self.ends = dict(landed=0, exhausted=0, lost=0)
        self.sums = dict(completed=0.0, used=0.0, episode_utility=0.0)      # of the episodes that ended since the last record
        self.multiplier = float(arm.get("multiplier0", 0.0))
        self.episodes = self.failures = 0
        self.trace = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode_failed" in info:
                self.episodes += 1
                self.failures += int(info["episode_failed"])
                for k in self.ends:
                    self.ends[k] += int(info[k])
                for k in self.sums:
                    self.sums[k] += float(info[k])
        return True

    def _on_rollout_end(self):
        if self.every is not None:
            if self.num_timesteps - self.last < self.every:
                return
            self.last = self.num_timesteps
        rate = self.failures / self.episodes if self.episodes else float("nan")
        if self.arm["kind"] == "adaptive" and self.episodes:
            self.multiplier = max(0.0, self.multiplier + self.arm["dual_step"] * (rate - self.arm["alpha"]))
            self.training_env.env_method("set_multiplier", self.multiplier)
        record = dict(steps=int(self.num_timesteps), multiplier=self.multiplier, episodes=self.episodes, failure_rate=rate,
                      **{k: (v / self.episodes if self.episodes else float("nan")) for k, v in self.ends.items()},
                      **{{"used": "fuel", "episode_utility": "U"}.get(k, k): (v / self.episodes if self.episodes else float("nan")) for k, v in self.sums.items()})
        self.trace.append(record)
        if self.log is not None:
            with open(self.log, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        self.episodes = self.failures = 0
        self.ends = dict(landed=0, exhausted=0, lost=0)
        self.sums = dict(completed=0.0, used=0.0, episode_utility=0.0)


def evaluate(sb3, budget, beta, *, n, seed, n_envs=32, deterministic=False, crash_fails=False, tank_until_touchdown=False, fuel_price=1.0, **formulation):
    """n independent episodes; failure = fuel exhausted (with crash_fails: or the lander lost; with tank_until_touchdown:
    exhausted before touchdown); utility discounted by beta."""
    envs = DummyVecEnv([lambda i=i: FuelLander(budget, "failure", 0.0, beta, crash_fails=crash_fails, tank_until_touchdown=tank_until_touchdown, fuel_price=fuel_price, **formulation) for i in range(n_envs)])
    envs.seed(seed)
    torch.manual_seed(seed)
    obs = envs.reset()
    done_count = 0
    records = []
    acc = [dict(U=0.0, t=0, failed=False, landed=False, used=0.0, crashed=False, exhausted=False, lost=False, empty=False, completed=False) for _ in range(n_envs)]
    while done_count < n:
        act, _ = sb3.predict(obs, deterministic=deterministic)
        obs, rew, dones, infos = envs.step(act)
        for i in range(n_envs):
            a = acc[i]
            a["U"] += beta ** a["t"] * infos[i]["utility"]
            a["t"] += 1
            if dones[i]:
                a["failed"] = infos[i]["failure"]; a["landed"] = infos[i]["landed"]; a["used"] = infos[i]["used"]
                a["crashed"] = infos[i]["crashed"]; a["exhausted"] = infos[i]["exhausted"]; a["lost"] = infos[i]["lost"]; a["empty"] = infos[i]["empty"]
                a["completed"] = infos[i]["completed"]
                if done_count < n:
                    records.append(a)
                done_count += 1
                acc[i] = dict(U=0.0, t=0, failed=False, landed=False, used=0.0, crashed=False, exhausted=False, lost=False, empty=False, completed=False)
    envs.close()
    fail = np.array([r["failed"] for r in records]); land = np.array([r["landed"] for r in records])
    used = np.array([r["used"] for r in records]); U = np.array([r["U"] for r in records]); T = np.array([r["t"] for r in records])
    return dict(n=n, seed=seed, deterministic=deterministic, risk=float(fail.mean()), failures=int(fail.sum()),
                landed=float(land.mean()), completed=float(np.mean([r["completed"] for r in records])), fuel_mean=float(used.mean()),
                crashed=float(np.mean([r["crashed"] for r in records])),
                exhausted=float(np.mean([r["exhausted"] for r in records])), lost=float(np.mean([r["lost"] for r in records])),
                landed_empty=float(np.mean([r["landed"] and r["empty"] for r in records])),      # landed with the tank empty at the end
                lost_empty=float(np.mean([r["lost"] and r["empty"] for r in records])),
                U=float(U.mean()), U_se=float(U.std(ddof=1) / np.sqrt(n)), length=float(T.mean()),
                fuel_quantiles=np.quantile(used, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist())


def run(arm, seed, steps, out, eval_n, hp, init=None):
    """init: path of a saved model to continue from (the unconstrained lander of the same seed);
    the learning-rate schedule restarts, the tank, price and objective are those of `arm`."""
    budget, beta = arm["budget"], arm.get("beta", 0.999)
    cf = dict(crash_fails=bool(arm.get("crash_fails", False)), tank_until_touchdown=bool(arm.get("tank_until_touchdown", False)), fuel_price=float(arm.get("fuel_price", 1.0)))
    if arm.get("revenue") is not None:                       # the ledger formulations of 2026-11-17
        cf.update(revenue=float(arm["revenue"]), progress=arm.get("progress", "aid"), aid_scale=float(arm.get("aid_scale", 1.0 / 12.0)))
    objective = {"fixed": "failure", "adaptive": "chance"}[arm["kind"]]
    algo = hp.get("algo", "ppo")
    n_envs = hp.get("n_envs", 16 if algo == "ppo" else 1)
    venv = SubprocVecEnv([lambda: FuelLander(budget, objective, arm.get("m", 0.0), beta, arm.get("multiplier0", 0.0), **cf) for _ in range(n_envs)]) \
        if hp.get("subproc", False) else DummyVecEnv([lambda: FuelLander(budget, objective, arm.get("m", 0.0), beta, arm.get("multiplier0", 0.0), **cf) for _ in range(n_envs)])
    venv.seed(seed)
    if hp.get("normalize", True):
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=1e9, gamma=beta)
    lr = (lambda p: 3e-4 * p) if hp.get("lr_linear", True) else 3e-4
    if algo == "sac":
        assert init is None, "the soft actor-critic is trained from scratch"
        sb3 = SAC("MlpPolicy", venv, seed=seed, gamma=beta, learning_rate=hp.get("learning_rate", 3e-4), buffer_size=hp.get("buffer_size", 1_000_000),
                  batch_size=hp.get("batch_size", 256), tau=hp.get("tau", 0.01), train_freq=hp.get("train_freq", 1), gradient_steps=hp.get("gradient_steps", 1),
                  learning_starts=hp.get("learning_starts", 10_000), ent_coef="auto", policy_kwargs=dict(net_arch=hp.get("net_arch", [400, 300])),
                  device="cpu", verbose=0)
    elif init is None:
        sb3 = PPO("MlpPolicy", venv, seed=seed, n_steps=hp.get("n_steps", 1024), batch_size=hp.get("batch_size", 64),
                  n_epochs=hp.get("n_epochs", 4), learning_rate=lr,
                  gamma=beta, gae_lambda=hp.get("gae_lambda", 0.98), clip_range=0.2, ent_coef=hp.get("ent_coef", 0.01),
                  max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    else:
        sb3 = PPO.load(init, env=venv, device="cpu", seed=seed, custom_objects=dict(learning_rate=lr))
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.jsonl").unlink(missing_ok=True)
    control = Control(arm, every=hp.get("trace_every", 10_000) if algo == "sac" else None, log=out / "progress.jsonl")
    t0 = time.monotonic()
    sb3.learn(total_timesteps=steps, callback=control, reset_num_timesteps=True)
    train_wall = time.monotonic() - t0
    out.mkdir(parents=True, exist_ok=True)
    sb3.save(out / "final_model.zip")
    t1 = time.monotonic()
    result = dict(arm=arm, seed=seed, steps=int(sb3.num_timesteps), train_wall=train_wall, trace=control.trace,
                  final_multiplier=control.multiplier,
                  stochastic=evaluate(sb3, budget, beta, n=eval_n, seed=10_000 + seed, **cf),
                  mean_action=evaluate(sb3, budget, beta, n=eval_n, seed=20_000 + seed, deterministic=True, **cf))
    result["eval_wall"] = time.monotonic() - t1
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--stage", default="campaign")
    ap.add_argument("--init", default=None, help="model to continue from; default from the stage's warm_from entry")
    args = ap.parse_args()
    protocol = json.loads((ROOT / "lunar-protocol.json").read_text())
    stage = protocol["stages"][args.stage]
    arm = {**protocol["arms"][args.arm], **stage.get("arm_overrides", {})}      # a stage may set options for all its arms
    out = ROOT / "lunar" / args.stage / f"{args.arm}__seed{args.seed}"
    if (out / "result.json").exists():
        print("exists", out.name); sys.exit(0)
    init = args.init
    if init is None and stage.get("warm_from") == "@same":      # continue the run of the same configuration and seed in another stage
        init = str(ROOT / "lunar" / stage["warm_stage"] / f"{args.arm}__seed{args.seed}" / "final_model.zip")
        if arm["kind"] == "adaptive" and stage.get("hold_multiplier", False):      # the multiplier keeps the value at which that run ended
            prev = json.loads((ROOT / "lunar" / stage["warm_stage"] / f"{args.arm}__seed{args.seed}" / "result.json").read_text())
            arm = {**arm, "multiplier0": prev["final_multiplier"], "dual_step": 0.0}
    elif init is None and stage.get("warm_from") and arm.get("warm", True) and args.arm != stage["warm_from"]:
        init = str(ROOT / "lunar" / stage.get("warm_stage", args.stage) / f"{stage['warm_from']}__seed{args.seed}" / "final_model.zip")
    steps = stage["warm_steps"] if init else stage["steps"]
    r = run(arm, args.seed, steps, out, stage["eval_n"], stage.get("learner", stage.get("ppo", {})), init=init)
    r["init"] = init
    (out / "result.json").write_text(json.dumps(r, indent=1) + "\n")
    s = r["stochastic"]
    print(f"{args.arm} seed {args.seed}: risk={s['risk']:.3f} landed={s['landed']:.3f} crashed={s['crashed']:.3f} U={s['U']:.1f} fuel q={[round(x,1) for x in s['fuel_quantiles']]} train={r['train_wall']:.0f}s eval={r['eval_wall']:.0f}s init={init}")
