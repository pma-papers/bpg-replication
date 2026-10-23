"""One training run of the revision-38 wear-allowance study (pinned Stable-Baselines3 PPO).

Arms (see protocol.json):
  fixed     failure-only price m (m = 0 is the unpriced learner)
  barrier   interior barrier with price m
  adaptive  failure indicator charged by a multiplier updated by projected dual ascent
            toward a prescribed breakdown probability alpha
  anchored  failure-only price m plus the stopped action relative entropy to an incumbent
            rule with weight eta (sample-wise log-density ratio added to the reward)
The final stochastic policy is the candidate; evaluation uses fresh trajectories.
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
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import Incumbent, WearEnv, model_policy, observe
from model import Model, simulate

torch.set_num_threads(1)


class Control(BaseCallback):
    """Dual ascent for the adaptive arm and reward shaping for the anchored arm."""

    def __init__(self, arm, incumbent=None):
        super().__init__(verbose=0)
        self.arm, self.incumbent = arm, incumbent
        self.multiplier = float(arm.get("multiplier0", 0.0))
        self.episodes, self.failures = 0, 0
        self.trace = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode_failed" in info:
                self.episodes += 1
                self.failures += int(info["episode_failed"])
        if self.arm["kind"] == "anchored" and self.arm["eta"] > 0:
            obs = self.model._last_obs                      # states at which the actions were drawn
            x = self.locals["actions"].reshape(-1)          # pre-clipping Gaussian samples
            logp = self.locals["log_probs"].detach().cpu().numpy().reshape(-1)
            z = (obs[:, 1] + 1.0) / 2.0 * self.incumbent.model.budget
            ratio = logp - self.incumbent.log_density(x, z)
            venv = self.model.get_vec_normalize_env()
            scale = np.sqrt(venv.ret_rms.var + venv.epsilon)
            self.locals["rewards"][:] -= (self.arm["eta"] * ratio / scale).astype(self.locals["rewards"].dtype)
        return True

    def _on_rollout_end(self):
        rate = self.failures / self.episodes if self.episodes else float("nan")
        if self.arm["kind"] == "adaptive" and self.episodes:
            self.multiplier = max(0.0, self.multiplier + self.arm["dual_step"] * (rate - self.arm["alpha"]))
            self.training_env.env_method("set_multiplier", self.multiplier)
        self.trace.append(dict(steps=int(self.num_timesteps), multiplier=self.multiplier,
                               episodes=self.episodes, failure_rate=rate))
        self.episodes, self.failures = 0, 0


def stopped_action_kl(model, sb3, incumbent, *, n, seed):
    """Monte Carlo estimate of A = E sum_{t <= tau and before access} KL(pi(.|y_t) || rho(.|y_t))."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    s = np.zeros(n); z = np.full(n, model.budget); alive = np.ones(n, dtype=bool); A = np.zeros(n)
    while alive.any():
        ids = np.flatnonzero(alive)
        obs = observe(model, s[ids], z[ids])
        with torch.no_grad():
            dist = sb3.policy.get_distribution(torch.as_tensor(obs)).distribution
            mean, std = dist.mean.numpy().reshape(-1), dist.stddev.numpy().reshape(-1)
        A[ids] += incumbent.kl_from(mean, std, z[ids])
        x = mean + std * rng.standard_normal(len(ids))
        a = (np.clip(x, -1, 1) + 1) / 2
        z_next = z[ids] - model.wear_rate * a * rng.gamma(model.wear_shape, 1 / model.wear_shape, len(ids))
        s[ids] = model.rho * s[ids] + (1 - model.rho) * rng.uniform(-1, 1, len(ids))
        z[ids] = z_next
        stop = (z_next < 0) | (rng.random(len(ids)) > model.beta)
        alive[ids[stop]] = False
    return dict(A=float(A.mean()), A_se=float(A.std(ddof=1) / np.sqrt(n)))


def run(arm, seed, steps, out, eval_n, lr_linear=False, hp=None):
    hp = hp or {}
    model = Model(**arm.get("model", {}))
    incumbent = Incumbent(model, **arm.get("incumbent", {})) if arm["kind"] == "anchored" else None
    objective = {"fixed": "failure", "anchored": "failure", "barrier": "barrier", "adaptive": "chance"}[arm["kind"]]
    n_envs = hp.get('n_envs', 8)
    venv = DummyVecEnv([lambda: WearEnv(model, objective, arm.get("m", 0.0), arm.get("multiplier0", 0.0)) for _ in range(n_envs)])
    venv.seed(seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=1e9, gamma=1.0)
    sb3 = PPO("MlpPolicy", venv, seed=seed, n_steps=hp.get("n_steps", 256), batch_size=hp.get("batch_size", 256), n_epochs=hp.get("n_epochs", 10), learning_rate=(lambda progress: 3e-4 * progress) if lr_linear else 3e-4,
              gamma=1.0, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5,
              policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0), device="cpu", verbose=0)
    control = Control(arm, incumbent)
    started = time.monotonic()
    sb3.learn(total_timesteps=steps, callback=control)
    train_wall = time.monotonic() - started
    m_eval = arm.get("m", 0.0)
    result = dict(arm=arm, seed=seed, steps=int(sb3.num_timesteps), train_wall=train_wall,
                  log_std=float(sb3.policy.log_std.detach().numpy().reshape(-1)[0]),
                  final_multiplier=control.multiplier, trace=control.trace,
                  stochastic=simulate(model, model_policy(model, sb3, False), n=eval_n, seed=10_000 + seed, m=m_eval),
                  mean_action=simulate(model, model_policy(model, sb3, True), n=eval_n, seed=20_000 + seed, m=m_eval),
                  robust90=simulate(model, model_policy(model, sb3, False), n=eval_n, seed=30_000 + seed, m=m_eval, budget_scale=0.9),
                  robust80=simulate(model, model_policy(model, sb3, False), n=eval_n, seed=40_000 + seed, m=m_eval, budget_scale=0.8))
    if incumbent is not None:
        result["action_kl"] = stopped_action_kl(model, sb3, incumbent, n=eval_n, seed=50_000 + seed)
    out.mkdir(parents=True, exist_ok=True)
    sb3.save(out / "final_model.zip")
    (out / "result.json").write_text(json.dumps(result, indent=1, allow_nan=True) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", default="campaign")
    args = ap.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    arm = protocol["arms"][args.arm]
    stage = protocol["stages"][args.stage]
    out = ROOT / args.stage / f"{args.arm}__seed{args.seed}"
    if (out / "result.json").exists():
        print("exists", out.name); sys.exit(0)
    r = run(arm, args.seed, stage["steps"], out, stage["eval_n"], stage.get("lr_linear", False), stage.get("ppo"))
    st = r["stochastic"]
    print(f"{args.arm} seed {args.seed}: U={st['U']:.3f} q={st['risk']:.4f} mean-action U={r['mean_action']['U']:.3f} q={r['mean_action']['risk']:.4f} wall={r['train_wall']:.0f}s")
