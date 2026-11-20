"""Compare PPO architectures in the fixed-price wear replication, one run per task."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from barrierpg.callbacks import actor_parameters
from barrierpg.sb3_policies import ResidualActorCriticPolicy
from barrierpg.validation import validate_direct
from env import WearEnv, model_policy
from model import Model, simulate


ARCHITECTURES = ("published", "normalized", "residual")
# SB3 initializes its 32 parallel environments at seed through seed+31.
SEEDS = tuple(range(97_000, 105_000, 1_000))
MODEL = Model(wear_shape=0.5)
PRICE = 1.0
ALPHA = 0.05
FAMILY_ERROR = 0.05
FAMILY = len(ARCHITECTURES) * len(SEEDS)
N_ENVS = 32
N_STEPS = 256
TRAINING_BUDGET = 8_000_000
PROBE_BUDGETS = (32_768, 500_000, 1_000_000, 2_000_000, 4_000_000)
PROBE_EPISODES = 4_000
FINAL_EPISODES = 20_000


def rollout_boundary(steps):
    rollout = N_ENVS * N_STEPS
    return ((steps + rollout - 1) // rollout) * rollout


class WearCheckpoints(BaseCallback):
    """Evaluate only after PPO trains on the corresponding rollout."""

    def __init__(self, seed_index, checkpoints, smoke):
        super().__init__()
        self.seed_index = seed_index
        self.checkpoints = checkpoints
        self.smoke = smoke
        self.rows = []
        self.evaluation_seconds = 0.0

    def _on_training_start(self):
        self.started = time.perf_counter()

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        if self.model.num_timesteps in self.checkpoints:
            self.record(final=False)

    def _on_training_end(self):
        self.record(final=True)

    def record(self, *, final):
        started = time.perf_counter()
        checkpoint_index = len(self.rows)
        seed_offset = 100_000 * self.seed_index + checkpoint_index
        environment_seed = 1_000_000 + seed_offset
        policy_seed = 2_000_000 + seed_offset
        episodes = 64 if self.smoke else (FINAL_EPISODES if final else PROBE_EPISODES)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(policy_seed)
            evidence = simulate(MODEL, model_policy(MODEL, self.model), n=episodes,
                                seed=environment_seed, m=PRICE, max_steps=None)

        row = dict(steps=self.model.num_timesteps, final=final, episodes=episodes,
                   environment_seed=environment_seed, policy_seed=policy_seed,
                   training_seconds=started - self.started - self.evaluation_seconds,
                   utility=evidence["U"], utility_se=evidence["U_se"], failures=evidence["failures"],
                   q_beta=evidence["risk"], priced_value=evidence["U"] - PRICE * evidence["risk"] / (1 - MODEL.beta))
        if final and not self.smoke:
            check = validate_direct(evidence["failures"], episodes, ALPHA, FAMILY_ERROR, FAMILY)
            row.update(q_beta_upper=check.upper, passes_risk_tolerance=check.passed)
        self.rows.append(row)
        self.evaluation_seconds += time.perf_counter() - started


def run(task, output, smoke):
    seed_index, architecture_index = divmod(task, len(ARCHITECTURES))
    seed, architecture = SEEDS[seed_index], ARCHITECTURES[architecture_index]
    policy = "MlpPolicy"
    policy_kwargs = dict(net_arch=[64, 64], log_std_init=-1.0)
    if architecture != "published":
        policy = ResidualActorCriticPolicy
        policy_kwargs = dict(width=32, depth=2, layer_norm=True,
                             residual=architecture == "residual", log_std_init=-1.0)

    requested_steps = 2 * N_ENVS * N_STEPS if smoke else TRAINING_BUDGET
    budgets = (N_ENVS * N_STEPS,) if smoke else PROBE_BUDGETS
    checkpoints = tuple(map(rollout_boundary, budgets))
    env = DummyVecEnv([lambda: WearEnv(MODEL, "failure", PRICE)] * N_ENVS)
    env.seed(seed)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=1e9, gamma=1.0)
    try:
        model = PPO(policy, env, seed=seed, n_steps=N_STEPS, batch_size=1024, n_epochs=10,
                    learning_rate=lambda progress: 3e-4 * progress, gamma=1.0, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5,
                    policy_kwargs=policy_kwargs, device="cpu")
        checks = WearCheckpoints(seed_index, checkpoints, smoke)
        model.learn(total_timesteps=requested_steps, callback=checks)
        actor_count = sum(parameter.numel() for parameter in actor_parameters(model))
        total_count = sum(parameter.numel() for parameter in model.policy.parameters())
        results = dict(task=task, seed=seed, architecture=architecture, smoke=smoke,
                       requested_steps=requested_steps, policy_kwargs=policy_kwargs,
                       actor_parameters=actor_count, critic_parameters=total_count - actor_count,
                       simulator=MODEL.to_dict(), m=PRICE, risk_tolerance=ALPHA,
                       final_family_error=FAMILY_ERROR, final_family_size=FAMILY,
                       evaluation_seconds=checks.evaluation_seconds, checkpoints=checks.rows,
                       software=dict(torch=torch.__version__, stable_baselines3=stable_baselines3.__version__,
                                     gymnasium=gymnasium.__version__, numpy=np.__version__))
        output.mkdir(parents=True, exist_ok=False)
        model.save(output / "final_model.zip")
        (output / "result.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=range(FAMILY), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for one run")
    parser.add_argument("--smoke", action="store_true", help="Two rollouts and small evaluation batches")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.task, args.output, args.smoke)


if __name__ == "__main__":
    main()
