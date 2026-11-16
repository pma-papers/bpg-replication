"""Benchmark-suite study, design of November 2026 (SafetyPointGoal1-v0, Safety-Gymnasium 1.0.0).

The task, the failure and the charge are those of safety_pilot_v63.py: a cost budget of 25 per episode of
H = 1000 steps, failure at the first step after which the cumulative cost exceeds the budget, the episode
ends there, and the fixed price m charges m (1 - beta^(H - t)) / (1 - beta) at the exhausting step t. The
training objective is U - nu * qt with nu = m / (1 - beta) and qt = E[(beta^tau - beta^H) 1{tau <= H - 1}].

What this runner adds to the stored ones (safety_pilot_v63.py, safety_ramp_v65.py, which stay as they are):

  * a run can continue from a stored controller (the controller trained without a price, x_0), with the
    observation normalization of that controller kept fixed and the reward normalization carried on;
  * a constant learning rate (the decayed one remains available for comparison);
  * a price that changes during training: constant stages ("0.16@0.5,0.08@1") or a linear rise from zero over
    a first fraction of the run (price_ramp);
  * the parameter anchor: kappa (x - x_0) is added to the actor gradient after PPO's gradient clipping and
    before the Adam step, with a strength that is constant or changes in stages ("1@0.33,0.1@0.66,0.01@1");
  * a trace after every rollout (price, strength, exhaustion frequency, return, cost and goals of the training
    episodes, action dispersion, distance to x_0, norms of the mean clipped actor gradient, of the mean pull
    and of their sum);
  * evaluations during training on fresh episodes that leave the random streams of training untouched, and
    stored copies of the learner at regular intervals;
  * per evaluation episode: exhausted or not, the exhausting step, return, stopped utility, cost (equal to
    the number of steps inside hazards), goals reached, distance travelled.

The runs of a stage and their settings are listed in safety-study-v76.json.

Usage (runtime ~/.venvs/constrained-rl-safety310/bin/python):
  safety_study_v76.py run --stage STAGE --config NAME --seed S
  safety_study_v76.py stage --stage STAGE [--jobs 4]      (all runs of the stage that have no result yet)
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from safety_pilot_v63 import BudgetGoal, H, TASK  # noqa: E402  (wrapper, task, horizon: unchanged)

torch.set_num_threads(1)
BETA, BUDGET = 0.999, 25.0
STUDY = HERE / "safety-study-v76.json"


class StudyGoal(BudgetGoal):
    """The stored wrapper with the fixed-price objective, a price the training may change, and a record of
    the goals reached and of the distance travelled."""

    def set_price(self, value):
        self.m = float(value)

    def _position(self):
        return np.array(self.inner.unwrapped.task.agent.pos[:2], dtype=float)

    def reset(self, *, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.goals, self.path, self.pos = 0, 0.0, self._position()
        return out

    def step(self, action):
        o, r, c, terminated, truncated, inner = self.inner.step(action)
        utility = float(r)
        self.reserve -= float(c)
        self.used += float(c)
        failed = self.reserve < 0.0
        if failed:
            reward = -self.m * (1 - self.beta ** (H - self.t)) / (1 - self.beta)
            utility = 0.0
            terminated = True
        else:
            reward = utility
        self.t += 1
        pos = self._position()
        self.path += float(np.linalg.norm(pos - self.pos)); self.pos = pos
        self.goals += int(bool(inner.get("goal_met", False)))
        info = dict(cost=float(c), failure=bool(failed), utility=utility, used=self.used)
        if terminated or truncated:
            info.update(episode_failed=bool(failed), episode_cost=self.used, episode_goals=self.goals, episode_path=self.path, episode_steps=self.t)
        return self._obs(o), float(reward), bool(terminated), bool(truncated), info


class FixedMeanStd(RunningMeanStd):
    """Statistics that are kept as they are (the observation normalization of the controller a run continues from)."""

    def update(self, arr):
        return None


def parse_schedule(text):
    """'10' -> [(1.0, 10.0)]; '100@0.25,10@0.5,1@1' -> [(0.25, 100.0), (0.5, 10.0), (1.0, 1.0)]."""
    stages = []
    for part in str(text).split(","):
        value, _, end = part.partition("@")
        stages.append((float(end) if end else 1.0, float(value)))
    assert all(a[0] < b[0] for a, b in zip(stages, stages[1:])) and abs(stages[-1][0] - 1.0) < 1e-12, text
    return stages


def value_at(stages, fraction):
    for end, value in stages:
        if fraction < end:
            return value
    return stages[-1][1]


def price_at(stages, ramp, fraction):
    if ramp > 0 and fraction < ramp:
        return stages[0][1] * fraction / ramp
    return value_at(stages, fraction)


def actor_params(policy):
    """All parameters that shape the policy: the mean network and the dispersion."""
    return list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters()) + [policy.log_std]


def distance(policy, theta0):
    with torch.no_grad():
        return float(torch.sqrt(sum(((p - p0) ** 2).sum() for p, p0 in zip(actor_params(policy), theta0))))


class State:
    """Shared between the optimizer hook and the callback."""

    def __init__(self):
        self.fraction = 0.0
        self.reset_window()

    def reset_window(self):
        self.n = 0; self.g = None; self.pull = None; self.g_norm = 0.0


def install_hook(sb3, theta0, st, kappa_sched):
    """Before every Adam step (after PPO's gradient clipping): record the actor gradient and add the pull kappa (x - x_0)."""
    params = actor_params(sb3.policy)
    opt = sb3.policy.optimizer
    base_step = opt.step

    def step(closure=None):
        kappa = value_at(kappa_sched, st.fraction)
        with torch.no_grad():
            g = torch.cat([(p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in params]).clone()
            st.n += 1; st.g_norm += float(g.norm())
            st.g = g if st.g is None else st.g + g
            if theta0 is not None:
                pull = torch.cat([(kappa * (p - p0)).reshape(-1) for p, p0 in zip(params, theta0)])
                st.pull = pull if st.pull is None else st.pull + pull
                if kappa > 0:
                    for p, p0 in zip(params, theta0):
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        p.grad.add_(kappa * (p - p0))
        return base_step(closure)
    opt.step = step


def summarize(eps, m):
    """Per policy: exhaustion probability, priced risk qt, stopped utility U, the objective at the price m, and how the episodes end."""
    n = len(eps)
    failed = np.array([e["failed"] for e in eps], dtype=bool)
    qt = np.array([(BETA ** e["tau"] - BETA ** H) if e["failed"] else 0.0 for e in eps])
    U = np.array([e["U"] for e in eps]); ret = np.array([e["ret"] for e in eps]); goals = np.array([e["goals"] for e in eps])
    J = U - m / (1 - BETA) * qt
    se = lambda x: float(np.std(x, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    kept = ~failed
    return dict(n=n, price=m, exhausted=float(failed.mean()), q_tilde=float(qt.mean()), q_tilde_se=se(qt), U=float(U.mean()), U_se=se(U),
                objective=float(J.mean()), objective_se=se(J), mean_return=float(ret.mean()), mean_return_se=se(ret),
                mean_cost=float(np.mean([e["cost"] for e in eps])), mean_goals=float(goals.mean()), mean_path=float(np.mean([e["path"] for e in eps])),
                mean_steps=float(np.mean([e["steps"] for e in eps])),
                share_exhausted=float(failed.mean()), share_goal_budget_kept=float((kept & (goals > 0)).mean()), share_no_goal_budget_kept=float((kept & (goals == 0)).mean()),
                return_budget_kept=float(ret[kept].mean()) if kept.any() else float("nan"), goals_budget_kept=float(goals[kept].mean()) if kept.any() else float("nan"))


class Evaluator:
    """Fresh episodes of the stopped task without a charge, the same number in each of n_envs environments
    (so that short episodes are not over-represented); the random streams of training are restored afterwards."""

    def __init__(self, n_envs=8):
        self.envs = [StudyGoal(BUDGET, "failure", 0.0, BETA) for _ in range(n_envs)]

    def episodes(self, sb3, venv, n, key):
        per = max(n // len(self.envs), 1)
        seeds = [int(s) for s in np.random.SeedSequence(key).generate_state(len(self.envs) + 1)]
        ts, ns = torch.get_rng_state(), np.random.get_state()
        torch.manual_seed(seeds[-1] % (2 ** 31))
        obs = np.stack([e.reset(seed=s % (2 ** 31))[0] for e, s in zip(self.envs, seeds)])
        k = len(self.envs)
        done = [0] * k; eps = []
        ret = np.zeros(k); util = np.zeros(k); cost = np.zeros(k); t = np.zeros(k, dtype=int)
        while min(done) < per:
            act, _ = sb3.policy.predict(venv.normalize_obs(obs), deterministic=False)
            for i, e in enumerate(self.envs):
                if done[i] >= per:
                    continue
                o, _, terminated, truncated, info = e.step(act[i])
                ret[i] += info["utility"]; util[i] += BETA ** t[i] * info["utility"]; cost[i] += info["cost"]
                if terminated or truncated:
                    failed = bool(info["episode_failed"])
                    eps.append(dict(failed=failed, tau=int(t[i]) if failed else None, ret=float(ret[i]), U=float(util[i]), cost=float(cost[i]),
                                    goals=int(info["episode_goals"]), path=float(info["episode_path"]), steps=int(info["episode_steps"])))
                    done[i] += 1; ret[i] = util[i] = cost[i] = 0.0; t[i] = 0
                    o, _ = e.reset()
                else:
                    t[i] += 1
                obs[i] = o
        torch.set_rng_state(ts); np.random.set_state(ns)
        return eps


class Trace(BaseCallback):
    """Sets the price, writes the trace after every rollout, evaluates and stores the learner at intervals."""

    def __init__(self, st, cfg, total, theta0, seed, out, scheds, evaluator):
        super().__init__(verbose=0)
        self.st, self.cfg, self.total, self.theta0, self.seed, self.out = st, cfg, total, theta0, seed, out
        self.price_sched, self.ramp, self.kappa_sched = scheds
        self.evaluator = evaluator
        self.trace = []; self.evals = []; self.next_eval = 0; self.next_copy = cfg["copy_every"]
        self.price_now = price_at(self.price_sched, self.ramp, 0.0)
        self.started = time.monotonic()
        self.reset_window()

    def reset_window(self):
        self.ep = dict(n=0, failed=0, ret=0.0, cost=0.0, goals=0, steps=0)

    def _on_training_start(self):
        self.training_env.env_method("set_price", self.price_now)
        self.acc = np.zeros(self.training_env.num_envs)

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            self.acc[i] += info["utility"]
            if "episode_failed" in info:
                e = self.ep
                e["n"] += 1; e["failed"] += int(info["episode_failed"]); e["ret"] += self.acc[i]; e["cost"] += info["episode_cost"]
                e["goals"] += info["episode_goals"]; e["steps"] += info["episode_steps"]
                self.acc[i] = 0.0
        return True

    def evaluate(self):
        k = len(self.evals)
        venv = self.model.get_vec_normalize_env()
        eps = self.evaluator.episodes(self.model, venv, self.cfg["trace_eval_n"], [self.seed, 7, k])
        row = dict(steps=int(self.num_timesteps), price_in_force=self.price_now, **summarize(eps, self.cfg["objective_price"]))
        self.evals.append(row)

    def _on_rollout_end(self):
        st, e = self.st, self.ep
        mean = lambda key: (e[key] / e["n"] if e["n"] else float("nan"))
        row = dict(steps=int(self.num_timesteps), price=self.price_now, kappa=value_at(self.kappa_sched, st.fraction),
                   lr=float(self.model.policy.optimizer.param_groups[0]["lr"]), episodes=e["n"], failure_rate=mean("failed"), mean_return=mean("ret"),
                   mean_cost=mean("cost"), mean_goals=mean("goals"), mean_steps=mean("steps"),
                   action_std=float(self.model.policy.log_std.detach().exp().mean()))
        if self.theta0 is not None:
            row["distance"] = distance(self.model.policy, self.theta0)
        if st.n:
            g = st.g / st.n
            row.update(g_mean_norm=float(g.norm()), g_step_norm=st.g_norm / st.n)
            if st.pull is not None:
                pull = st.pull / st.n
                row.update(pull_mean_norm=float(pull.norm()), residual_norm=float((g + pull).norm()))
        self.trace.append(row)
        st.reset_window(); self.reset_window()
        if self.num_timesteps >= self.next_eval:
            self.evaluate(); self.next_eval += self.cfg["eval_every"]
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "progress.json").write_text(json.dumps(dict(steps=int(self.num_timesteps), wall=time.monotonic() - self.started,
                                                                    evals=self.evals, trace=self.trace), allow_nan=True) + "\n")
        if self.num_timesteps >= self.next_copy:
            save_learner(self.model, self.out / "copies", f"-{self.next_copy // 1_000_000}M"); self.next_copy += self.cfg["copy_every"]
        st.fraction = self.num_timesteps / self.total          # the price and strength of the next rollout and of the updates before it
        self.price_now = price_at(self.price_sched, self.ramp, st.fraction)
        self.training_env.env_method("set_price", self.price_now)


def save_learner(sb3, folder, tag=""):
    """The model and its normalization, the latter with plain statistics so that it loads without this module."""
    folder.mkdir(parents=True, exist_ok=True)
    venv = sb3.get_vec_normalize_env()
    kept = venv.obs_rms
    if isinstance(kept, FixedMeanStd):
        plain = RunningMeanStd(shape=kept.mean.shape); plain.mean, plain.var, plain.count = kept.mean.copy(), kept.var.copy(), kept.count
        venv.obs_rms = plain
    sb3.save(folder / f"model{tag}.zip"); venv.save(folder / f"vecnormalize{tag}.pkl")
    venv.obs_rms = kept


def settings(stage_name, config_name):
    study = json.loads(STUDY.read_text())
    stage = study["stages"][stage_name]
    cfg = dict(study["defaults"]); cfg.update(stage.get("defaults", {})); cfg.update(stage["configurations"][config_name])
    return study, stage, cfg


def run(stage_name, config_name, seed):
    study, stage, cfg = settings(stage_name, config_name)
    hp = study["learner"]
    out = HERE / "safety" / stage_name / f"{config_name}__seed{seed}"
    if (out / "result.json").exists():
        print("exists", out); return
    steps, lr0 = int(cfg["steps"]), float(cfg.get("learning_rate", hp["learning_rate"]))
    lr_fn = (lambda progress: lr0 * progress) if cfg["lr_schedule"] == "linear" else (lambda progress: lr0)
    price_sched, ramp, kappa_sched = parse_schedule(cfg["price"]), float(cfg.get("price_ramp", 0.0)), parse_schedule(cfg.get("kappa", "0"))
    cfg.setdefault("objective_price", price_sched[-1][1])          # evaluations report the objective at the final price
    venv = DummyVecEnv([lambda: StudyGoal(BUDGET, "failure", 0.0, BETA) for _ in range(hp["n_envs"])])
    init, theta0, source = cfg.get("init"), None, None
    if init is None:
        assert all(v == 0 for _, v in kappa_sched), "the anchor needs a controller to continue from"
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_reward=1e9, gamma=BETA)
        sb3 = PPO("MlpPolicy", venv, seed=seed, n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"], learning_rate=lr_fn,
                  gamma=BETA, gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"], ent_coef=hp["ent_coef"], max_grad_norm=hp["max_grad_norm"],
                  policy_kwargs=dict(net_arch=hp["net_arch"]), device="cpu", verbose=0)
        venv.seed(100 * seed)                                      # environment seeds 100 s, ..., 100 s + 7: no overlap between the seeds of a stage
    else:
        src = HERE / "safety" / init["stage"] / f"{init['config']}__seed{seed}"
        tag = f"-{init['steps'] // 1_000_000}M" if init.get("steps") else ""
        source = dict(model=str((src / ("copies" if tag else "") / f"model{tag}.zip").relative_to(HERE)),
                      normalization=str((src / ("copies" if tag else "") / f"vecnormalize{tag}.pkl").relative_to(HERE)))
        venv = VecNormalize.load(str(HERE / source["normalization"]), venv)
        venv.training, venv.norm_reward = True, True
        if cfg.get("fix_observation_normalization", True):
            fixed = FixedMeanStd(shape=venv.obs_rms.mean.shape); fixed.mean, fixed.var, fixed.count = venv.obs_rms.mean.copy(), venv.obs_rms.var.copy(), venv.obs_rms.count
            venv.obs_rms = fixed
        sb3 = PPO.load(str(HERE / source["model"]), env=venv, device="cpu", seed=seed + 500_000, custom_objects=dict(learning_rate=lr_fn, lr_schedule=lr_fn))
        theta0 = [p.detach().clone() for p in actor_params(sb3.policy)]
        venv.seed(100 * seed + 50)
    st = State()
    install_hook(sb3, theta0, st, kappa_sched)
    cb = Trace(st, cfg, steps, theta0, seed, out, (price_sched, ramp, kappa_sched), Evaluator())
    started = time.monotonic()
    sb3.learn(total_timesteps=steps, callback=cb)
    train_wall = time.monotonic() - started
    save_learner(sb3, out)
    t1 = time.monotonic()
    eps = cb.evaluator.episodes(sb3, sb3.get_vec_normalize_env(), cfg["eval_n"], [seed, 11])
    result = dict(stage=stage_name, config=config_name, settings=cfg, learner=hp, seed=seed, source=source, steps=int(sb3.num_timesteps),
                  task=dict(task=TASK, H=H, budget=BUDGET, beta=BETA), train_wall=train_wall, eval_wall=time.monotonic() - t1,
                  versions=dict(sb3=stable_baselines3.__version__, torch=torch.__version__, python=sys.version.split()[0]),
                  final=summarize(eps, cfg["objective_price"]), action_std=float(sb3.policy.log_std.detach().exp().mean()),
                  distance=(distance(sb3.policy, theta0) if theta0 is not None else None), n_actor_params=sum(p.numel() for p in actor_params(sb3.policy)),
                  evals=cb.evals, trace=cb.trace, episodes=eps)
    (out / "result.json").write_text(json.dumps(result, allow_nan=True) + "\n")
    f = result["final"]
    print(f"{stage_name} {config_name} seed {seed}: exhausted={f['exhausted']:.3f} qt={f['q_tilde']:.4f} U={f['U']:.2f} objective={f['objective']:.2f} "
          f"return={f['mean_return']:.2f} goals={f['mean_goals']:.2f} train={train_wall:.0f}s eval={result['eval_wall']:.0f}s")


def launch(stage_name, jobs):
    stage = json.loads(STUDY.read_text())["stages"][stage_name]
    todo = [(c, s) for s in stage["seeds"] for c in stage["configurations"]
            if not (HERE / "safety" / stage_name / f"{c}__seed{s}" / "result.json").exists()]
    (HERE / "logs").mkdir(exist_ok=True)

    def one(cs):
        c, s = cs
        with open(HERE / "logs" / f"study-v76_{stage_name}_{c}_{s}.log", "a") as log:
            return subprocess.run([sys.executable, str(Path(__file__).resolve()), "run", "--stage", stage_name, "--config", c, "--seed", str(s)],
                                  stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")).returncode
    with ThreadPoolExecutor(jobs) as pool:
        codes = list(pool.map(one, todo))
    print(f"stage {stage_name}: {len(todo)} runs, return codes {codes}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "stage"])
    ap.add_argument("--stage", required=True); ap.add_argument("--config"); ap.add_argument("--seed", type=int); ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "run":
        run(a.stage, a.config, a.seed)
    else:
        launch(a.stage, a.jobs)
