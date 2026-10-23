"""Follow-up to the benchmark-suite campaign (protocol PROTOCOL-ramp-v65.md): a fixed price that is raised
linearly from zero to its final value over the first third of training ("ramped"), lower fixed prices, and an
evaluation that records the failure step of every episode, so that the priced risk and the stopped utility are
measured exactly. The task, wrapper, learner and hyperparameters are those of safety_pilot_v63.py, which is
imported unchanged.

The wrapper's charge at the exhausting step t is m(1 - beta^(H-t))/(1 - beta), so the training objective is
  U - nu * qt,   nu = m/(1 - beta),   qt = E[(beta^tau - beta^H) 1{tau <= H-1}],
the frozen-multiplier form of the paper's identity with no interior barrier (I_m = 0 and no clipping before
failure); Proposition 3.3(i) holds for it with qt in place of the geometric risk.

Usage (runtime ~/.venvs/constrained-rl-safety310/bin/python):
  safety_ramp_v65.py train --kind ramp|fixed --m M --seed S [--steps 2000000] [--stage campaign-v65]
  safety_ramp_v65.py reeval --run DIR      (re-evaluates a saved v63 run with failure times; writes DIR/eval-times-v65.json)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import safety_pilot_v63 as V63   # noqa: E402  (wrapper, task, horizon: unchanged)
from safety_pilot_v63 import BudgetGoal, H, TASK  # noqa: E402

BETA, BUDGET, EVAL_N = 0.999, 25.0, 200


class PricedGoal(BudgetGoal):
    """The v63 wrapper with the fixed-price objective and a price that the training callback may change."""
    def set_price(self, value):
        self.m = float(value)


class Ramp(BaseCallback):
    """Sets the price after every rollout: m_final * min(1, steps / ramp_steps); ramp_steps = 0 means fixed."""
    def __init__(self, m_final, ramp_steps):
        super().__init__(verbose=0)
        self.m_final, self.ramp_steps = float(m_final), int(ramp_steps)
        self.episodes = self.failures = 0; self.cost_sum = 0.0; self.trace = []

    def price(self, steps):
        return self.m_final if self.ramp_steps <= 0 else self.m_final * min(1.0, steps / self.ramp_steps)

    def _on_training_start(self):
        self.training_env.env_method("set_price", self.price(0))

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode_failed" in info:
                self.episodes += 1; self.failures += int(info["episode_failed"]); self.cost_sum += float(info.get("episode_cost", 0.0))
        return True

    def _on_rollout_end(self):
        m_now = self.price(self.num_timesteps)
        self.training_env.env_method("set_price", m_now)
        self.trace.append(dict(steps=int(self.num_timesteps), price=m_now, episodes=self.episodes,
                               failure_rate=self.failures / self.episodes if self.episodes else float("nan"),
                               mean_cost=self.cost_sum / self.episodes if self.episodes else float("nan")))
        self.episodes = self.failures = 0; self.cost_sum = 0.0


def evaluate_times(sb3, envs, *, n, seed, deterministic=False):
    """The loop of safety_pilot_v63.evaluate_norm, with the failure step and the discounted utility of every episode."""
    envs.seed(seed); torch.manual_seed(seed)
    obs = envs.reset()
    eps = []
    ep_ret = ep_cost = ep_util = 0.0; ep_t = 0
    while len(eps) < n:
        act, _ = sb3.predict(obs, deterministic=deterministic)
        obs, rew, dones, infos = envs.step(act)
        info = infos[0]
        ep_ret += info["utility"]; ep_cost += info["cost"]; ep_util += BETA ** ep_t * info["utility"]
        if dones[0]:
            failed = bool(info["episode_failed"])
            eps.append(dict(failed=failed, tau=ep_t if failed else None, ret=ep_ret, cost=ep_cost, U=ep_util))
            ep_ret = ep_cost = ep_util = 0.0; ep_t = 0
        else:
            ep_t += 1
    fails = [e for e in eps if e["failed"]]
    qt = [(BETA ** e["tau"] - BETA ** H) if e["failed"] else 0.0 for e in eps]
    qb = [BETA ** e["tau"] if e["failed"] else 0.0 for e in eps]
    return dict(n=len(eps), exhausted=len(fails) / len(eps), mean_return=float(np.mean([e["ret"] for e in eps])),
                mean_cost=float(np.mean([e["cost"] for e in eps])), stopped_utility=float(np.mean([e["U"] for e in eps])),
                cost_q90=float(np.quantile([e["cost"] for e in eps], 0.9)), q_tilde=float(np.mean(qt)), q_beta=float(np.mean(qb)),
                failure_steps=[e["tau"] for e in fails], episodes=eps)


def eval_envs_for(run_dir):
    envs = VecNormalize.load(run_dir / "vecnormalize.pkl", DummyVecEnv([lambda: BudgetGoal(BUDGET, "failure", 0.0, BETA)]))
    envs.training = False; envs.norm_reward = False
    return envs


def train(kind, m, seed, steps, stage):
    out = HERE / "safety" / stage / f"{kind}_m{m:g}_b{BUDGET:g}__seed{seed}"
    if (out / "result.json").exists():
        print("exists", out); return
    ramp_steps = steps // 3 if kind == "ramp" else 0
    make = lambda: PricedGoal(BUDGET, "failure", m, BETA, 0.0)
    venv = DummyVecEnv([make for _ in range(8)])
    venv.seed(seed)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_reward=1e9, gamma=BETA)
    sb3 = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=64, n_epochs=4, learning_rate=(lambda p: 3e-4 * p),
              gamma=BETA, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5,
              policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    cb = Ramp(m, ramp_steps)
    t0 = time.monotonic(); sb3.learn(total_timesteps=steps, callback=cb); train_wall = time.monotonic() - t0
    out.mkdir(parents=True, exist_ok=True)
    sb3.save(out / "final_model.zip"); venv.save(out / "vecnormalize.pkl")
    t1 = time.monotonic()
    ev = evaluate_times(sb3, eval_envs_for(out), n=EVAL_N, seed=10_000 + seed)
    arm = dict(kind=kind, m=m, ramp_steps=ramp_steps, budget=BUDGET, beta=BETA, task=TASK, H=H)
    res = dict(arm=arm, seed=seed, steps=int(sb3.num_timesteps), train_wall=train_wall, trace=cb.trace,
               stochastic=ev, eval_wall=time.monotonic() - t1)
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: ev[k] for k in ("exhausted", "mean_return", "stopped_utility", "q_tilde", "mean_cost")}))


def reeval(run_dir):
    run_dir = Path(run_dir).resolve()
    dst = run_dir / "eval-times-v65.json"
    if dst.exists():
        print("exists", dst); return
    old = json.loads((run_dir / "result.json").read_text())
    envs = eval_envs_for(run_dir)
    sb3 = PPO.load(run_dir / "final_model.zip", env=envs, device="cpu")
    ev = evaluate_times(sb3, envs, n=EVAL_N, seed=10_000 + old["seed"])
    same = abs(ev["exhausted"] - old["stochastic"]["exhausted"]) < 1e-12 and abs(ev["mean_return"] - old["stochastic"]["mean_return"]) < 1e-9
    dst.write_text(json.dumps(dict(arm=old["arm"], seed=old["seed"], reproduces_v63_evaluation=same, stochastic=ev), indent=1) + "\n")
    print(run_dir.name, "reproduces" if same else "DIFFERS", ev["exhausted"], old["stochastic"]["exhausted"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "reeval"])
    ap.add_argument("--kind", choices=["ramp", "fixed"]); ap.add_argument("--m", type=float); ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int, default=2_000_000); ap.add_argument("--stage", default="campaign-v65"); ap.add_argument("--run")
    a = ap.parse_args()
    if a.mode == "train": train(a.kind, a.m, a.seed, a.steps, a.stage)
    else: reeval(a.run)
