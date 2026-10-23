"""Parameter anchoring in PPO for the wear-allowance example (revision 39; runtime ~/.venvs/constrained-rl-numerics/bin/python).

The weight-decay part of relative entropy to a Gaussian prior N(theta_0, sigma^2 I)
(Proposition prop:prior) applied to a finite actor: the gradient kappa_p (theta - theta_0) is
added to the actor's gradient before every optimizer step (anchoring in parameter space),
at the mispriced failure risk m = 0.03 of the action-anchored arms of the wear campaign, so the
two anchors can be compared. theta_0 are the parameters of the campaign's actor architecture
fitted by least squares to the incumbent rule's mean action; the actor starts at theta_0 with
the campaign's exploration dispersion (log_std_init = -1), and log_std is not anchored. The
prior policy, the actor at theta_0 with that dispersion, is evaluated like every other policy.
Everything else (environment, PPO settings, evaluation) is the wear campaign's (wear-allowance).
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
sys.path.insert(0, str(ROOT.parent / "wear-allowance"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from env import Incumbent, WearEnv, model_policy, observe, to_rate  # noqa: E402
from model import Model, simulate  # noqa: E402
from train import Control, stopped_action_kl  # noqa: E402

torch.set_num_threads(1)


def actor_params(policy, with_logstd=True):
    """All parameters that shape the policy: the mean network and, unless excluded, the dispersion."""
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    return params + [policy.log_std] if with_logstd else params


def actor_mean(policy, obs):
    feats = policy.extract_features(obs, policy.pi_features_extractor)
    return policy.action_net(policy.mlp_extractor.forward_actor(feats))


def fit_incumbent(sb3, incumbent, model, seed, n=20000, steps=4000):
    """Least-squares fit of the actor's mean (pre-clipping variable) to the incumbent rule."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1.0, 1.0, n); z = rng.uniform(0.0, model.budget, n)
    obs = torch.as_tensor(observe(model, s, z))
    target = torch.as_tensor(incumbent.mean(z), dtype=torch.float32).reshape(-1, 1)
    params = actor_params(sb3.policy, with_logstd=False)
    opt = torch.optim.Adam(params, lr=1e-3)
    for it in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, 512))
        loss = ((actor_mean(sb3.policy, obs[idx]) - target[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        final = float(((actor_mean(sb3.policy, obs) - target) ** 2).mean())
    return final


def anchor(sb3, theta0, kappa, with_logstd):
    """Add kappa (theta - theta_0) to the actor's gradient before each optimizer step."""
    params = actor_params(sb3.policy, with_logstd)
    opt = sb3.policy.optimizer
    base_step = opt.step

    def step(closure=None):
        with torch.no_grad():
            for p, p0 in zip(params, theta0):
                if p.grad is not None:
                    p.grad.add_(kappa * (p - p0))
        return base_step(closure)

    opt.step = step


def distance(sb3, theta0, with_logstd):
    with torch.no_grad():
        return float(torch.sqrt(sum(((p - p0) ** 2).sum() for p, p0 in zip(actor_params(sb3.policy, with_logstd), theta0))))


def run(arm, seed, steps, out, eval_n, hp):
    model = Model(**arm.get("model", {}))
    incumbent = Incumbent(model, **arm.get("incumbent", {}))
    n_envs = hp.get("n_envs", 32)
    venv = DummyVecEnv([lambda: WearEnv(model, "failure", arm["m"]) for _ in range(n_envs)])
    venv.seed(seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=1e9, gamma=1.0)
    sb3 = PPO("MlpPolicy", venv, seed=seed, n_steps=hp.get("n_steps", 256), batch_size=hp.get("batch_size", 1024),
              n_epochs=hp.get("n_epochs", 10), learning_rate=(lambda progress: 3e-4 * progress), gamma=1.0, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0), device="cpu", verbose=0)
    fit_mse = fit_incumbent(sb3, incumbent, model, seed)
    with_logstd = arm.get("anchor_logstd", True)
    theta0 = [p.detach().clone() for p in actor_params(sb3.policy, with_logstd)]
    sb3.policy.optimizer = type(sb3.policy.optimizer)(sb3.policy.parameters(), lr=3e-4, eps=1e-5)   # fresh optimizer state after the fit
    prior_policy = dict(stochastic=simulate(model, model_policy(model, sb3, False), n=eval_n, seed=60_000 + seed, m=arm["m"]),
                        mean_action=simulate(model, model_policy(model, sb3, True), n=eval_n, seed=61_000 + seed, m=arm["m"]),
                        action_kl=stopped_action_kl(model, sb3, incumbent, n=eval_n, seed=62_000 + seed), fit_mse=fit_mse)
    if arm["kappa_p"] > 0:
        anchor(sb3, theta0, arm["kappa_p"], with_logstd)
    control = Control(dict(kind="fixed"), None)
    started = time.monotonic()
    sb3.learn(total_timesteps=steps, callback=control)
    train_wall = time.monotonic() - started
    result = dict(arm=arm, seed=seed, steps=int(sb3.num_timesteps), train_wall=train_wall, prior_policy=prior_policy,
                  log_std=float(sb3.policy.log_std.detach().numpy().reshape(-1)[0]), distance=distance(sb3, theta0, with_logstd),
                  stochastic=simulate(model, model_policy(model, sb3, False), n=eval_n, seed=10_000 + seed, m=arm["m"]),
                  mean_action=simulate(model, model_policy(model, sb3, True), n=eval_n, seed=20_000 + seed, m=arm["m"]),
                  action_kl=stopped_action_kl(model, sb3, incumbent, n=eval_n, seed=50_000 + seed))
    out.mkdir(parents=True, exist_ok=True)
    sb3.save(out / "final_model.zip")
    (out / "result.json").write_text(json.dumps(result, indent=1, allow_nan=True) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", default="campaign"); ap.add_argument("--steps", type=int, default=8_000_000); ap.add_argument("--eval", type=int, default=20_000)
    ap.add_argument("--mean-only", action="store_true", help="anchor the mean network only (the dispersion stays free)")
    args = ap.parse_args()
    arm = dict(kind="param-anchored", m=0.03, kappa_p=args.kappa, model=dict(wear_shape=0.5), incumbent=dict(span=3.0, std=0.1), anchor_logstd=not args.mean_only)
    out = ROOT / "wear-anchor" / args.stage / f"kappa{args.kappa:g}__seed{args.seed}"
    if (out / "result.json").exists():
        print("exists", out); sys.exit(0)
    r = run(arm, args.seed, args.steps, out, args.eval, dict(n_envs=32, n_steps=256, batch_size=1024, n_epochs=10))
    s, pp = r["stochastic"], r["prior_policy"]["stochastic"]
    print(f"kappa_p={args.kappa:g} seed={args.seed}: risk={s['risk']:.4f} U={s['U']:.3f} A={r['action_kl']['A']:.3f} dist={r['distance']:.3f} "
          f"| prior policy: risk={pp['risk']:.4f} U={pp['U']:.3f} A={r['prior_policy']['action_kl']['A']:.3f} fit_mse={r['prior_policy']['fit_mse']:.2e} train={r['train_wall']:.0f}s")
