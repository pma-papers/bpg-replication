"""Revision 68: the zero-objective control of PROTOCOL-optimizers-v68.md, run directly on the actor's parameters.

With the objective's gradient replaced by zero, a PPO run of wear_param_optimizers_v68.py --zero-grad moves the
parameters by the pull and the noise through the optimizer alone; the environment, the rollouts and the critic have no
effect on them. This script reproduces that run without the environment: it builds the same learner from the same seed
(the fit of theta_0 to the incumbent rule included), replaces the optimizer as the runner does, and then applies the
same number of optimizer steps with the same learning-rate schedule (80 steps per rollout of 8192 environment steps,
lr_k = lr_0 (1 - k 8192 / steps) set once per rollout as Stable-Baselines3 does, negative on the last rollout, clipped at
zero in the pull and the noise as in the runner), the same gradient (zero for the objective, plus the pull in the
gradient for sgd and adam-coupled), the same post-step pull for adam-decoupled, and the same noise from the same
generator. It then evaluates the resulting actor as the runner does (stochastic and mean-action policy on 20,000
episodes, the action relative entropy to the incumbent) and the prior policy, the reference point of this control.
The per-rollout spread is recorded. A --zero-grad run of the runner with the same arguments reproduces it (checked on
the smoke stage). Runtime: ~/.venvs/constrained-rl-numerics/bin/python.

  python regularizer_dynamics_v68.py --optimizer adam-coupled --lr 3e-4 --kappa1 0.101 --kappa2 10 --seed 94001
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import stable_baselines3  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
import wear_param_anchor as W  # noqa: E402
from env import Incumbent, WearEnv, model_policy  # noqa: E402
from model import Model, simulate  # noqa: E402
from train import stopped_action_kl  # noqa: E402

torch.set_num_threads(1)
ROLLOUT, STEPS_PER_ROLLOUT = 32 * 256, 10 * (32 * 256 // 1024)


def run(arm, seed, steps, out, eval_n):
    model = Model(**arm["model"]); incumbent = Incumbent(model, **arm["incumbent"])
    venv = DummyVecEnv([lambda: WearEnv(model, "failure", arm["m"]) for _ in range(32)]); venv.seed(seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=1e9, gamma=1.0)
    lr0 = arm["lr0"]
    sb3 = PPO("MlpPolicy", venv, seed=seed, n_steps=256, batch_size=1024, n_epochs=10, learning_rate=(lambda progress: lr0 * progress), gamma=1.0, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64], log_std_init=arm["prior_logstd"]), device="cpu", verbose=0)
    fit_mse = W.fit_incumbent(sb3, incumbent, model, seed)
    params = W.actor_params(sb3.policy, arm["anchor_logstd"]); all_params = list(sb3.policy.parameters())
    theta0 = [p.detach().clone() for p in params]
    opt = (torch.optim.SGD(all_params, lr=lr0, momentum=0.0) if arm["optimizer"] == "sgd" else torch.optim.Adam(all_params, lr=lr0, eps=1e-5))
    prior_policy = dict(stochastic=simulate(model, model_policy(model, sb3, False), n=eval_n, seed=60_000 + seed, m=arm["m"]),
                        mean_action=simulate(model, model_policy(model, sb3, True), n=eval_n, seed=61_000 + seed, m=arm["m"]),
                        action_kl=stopped_action_kl(model, sb3, incumbent, n=eval_n, seed=62_000 + seed), fit_mse=fit_mse)
    gen = torch.Generator().manual_seed(1_000_000 + seed)
    k1, k2, form = arm["kappa_1"], arm["kappa_p"], arm["optimizer"]
    n_roll = math.ceil(steps / ROLLOUT)
    d = sum(p.numel() for p in params)
    trace = []; lr_sum = 0.0; n_steps = 0; t0 = time.monotonic()
    for k in range(1, n_roll + 1):
        lr_k = lr0 * (1.0 - k * ROLLOUT / steps)           # Stable-Baselines3's progress_remaining after the k-th rollout
        for g in opt.param_groups:
            g["lr"] = lr_k
        for _ in range(STEPS_PER_ROLLOUT):
            with torch.no_grad():
                lr = max(lr_k, 0.0); lr_sum += lr; n_steps += 1
                for p in all_params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    else:
                        p.grad.zero_()
                if form in ("sgd", "adam-coupled") and k2 > 0:
                    for p, p0 in zip(params, theta0):
                        p.grad.add_(k2 * (p - p0))
            opt.step()
            with torch.no_grad():
                if form == "adam-decoupled" and k2 > 0 and lr > 0:
                    c = lr * k2
                    for p, p0 in zip(params, theta0):
                        p.sub_(c * (p - p0))
                if k1 > 0 and lr > 0:
                    std = math.sqrt(2.0 * k1 * lr)
                    for p in params:
                        p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * std)
        with torch.no_grad():
            trace.append(float(torch.sqrt(sum(((p - p0) ** 2).sum() for p, p0 in zip(params, theta0)))) / math.sqrt(d))
    dyn_wall = time.monotonic() - t0
    dist = W.distance(sb3, theta0, arm["anchor_logstd"])
    with torch.no_grad():
        x = torch.cat([(p - p0).reshape(-1) for p, p0 in zip(params, theta0)]).numpy()
    # the stationary law per coordinate: Gaussian (kurtosis 3) under a gradient pull, Laplace (kurtosis 6) under the coupled Adam pull
    kurt = float(np.mean(x ** 4) / np.mean(x ** 2) ** 2) if np.mean(x ** 2) > 0 else float("nan")
    result = dict(arm=arm, seed=seed, steps=int(n_roll * ROLLOUT), optimizer_steps=n_steps, dynamics_wall=dyn_wall, prior_policy=prior_policy, control="zero objective, direct simulation",
                  log_std=float(sb3.policy.log_std.detach().numpy().reshape(-1)[0]), distance=dist, n_actor_params=d, spread=dist / math.sqrt(d), spread_trace_per_rollout=trace,
                  kurtosis=kurt, mean_abs_over_rms=float(np.mean(np.abs(x)) / np.sqrt(np.mean(x ** 2))) if np.mean(x ** 2) > 0 else float("nan"),
                  predicted_spread=dict(particle_form=(math.sqrt(k1 / k2) if k1 > 0 and k2 > 0 else (0.0 if k1 == 0 else None)), adam_laplace_coupled=math.sqrt(2.0) * k1),
                  hook=dict(form=form, lr0=lr0, lr_sum_flow_time=lr_sum, noise_generator=f"torch.Generator seeded {1_000_000 + seed}",
                            versions=dict(sb3=stable_baselines3.__version__, torch=torch.__version__, python=sys.version.split()[0])),
                  stochastic=simulate(model, model_policy(model, sb3, False), n=eval_n, seed=10_000 + seed, m=arm["m"]),
                  mean_action=simulate(model, model_policy(model, sb3, True), n=eval_n, seed=20_000 + seed, m=arm["m"]),
                  action_kl=stopped_action_kl(model, sb3, incumbent, n=eval_n, seed=50_000 + seed))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "coordinates.npz", x=x, theta0=torch.cat([p0.reshape(-1) for p0 in theta0]).numpy())
    (out / "result.json").write_text(json.dumps(result, indent=1, allow_nan=True) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["sgd", "adam-coupled", "adam-decoupled"], required=True); ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--kappa1", type=float, required=True); ap.add_argument("--kappa2", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--stage", default="campaign")
    ap.add_argument("--steps", type=int, default=8_000_000); ap.add_argument("--eval", type=int, default=20_000)
    args = ap.parse_args()
    arm = dict(kind=f"zero-objective-{args.optimizer}", m=0.03, kappa_p=args.kappa2, kappa_1=args.kappa1, model=dict(wear_shape=0.5), incumbent=dict(span=3.0, std=0.1),
               anchor_logstd=True, prior_logstd=math.log(0.1), optimizer=args.optimizer, lr0=args.lr, zero_grad=True)
    out = HERE / "wear-optimizers-v68" / args.stage / f"{args.optimizer}__lr{args.lr:g}__k1_{args.kappa1:g}__k2_{args.kappa2:g}__zerograd-direct__seed{args.seed}"
    if (out / "result.json").exists():
        print("exists", out); sys.exit(0)
    r = run(arm, args.seed, args.steps, out, args.eval)
    s, pp = r["stochastic"], r["prior_policy"]["stochastic"]; pred = r["predicted_spread"]["particle_form"]
    print(f"zero-objective {args.optimizer} lr={args.lr:g} kappa1={args.kappa1:g} kappa2={args.kappa2:g} seed={args.seed}: spread={r['spread']:.4f} "
          f"predicted={pred if pred is None else round(pred, 4)} kurtosis={r['kurtosis']:.2f} risk={s['risk']:.4f} U={s['U']:.3f} A={r['action_kl']['A']:.3f} "
          f"| prior: risk={pp['risk']:.4f} U={pp['U']:.3f} | opt_steps={r['optimizer_steps']} dyn={r['dynamics_wall']:.0f}s", flush=True)
