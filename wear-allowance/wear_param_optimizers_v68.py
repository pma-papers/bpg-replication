"""Revision 68: the regularizers on PPO under three optimizers (PROTOCOL-optimizers-v68.md).

Same problem, start, PPO settings, seeds and evaluation as the roles study of revision 60 (wear_param_noise_v60.py, which
reuses the revision-41 runner). This runner copies the revision-41 run() with the optimizer as an argument:

  sgd              plain stochastic gradient descent, no momentum, learning rate --lr decayed linearly to zero. The pull
                   kappa_2 (theta - theta_0) is added to the actor gradient after gradient clipping, so the step on the
                   pull is exactly lr_t kappa_2 (theta - theta_0): the Euler step of the particle form. Noise of variance
                   2 kappa_1 lr_t after the step.
  adam-coupled     Adam (eps 1e-5, the learner's default), the pull added to the gradient before the step (the paper's
                   recorded form); noise after the step. Identical in arithmetic to wear_param_noise_v60.py.
  adam-decoupled   Adam, the pull applied to the parameters after the step, theta <- theta - lr_t kappa_2 (theta -
                   theta_0), then the noise. Identical in arithmetic to wear_param_noise_decoupled_v68.py.

--zero-grad replaces the objective's gradient (actor, critic and dispersion) by zero before the pull is added, so that
pull and noise alone drive the parameters through the optimizer, with the environment still running (a check of the
direct simulation regularizer_dynamics_v68.py). One optimizer over actor, critic and dispersion, as in the learner.
The noise comes from a torch.Generator seeded with 1_000_000 + seed. Runtime: ~/.venvs/constrained-rl-numerics.

  python wear_param_optimizers_v68.py --optimizer sgd --lr 0.01 --kappa1 0.101 --kappa2 10 --seed 94001
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import stable_baselines3  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
import wear_param_anchor as W  # noqa: E402
from env import Incumbent, WearEnv, model_policy  # noqa: E402
from model import Model, simulate  # noqa: E402
from train import Control, stopped_action_kl  # noqa: E402

torch.set_num_threads(1)
HOOK = {"steps": 0, "lr_sum": 0.0, "lr_max": 0.0, "noise_var_sum": 0.0, "grad_norm_sum": 0.0, "g2": None}


def install_hook(sb3, theta0, arm, seed):
    params = W.actor_params(sb3.policy, arm["anchor_logstd"])
    all_params = list(sb3.policy.parameters())
    opt = sb3.policy.optimizer
    base_step = opt.step
    gen = torch.Generator().manual_seed(1_000_000 + seed)
    k1, k2, form, zero = arm["kappa_1"], arm["kappa_p"], arm["optimizer"], arm["zero_grad"]
    assert arm["lr0"] * k2 < 1.0, "lr kappa_2 must be below one"

    def step(closure=None):
        with torch.no_grad():
            lr = max(float(opt.param_groups[0]["lr"]), 0.0)
            HOOK["steps"] += 1; HOOK["lr_sum"] += lr; HOOK["lr_max"] = max(HOOK["lr_max"], lr); HOOK["noise_var_sum"] += 2.0 * k1 * lr
            if zero:
                for p in all_params:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                HOOK["grad_norm_sum"] += float(torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None)))
            g2 = torch.cat([(p.grad.reshape(-1) ** 2 if p.grad is not None else torch.zeros(p.numel())) for p in params])
            HOOK["g2"] = g2 if HOOK["g2"] is None else HOOK["g2"] + g2      # per-coordinate sum of squared clipped objective gradients
            if form in ("sgd", "adam-coupled") and k2 > 0:
                for p, p0 in zip(params, theta0):
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.add_(k2 * (p - p0))
        out = base_step(closure)
        with torch.no_grad():
            if form == "adam-decoupled" and k2 > 0 and lr > 0:
                c = lr * k2
                for p, p0 in zip(params, theta0):
                    p.sub_(c * (p - p0))
            if k1 > 0 and lr > 0:
                std = math.sqrt(2.0 * k1 * lr)
                for p in params:
                    p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * std)
        return out
    opt.step = step


def run(arm, seed, steps, out, eval_n, hp):
    model = Model(**arm.get("model", {}))
    incumbent = Incumbent(model, **arm.get("incumbent", {}))
    n_envs = hp.get("n_envs", 32)
    venv = DummyVecEnv([lambda: WearEnv(model, "failure", arm["m"]) for _ in range(n_envs)])
    venv.seed(seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=1e9, gamma=1.0)
    lr0 = arm["lr0"]
    sb3 = PPO("MlpPolicy", venv, seed=seed, n_steps=hp.get("n_steps", 256), batch_size=hp.get("batch_size", 1024),
              n_epochs=hp.get("n_epochs", 10), learning_rate=(lambda progress: lr0 * progress), gamma=1.0, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64], log_std_init=hp.get("log_std_init", -1.0)), device="cpu", verbose=0)
    fit_mse = W.fit_incumbent(sb3, incumbent, model, seed)
    with_logstd = arm["anchor_logstd"]
    theta0 = [p.detach().clone() for p in W.actor_params(sb3.policy, with_logstd)]
    if arm["optimizer"] == "sgd":                       # fresh optimizer state after the fit, as in the revision-41 runner
        sb3.policy.optimizer = torch.optim.SGD(sb3.policy.parameters(), lr=lr0, momentum=0.0)
    else:
        sb3.policy.optimizer = torch.optim.Adam(sb3.policy.parameters(), lr=lr0, eps=1e-5)
    prior_policy = dict(stochastic=simulate(model, model_policy(model, sb3, False), n=eval_n, seed=60_000 + seed, m=arm["m"]),
                        mean_action=simulate(model, model_policy(model, sb3, True), n=eval_n, seed=61_000 + seed, m=arm["m"]),
                        action_kl=stopped_action_kl(model, sb3, incumbent, n=eval_n, seed=62_000 + seed), fit_mse=fit_mse)
    install_hook(sb3, theta0, arm, seed)
    control = Control(dict(kind="fixed"), None)
    started = time.monotonic()
    sb3.learn(total_timesteps=steps, callback=control)
    train_wall = time.monotonic() - started
    d = W.distance(sb3, theta0, with_logstd)
    n_par = sum(p.numel() for p in W.actor_params(sb3.policy, with_logstd))
    k1, k2 = arm["kappa_1"], arm["kappa_p"]
    result = dict(arm=arm, seed=seed, steps=int(sb3.num_timesteps), train_wall=train_wall, prior_policy=prior_policy,
                  log_std=float(sb3.policy.log_std.detach().numpy().reshape(-1)[0]), distance=d, n_actor_params=n_par, spread=d / math.sqrt(n_par),
                  predicted_spread=dict(particle_form=(math.sqrt(k1 / k2) if k1 > 0 and k2 > 0 else (0.0 if k1 == 0 else None)),
                                        adam_laplace_coupled=math.sqrt(2.0) * k1),
                  hook=dict(form=arm["optimizer"], zero_grad=arm["zero_grad"], order=("gradient (clipped), pull in the gradient, step, noise" if arm["optimizer"] != "adam-decoupled" else "gradient (clipped), step, pull, noise"),
                            optimizer=("SGD, no momentum" if arm["optimizer"] == "sgd" else "Adam, eps 1e-5") + ", one optimizer over actor, critic and dispersion",
                            lr0=lr0, schedule="linear to zero, updated once per rollout, clipped at zero in the hook", noise_generator=f"torch.Generator seeded {1_000_000 + seed}",
                            optimizer_steps=HOOK["steps"], lr_sum_flow_time=HOOK["lr_sum"], lr_max=HOOK["lr_max"], contraction_per_step_at_lr_max=1.0 - HOOK["lr_max"] * k2,
                            integrated_noise_variance_per_parameter=HOOK["noise_var_sum"], mean_clipped_actor_grad_norm=(HOOK["grad_norm_sum"] / HOOK["steps"] if HOOK["steps"] and not arm["zero_grad"] else None),
                            versions=dict(sb3=stable_baselines3.__version__, torch=torch.__version__, python=sys.version.split()[0])),
                  stochastic=simulate(model, model_policy(model, sb3, False), n=eval_n, seed=10_000 + seed, m=arm["m"]),
                  mean_action=simulate(model, model_policy(model, sb3, True), n=eval_n, seed=20_000 + seed, m=arm["m"]),
                  action_kl=stopped_action_kl(model, sb3, incumbent, n=eval_n, seed=50_000 + seed))
    out.mkdir(parents=True, exist_ok=True)
    sb3.save(out / "final_model.zip")
    # per-coordinate record: theta_0, final |theta - theta_0|, the sum of squared clipped objective gradients, and Adam's
    # second-moment estimate (exp_avg_sq, which under the coupled form includes the pull) for every actor coordinate
    import numpy as np
    params = W.actor_params(sb3.policy, with_logstd)
    with torch.no_grad():
        x = torch.cat([(p - p0).reshape(-1) for p, p0 in zip(params, theta0)]).numpy()
        t0 = torch.cat([p0.reshape(-1) for p0 in theta0]).numpy()
        st = sb3.policy.optimizer.state
        v = (torch.cat([st[p]["exp_avg_sq"].reshape(-1) if p in st and "exp_avg_sq" in st[p] else torch.full((p.numel(),), float("nan")) for p in params]).numpy()
             if arm["optimizer"] != "sgd" else np.full(x.shape, np.nan))
    np.savez_compressed(out / "coordinates.npz", theta0=t0, x=x, g2_sum=(HOOK["g2"].numpy() if HOOK["g2"] is not None else np.zeros_like(x)),
                        adam_v=v, steps=HOOK["steps"], names=np.array([f"{i}:{tuple(p.shape)}" for i, p in enumerate(params)]))
    (out / "result.json").write_text(json.dumps(result, indent=1, allow_nan=True) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["sgd", "adam-coupled", "adam-decoupled"], required=True)
    ap.add_argument("--lr", type=float, required=True, help="initial learning rate of the linear schedule")
    ap.add_argument("--kappa1", type=float, required=True); ap.add_argument("--kappa2", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--stage", default="campaign")
    ap.add_argument("--steps", type=int, default=8_000_000); ap.add_argument("--eval", type=int, default=20_000)
    ap.add_argument("--zero-grad", action="store_true")
    args = ap.parse_args()
    arm = dict(kind=f"param-{args.optimizer}", m=0.03, kappa_p=args.kappa2, kappa_1=args.kappa1, model=dict(wear_shape=0.5), incumbent=dict(span=3.0, std=0.1),
               anchor_logstd=True, prior_logstd=math.log(0.1), optimizer=args.optimizer, lr0=args.lr, zero_grad=bool(args.zero_grad))
    out = HERE / "wear-optimizers-v68" / args.stage / f"{args.optimizer}__lr{args.lr:g}__k1_{args.kappa1:g}__k2_{args.kappa2:g}{'__zerograd' if args.zero_grad else ''}__seed{args.seed}"
    if (out / "result.json").exists():
        print("exists", out); sys.exit(0)
    r = run(arm, args.seed, args.steps, out, args.eval, dict(n_envs=32, n_steps=256, batch_size=1024, n_epochs=10, log_std_init=math.log(0.1)))
    s = r["stochastic"]; pred = r["predicted_spread"]["particle_form"]
    print(f"{args.optimizer} lr={args.lr:g} kappa1={args.kappa1:g} kappa2={args.kappa2:g}{' zero-grad' if args.zero_grad else ''} seed={args.seed}: "
          f"risk={s['risk']:.4f} U={s['U']:.3f} A={r['action_kl']['A']:.3f} dist={r['distance']:.3f} spread={r['spread']:.4f} "
          f"predicted={pred if pred is None else round(pred, 4)} logstd={r['log_std']:.2f} opt_steps={r['hook']['optimizer_steps']} flow_time={r['hook']['lr_sum_flow_time']:.2f} train={r['train_wall']:.0f}s", flush=True)
