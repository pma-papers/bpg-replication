"""Revision 68: the roles of the regularizers on PPO with the pull DECOUPLED from Adam (PROTOCOL-decoupled-v68.md).

The recorded campaign of revision 60 (wear_param_noise_v60.py) adds the pull kappa_2 (theta - theta_0) to the actor
gradient before the Adam step, so the pull is rescaled by Adam's per-coordinate normalizer. This script keeps every
setting of that campaign (runner, environment, PPO, seeds, evaluation) and replaces the one optimizer hook. After every
optimizer step, in this order,

    theta <- theta - lr_t kappa_2 (theta - theta_0)          the decoupled pull (the weight decay of AdamW)
    theta <- theta + sqrt(2 kappa_1 lr_t) xi,  xi ~ N(0, I)  the noise of the particle form

on every actor parameter (the mean network and the log dispersion), lr_t the current learning rate of the optimizer's
first parameter group, clipped at zero (the linear schedule dips below zero on the last iteration). Nothing is added to
the gradient, so the actor gradient reaches Adam unchanged. The noise is drawn from a dedicated torch.Generator seeded
with 1_000_000 + seed, not from the global generator that Stable-Baselines3 seeds (the revision-60 script drew from
the global one). The recorded scripts are not modified. Runtime: ~/.venvs/constrained-rl-numerics/bin/python.

    python wear_param_noise_decoupled_v68.py --kappa1 0.101 --kappa2 10 --seed 94001
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import stable_baselines3  # noqa: E402
import wear_param_anchor as W  # noqa: E402

CFG = {"kappa1": 0.0, "seed": 0}
HOOK = {"steps": 0, "lr_sum": 0.0, "lr_max": 0.0, "noise_var_sum": 0.0}


def decoupled_hook(sb3, theta0, kappa2, kappa1, with_logstd, seed):
    """Pull on the parameters after the step, then the noise; the gradient is left alone."""
    params = W.actor_params(sb3.policy, with_logstd)
    opt = sb3.policy.optimizer
    base_step = opt.step
    gen = torch.Generator().manual_seed(1_000_000 + seed)
    assert float(opt.param_groups[0]["lr"]) * kappa2 < 1.0, "lr kappa_2 must be below one for the contraction to be a contraction"

    def step(closure=None):
        out = base_step(closure)
        with torch.no_grad():
            lr = max(float(opt.param_groups[0]["lr"]), 0.0)
            HOOK["steps"] += 1; HOOK["lr_sum"] += lr; HOOK["lr_max"] = max(HOOK["lr_max"], lr); HOOK["noise_var_sum"] += 2.0 * kappa1 * lr
            if kappa2 > 0 and lr > 0:
                c = lr * kappa2
                for p, p0 in zip(params, theta0):
                    p.sub_(c * (p - p0))
            if kappa1 > 0 and lr > 0:
                std = math.sqrt(2.0 * kappa1 * lr)
                for p in params:
                    p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * std)
        return out
    opt.step = step


def anchor_decoupled(sb3, theta0, kappa, with_logstd):
    kappa2 = 0.0 if kappa < 1e-100 else kappa          # the runner's placeholder for "no pull"
    decoupled_hook(sb3, theta0, kappa2, CFG["kappa1"], with_logstd, CFG["seed"])


W.anchor = anchor_decoupled                           # the revision-41 runner calls anchor() once, before learn()
_orig_run = W.run


def run(arm, seed, steps, out, eval_n, hp):
    if arm["kappa_p"] <= 0:                           # the runner installs the hook only when kappa_p > 0
        arm = dict(arm); arm["kappa_p"] = 1e-300
    r = _orig_run(arm, seed, steps, out, eval_n, hp)
    d = r["distance"]
    n_par = sum(p.numel() for p in W.actor_params(W.PPO.load(out / "final_model.zip").policy, arm.get("anchor_logstd", True)))
    r["arm"]["kappa_1"] = CFG["kappa1"]; r["arm"]["kappa_p"] = 0.0 if arm["kappa_p"] < 1e-100 else arm["kappa_p"]
    r["arm"]["pull_form"] = "decoupled"
    k2 = r["arm"]["kappa_p"]; k1 = CFG["kappa1"]
    r["n_actor_params"] = n_par; r["spread"] = d / math.sqrt(n_par)
    r["predicted_spread"] = dict(particle_form_no_pull_in_noise=(math.sqrt(k1 / k2) if k2 > 0 else None),
                                 particle_form_pull_in_noise=(math.sqrt(k1 / (k1 + k2)) if k1 + k2 > 0 else None),
                                 adam_laplace_coupled=math.sqrt(2.0) * k1)
    r["hook"] = dict(form="decoupled", order="optimizer step, pull, noise", optimizer="Adam (SB3 default, eps 1e-5, one optimizer over actor, critic and dispersion)",
                     gradient_clipping="max_grad_norm 0.5 before the step; the pull is not in the gradient", noise_generator=f"torch.Generator seeded {1_000_000 + seed}",
                     optimizer_steps=HOOK["steps"], lr_mean=(HOOK["lr_sum"] / HOOK["steps"] if HOOK["steps"] else None), lr_max=HOOK["lr_max"],
                     contraction_per_step_at_lr_max=1.0 - HOOK["lr_max"] * k2, integrated_noise_variance_per_parameter=HOOK["noise_var_sum"],
                     versions=dict(sb3=stable_baselines3.__version__, torch=torch.__version__, python=sys.version.split()[0]))
    (out / "result.json").write_text(json.dumps(r, indent=1, allow_nan=True) + "\n")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa1", type=float, required=True); ap.add_argument("--kappa2", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--stage", default="campaign")
    ap.add_argument("--steps", type=int, default=8_000_000); ap.add_argument("--eval", type=int, default=20_000)
    args = ap.parse_args()
    CFG["kappa1"] = args.kappa1; CFG["seed"] = args.seed
    arm = dict(kind="param-noise-decoupled", m=0.03, kappa_p=args.kappa2, model=dict(wear_shape=0.5), incumbent=dict(span=3.0, std=0.1), anchor_logstd=True, prior_logstd=math.log(0.1))
    out = HERE / "wear-noise-decoupled-v68" / args.stage / f"k1_{args.kappa1:g}__k2_{args.kappa2:g}__seed{args.seed}"
    if (out / "result.json").exists():
        print("exists", out); sys.exit(0)
    r = run(arm, args.seed, args.steps, out, args.eval, dict(n_envs=32, n_steps=256, batch_size=1024, n_epochs=10, log_std_init=math.log(0.1)))
    s = r["stochastic"]; pred = r["predicted_spread"]["particle_form_no_pull_in_noise"]
    print(f"decoupled kappa1={args.kappa1:g} kappa2={args.kappa2:g} seed={args.seed}: risk={s['risk']:.4f} U={s['U']:.3f} A={r['action_kl']['A']:.3f} "
          f"dist={r['distance']:.3f} spread={r['spread']:.4f} predicted={pred if pred is None else round(pred, 4)} logstd={r['log_std']:.2f} "
          f"opt_steps={r['hook']['optimizer_steps']} train={r['train_wall']:.0f}s", flush=True)
