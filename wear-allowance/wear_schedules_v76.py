"""Revision 76: convergence check and strength schedules of the regularized PPO runs of the wear-allowance example.

The recorded anchor, noise and action-regularization runs (wear_param_anchor.py, wear_param_noise_v60.py,
wear_param_optimizers_v68.py, train.py) kept the final policy only. This runner repeats their arithmetic and adds

  * a trace after every rollout (learning rate, strengths in force, parameter distance to theta_0, dispersion,
    breakdown frequency and mean stopped utility of the episodes completed in the rollout, and the stationarity
    residual of the anchored objective: the norms of the mean clipped actor gradient g, of the mean pull
    kappa_2 (theta - theta_0) and of their mean sum over the optimizer steps that followed the previous rollout);
  * an evaluation of the current policy on fresh episodes every --eval-every steps (breakdown probability, stopped
    utility, action relative entropy to the incumbent rule), which leaves the random streams of training untouched;
  * a learning rate that is constant (--lr-schedule constant) instead of decayed linearly to zero;
  * strengths that change during training: --kappa2, --kappa1 and --eta take a constant ("10") or a
    piecewise-constant schedule "100@0.25,10@0.5,1@1" (value@end of the stage as a fraction of the training steps).

--start incumbent   the anchor design: the actor's mean network is fitted to the incumbent rule, dispersion 0.1,
                    theta_0 the fitted parameters (seeds 94001-94008 in the records).
--anchor-part dispersion   the pull holds the dispersion alone and the mean network is free (the mirror image of the
                    recorded configuration with the anchor on the mean network only).
--start default     the action-regularization design of the first campaign: default initialization, dispersion
                    exp(-1) (seeds 92001-92008 in the records). With --eta 0 and --m it is the fixed-price design.

With a constant strength, --lr-schedule linear and 8,000,000 steps a run repeats the recorded run of the same seed
(same calls in the same order; the evaluations during training save and restore the random states).
Runtime: ~/.venvs/constrained-rl-numerics.

  python wear_schedules_v76.py --optimizer adam-coupled --kappa2 100@0.33,10@0.66,1@1 --lr-schedule constant --seed 94001

--rule (added on 2026-11-17 for revision 77) lowers the pull by a rule in place of stages fixed in advance:
"start=100,factor=10,min=0.01,window=4,tol=0.04,sigma2=0". The pull starts at `start`. After every evaluation the
monitored value U - m q / (1 - beta) of the current policy is recorded. When the stage holds at least two windows of
`window` evaluations and the mean over the last window differs from the mean over the window before it by less than
`tol`, the pull is divided by `factor`; below `min` it is removed. With sigma2 > 0 the noise strength is sigma2
times the pull (relative entropy to a prior of variance sigma2). The changes are stored as rule_changes.
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
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
import wear_param_anchor as W  # noqa: E402
from env import Incumbent, WearEnv, model_policy  # noqa: E402
from model import Model, simulate  # noqa: E402
from train import stopped_action_kl  # noqa: E402

torch.set_num_threads(1)


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


class State:
    """Shared between the optimizer hook and the callback."""
    def __init__(self):
        self.rule = None; self.k2 = 0.0; self.k1 = 0.0      # with --rule the strengths in force are set by the callback
        self.fraction = 0.0
        self.steps = 0; self.lr_sum = 0.0; self.noise_var_sum = 0.0
        self.reset_window()

    def reset_window(self):
        self.n = 0; self.g = None; self.pull = None; self.g_norm = 0.0


def install_hook(sb3, theta0, arm, seed, st, k2_sched, k1_sched):
    params = W.actor_params(sb3.policy, True)
    opt = sb3.policy.optimizer
    base_step = opt.step
    gen = torch.Generator().manual_seed(1_000_000 + seed)
    form = arm["optimizer"]
    # which parameters the pull acts on: all actor parameters, or the dispersion alone (the mean network is then free)
    held = [True] * len(params) if arm.get("anchor_part", "all") == "all" else [p is sb3.policy.log_std for p in params]

    def step(closure=None):
        k2, k1 = (st.k2, st.k1) if st.rule else (value_at(k2_sched, st.fraction), value_at(k1_sched, st.fraction))
        with torch.no_grad():
            lr = max(float(opt.param_groups[0]["lr"]), 0.0)
            st.steps += 1; st.lr_sum += lr; st.noise_var_sum += 2.0 * k1 * lr
            g = torch.cat([(p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in params]).clone()
            pull = torch.cat([((k2 if h else 0.0) * (p - p0)).reshape(-1) for p, p0, h in zip(params, theta0, held)]) if theta0 is not None else torch.zeros_like(g)
            st.n += 1; st.g_norm += float(g.norm())
            st.g = g if st.g is None else st.g + g
            st.pull = pull if st.pull is None else st.pull + pull
            if form in ("sgd", "adam-coupled") and k2 > 0:
                for p, p0, h in zip(params, theta0, held):
                    if not h:
                        continue
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.add_(k2 * (p - p0))
        out = base_step(closure)
        with torch.no_grad():
            if form == "adam-decoupled" and k2 > 0 and lr > 0:
                c = lr * k2
                for p, p0, h in zip(params, theta0, held):
                    if h:
                        p.sub_(c * (p - p0))
            if k1 > 0 and lr > 0:
                std = math.sqrt(2.0 * k1 * lr)
                for p in params:
                    p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * std)
        return out
    opt.step = step


class Trace(BaseCallback):
    """Per-rollout trace, periodic evaluation, and the reward shaping of the action-regularized design."""

    def __init__(self, st, model, incumbent, theta0, total, scheds, eval_every, eval_n, m, seed):
        super().__init__(verbose=0)
        self.st, self.mdl, self.incumbent, self.theta0, self.total = st, model, incumbent, theta0, total
        self.k2_sched, self.k1_sched, self.eta_sched = scheds
        self.eval_every, self.eval_n, self.m, self.seed = eval_every, eval_n, m, seed
        self.episodes = 0; self.failures = 0; self.utility = 0.0
        self.trace = []; self.evals = []; self.next_eval = 0
        self.stage_values = []; self.rule_changes = []

    def _on_step(self):
        for info in self.locals["infos"]:
            self.utility += info["utility"]
            if "episode_failed" in info:
                self.episodes += 1; self.failures += int(info["episode_failed"])
        eta = value_at(self.eta_sched, self.st.fraction)
        if eta > 0:                                         # as train.Control for the arm kind "anchored"
            obs = self.model._last_obs
            x = self.locals["actions"].reshape(-1)
            logp = self.locals["log_probs"].detach().cpu().numpy().reshape(-1)
            z = (obs[:, 1] + 1.0) / 2.0 * self.incumbent.model.budget
            ratio = logp - self.incumbent.log_density(x, z)
            venv = self.model.get_vec_normalize_env()
            scale = np.sqrt(venv.ret_rms.var + venv.epsilon)
            self.locals["rewards"][:] -= (eta * ratio / scale).astype(self.locals["rewards"].dtype)
        return True

    def evaluate(self):
        ts, ns = torch.get_rng_state(), np.random.get_state()
        k = len(self.evals)
        s = simulate(self.mdl, model_policy(self.mdl, self.model, False), n=self.eval_n, seed=7_000_000 + 1000 * k + self.seed % 1000, m=self.m)
        a = stopped_action_kl(self.mdl, self.model, self.incumbent, n=max(self.eval_n // 2, 1), seed=8_000_000 + 1000 * k + self.seed % 1000)
        torch.set_rng_state(ts); np.random.set_state(ns)
        self.evals.append(dict(steps=int(self.num_timesteps), risk=s["risk"], U=s["U"], U_se=s["U_se"], A=a["A"]))
        rule = self.st.rule
        if rule and self.st.k2 > 0:                         # lower the pull when the monitored value has stopped changing
            self.stage_values.append(s["U"] - self.m * s["risk"] / (1.0 - self.mdl.beta))
            w = int(rule["window"])
            if len(self.stage_values) >= 2 * w and abs(np.mean(self.stage_values[-w:]) - np.mean(self.stage_values[-2 * w:-w])) < rule["tol"]:
                k2 = self.st.k2 / rule["factor"]
                self.st.k2 = k2 if k2 >= rule["min"] * (1 - 1e-9) else 0.0
                self.st.k1 = rule["sigma2"] * self.st.k2
                self.rule_changes.append(dict(steps=int(self.num_timesteps), kappa2=self.st.k2, kappa1=self.st.k1))
                self.stage_values = []

    def _on_rollout_end(self):
        st = self.st
        row = dict(steps=int(self.num_timesteps), lr=float(self.model.policy.optimizer.param_groups[0]["lr"]), lr_sum=st.lr_sum,
                   kappa2=(st.k2 if st.rule else value_at(self.k2_sched, st.fraction)), kappa1=(st.k1 if st.rule else value_at(self.k1_sched, st.fraction)), eta=value_at(self.eta_sched, st.fraction),
                   log_std=float(self.model.policy.log_std.detach().numpy().reshape(-1)[0]),
                   episodes=self.episodes, failure_rate=(self.failures / self.episodes if self.episodes else float("nan")),
                   utility=(self.utility / self.episodes if self.episodes else float("nan")))
        if self.theta0 is not None:
            row["distance"] = W.distance(self.model, self.theta0, True)
        if st.n:
            g, pull = st.g / st.n, st.pull / st.n
            row.update(g_mean_norm=float(g.norm()), pull_mean_norm=float(pull.norm()), residual_norm=float((g + pull).norm()), g_step_norm=st.g_norm / st.n)
        self.trace.append(row)
        st.reset_window()
        self.episodes = 0; self.failures = 0; self.utility = 0.0
        if self.num_timesteps >= self.next_eval:
            self.evaluate(); self.next_eval += self.eval_every
        st.fraction = self.num_timesteps / self.total      # the strengths of the next rollout and of the updates before it


def run(arm, seed, steps, out, eval_n, eval_every, trace_eval_n):
    model = Model(**arm["model"])
    incumbent = Incumbent(model, **arm["incumbent"])
    start, lr0 = arm["start"], arm["lr0"]
    k2_sched, k1_sched, eta_sched = (parse_schedule("0" if arm[k] == "rule" else arm[k]) for k in ("kappa2", "kappa1", "eta"))
    venv = DummyVecEnv([lambda: WearEnv(model, "failure", arm["m"]) for _ in range(32)])
    venv.seed(seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=1e9, gamma=1.0)
    learning_rate = (lambda progress: lr0 * progress) if arm["lr_schedule"] == "linear" else (lambda progress: lr0)
    log_std_init = math.log(0.1) if start == "incumbent" else -1.0
    sb3 = PPO("MlpPolicy", venv, seed=seed, n_steps=256, batch_size=1024, n_epochs=10, learning_rate=learning_rate, gamma=1.0, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64], log_std_init=log_std_init), device="cpu", verbose=0)
    theta0, prior_policy = None, None
    if start == "incumbent":
        fit_mse = W.fit_incumbent(sb3, incumbent, model, seed)
        theta0 = [p.detach().clone() for p in W.actor_params(sb3.policy, True)]
        if arm["optimizer"] == "sgd":
            sb3.policy.optimizer = torch.optim.SGD(sb3.policy.parameters(), lr=lr0, momentum=0.0)
        else:
            sb3.policy.optimizer = torch.optim.Adam(sb3.policy.parameters(), lr=lr0, eps=1e-5)
        prior_policy = dict(stochastic=simulate(model, model_policy(model, sb3, False), n=eval_n, seed=60_000 + seed, m=arm["m"]),
                            mean_action=simulate(model, model_policy(model, sb3, True), n=eval_n, seed=61_000 + seed, m=arm["m"]),
                            action_kl=stopped_action_kl(model, sb3, incumbent, n=eval_n, seed=62_000 + seed), fit_mse=fit_mse)
    else:
        assert arm["optimizer"] == "adam-coupled" and all(v == 0 for _, v in k2_sched + k1_sched), "the default start has no theta_0"
    st = State()
    if arm.get("rule"):
        st.rule = {k: float(v) for k, v in (item.split("=") for item in arm["rule"].split(","))}
        st.k2 = st.rule["start"]; st.k1 = st.rule["sigma2"] * st.k2
    install_hook(sb3, theta0, arm, seed, st, k2_sched, k1_sched)
    cb = Trace(st, model, incumbent, theta0, steps, (k2_sched, k1_sched, eta_sched), eval_every, trace_eval_n, arm["m"], seed)
    started = time.monotonic()
    sb3.learn(total_timesteps=steps, callback=cb)
    train_wall = time.monotonic() - started
    result = dict(arm=arm, seed=seed, steps=int(sb3.num_timesteps), train_wall=train_wall, prior_policy=prior_policy,
                  log_std=float(sb3.policy.log_std.detach().numpy().reshape(-1)[0]),
                  optimizer_steps=st.steps, lr_sum_flow_time=st.lr_sum, integrated_noise_variance_per_parameter=st.noise_var_sum,
                  versions=dict(sb3=stable_baselines3.__version__, torch=torch.__version__, python=sys.version.split()[0]),
                  stochastic=simulate(model, model_policy(model, sb3, False), n=eval_n, seed=10_000 + seed, m=arm["m"]),
                  mean_action=simulate(model, model_policy(model, sb3, True), n=eval_n, seed=20_000 + seed, m=arm["m"]))
    if start == "default":                                  # the first campaign evaluated two more quantities before the action relative entropy
        result["robust90"] = simulate(model, model_policy(model, sb3, False), n=eval_n, seed=30_000 + seed, m=arm["m"], budget_scale=0.9)
        result["robust80"] = simulate(model, model_policy(model, sb3, False), n=eval_n, seed=40_000 + seed, m=arm["m"], budget_scale=0.8)
    result["action_kl"] = stopped_action_kl(model, sb3, incumbent, n=eval_n, seed=50_000 + seed)
    if theta0 is not None:
        d = W.distance(sb3, theta0, True); n_par = sum(p.numel() for p in theta0)
        result.update(distance=d, n_actor_params=n_par, spread=d / math.sqrt(n_par))
    result["evals"] = cb.evals; result["trace"] = cb.trace
    if st.rule:
        result["rule_changes"] = cb.rule_changes
    out.mkdir(parents=True, exist_ok=True)
    sb3.save(out / "final_model.zip")
    (out / "result.json").write_text(json.dumps(result, allow_nan=True) + "\n")
    return result


def tag(text):
    return str(text).replace(",", "_").replace("@", "at")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["sgd", "adam-coupled", "adam-decoupled"], default="adam-coupled")
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--lr-schedule", choices=["linear", "constant"], default="linear")
    ap.add_argument("--start", choices=["incumbent", "default"], default="incumbent")
    ap.add_argument("--kappa2", default="0"); ap.add_argument("--kappa1", default="0"); ap.add_argument("--eta", default="0")
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--stage", default="campaign")
    ap.add_argument("--steps", type=int, default=8_000_000); ap.add_argument("--eval", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=200_000); ap.add_argument("--trace-eval", type=int, default=4_000)
    ap.add_argument("--anchor-part", choices=["all", "dispersion"], default="all", help="the parameters the pull holds: all actor parameters, or the dispersion alone")
    ap.add_argument("--m", type=float, default=0.03, help="price of a breakdown (0.03 in the regularization studies)")
    ap.add_argument("--rule", default="", help="lower the pull by a rule, e.g. start=100,factor=10,min=0.01,window=4,tol=0.04,sigma2=0")
    args = ap.parse_args()
    arm = dict(kind="schedules-v76", m=args.m, model=dict(wear_shape=0.5), incumbent=dict(span=3.0, std=0.1), start=args.start, optimizer=args.optimizer,
               lr0=args.lr, lr_schedule=args.lr_schedule, kappa2=args.kappa2, kappa1=args.kappa1, eta=args.eta, anchor_part=args.anchor_part)
    if args.rule:
        arm.update(rule=args.rule, kappa2="rule", kappa1="rule")
    name = (f"{args.start}__{args.optimizer}__lr{args.lr:g}-{args.lr_schedule}__steps{args.steps // 1_000_000}M"
            f"__k2_{tag(args.kappa2)}__k1_{tag(args.kappa1)}__eta_{tag(args.eta)}{'' if args.m == 0.03 else f'__m{args.m:g}'}{'' if args.anchor_part == 'all' else '__anchor-' + args.anchor_part}{'__rule_' + args.rule.replace(',', '_').replace('=', '') if args.rule else ''}__seed{args.seed}")
    out = HERE / "wear-schedules-v76" / args.stage / name
    if (out / "result.json").exists():
        print("exists", out); sys.exit(0)
    r = run(arm, args.seed, args.steps, out, args.eval, args.eval_every, args.trace_eval)
    s = r["stochastic"]
    print(f"{name}: risk={s['risk']:.4f} U={s['U']:.3f} A={r['action_kl']['A']:.3f} dist={r.get('distance', float('nan')):.4f} "
          f"sigma={math.exp(r['log_std']):.3f} flow_time={r['lr_sum_flow_time']:.2f} train={r['train_wall']:.0f}s", flush=True)
