"""Revision 68: per-coordinate records of the reused coupled runs (PROTOCOL-optimizers-v68.md, check (E6)).

The recorded runs of revision 60 (wear-noise/campaign, kappa_2 = 10), of the matched anchor campaign of revision 41
(wear-anchor/campaign-matched, kappa_p = 10 and 0) and of the decoupled campaign of the first protocol of revision 68
(wear-noise-decoupled-v68/campaign) saved the trained policy with its optimizer state but not theta_0.
This script rebuilds theta_0 from the seed exactly as the runner did (the same learner, the same least-squares fit
from the same generator), checks the fit's mean squared error against the record, loads the saved model (which restores
Adam's state) and writes theta_0, the final theta - theta_0 and Adam's exp_avg_sq for every actor coordinate to
wear-optimizers-v68/derived-coordinates/<study>__<stage>__<run>.npz. Nothing in the record folders is modified.
Runtime: ~/.venvs/constrained-rl-numerics/bin/python.
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
import wear_param_anchor as W  # noqa: E402
from env import Incumbent, WearEnv  # noqa: E402
from model import Model  # noqa: E402

torch.set_num_threads(1)
OUT = HERE / "wear-optimizers-v68" / "derived-coordinates"


def rebuild_theta0(arm, seed):
    model = Model(**arm.get("model", {})); incumbent = Incumbent(model, **arm.get("incumbent", {}))
    venv = DummyVecEnv([lambda: WearEnv(model, "failure", arm["m"]) for _ in range(32)]); venv.seed(seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=1e9, gamma=1.0)
    sb3 = PPO("MlpPolicy", venv, seed=seed, n_steps=256, batch_size=1024, n_epochs=10, learning_rate=(lambda p: 3e-4 * p), gamma=1.0, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64], log_std_init=arm["prior_logstd"]), device="cpu", verbose=0)
    fit_mse = W.fit_incumbent(sb3, incumbent, model, seed)
    return [p.detach().clone() for p in W.actor_params(sb3.policy, arm.get("anchor_logstd", True))], fit_mse


def extract(run_dir):
    r = json.loads((run_dir / "result.json").read_text())
    arm, seed = r["arm"], r["seed"]
    theta0, fit_mse = rebuild_theta0(arm, seed)
    rec = r["prior_policy"]["fit_mse"]
    assert math.isclose(fit_mse, rec, rel_tol=1e-6), (run_dir.name, fit_mse, rec)
    m = PPO.load(str(run_dir / "final_model.zip"), device="cpu")
    params = W.actor_params(m.policy, arm.get("anchor_logstd", True))
    st = m.policy.optimizer.state
    with torch.no_grad():
        x = torch.cat([(p - p0).reshape(-1) for p, p0 in zip(params, theta0)]).numpy()
        v = torch.cat([st[p]["exp_avg_sq"].reshape(-1) if p in st and "exp_avg_sq" in st[p] else torch.full((p.numel(),), float("nan")) for p in params]).numpy()
        t0 = torch.cat([p0.reshape(-1) for p0 in theta0]).numpy()
    d_rec = r["distance"]; d_now = float(np.sqrt((x ** 2).sum()))
    assert math.isclose(d_rec, d_now, rel_tol=1e-4), (run_dir.name, d_rec, d_now)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{run_dir.parents[1].name}__{run_dir.parent.name}__{run_dir.name}.npz", theta0=t0, x=x, adam_v=v, g2_sum=np.full(x.shape, np.nan),
                        source=str(run_dir.relative_to(HERE)), kappa1=float(arm.get("kappa_1", 0.0)), kappa2=float(arm["kappa_p"]), seed=seed, form=("adam-decoupled" if arm.get("pull_form") == "decoupled" else "adam-coupled"))
    print(f"{run_dir.parent.name}/{run_dir.name}: fit_mse {fit_mse:.3e} (record {rec:.3e}), distance {d_now:.4f} (record {d_rec:.4f}), "
          f"rank corr |x| vs sqrt(v) {spearman(np.abs(x), np.sqrt(v)):.3f}", flush=True)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    runs = (sorted((HERE / "wear-noise" / "campaign").glob("k1_*__k2_10__seed9400*")) + sorted((HERE / "wear-anchor" / "campaign-matched").glob("kappa*__seed9400*"))
            + sorted((HERE / "wear-noise-decoupled-v68" / "campaign").glob("k1_*__seed9400*")))
    if len(sys.argv) > 1:
        runs = [Path(p) for p in sys.argv[1:]]
    for run_dir in runs:
        if (OUT / f"{run_dir.parents[1].name}__{run_dir.parent.name}__{run_dir.name}.npz").exists():
            continue
        extract(run_dir)
