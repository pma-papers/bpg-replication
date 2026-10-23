# Errata and additions to the records

The protocols, reports and logs in this repository are records and are kept as written. Where a record states something
that the run records or the code contradict, or leaves out a setting that matters for reading the results, the
correction is listed here with the evidence. Dates are those of the entries.

## 2026-10-20, after the regularizer--optimizer audit of revision 68

1. **Warm start of the fuel-tank campaign carries the optimizer state.** In `fuel-tank/lunar.py` the constrained arms
   continue from the unconstrained controller of the same seed through `PPO.load(init, env=venv, seed=seed,
   custom_objects=dict(learning_rate=lr))` and `learn(..., reset_num_timesteps=True)`. `PPO.load` restores the
   saved `policy.optimizer` state (the saved archives hold `policy.optimizer.pth`; `set_parameters` loads it), so the
   Adam moments and step count of the unconstrained run carry over into the constrained run, while the
   learning-rate schedule and the timestep count restart. The records do not state this. Its effect is that the
   first constrained steps are normalized by the unconstrained run's running root-mean-square gradients, with the
   bias correction already complete.

2. **Random numbers of the injected parameter noise (revision 60).** `wear-allowance/wear_param_noise_v60.py` draws the
   noise with `torch.randn_like`, that is, from PyTorch's global generator, which Stable-Baselines3 seeds with the
   run's seed at construction. The noise therefore shares one stream with the learner's own draws (action sampling).
   Runs are reproducible given the seed and one thread, but the noise stream is not separable from the learner's.
   The toolbox's `ParameterNoise` and the decoupled runner `wear_param_noise_decoupled_v68.py` draw from a dedicated
   generator seeded with 1,000,000 + seed.

3. **Form of the parameter pull in every recorded campaign.** All recorded pulls (`wear_param_anchor.py`,
   `wear_param_noise_v60.py`) add kappa (theta - theta_0) to the actor gradient after gradient clipping and before the
   Adam step (the coupled form). The decoupled form, applied to the parameters after the step, was run on 2026-10-20
   under `wear-allowance/PROTOCOL-decoupled-v68.md`; results in `wear-allowance/REPORT-decoupled-v68.md`. Toolbox 0.6.1
   (2026-10-20) gives `ParameterAnchor` a `decoupled` option with the decoupled form as its default; the toolbox is now
   called `barrierpg`.
