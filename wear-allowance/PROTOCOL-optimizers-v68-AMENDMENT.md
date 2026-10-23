# Amendment to PROTOCOL-optimizers-v68 (2026-10-20, 19:38 EEST, after the campaign of the main protocol)

Fixed and hashed before any of its runs. The main protocol (`PROTOCOL-optimizers-v68.md`, SHA-256
a138bd9de66b084b17539cb0df42b138c3ff278fbd41008e317afd9fff0284c9) and its scripts keep their hashes; no script of the
main protocol is changed. Reason: the three-arm study is to replace the paper's roles-on-PPO experiment of revision 60
(`PROTOCOL-noise-v60.md`), and three of that experiment's rows exist only under the coupled Adam form; the application
of the main protocol was run on one seed per new arm where the recorded coupled arm has six. This amendment completes
both so that every row of the replaced experiment, and every seed of the application, is reported under the three arms.
Nothing of the amendment was run or inspected before this file was hashed (the main campaign's records, analysed in
`REPORT-optimizers-v68.md`, were).

## Part A, the rows of revision 60 that exist only coupled

Setting, arms, seeds, budget, evaluation, runner and hooks exactly as in the main protocol. The sgd arm at lr_sgd =
0.001 and the adam-decoupled arm at 3e-4; the adam-coupled arm of each row is the recorded run of revision 60
(`wear-noise/campaign/`, drawn from the global generator, as noted in the main protocol) and is reused without rerun.

| row | kappa_2 | kappa_1 | revision-60 role | recorded coupled spread (eight seeds) |
|---|---|---|---|---|
| C7 | 10 | 0.001 | nominal sigma 0.01, the paper's first noise row | 0.0012 |
| C8 | 10 | 0.989 | nominal sigma 0.3, the paper's fourth noise row | 1.318 |
| C9 | 0 | 0.101 | noise alone, the paper's fifth noise row | 1.533 |

Learning runs: 2 arms x 3 rows x 8 seeds (94001-94008) = 48, `wear_param_optimizers_v68.py` through
`optimizers_run_one_v68.sh`, stage `campaign`, records in `wear-optimizers-v68/campaign/` next to the main campaign's.
Zero-objective control: the same 2 arms x 3 rows x 8 seeds = 48 direct simulations, `regularizer_dynamics_v68.py`
through `zero_run_one_v68.sh`. The adam-coupled arm has no new runs in either part; its zero-objective control at
these rows is not run (the main protocol's coupled control at C1-C3 established the coupled law, and at C9 the coupled
and decoupled forms coincide since there is no pull).

Expected spreads, as in the main protocol. With the pull (C7, C8) the stationary spread of the particle form is
sqrt(kappa_1 / kappa_2) (the exact value of the autoregression at lr_sgd kappa_2 = 0.01 is larger by the factor
1 / sqrt(1 - lr kappa_2 / 2) = 1.0025):

    C7:  sigma = sqrt(0.001 / 10) = 0.0100        C8:  sigma = sqrt(0.989 / 10) = 0.3145

Without the pull (C9) there is no stationary law: each coordinate is a Gaussian random walk of variance 2 kappa_1 lr_t
per step, so the per-parameter spread after the run is the random-walk value

    sqrt(2 kappa_1 sum_t lr_t) = sqrt(2 x 0.101 x 39.02) = 2.808 (sgd, flow time 39.02)
                               = sqrt(2 x 0.101 x 11.71) = 1.538 (adam-decoupled, flow time 11.71),

and along the run the spread after rollout k is sqrt(2 kappa_1 S_k) with S_k the flow time accumulated by then. The
flow time is the sum of the schedule's learning rates over the 78,160 steps (976 rollouts at a positive rate, the last
clipped at zero), 39.02 and 11.71 as recorded in the main campaign's hooks. The recorded coupled C9 spread 1.533 is
this random-walk value to 0.3%, as it must be whatever Adam does to a zero pull. Under the coupled form the Laplace
value sqrt(2) kappa_1 is 0.0014 (C7) and 1.399 (C8); the recorded spreads are 0.0012 and 1.318 (ratios 0.85 and 0.94),
consistent with the main campaign's 0.83-0.90.

Checks (fixed before the runs; observations, with the same thresholds as the main protocol's (E1)-(E3)):

(F1) Zero-objective control, sgd and adam-decoupled arms. C7 and C8: spread within 10% of sigma, kurtosis between 2.7
     and 3.3. C9: spread within 10% of the random-walk value, kurtosis between 2.7 and 3.3 (a sum of Gaussians), and
     the recorded per-rollout spread trace within 5% of sqrt(2 kappa_1 S_k) from the tenth rollout on.
(F2) Learning runs, sgd arm: C7 and C8 within a factor 1.3 of sqrt(sigma^2 + s_0^2), s_0 the spread of the anchor-only
     row C4 of the main campaign (3.2e-4); C9 within a factor 1.3 of the random-walk value 2.808 (the unregularized
     row C6's displacement 0.016 is negligible against it).
(F3) Learning runs, adam-decoupled arm: as (F2) with its own s_0 (C4, 0.0142) and random-walk value 1.538.
(F4) The sequence of revision 60 under each arm. At kappa_2 = 10 the rows C4, C7, C1, C2, C8 (kappa_1 = 0, 0.001,
     0.009, 0.101, 0.989) are reported in that order per arm with spread, q_beta, U, A and the final dispersion, next
     to the recorded coupled sequence, and the noise-alone row C9 next to the unregularized C6. No numerical
     prediction; the two statements of revision 60 that the sequence tests are reported as observations: whether
     the spread and A increase with kappa_1 in every arm ((R1), (R4) of revision 60), and whether the seed-mean
     breakdown probability rises with kappa_1 from the anchor-only value toward the arm's unregularized value
     ((R2)), or splits as it did at C2 and C3 in the main campaign.
(F5) (E5) of the main protocol for the new rows: the position in the risk-utility plane, the regularizer-bias
     comparison of the sgd arm against the unregularized sgd policy of the same seed, and the optimizer-effect
     comparison of each Adam arm against the sgd arm at the same (kappa_1, kappa_2), with the same columns.
(F6) The per-coordinate diagnostic (E6) for the new runs of the sgd and adam-decoupled arms (their records hold the
     per-coordinate gradient sums and Adam's second moment); the reused coupled rows are reported without it.

## Part B, the application on six seeds per arm

The anchored configuration of `../application/PROTOCOL-application-v42.md` (F = 50, kappa_p = 10, residual action
map, 2.08 million steps, constant learning rate) and its residual control (kappa_p = 0) on seeds 1-6, under the three
arms, through `../application/train_app_optimizers_v68.py` with `--out runs-optimizers-v68` and `--n-eval-wrapper
2000` as the seed-1 runs of the main protocol:

| arm | lr | kappa_p = 10 | kappa_p = 0 (residual control) |
|---|---|---|---|
| adam-coupled | 3e-4 | the records `runs/anchored_F50_k10_s1` to `_s6` (reused) | the records `runs/anchored_F50_k0_s1` to `_s6` (reused) |
| adam-decoupled | 3e-4 | seed 1 `runs-decoupled-v68/anchored-decoupled_F50_k10_s1` (reused), seeds 2-6 new | seeds 1-6 new |
| sgd | 0.01 | seed 1 `runs-optimizers-v68/sgd_lr0.01_anchored_F50_k10_s1` (reused), seeds 2-6 new | seed 1 reused, seeds 2-6 new |

New runs: 5 + 6 + 5 + 5 = 21, plus one check: the adam-decoupled anchored seed 1 run again through
`train_app_optimizers_v68.py` (tag `adam-decoupled_lr0.0003_anchored_F50_k10_s1`), which has the same arithmetic as
the first protocol's runner and is expected to reproduce its record to the digit. Twenty-two runs, three in parallel,
about four minutes each on a loaded core. The decoupled residual control has the same arithmetic as the recorded Adam
control (with kappa_p = 0 neither form applies any pull; Adam's eps is 1e-5 in both), and differs from it only in the
toolbox version of the pipeline around the learner (0.1.0 in the records, 0.6.1 now); whether it reproduces the
records to the digit is reported as an observation.

Expected: the seed means of each anchored arm inside the (A5) band of the application protocol (miss probability
within 0.005 of the prior policy's 0.0128, abatement cost within 0.4 of the prior's 5.298, value within 0.5 of the
prior's -5.893), as the seed-1 runs were; anchor distances of the order of the seed-1 values (0.007 coupled, 0.42
decoupled, 0.006 sgd); every candidate validated (pass) at the family error, as every recorded run was.

(F7) Reported per arm and kappa_p over the six seeds: anchor distance, miss probability p_51, abatement cost, stopped
     utility U, value V, the gap to the dynamic-programming optimum, the validated upper bound and the pass count,
     mean (SD) and every seed; the prior policy's row; the (A5) band per arm; the regularizer's effect (anchored
     minus control, same arm and seed, paired) and the optimizer's effect (each Adam arm minus sgd at the same
     kappa_p and seed, paired) in miss probability, cost and V.

## Budget and parallelism

The machine is shared with other work: at most eight wear-allowance processes at a time (six learning runs and two
zero-objective runs), and three application runs. Learning runs about 5 minutes each (training 250-300 s, evaluation
60 s), zero-objective runs about half a minute; expected wall time about 45 minutes for Part A and about 35 minutes
for Part B, run concurrently.

## Reporting

`analyze_optimizers_v68.py` is extended to read the new rows and the application records: the rows C7-C9 join
`wear-optimizers-v68.json`, `wear-optimizers-v68.tex` and `wear-optimizers-v68.pdf/png` (the existing rows' entries
unchanged), and the application table over six seeds per arm is written to `wear-optimizers-v68-application.tex` and
the key `application` of the JSON. `REPORT-optimizers-v68.md` gains a dated section with the results and the checks
(F1)-(F7). Every run and seed is reported.
