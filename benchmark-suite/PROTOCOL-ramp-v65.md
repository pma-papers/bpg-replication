# Protocol: ramped price and low prices on the benchmark-suite task (revision 65 candidate), 2026-10-13

FIXED on 2026-10-13 before any campaign run (authors: "run the ramped-price follow-up campaign"); hashed in
PROTOCOL-ramp-v65.md.sha256 together with the runner and the launcher. Motivation: the research note on the
first campaign (notes/safety-gym-benchmark-note-v1.tex). There, at equal exhaustion, the adaptive multiplier
at alpha = 0.05 returned 7.5 and the fixed price m = 0.03 returned 1.7 although both end with a charge near
20 for a failure at the first step; three of four prices lay above the stakes (the charge at the first step
is 632 m against an unpriced return near 16); and the evaluation did not record failure times.

Task, wrapper, learner, budget (25), discount (beta = 0.999), horizon (H = 1000), training length (2e6 steps),
evaluation (200 fresh episodes, seed 10000 + seed, stochastic policy) and seeds (95001-95008, paired by seed
with the first campaign) are those of PROTOCOL-safety-v63.md; the runner safety_ramp_v65.py imports the
v63 wrapper unchanged and copies the learner settings.

Configurations (eight seeds each, 32 runs):
  ramped price, m in {0.01, 0.03}: the price rises linearly from 0 to m over the first third of training
    (updated after every rollout of 8 x 1024 steps) and stays at m;
  fixed price from the start, m in {0.003, 0.01}.
Re-evaluation: the 72 saved policies of the first campaign are evaluated again with the same seeds, recording
the failure step of every episode (smoke test: the re-evaluation of fixed m = 0.03, seed 95001, reproduces its
original exhaustion frequency and return exactly). New runs are evaluated the same way.

Measured per policy: exhaustion frequency, priced risk qt = E[(beta^tau - beta^H) 1{tau <= H-1}] (the wrapper
charges m(1 - beta^(H-tau))/(1 - beta) at the exhausting step, so the training objective is U - nu qt with
nu = m/(1 - beta)), geometric risk q_beta, stopped discounted utility U, return, cost, validation passes at
0.2 and 0.05 (exact binomial upper bound at error 0.05/64, as before).

Predictions (falsifiers in brackets):
  (R1) The ramp carries the adaptive arm's advantage: the ramped price m = 0.03 reaches a mean return of at
       least 5 with mean exhaustion at most 0.15 [mean return below 3.5, twice the fixed price's 1.74].
  (R2) The ramp matters less below the stakes: the gap in mean return between ramped and fixed prices is
       smaller at m = 0.01 than at m = 0.03 [larger at m = 0.01].
  (R3) Low prices trace the trade-off: over the fixed prices m in {0, 0.003, 0.01, 0.03} (0 and 0.03 from
       the first campaign) mean exhaustion and mean return are nonincreasing in m within two standard errors,
       and m = 0.003 lies strictly between m = 0 and m = 0.03 in exhaustion [a reversal beyond two standard
       errors].
  (R4) The optimization error of the fixed-start learner is detectable by Proposition 3.3(i): for the fixed
       price m = 0.03 from the start, in at least 6 of 8 seeds some evaluated policy of the same class (any arm,
       any seed, either campaign) has qt not larger and stopped utility larger, with the 95% bootstrap lower
       bound of the utility difference above zero; that difference is a lower bound on the learner's
       optimization error. For the ramped price m = 0.03 the median of this bound is smaller
       [fewer than 6 of 8 seeds, or a larger median for the ramped price].
Analysis: one table (every configuration of both campaigns, with qt and U), one figure (U against qt for every
policy; the price path of the ramped runs), the R4 diagnostic per seed. Every seed is reported. Deviations are
reported as amendments. Runtime: about 3.8 hours at 6 runs in parallel; the machine is kept awake with
caffeinate while the campaign runs.
