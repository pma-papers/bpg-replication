# Protocol: the paper's class on a benchmark-suite task (revision 63 candidate), 2026-10-12

FIXED on 2026-10-12 (authors: "run it with the price grid as proposed") before any campaign run; hashed in PROTOCOL-safety-v63.md.sha256.

Task: SafetyPointGoal1-v0 of Safety-Gymnasium 1.0.0 (MuJoCo 2.3.3), episode length H = 1000, native
reward (goal progress and goal bonus), per-step cost c_t in {0, 1} for hazard contact.

Reading in the paper's class: resource = cost budget b = 25 per episode (the suite's customary cost
limit), consumption = c_t, failure = the first step after which the cumulative cost exceeds b, horizon
= the episode, requirement = tolerated exhaustion probability alpha within the episode. Wrapper as in
safety_pilot_v63.py: observation augmented with the remaining fraction; fixed price m charged as the tail
-m (1 - beta^(H-t))/(1 - beta) at the exhausting step (beta = 0.999), termination at exhaustion, utility
after failure zero (the fuel-tank convention, Section 6.2 of the paper).

Configurations (eight seeds 95001-95008 each unless stated):
  unpriced control m = 0;
  fixed prices m in {0.03, 0.1, 0.3, 1} (the pilot shows m = 1 already far above the stakes: return ~0);
  adaptive multiplier at alpha in {0.05, 0.2} with dual step 0.5 (dual ascent on the exhaustion frequency);
  CMDP reading of the same task: Lagrangian penalty nu on every unit of cost, nu by dual ascent on the mean
  episode cost toward d = 25, no termination (the suite's own formulation), to measure its exhaustion
  probability.
Learner: PPO (SB3 2.2.1), 8 environments, 1024 steps per environment per rollout, minibatch 64, 4 epochs,
learning rate 3e-4 decayed linearly, gamma = 0.999, GAE 0.95, clip 0.2, no entropy bonus, observation and
reward normalization, two layers of 64 units, 2 x 10^6 steps per run (pilot: about 800 steps/s, so ~42 min
per run; 8 seeds x 8 configurations = 64 runs, ~45 processor-hours, ~8 hours wall at 6 in parallel).
Evaluation: 200 fresh episodes per policy (exhaustion frequency with its exact binomial upper bound at
error 0.05/64, mean return, mean and 90th-percentile cost, stopped utility), plus the lump-sanction value
certificate of Section 5.2 at the measured return.

Predictions (falsifiers in brackets):
  (S1) The exhaustion probability falls with m from the control's [non-monotone in m];
  (S2) a price above the stakes of the task (m = 1) degrades the return to near zero (as in the fuel tank)
       [return within half of the control's];
  (S3) the CMDP policy that meets E[cost] <= 25 exhausts the budget in at least a third of its episodes,
       whereas the priced policies at the same mean cost exhaust it less often [CMDP exhaustion below
       the priced policy of equal mean cost];
  (S4) every value certificate holds and validation at alpha = 0.2 passes for at least one price,
       at alpha = 0.05 for none [a violated certificate];
  (S5) the adaptive multiplier at alpha = 0.05 winds up (no policy at that level), at alpha = 0.2 settles.
Pilot facts (100k steps, one seed, 50 evaluation episodes): unpriced exhaustion 0.70, mean cost 19.4,
return 4.75; m = 1: exhaustion 0.22, mean cost 5.7, return 0.02. Training throughput ~800 steps/s.
