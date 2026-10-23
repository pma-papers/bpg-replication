# Protocol: injected parameter noise in PPO (the roles of regularizers on a learned controller), revision 60

Fixed on 2026-10-10 before any campaign run. Setting: the parameter-anchor study of Section 6.4
(wear-allowance example, mispriced m = 0.03, PPO as in the matched-prior campaign of revision 41:
actor started at theta_0 fitted to the incumbent rule with dispersion 0.1, 8M steps, 32 environments,
evaluation on 20,000 fresh episodes). The quadratic anchor kappa_2 (theta - theta_0) is the finite-network
counterpart of the potential term of the particle form (eq. particle); this campaign adds the counterpart
of the entropy term: Gaussian noise injected into all actor parameters after every optimizer step, of
variance 2 kappa_1 alpha_t per step, where alpha_t is the current learning rate (the step of flow time).
Under the quadratic pull alone the stationary spread per parameter would be sigma = sqrt(kappa_1 / (kappa_1 + kappa_2)).

Configurations (8 seeds each, 94001-94008, the seeds of the matched campaign):
  pull kappa_2 = 10 (the strength that keeps the incumbent in Section 6.4) with noise kappa_1 chosen so that
  sigma in {0.01, 0.03, 0.1, 0.3}: kappa_1 = sigma^2 kappa_2 / (1 - sigma^2) = 0.0010, 0.0090, 0.1010, 0.9890;
  noise alone: kappa_2 = 0, kappa_1 = 0.1010.
Reused without rerun: pull alone (kappa_2 = 10, kappa_1 = 0) and the unanchored control (kappa_2 = kappa_1 = 0)
from numerics-v41/wear-anchor/campaign-matched.

Reported per run: breakdown probability q_beta and stopped utility U of the stochastic policy, the
mean-action policy's q_beta, the action relative entropy A to the incumbent rule, the parameter distance
||theta - theta_0||, the per-parameter spread ||theta - theta_0|| / sqrt(d), and the final log dispersion.

Predictions (falsifiers in brackets):
  (R1) The per-parameter spread increases with kappa_1 at fixed pull and, for sigma <= 0.1, lies within a
       factor of three of sigma [falsified by a non-monotone spread or a spread outside that factor].
  (R2) At fixed pull, the breakdown probability rises with kappa_1 from the anchored value (about 1.3%)
       toward the control's (about 35%), and the utility rises with it [falsified by a non-monotone risk].
  (R3) Noise alone ends far from the incumbent, with breakdown probability and utility near the control's,
       and a larger action relative entropy than the control [falsified by risk below half the control's].
  (R4) The action relative entropy to the incumbent increases with kappa_1 at fixed pull.
Prediction (R2) is a prediction of the roles: noise is exploration that the pull must contain; the sum
decouples how far the units spread (kappa_1) from where they are held (kappa_2).
Adam is not a gradient step, so sigma is nominal; (R1) allows for that.
