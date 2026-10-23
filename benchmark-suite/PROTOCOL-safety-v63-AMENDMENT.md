# Amendment to PROTOCOL-safety-v63 (2026-10-12, during the campaign)

Launcher `run_one.sh` only; `safety_pilot_v63.py` and `PROTOCOL-safety-v63.md` keep their hashes.

1. The runner names a run's folder `<kind>_m<m>_b<budget>__seed<seed>`, without the adaptive
   target alpha. The two adaptive arms (alpha 0.05 and 0.2) of the same seed would therefore write
   to the same folder, the later one overwriting the earlier. The alpha = 0.2 arm now runs with
   `--stage campaign-a0.2` and writes under `safety/campaign-a0.2/`; the alpha = 0.05 arm and all
   other arms stay under `safety/campaign/`. No run of the alpha = 0.2 arm had started when the
   launcher was replaced (the first six runs, seed 95001 of the five fixed prices and of
   alpha = 0.05, were in progress and are unaffected; `sh` had read the old script completely).
2. A run whose `result.json` exists is skipped, so `xargs -P 6 -n 3 ./run_one.sh < campaign.txt`
   can be relaunched after an interruption (sleep does not interrupt; a shutdown or crash would)
   without repeating finished runs. Interrupted runs leave no `result.json` and are rerun from
   scratch.

The analysis reads both folders. Predictions, configurations, seeds and evaluation are unchanged.

3. Added after the campaign (2026-10-12, 22:40 EEST, after the analysis of the 64 pre-registered
   runs): a second CMDP arm with the dual step 0.01 instead of 0.5, eight seeds 95001-95008,
   stage `campaign-cmdp-s0.01` (launcher kind `cmdp-s0.01`; runner unchanged). Reason: at the
   pre-registered step the multiplier rose to 146-429 within the first tenth of training, because
   the initial policies incur a mean episode cost near 170 against the limit 25; the policies
   collapsed to near inaction (mean cost about 3, return about -2), the multiplier then decayed to
   zero and the policies did not recover. The pre-registered arm therefore never realised a policy
   with an active expected-cost constraint, and prediction (S3) could not be tested on it. The
   added arm is reported as an addition, next to the pre-registered one, exactly as the fuel-tank
   study reports its added dual step.

4. (2026-10-13, after the campaign) The Python runtime moved out of the synced folder, from
   `numerics-v63/.venv310` to `~/.venvs/constrained-rl-safety310`, recreated from the frozen package list
   `environments/constrained-rl-safety310.txt` (Python 3.10.22, identical versions). The launcher's
   interpreter path changed accordingly; the runner and its hash are unchanged. The usage line in the
   runner's docstring still names the old path.
