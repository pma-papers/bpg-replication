# The lander pilot across formulations on CSC's Roihu (17 November 2026)

- `env-numerics313.yml`, `post-install-numerics313.sh`: the environment, built on the login node with
  `conda-containerize new --prefix /projappl/${CSC_PROJECT}/${USER}-bpg/numerics313 --post-install post-install-numerics313.sh env-numerics313.yml`
  (module tykky/0.6.0). The packages are those of `environments/constrained-rl-numerics.txt`, torch in its CPU build.
- `run_lunar_test.sh`: one run of 500,000 steps on the partition test (job 3177888: 69 s of training, 0.8 GB).
- `run_lunar_array.sh`, `runs-formulations.txt`: the pilot, 240 runs as an array of 120 tasks with two runs each
  (the account may have 200 jobs submitted and 100 running on the partition small). Submitted as
  `sbatch --array=1-120%100 run_lunar_array.sh runs-formulations.txt 2` (job 3177960).
- Working folder on Roihu: `/scratch/${CSC_PROJECT}/${USER}-bpg/fuel-tank` (lunar.py, lunar-protocol.json, the five
  controllers of `lunar/campaign/lunar-unlimited__seed*`, logs, records).

All computation runs through Slurm on compute nodes. The login node is used for building, submitting and light checks.
