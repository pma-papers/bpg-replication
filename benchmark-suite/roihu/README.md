# Runs of the benchmark-suite study on CSC's supercomputer Roihu

From 17 November 2026 the runs of `safety_study_v76.py` with the soft actor-critic are made on Roihu
(CPU nodes, AMD EPYC 9965), as batch jobs on compute nodes. Nothing of the study is computed on a login node.

- `env-safety310.yml`, `post-install-safety310.sh`: the Python environment, built as a container with CSC's
  tool (`conda-containerize new --post-install post-install-safety310.sh --prefix <dir> env-safety310.yml`):
  Python 3.10 and the pinned packages of `environments/constrained-rl-safety310.txt`, installed as listed,
  with the CPU build of torch 2.14.1.
- `run_array.sh`: one run per task of an array job, two cores and 5000 MB each
  (`sbatch --array=1-N --time=20:00:00 run_array.sh RUNS.txt`).
- `runs-sac-first.txt`: the ten runs of the stages `sac-no-price` and `sac-scratch` (job 3177345).
- `diagnose.py`, `diagnose2.py`: the two timing checks that set the number of cores. One update of the
  learner takes 2.8 ms on the laptop the earlier stages ran on, and on these nodes 11.9 ms with one thread,
  7.7 ms with two and 6.9 ms with four; the simulator makes 892 steps per second against 1255.

A run's record names the machine and the number of threads (`versions` in `result.json`). Runs on Roihu and
runs on the laptop differ in the last digits of their arithmetic; within a stage all runs come from one machine.
