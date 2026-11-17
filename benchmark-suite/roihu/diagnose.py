"""Where do time and memory go in one run of the study? Prints thread settings, memory after each stage, and
the speed of the environment and of the learner's updates."""
import os, sys, time, platform
def rss():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"): return int(line.split()[1]) / 1024
    except OSError:
        import resource; return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print("host", platform.node(), platform.machine(), "cpus seen", os.cpu_count(), "affinity", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "n/a")
print("env", {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "SLURM_CPUS_PER_TASK")})
print("rss start %.0f MB" % rss())
import numpy as np; print("rss numpy %.0f MB" % rss())
import torch; print("rss torch %.0f MB" % rss(), "threads", torch.get_num_threads(), "interop", torch.get_num_interop_threads())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safety_study_v76 as s
print("rss study module %.0f MB" % rss(), "threads", torch.get_num_threads())
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
env = s.StudyGoalTime(s.BUDGET, "failure", 0.0, s.BETA); o, _ = env.reset(seed=1)
print("rss one env %.0f MB" % rss())
t = time.perf_counter(); n = 0
for _ in range(3000):
    o, r, te, tr, i = env.step(env.action_space.sample()); n += 1
    if te or tr: o, _ = env.reset()
dt = time.perf_counter() - t; print("env: %.0f steps/s" % (n / dt))
ev = s.Evaluator(make=s.StudyGoalTime); print("rss evaluator (8 envs) %.0f MB" % rss())
venv = VecNormalize(DummyVecEnv([lambda: s.StudyGoalTime(s.BUDGET, "failure", 0.0, s.BETA)]), norm_obs=True, norm_reward=False, gamma=s.BETA)
m = s.StudySAC("MlpPolicy", venv, seed=1, gamma=s.BETA, buffer_size=1_000_000, batch_size=256, learning_starts=1000, policy_kwargs=dict(net_arch=[256, 256]), device="cpu", verbose=0)
print("rss learner with replay memory %.0f MB" % rss())
t = time.perf_counter(); m.learn(total_timesteps=1000); dt = time.perf_counter() - t; print("collect only: %.0f steps/s" % (1000 / dt), "rss %.0f MB" % rss())
t = time.perf_counter(); m.learn(total_timesteps=3000, reset_num_timesteps=False); dt = time.perf_counter() - t; print("collect + update: %.0f steps/s" % (3000 / dt), "rss %.0f MB" % rss())
t = time.perf_counter(); m.train(gradient_steps=1000, batch_size=256); dt = time.perf_counter() - t; print("updates alone: %.0f per s" % (1000 / dt), "rss %.0f MB" % rss())
