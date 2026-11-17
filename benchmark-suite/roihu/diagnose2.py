"""Why are the learner's updates slow? Times the parts of one update at 1, 2 and 4 threads."""
import os, sys, time, platform
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safety_study_v76 as s
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
print("host", platform.node(), platform.processor() or platform.machine(), "affinity", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "n/a")
try:
    print([l.strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0])
except Exception: pass
print(torch.__config__.show().replace("\n", " | ")[:700])
venv = VecNormalize(DummyVecEnv([lambda: s.StudyGoalTime(s.BUDGET, "failure", 0.0, s.BETA)]), norm_obs=True, norm_reward=False, gamma=s.BETA)
m = s.StudySAC("MlpPolicy", venv, seed=1, gamma=s.BETA, buffer_size=1_000_000, batch_size=256, learning_starts=1000, policy_kwargs=dict(net_arch=[256, 256]), device="cpu", verbose=0)
m.learn(total_timesteps=2000)
t = time.perf_counter()
for _ in range(1000): b = m.replay_buffer.sample(256, env=m._vec_normalize_env)
print("sampling from the replay memory: %.2f ms each" % (time.perf_counter() - t))
net = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))
opt = torch.optim.Adam(net.parameters(), lr=3e-4); x = torch.randn(256, 64)
for flush in (False, True):
    torch.set_flush_denormal(flush)
    for k in (1, 2, 4):
        torch.set_num_threads(k)
        t = time.perf_counter()
        for _ in range(1000):
            opt.zero_grad(); net(x).pow(2).mean().backward(); opt.step()
        a = (time.perf_counter() - t)
        t = time.perf_counter(); m.train(gradient_steps=300, batch_size=256); b = (time.perf_counter() - t) / 300 * 1000
        print("flush denormals %s, threads %d: plain network step %.2f ms, full update of the learner %.2f ms" % (flush, k, a, b))
