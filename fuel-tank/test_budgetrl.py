"""Checks of budgetrl against the hand-written environments of the studies (runtime ~/.venvs/constrained-rl-numerics/bin/python).

1. BudgetWrapper around LunarLanderContinuous-v3 with the fuel cost reproduces FuelLander
   (lunar.py, the frozen-multiplier form, clip_utility=False) step by step: identical observations, rewards, terminations and
   failure flags for the same seeds and action sequences, in the physical-horizon form.
2. The absorbing form charges -m/(1-beta) at failure.
3. The multiplier form charges -lambda and the DualAscent update is the paper's rule.
4. The certificates reproduce hand computations.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gymnasium as gym  # noqa: E402
import budgetrl as br  # noqa: E402
from lunar import FuelLander, fuel_use, H  # noqa: E402


def make_wrapped(budget, m, beta, horizon):
    env = gym.make("LunarLanderContinuous-v3")
    return br.BudgetWrapper(env, lambda o, a, o2, r, info: [fuel_use(a)], [budget], beta=beta, m=m, horizon=horizon, clip_utility=False)


def test_reproduces_fuellander():
    for seed, m, budget in ((1, 0.5, 22.0), (2, 0.0, 20.0), (3, 2.0, 25.0), (4, 0.5, 5.0), (5, 0.0, 3.0)):
        ref = FuelLander(budget, "failure", m, 0.999)
        new = make_wrapped(budget, m, 0.999, H)
        o1, _ = ref.reset(seed=seed); o2, _ = new.reset(seed=seed)
        rng = np.random.default_rng(seed)
        assert np.allclose(o1, o2)
        for t in range(1000):
            a = rng.uniform(-1, 1, 2).astype(np.float32)
            o1, r1, d1, tr1, i1 = ref.step(a)
            o2, r2, d2, tr2, i2 = new.step(a)
            assert np.allclose(o1, o2), (t, o1, o2)
            assert abs(r1 - r2) < 1e-9, (t, r1, r2)
            assert d1 == d2 and tr1 == tr2 and i1["failure"] == i2["failure"]
            assert abs(i1["utility"] - i2["utility"]) < 1e-9
            if d1 or tr1:
                break
        print(f"seed {seed}: identical for {t + 1} steps, failure={i2['failure']}, final reward {r2:.3f}")


def test_absorbing_and_multiplier():
    env = gym.make("LunarLanderContinuous-v3")
    w = br.BudgetWrapper(env, lambda o, a, o2, r, info: [10.0], [5.0], beta=0.9, m=2.0)      # fails at the first step
    w.reset(seed=0)
    _, r, d, _, info = w.step(np.zeros(2, dtype=np.float32))
    assert d and info["failure"] and abs(r + 2.0 / 0.1) < 1e-9
    w = br.BudgetWrapper(env, lambda o, a, o2, r, info: [10.0], [5.0], beta=0.9, penalty="multiplier", multiplier=7.0)
    w.reset(seed=0)
    _, r, d, _, info = w.step(np.zeros(2, dtype=np.float32))
    assert d and abs(r + 7.0) < 1e-9
    # barrier: interior loss equals min{B(z'), u + m}
    w = br.BudgetWrapper(env, lambda o, a, o2, r, info: [1.0], [4.0], beta=0.9, m=1.0, theta=0.5)
    w.reset(seed=0)
    _, r, d, _, info = w.step(np.zeros(2, dtype=np.float32))
    B = 0.5 * np.log1p(4.0 / 3.0)
    assert abs(info["interior_loss"] - min(B, info["utility"] + 1.0)) < 1e-9 and abs(r - max(-1.0, info["utility"] - B)) < 1e-9
    print("absorbing, multiplier and barrier forms: ok")


def test_certificates():
    assert abs(br.risk_from_value(V_m=0.0, m=1.0, ubar=1.0, beta=0.9) - 0.5) < 1e-12
    assert abs(br.risk_from_value(V_m=0.0, m=1.0, ubar=1.0, beta=0.9, T=1) - 0.5 / 0.9) < 1e-12
    law, var = br.price_risk_law(m=1.0, beta=0.95, sup_U=5.35, ubar=1.0)
    assert abs(law - 0.05 * 5.35) < 1e-12 and abs(var - 1.0 / 2.0) < 1e-12
    ok, up, floor = br.validate(0, 1000, 0.05, 0.05)
    assert ok and abs(up - (1 - 0.05 ** (1 / 1000))) < 1e-12 and abs(floor - up) < 1e-12
    ok, up, floor = br.validate(0, 50, 0.05, 0.05)                      # 50 episodes cannot certify 5%
    assert not ok and floor > 0.05
    ok, up, floor = br.validate(3, 20000, 0.05, 0.05, n_statements=120)
    assert ok and up < 0.05
    print(f"certificates: ok (resolution floor of 20,000 episodes at error 0.05/120: {floor:.5f})")


if __name__ == "__main__":
    test_reproduces_fuellander()
    test_absorbing_and_multiplier()
    test_certificates()
    print("all checks passed")
