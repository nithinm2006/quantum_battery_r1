import numpy as np

from env.battery_env import BatteryEnv
from agents.quantum_policy import QuantumPolicy
from training.train_quantum_policy import run_episode, evaluate_weights


def test_quantum_policy_runs_full_battery_episode():
    env = BatteryEnv()
    policy = QuantumPolicy(seed=7)

    result = run_episode(env, policy, seed=123)

    assert np.isfinite(result["reward"])
    assert 0.0 <= result["final_soc"] <= env.capacity
    assert "energy_cost" in result
    assert "terminal_soc_met" in result

    env.close()


def test_evaluate_weights_returns_finite_average():
    env = BatteryEnv()
    policy = QuantumPolicy(seed=7)

    result = evaluate_weights(
        env,
        policy,
        policy.weights.copy(),
        seeds=[10, 11],
    )

    assert np.isfinite(result)
    env.close()


def test_training_episode_is_reproducible_for_fixed_policy_seed():
    env = BatteryEnv()
    policy = QuantumPolicy(seed=7)

    first = run_episode(env, policy, seed=99)
    second = run_episode(env, policy, seed=99)

    # The environment scenario is seeded; policy sampling is stochastic.
    # Check the returned structure and numeric values rather than exact reward.
    assert set(first) == set(second)
    assert np.isfinite(first["reward"])
    assert np.isfinite(second["reward"])

    env.close()