
import numpy as np
import pytest

from env.battery_env import BatteryEnv
from evaluation.classical_quantum_benchmark import (
    ClassicalNeuralPolicy,
    run_episode,
    evaluate_policy,
    compare_policies_paired,
)


def test_classical_policy_probabilities_are_valid():
    policy = ClassicalNeuralPolicy(seed=42)
    observation = np.array([0.5, 0.4, 0.6, 0.2])

    probabilities = policy.action_probabilities(observation)

    assert probabilities.shape == (3,)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.isclose(probabilities.sum(), 1.0)


def test_classical_policy_rejects_invalid_observation():
    policy = ClassicalNeuralPolicy(seed=42)

    with pytest.raises(ValueError):
        policy.action_probabilities(np.array([0.5, 0.4]))

    with pytest.raises(ValueError):
        policy.action_probabilities(
            np.array([0.5, np.nan, 0.6, 0.2])
        )


def test_classical_policy_weight_round_trip():
    policy = ClassicalNeuralPolicy(seed=42)
    original_weights = policy.weights.copy()

    policy.set_weights(np.zeros_like(original_weights))
    assert np.allclose(policy.weights, 0.0)

    policy.set_weights(original_weights)
    assert np.allclose(policy.weights, original_weights)


def test_classical_policy_rejects_invalid_weights():
    policy = ClassicalNeuralPolicy(seed=42)

    with pytest.raises(ValueError):
        policy.set_weights(np.zeros(5))

    invalid_weights = policy.weights.copy()
    invalid_weights[0] = np.nan

    with pytest.raises(ValueError):
        policy.set_weights(invalid_weights)


def test_stochastic_action_sampling_is_reproducible():
    policy = ClassicalNeuralPolicy(seed=42)
    observation = np.array([0.5, 0.4, 0.6, 0.2])

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    actions1 = [
        policy.sample_action(observation, rng=rng1)
        for _ in range(100)
    ]

    actions2 = [
        policy.sample_action(observation, rng=rng2)
        for _ in range(100)
    ]

    assert actions1 == actions2
    assert all(action in (0, 1, 2) for action in actions1)


def test_run_episode_returns_expected_metrics():
    env = BatteryEnv()
    policy = ClassicalNeuralPolicy(seed=42)

    result = run_episode(
        env,
        policy,
        seed=100,
        stochastic=False,
    )

    expected_keys = {
        "reward",
        "energy_cost",
        "wear_penalty",
        "terminal_penalty",
        "final_soc",
        "terminal_soc_met",
    }

    assert expected_keys.issubset(result.keys())
    assert all(
        np.isfinite(result[key])
        for key in (
            "reward",
            "energy_cost",
            "wear_penalty",
            "terminal_penalty",
            "final_soc",
        )
    )
    assert isinstance(result["terminal_soc_met"], bool)


def test_stochastic_episode_is_reproducible():
    env = BatteryEnv()
    policy = ClassicalNeuralPolicy(seed=42)

    first = run_episode(
        env,
        policy,
        seed=200,
        stochastic=True,
        action_seed=500,
    )

    second = run_episode(
        env,
        policy,
        seed=200,
        stochastic=True,
        action_seed=500,
    )

    assert first == second


def test_evaluate_policy_returns_one_result_per_seed():
    env = BatteryEnv()
    policy = ClassicalNeuralPolicy(seed=42)
    seeds = [10, 20, 30]

    results = evaluate_policy(
        env,
        policy,
        seeds,
        stochastic=True,
    )

    assert len(results) == len(seeds)
    assert all("reward" in result for result in results)


def test_paired_comparison_calculates_correct_statistics():
    quantum_results = [
        {"reward": 10.0},
        {"reward": 20.0},
        {"reward": 30.0},
        {"reward": 40.0},
    ]

    classical_results = [
        {"reward": 5.0},
        {"reward": 20.0},
        {"reward": 25.0},
        {"reward": 50.0},
    ]

    comparison = compare_policies_paired(
        quantum_results,
        classical_results,
        n_bootstrap=1000,
        seed=42,
    )

    assert comparison["scenarios_compared"] == 4
    assert comparison["quantum_wins"] == 2
    assert comparison["classical_wins"] == 1
    assert comparison["ties"] == 1
    assert comparison["quantum_win_rate"] == 0.5
    assert comparison["mean_reward_difference"] == 0.0
    assert comparison["median_reward_difference"] == 2.5


def test_paired_comparison_rejects_different_lengths():
    quantum_results = [{"reward": 10.0}]
    classical_results = [
        {"reward": 5.0},
        {"reward": 8.0},
    ]

    with pytest.raises(ValueError):
        compare_policies_paired(
            quantum_results,
            classical_results,
        )


def test_paired_comparison_rejects_empty_results():
    with pytest.raises(ValueError):
        compare_policies_paired([], [])
