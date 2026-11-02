import numpy as np
import pytest

from agents.quantum_policy import QuantumPolicy


@pytest.fixture
def policy():
    return QuantumPolicy(n_qubits=4, n_layers=2, seed=42)


def test_circuit_is_created(policy):
    observation = np.array([0.5, 0.4, 0.6, 0.25])

    circuit = policy.build_circuit(observation)

    assert circuit.num_qubits == 4
    assert circuit.num_clbits == 0


def test_action_probabilities_are_valid(policy):
    observation = np.array([0.5, 0.4, 0.6, 0.25])

    probabilities = policy.action_probabilities(observation)

    assert probabilities.shape == (3,)
    assert np.all(probabilities >= 0.0)
    assert np.isclose(probabilities.sum(), 1.0)


def test_action_is_valid(policy):
    observation = np.array([0.5, 0.4, 0.6, 0.25])

    action = policy.select_action(observation)

    assert action in [0, 1, 2]


def test_same_seed_gives_same_policy():
    observation = np.array([0.5, 0.4, 0.6, 0.25])

    policy_a = QuantumPolicy(seed=123)
    policy_b = QuantumPolicy(seed=123)

    probs_a = policy_a.action_probabilities(observation)
    probs_b = policy_b.action_probabilities(observation)

    assert np.allclose(probs_a, probs_b)


def test_invalid_observation_shape_is_rejected(policy):
    with pytest.raises(ValueError):
        policy.action_probabilities(np.array([0.5, 0.4]))


def test_weights_can_be_updated(policy):
    new_weights = np.zeros_like(policy.weights)

    policy.set_weights(new_weights)

    assert np.allclose(policy.weights, new_weights)


def test_invalid_weight_shape_is_rejected(policy):
    with pytest.raises(ValueError):
        policy.set_weights(np.zeros((2, 4)))