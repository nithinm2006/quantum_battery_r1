
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QuantumPolicy:
    """
    Parameterized quantum policy for BatteryEnv.

    Observation:
        [normalized_soc, normalized_price, normalized_demand, normalized_hour]

    Actions:
        0 = charge
        1 = discharge
        2 = hold
    """

    ACTIONS = {0: "charge", 1: "discharge", 2: "hold"}

    def __init__(self, n_qubits=4, n_layers=2, seed=42):
        if n_qubits < 2:
            raise ValueError("QuantumPolicy requires at least 2 qubits.")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rng = np.random.default_rng(seed)

        self.weights = self.rng.uniform(
            -np.pi, np.pi, size=(n_layers, n_qubits, 3)
        )

    def _validate_observation(self, observation):
        observation = np.asarray(observation, dtype=float)

        if observation.shape != (4,):
            raise ValueError(
                f"Expected observation shape (4,), got {observation.shape}"
            )

        if not np.all(np.isfinite(observation)):
            raise ValueError("Observation must contain finite values.")

        return np.clip(observation, 0.0, 1.0)

    def build_circuit(self, observation):
        observation = self._validate_observation(observation)
        circuit = QuantumCircuit(self.n_qubits)

        for qubit in range(self.n_qubits):
            feature = observation[qubit % 4]
            circuit.ry(np.pi * feature, qubit)
            circuit.rz(np.pi * feature, qubit)

        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                theta_y, theta_z, theta_x = self.weights[layer, qubit]
                circuit.ry(theta_y, qubit)
                circuit.rz(theta_z, qubit)
                circuit.rx(theta_x, qubit)

            for qubit in range(self.n_qubits - 1):
                circuit.cx(qubit, qubit + 1)

            circuit.cx(self.n_qubits - 1, 0)

        return circuit

    def action_probabilities(self, observation):
        circuit = self.build_circuit(observation)
        state = Statevector.from_instruction(circuit)
        basis_probabilities = state.probabilities()

        action_probs = np.zeros(3, dtype=float)

        for basis_index, probability in enumerate(basis_probabilities):
            action_code = basis_index & 0b11

            if action_code == 0:
                action_probs[0] += probability
            elif action_code == 1:
                action_probs[1] += probability
            else:
                action_probs[2] += probability

        total = action_probs.sum()

        if total <= 0:
            return np.full(3, 1 / 3, dtype=float)

        return action_probs / total

    def select_action(self, observation):
        """Deterministic action selection for evaluation."""
        return int(np.argmax(self.action_probabilities(observation)))

    def sample_action(self, observation, rng=None):
        """
        Sample an action from the policy.

        An external RNG can be supplied to reproduce action sequences
        across different policy evaluations.
        """
        probabilities = self.action_probabilities(observation)

        generator = self.rng if rng is None else rng

        return int(generator.choice(3, p=probabilities))

    def set_weights(self, weights):
        weights = np.asarray(weights, dtype=float)

        if weights.shape != self.weights.shape:
            raise ValueError(
                f"Expected weights shape {self.weights.shape}, "
                f"got {weights.shape}"
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError("Weights must contain finite values.")

        self.weights = weights.copy()
