import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QuantumPolicy:
    """
    Parameterized quantum policy for the battery RL environment.

    Observation features:
        [normalized_soc, normalized_price, normalized_demand, normalized_hour]

    Actions:
        0 = charge
        1 = discharge
        2 = hold
    """

    ACTIONS = {
        0: "charge",
        1: "discharge",
        2: "hold",
    }

    def __init__(self, n_qubits=4, n_layers=2, seed=42):
        if n_qubits < 2:
            raise ValueError("QuantumPolicy requires at least 2 qubits.")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rng = np.random.default_rng(seed)

        # Three trainable rotation angles per qubit per layer.
        self.weights = self.rng.uniform(
            -np.pi,
            np.pi,
            size=(n_layers, n_qubits, 3)
        )

    def _validate_observation(self, observation):
        observation = np.asarray(observation, dtype=float)

        if observation.shape != (4,):
            raise ValueError(
                f"Expected observation shape (4,), got {observation.shape}"
            )

        if not np.all(np.isfinite(observation)):
            raise ValueError("Observation must contain finite values.")

        # The environment provides normalized features.
        return np.clip(observation, 0.0, 1.0)

    def build_circuit(self, observation):
        """Build the parameterized circuit for one battery observation."""
        observation = self._validate_observation(observation)

        circuit = QuantumCircuit(self.n_qubits)

        # Encode the four observation features into the qubits.
        for qubit in range(self.n_qubits):
            feature = observation[qubit % len(observation)]
            circuit.ry(np.pi * feature, qubit)
            circuit.rz(np.pi * feature, qubit)

        # Trainable variational layers with ring entanglement.
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                theta_y, theta_z, theta_x = self.weights[layer, qubit]

                circuit.ry(theta_y, qubit)
                circuit.rz(theta_z, qubit)
                circuit.rx(theta_x, qubit)

            for qubit in range(self.n_qubits - 1):
                circuit.cx(qubit, qubit + 1)

            # Close the entanglement ring.
            circuit.cx(self.n_qubits - 1, 0)

        return circuit

    def action_probabilities(self, observation):
        """
        Return probabilities in action order:
        [charge, discharge, hold].
        """
        circuit = self.build_circuit(observation)
        state = Statevector.from_instruction(circuit)
        basis_probabilities = state.probabilities()

        action_probs = np.zeros(3, dtype=float)

        # Use the first two qubits as the action register.
        # 00 -> charge, 01 -> discharge, 10/11 -> hold.
        for basis_index, probability in enumerate(basis_probabilities):
            action_code = basis_index & 0b11

            if action_code == 0:
                action_probs[0] += probability
            elif action_code == 1:
                action_probs[1] += probability
            else:
                action_probs[2] += probability

        # Protect against small floating-point deviations.
        action_probs /= action_probs.sum()
        return action_probs

    def select_action(self, observation):
        """Choose the action with the highest quantum-policy probability."""
        probabilities = self.action_probabilities(observation)
        return int(np.argmax(probabilities))

    def set_weights(self, weights):
        """Set trainable circuit parameters, useful for optimization."""
        weights = np.asarray(weights, dtype=float)

        if weights.shape != self.weights.shape:
            raise ValueError(
                f"Expected weights shape {self.weights.shape}, "
                f"got {weights.shape}"
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError("Weights must contain finite values.")

        self.weights = weights.copy()