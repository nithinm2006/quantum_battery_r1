
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.experiment_config import (
    TRAINING_SEED,
    TRAINING_ITERATIONS,
    EPISODES_PER_ESTIMATE,
    SPSA_LEARNING_RATE,
    SPSA_PERTURBATION,
    VALIDATION_SEED_OFFSET,
    VALIDATION_EPISODES,
    EVALUATION_SEED,
    EVALUATION_EPISODES,
)

from env.battery_env import BatteryEnv
from agents.quantum_policy import QuantumPolicy


RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "classical_quantum_benchmark.csv"
DETAILS_PATH = RESULTS_DIR / "classical_quantum_benchmark_details.csv"

HIDDEN_UNITS = 8


class ClassicalNeuralPolicy:
    """Small feed-forward neural policy: 4 inputs -> 8 hidden -> 3 actions."""

    def __init__(self, input_size=4, hidden_size=HIDDEN_UNITS,
                 output_size=3, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        rng = np.random.default_rng(seed)

        # Xavier-style initialization.
        self.w1 = rng.normal(
            0.0, np.sqrt(2.0 / (input_size + hidden_size)),
            size=(input_size, hidden_size)
        )
        self.b1 = np.zeros(hidden_size)

        self.w2 = rng.normal(
            0.0, np.sqrt(2.0 / (hidden_size + output_size)),
            size=(hidden_size, output_size)
        )
        self.b2 = np.zeros(output_size)

    @property
    def weights(self):
        return np.concatenate([
            self.w1.ravel(),
            self.b1.ravel(),
            self.w2.ravel(),
            self.b2.ravel(),
        ])

    def set_weights(self, weights):
        weights = np.asarray(weights, dtype=float)

        expected_size = (
            self.input_size * self.hidden_size
            + self.hidden_size
            + self.hidden_size * self.output_size
            + self.output_size
        )

        if weights.shape != (expected_size,):
            raise ValueError(
                f"Expected flattened weight vector of length {expected_size}, "
                f"got {weights.shape}"
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError("Weights must contain finite values.")

        index = 0

        size = self.input_size * self.hidden_size
        self.w1 = weights[index:index + size].reshape(
            self.input_size, self.hidden_size
        ).copy()
        index += size

        self.b1 = weights[index:index + self.hidden_size].copy()
        index += self.hidden_size

        size = self.hidden_size * self.output_size
        self.w2 = weights[index:index + size].reshape(
            self.hidden_size, self.output_size
        ).copy()
        index += size

        self.b2 = weights[index:index + self.output_size].copy()

    def action_probabilities(self, observation):
        observation = np.asarray(observation, dtype=float)

        if observation.shape != (self.input_size,):
            raise ValueError(
                f"Expected observation shape ({self.input_size},), "
                f"got {observation.shape}"
            )

        if not np.all(np.isfinite(observation)):
            raise ValueError("Observation must contain finite values.")

        hidden = np.tanh(observation @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2

        # Numerically stable softmax.
        logits = logits - np.max(logits)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum()

    def select_action(self, observation):
        return int(np.argmax(self.action_probabilities(observation)))


def run_episode(env, policy, seed):
    observation, _ = env.reset(seed=int(seed))

    total_reward = 0.0
    total_energy_cost = 0.0
    total_wear_penalty = 0.0
    total_terminal_penalty = 0.0

    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        action = policy.select_action(observation)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_energy_cost += info["energy_cost"]
        total_wear_penalty += info["wear_penalty"]
        total_terminal_penalty += info["terminal_penalty"]

    return {
        "reward": float(total_reward),
        "energy_cost": float(total_energy_cost),
        "wear_penalty": float(total_wear_penalty),
        "terminal_penalty": float(total_terminal_penalty),
        "final_soc": float(info["soc"]),
        "terminal_soc_met": bool(info["terminal_soc_met"]),
    }


def evaluate_policy(env, policy, seeds):
    return [
        run_episode(env, policy, seed)
        for seed in seeds
    ]


def train_with_spsa(policy, env, validation_seeds, label):
    """Train either policy with the same SPSA settings."""

    rng = np.random.default_rng(TRAINING_SEED)

    weights = policy.weights.copy()
    best_weights = weights.copy()

    initial_results = evaluate_policy(
        env, policy, validation_seeds
    )
    best_reward = float(np.mean([
        result["reward"] for result in initial_results
    ]))

    start_time = time.perf_counter()

    for iteration in range(1, TRAINING_ITERATIONS + 1):
        delta = rng.choice([-1.0, 1.0], size=weights.shape)

        plus_weights = weights + SPSA_PERTURBATION * delta
        minus_weights = weights - SPSA_PERTURBATION * delta

        # Use the same environment seeds for both perturbations.
        episode_seeds = rng.integers(
            0,
            1_000_000,
            size=EPISODES_PER_ESTIMATE,
        )

        policy.set_weights(plus_weights)
        plus_results = evaluate_policy(
            env, policy, episode_seeds
        )
        reward_plus = float(np.mean([
            result["reward"] for result in plus_results
        ]))

        policy.set_weights(minus_weights)
        minus_results = evaluate_policy(
            env, policy, episode_seeds
        )
        reward_minus = float(np.mean([
            result["reward"] for result in minus_results
        ]))

        gradient = (
            (reward_plus - reward_minus)
            / (2.0 * SPSA_PERTURBATION * 1000.0)
        ) * delta

        weights = weights + SPSA_LEARNING_RATE * gradient
        policy.set_weights(weights)

        validation_results = evaluate_policy(
            env, policy, validation_seeds
        )
        validation_reward = float(np.mean([
            result["reward"] for result in validation_results
        ]))

        if validation_reward > best_reward:
            best_reward = validation_reward
            best_weights = weights.copy()

        if iteration % 10 == 0 or iteration == 1:
            print(
                f"{label} | Iteration {iteration}/{TRAINING_ITERATIONS} "
                f"| Validation reward: {validation_reward:.2f} "
                f"| Best: {best_reward:.2f}"
            )

    training_time = time.perf_counter() - start_time
    policy.set_weights(best_weights)

    return {
        "best_weights": best_weights,
        "best_validation_reward": best_reward,
        "training_time_seconds": training_time,
    }


def benchmark_inference(policy, evaluation_seeds):
    """Measure average policy decision time over actual environment states."""

    env = BatteryEnv()
    timings = []

    for seed in evaluation_seeds:
        observation, _ = env.reset(seed=int(seed))

        terminated = False
        truncated = False

        while not (terminated or truncated):
            start = time.perf_counter()
            action = policy.select_action(observation)
            timings.append(time.perf_counter() - start)

            observation, _, terminated, truncated, _ = env.step(action)

    env.close()

    return float(np.mean(timings) * 1000.0) if timings else 0.0


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    validation_seeds = np.random.default_rng(
        TRAINING_SEED + VALIDATION_SEED_OFFSET
    ).integers(
        0,
        1_000_000,
        size=VALIDATION_EPISODES,
    )

    evaluation_seeds = (
        EVALUATION_SEED
        + np.arange(EVALUATION_EPISODES)
    )

    env = BatteryEnv()

    policies = {
        "Quantum Policy": QuantumPolicy(
            n_qubits=4,
            n_layers=2,
            seed=TRAINING_SEED,
        ),
        "Classical Neural Policy": ClassicalNeuralPolicy(
            input_size=4,
            hidden_size=HIDDEN_UNITS,
            output_size=3,
            seed=TRAINING_SEED,
        ),
    }

    summary_rows = []
    detail_rows = []

    print("\nCLASSICAL VS QUANTUM POLICY BENCHMARK")
    print("=" * 65)

    for name, policy in policies.items():
        print(f"\nTraining {name}...")

        training_result = train_with_spsa(
            policy,
            env,
            validation_seeds,
            name,
        )

        # Evaluate the selected best weights on unseen evaluation seeds.
        evaluation_start = time.perf_counter()
        episode_results = evaluate_policy(
            env,
            policy,
            evaluation_seeds,
        )
        evaluation_time = time.perf_counter() - evaluation_start

        rewards = [item["reward"] for item in episode_results]
        energy_costs = [item["energy_cost"] for item in episode_results]
        final_socs = [item["final_soc"] for item in episode_results]
        success_rates = [
            item["terminal_soc_met"] for item in episode_results
        ]

        inference_ms = benchmark_inference(
            policy,
            evaluation_seeds,
        )

        summary_rows.append({
            "agent": name,
            "trainable_parameters": int(policy.weights.size),
            "training_iterations": TRAINING_ITERATIONS,
            "best_validation_reward": training_result[
                "best_validation_reward"
            ],
            "mean_evaluation_reward": float(np.mean(rewards)),
            "std_evaluation_reward": float(np.std(rewards)),
            "mean_energy_cost": float(np.mean(energy_costs)),
            "mean_final_soc": float(np.mean(final_socs)),
            "terminal_soc_success_rate": float(np.mean(success_rates)),
            "training_time_seconds": training_result[
                "training_time_seconds"
            ],
            "evaluation_time_seconds": float(evaluation_time),
            "mean_inference_time_ms": inference_ms,
        })

        for episode_number, (seed, result) in enumerate(
            zip(evaluation_seeds, episode_results),
            start=1,
        ):
            detail_rows.append({
                "agent": name,
                "episode": episode_number,
                "seed": int(seed),
                **result,
            })

        print(
            f"{name} | Mean reward: {np.mean(rewards):.2f} "
            f"| Parameters: {policy.weights.size} "
            f"| Training time: "
            f"{training_result['training_time_seconds']:.2f}s"
        )

    env.close()

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.DataFrame(detail_rows)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    details_df.to_csv(DETAILS_PATH, index=False)

    print("\nBENCHMARK SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))

    print(f"\nSummary saved to: {SUMMARY_PATH}")
    print(f"Episode details saved to: {DETAILS_PATH}")


if __name__ == "__main__":
    main()
