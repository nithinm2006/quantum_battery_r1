
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


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results"

SUMMARY_PATH = RESULTS_DIR / "classical_quantum_benchmark.csv"
DETAILS_PATH = RESULTS_DIR / "classical_quantum_benchmark_details.csv"
COMPARISON_PATH = RESULTS_DIR / "classical_quantum_paired_comparison.csv"

STOCHASTIC_SUMMARY_PATH = (
    RESULTS_DIR / "classical_quantum_stochastic_summary.csv"
)
STOCHASTIC_DETAILS_PATH = (
    RESULTS_DIR / "classical_quantum_stochastic_details.csv"
)
STOCHASTIC_COMPARISON_PATH = (
    RESULTS_DIR / "classical_quantum_stochastic_comparison.csv"
)

HIDDEN_UNITS = 8
BOOTSTRAP_SAMPLES = 5000


# --------------------------------------------------
# Classical Neural Policy
# --------------------------------------------------

class ClassicalNeuralPolicy:
    """Feed-forward neural policy: 4 inputs -> 8 hidden -> 3 actions."""

    def __init__(
        self,
        input_size=4,
        hidden_size=HIDDEN_UNITS,
        output_size=3,
        seed=42
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed)

        # Xavier-style initialization.
        self.w1 = rng.normal(
            0.0,
            np.sqrt(2.0 / (input_size + hidden_size)),
            size=(input_size, hidden_size)
        )
        self.b1 = np.zeros(hidden_size)

        self.w2 = rng.normal(
            0.0,
            np.sqrt(2.0 / (hidden_size + output_size)),
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
                f"Expected flattened weight vector of length "
                f"{expected_size}, got {weights.shape}"
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
        """Deterministic action selection."""
        return int(
            np.argmax(self.action_probabilities(observation))
        )

    def sample_action(self, observation, rng=None):
        """Sample an action using the policy probability distribution."""

        probabilities = self.action_probabilities(observation)

        generator = self.rng if rng is None else rng

        return int(
            generator.choice(self.output_size, p=probabilities)
        )


# --------------------------------------------------
# Episode Evaluation
# --------------------------------------------------

def run_episode(
    env,
    policy,
    seed,
    stochastic=False,
    action_seed=None
):
    """
    Run one battery episode.

    Deterministic mode uses the highest-probability action.
    Stochastic mode samples from the policy distribution.

    Environment and action randomness are independently seeded.
    """

    observation, _ = env.reset(seed=int(seed))

    actual_action_seed = (
        int(seed) if action_seed is None else int(action_seed)
    )

    action_rng = np.random.default_rng(actual_action_seed)

    total_reward = 0.0
    total_energy_cost = 0.0
    total_wear_penalty = 0.0
    total_terminal_penalty = 0.0

    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):

        if stochastic:
            action = policy.sample_action(
                observation,
                rng=action_rng
            )
        else:
            action = policy.select_action(observation)

        observation, reward, terminated, truncated, info = env.step(
            action
        )

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


def evaluate_policy(
    env,
    policy,
    seeds,
    stochastic=False
):
    """Evaluate a policy on a fixed set of environment seeds."""

    return [
        run_episode(
            env,
            policy,
            seed,
            stochastic=stochastic,
            action_seed=int(seed)
        )
        for seed in seeds
    ]


# --------------------------------------------------
# Paired Statistical Comparison
# --------------------------------------------------

def compare_policies_paired(
    quantum_results,
    classical_results,
    n_bootstrap=BOOTSTRAP_SAMPLES,
    seed=42
):
    """
    Compare policies episode-by-episode.

    Positive reward differences favor the quantum policy.
    """

    if len(quantum_results) != len(classical_results):
        raise ValueError(
            "Both policies must have the same number of evaluation episodes."
        )

    if not quantum_results:
        raise ValueError("Evaluation results cannot be empty.")

    quantum_rewards = np.array(
        [item["reward"] for item in quantum_results],
        dtype=float
    )

    classical_rewards = np.array(
        [item["reward"] for item in classical_results],
        dtype=float
    )

    differences = quantum_rewards - classical_rewards

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sampled = rng.choice(
            differences,
            size=len(differences),
            replace=True
        )

        bootstrap_means[i] = np.mean(sampled)

    ci_lower, ci_upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    quantum_wins = int(np.sum(differences > 1e-9))
    classical_wins = int(np.sum(differences < -1e-9))
    ties = int(
        np.sum(np.isclose(differences, 0.0, atol=1e-9))
    )

    return {
        "scenarios_compared": len(differences),
        "quantum_wins": quantum_wins,
        "classical_wins": classical_wins,
        "ties": ties,
        "quantum_win_rate": quantum_wins / len(differences),
        "mean_reward_difference": float(np.mean(differences)),
        "median_reward_difference": float(np.median(differences)),
        "reward_difference_ci_95_lower": float(ci_lower),
        "reward_difference_ci_95_upper": float(ci_upper),
    }


# --------------------------------------------------
# SPSA Training
# --------------------------------------------------

def train_with_spsa(policy, env, validation_seeds, label):
    """Train either policy with identical SPSA settings."""

    rng = np.random.default_rng(TRAINING_SEED)

    weights = policy.weights.copy()
    best_weights = weights.copy()

    initial_results = evaluate_policy(
        env,
        policy,
        validation_seeds
    )

    best_reward = float(np.mean([
        result["reward"] for result in initial_results
    ]))

    start_time = time.perf_counter()

    for iteration in range(1, TRAINING_ITERATIONS + 1):

        delta = rng.choice(
            [-1.0, 1.0],
            size=weights.shape
        )

        plus_weights = weights + SPSA_PERTURBATION * delta
        minus_weights = weights - SPSA_PERTURBATION * delta

        # Common random numbers for both perturbations.
        episode_seeds = rng.integers(
            0,
            1_000_000,
            size=EPISODES_PER_ESTIMATE
        )

        policy.set_weights(plus_weights)

        plus_results = evaluate_policy(
            env,
            policy,
            episode_seeds
        )

        reward_plus = float(np.mean([
            result["reward"] for result in plus_results
        ]))

        policy.set_weights(minus_weights)

        minus_results = evaluate_policy(
            env,
            policy,
            episode_seeds
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
            env,
            policy,
            validation_seeds
        )

        validation_reward = float(np.mean([
            result["reward"] for result in validation_results
        ]))

        if validation_reward > best_reward:
            best_reward = validation_reward
            best_weights = weights.copy()

        if iteration % 10 == 0 or iteration == 1:
            print(
                f"{label} | Iteration "
                f"{iteration}/{TRAINING_ITERATIONS} "
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


# --------------------------------------------------
# Inference Benchmark
# --------------------------------------------------

def benchmark_inference(policy, evaluation_seeds):
    """Measure average deterministic policy decision time."""

    env = BatteryEnv()
    timings = []

    for seed in evaluation_seeds:

        observation, _ = env.reset(seed=int(seed))

        terminated = False
        truncated = False

        while not (terminated or truncated):

            start = time.perf_counter()

            action = policy.select_action(observation)

            timings.append(
                time.perf_counter() - start
            )

            observation, _, terminated, truncated, _ = env.step(
                action
            )

    env.close()

    return (
        float(np.mean(timings) * 1000.0)
        if timings else 0.0
    )


# --------------------------------------------------
# Main Experiment
# --------------------------------------------------

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    validation_seeds = np.random.default_rng(
        TRAINING_SEED + VALIDATION_SEED_OFFSET
    ).integers(
        0,
        1_000_000,
        size=VALIDATION_EPISODES
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
            seed=TRAINING_SEED
        ),

        "Classical Neural Policy": ClassicalNeuralPolicy(
            input_size=4,
            hidden_size=HIDDEN_UNITS,
            output_size=3,
            seed=TRAINING_SEED
        ),
    }

    summary_rows = []
    detail_rows = []
    stochastic_summary_rows = []
    stochastic_detail_rows = []

    evaluation_results_by_agent = {}
    stochastic_results_by_agent = {}

    print("\nCLASSICAL VS QUANTUM POLICY BENCHMARK")
    print("=" * 65)

    # --------------------------------------------------
    # Train and evaluate both policies
    # --------------------------------------------------

    for name, policy in policies.items():

        print(f"\nTraining {name}...")

        training_result = train_with_spsa(
            policy,
            env,
            validation_seeds,
            name
        )

        # Deterministic evaluation on unseen seeds.
        evaluation_start = time.perf_counter()

        episode_results = evaluate_policy(
            env,
            policy,
            evaluation_seeds,
            stochastic=False
        )

        evaluation_time = (
            time.perf_counter() - evaluation_start
        )

        evaluation_results_by_agent[name] = episode_results

        # Stochastic evaluation on the same environment seeds.
        print(f"\nRunning stochastic evaluation for {name}...")

        stochastic_results = evaluate_policy(
            env,
            policy,
            evaluation_seeds,
            stochastic=True
        )

        stochastic_results_by_agent[name] = stochastic_results

        # Deterministic metrics.
        rewards = [
            item["reward"] for item in episode_results
        ]

        energy_costs = [
            item["energy_cost"] for item in episode_results
        ]

        final_socs = [
            item["final_soc"] for item in episode_results
        ]

        success_rates = [
            item["terminal_soc_met"]
            for item in episode_results
        ]

        inference_ms = benchmark_inference(
            policy,
            evaluation_seeds
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
            "terminal_soc_success_rate": float(
                np.mean(success_rates)
            ),
            "training_time_seconds": training_result[
                "training_time_seconds"
            ],
            "evaluation_time_seconds": float(evaluation_time),
            "mean_inference_time_ms": inference_ms,
        })

        # Store deterministic episode details separately.
        for episode_number, (seed, result) in enumerate(
            zip(evaluation_seeds, episode_results),
            start=1
        ):
            detail_rows.append({
                "agent": name,
                "evaluation_mode": "deterministic",
                "episode": episode_number,
                "seed": int(seed),
                **result,
            })

        # Stochastic metrics.
        stochastic_rewards = [
            item["reward"] for item in stochastic_results
        ]

        stochastic_energy_costs = [
            item["energy_cost"] for item in stochastic_results
        ]

        stochastic_summary_rows.append({
            "agent": name,
            "episodes": len(stochastic_results),
            "mean_reward": float(
                np.mean(stochastic_rewards)
            ),
            "std_reward": float(
                np.std(stochastic_rewards)
            ),
            "mean_energy_cost": float(
                np.mean(stochastic_energy_costs)
            ),
            "mean_final_soc": float(np.mean([
                item["final_soc"]
                for item in stochastic_results
            ])),
            "terminal_soc_success_rate": float(np.mean([
                item["terminal_soc_met"]
                for item in stochastic_results
            ])),
        })

        # Store stochastic episode details separately.
        for episode_number, (seed, result) in enumerate(
            zip(evaluation_seeds, stochastic_results),
            start=1
        ):
            stochastic_detail_rows.append({
                "agent": name,
                "evaluation_mode": "stochastic",
                "episode": episode_number,
                "seed": int(seed),
                **result,
            })

        print(
            f"\n{name} | Mean deterministic reward: "
            f"{np.mean(rewards):.2f} "
            f"| Mean stochastic reward: "
            f"{np.mean(stochastic_rewards):.2f}"
        )

        print(
            f"Parameters: {policy.weights.size} "
            f"| Training time: "
            f"{training_result['training_time_seconds']:.2f}s"
        )

    env.close()

    # --------------------------------------------------
    # Deterministic paired comparison
    # --------------------------------------------------

    paired_result = compare_policies_paired(
        quantum_results=evaluation_results_by_agent[
            "Quantum Policy"
        ],
        classical_results=evaluation_results_by_agent[
            "Classical Neural Policy"
        ],
        n_bootstrap=BOOTSTRAP_SAMPLES,
        seed=EVALUATION_SEED
    )

    paired_result["comparison"] = (
        "Quantum Policy vs Classical Neural Policy"
    )

    comparison_df = pd.DataFrame([paired_result])

    # --------------------------------------------------
    # Stochastic paired comparison
    # --------------------------------------------------

    stochastic_paired_result = compare_policies_paired(
        quantum_results=stochastic_results_by_agent[
            "Quantum Policy"
        ],
        classical_results=stochastic_results_by_agent[
            "Classical Neural Policy"
        ],
        n_bootstrap=BOOTSTRAP_SAMPLES,
        seed=EVALUATION_SEED
    )

    stochastic_paired_result["comparison"] = (
        "Stochastic Quantum Policy vs Classical Neural Policy"
    )

    stochastic_comparison_df = pd.DataFrame([
        stochastic_paired_result
    ])

    # --------------------------------------------------
    # Create DataFrames
    # --------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.DataFrame(detail_rows)

    stochastic_summary_df = pd.DataFrame(
        stochastic_summary_rows
    )

    stochastic_details_df = pd.DataFrame(
        stochastic_detail_rows
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    summary_df.to_csv(SUMMARY_PATH, index=False)
    details_df.to_csv(DETAILS_PATH, index=False)

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False
    )

    stochastic_summary_df.to_csv(
        STOCHASTIC_SUMMARY_PATH,
        index=False
    )

    stochastic_details_df.to_csv(
        STOCHASTIC_DETAILS_PATH,
        index=False
    )

    stochastic_comparison_df.to_csv(
        STOCHASTIC_COMPARISON_PATH,
        index=False
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\nDETERMINISTIC BENCHMARK SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))

    print("\nDETERMINISTIC PAIRED COMPARISON")
    print("=" * 65)
    print(comparison_df.to_string(index=False))

    print("\nSTOCHASTIC BENCHMARK SUMMARY")
    print("=" * 65)
    print(stochastic_summary_df.to_string(index=False))

    print("\nSTOCHASTIC PAIRED COMPARISON")
    print("=" * 65)
    print(stochastic_comparison_df.to_string(index=False))

    print("\nSAVED EXPERIMENT RESULTS")
    print("=" * 65)

    print(f"Deterministic summary: {SUMMARY_PATH}")
    print(f"Deterministic details: {DETAILS_PATH}")
    print(f"Deterministic comparison: {COMPARISON_PATH}")

    print(f"Stochastic summary: {STOCHASTIC_SUMMARY_PATH}")
    print(f"Stochastic details: {STOCHASTIC_DETAILS_PATH}")
    print(f"Stochastic comparison: {STOCHASTIC_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
