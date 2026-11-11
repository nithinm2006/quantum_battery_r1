
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
    VALIDATION_SEED_OFFSET,
    VALIDATION_EPISODES,
)

from env.battery_env import BatteryEnv
from agents.quantum_policy import QuantumPolicy


RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "quantum_ablation_summary.csv"

# Keep the number of qubits fixed and investigate circuit depth.
ARCHITECTURES = [
    {"name": "shallow_1_layer", "n_qubits": 4, "n_layers": 1},
    {"name": "baseline_2_layers", "n_qubits": 4, "n_layers": 2},
    {"name": "deep_3_layers", "n_qubits": 4, "n_layers": 3},
]

# Ablation-specific training settings.
ABLATION_ITERATIONS = 40
EPISODES_PER_ESTIMATE = 2
LEARNING_RATE = 0.08
PERTURBATION = 0.1


def run_episode(env, policy, seed):
    observation, _ = env.reset(seed=int(seed))

    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = policy.sample_action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    return float(total_reward)


def evaluate_policy(env, policy, seeds):
    rewards = []

    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))

        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = policy.select_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        rewards.append(float(total_reward))

    return np.asarray(rewards, dtype=float)


def train_architecture(config, validation_seeds):
    n_qubits = config["n_qubits"]
    n_layers = config["n_layers"]

    policy = QuantumPolicy(
        n_qubits=n_qubits,
        n_layers=n_layers,
        seed=TRAINING_SEED,
    )

    env = BatteryEnv()
    rng = np.random.default_rng(TRAINING_SEED)

    weights = policy.weights.copy()
    best_weights = weights.copy()

    initial_rewards = evaluate_policy(env, policy, validation_seeds)
    best_reward = float(np.mean(initial_rewards))

    training_start = time.perf_counter()

    for iteration in range(1, ABLATION_ITERATIONS + 1):
        delta = rng.choice([-1.0, 1.0], size=weights.shape)

        plus_weights = weights + PERTURBATION * delta
        minus_weights = weights - PERTURBATION * delta

        # Common random numbers for both perturbations.
        episode_seeds = rng.integers(
            0,
            1_000_000,
            size=EPISODES_PER_ESTIMATE,
        )

        policy.set_weights(plus_weights)
        reward_plus = np.mean([
            run_episode(env, policy, seed)
            for seed in episode_seeds
        ])

        policy.set_weights(minus_weights)
        reward_minus = np.mean([
            run_episode(env, policy, seed)
            for seed in episode_seeds
        ])

        gradient = (
            (reward_plus - reward_minus)
            / (2.0 * PERTURBATION * 1000.0)
        ) * delta

        weights = weights + LEARNING_RATE * gradient
        policy.set_weights(weights)

        validation_rewards = evaluate_policy(
            env,
            policy,
            validation_seeds,
        )

        validation_reward = float(np.mean(validation_rewards))

        if validation_reward > best_reward:
            best_reward = validation_reward
            best_weights = weights.copy()

        if iteration % 10 == 0:
            print(
                f"{config['name']} | "
                f"Iteration {iteration}/{ABLATION_ITERATIONS} | "
                f"Validation reward: {validation_reward:.2f} | "
                f"Best: {best_reward:.2f}"
            )

    training_time = time.perf_counter() - training_start

    # Evaluate the best checkpoint, not the final training weights.
    policy.set_weights(best_weights)

    evaluation_start = time.perf_counter()
    final_rewards = evaluate_policy(env, policy, validation_seeds)
    evaluation_time = time.perf_counter() - evaluation_start

    env.close()

    return {
        "architecture": config["name"],
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "trainable_parameters": int(best_weights.size),
        "training_iterations": ABLATION_ITERATIONS,
        "mean_validation_reward": float(np.mean(final_rewards)),
        "std_validation_reward": float(np.std(final_rewards)),
        "best_validation_reward": float(best_reward),
        "training_time_seconds": float(training_time),
        "evaluation_time_seconds": float(evaluation_time),
    }


def run_ablation_study():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    validation_seeds = np.random.default_rng(
        TRAINING_SEED + VALIDATION_SEED_OFFSET
    ).integers(
        0,
        1_000_000,
        size=VALIDATION_EPISODES,
    )

    results = []

    print("\nQUANTUM POLICY ABLATION STUDY")
    print("=" * 60)

    for config in ARCHITECTURES:
        print(
            f"\nTesting {config['name']} "
            f"({config['n_qubits']} qubits, "
            f"{config['n_layers']} layers)"
        )

        result = train_architecture(config, validation_seeds)
        results.append(result)

        print(
            f"Mean validation reward: "
            f"{result['mean_validation_reward']:.2f}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nABLATION RESULTS")
    print("=" * 60)
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    best_architecture = results_df.loc[
        results_df["mean_validation_reward"].idxmax()
    ]

    print("\nBest architecture:")
    print(
        f"{best_architecture['architecture']} "
        f"with mean reward "
        f"{best_architecture['mean_validation_reward']:.2f}"
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")

    return results_df


if __name__ == "__main__":
    run_ablation_study()
