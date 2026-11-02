import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.battery_env import BatteryEnv
from agents.quantum_policy import QuantumPolicy


def run_episode(env, policy, seed):
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    total_energy_cost = 0.0
    total_wear_penalty = 0.0
    total_terminal_penalty = 0.0

    terminated = False
    while not terminated:
        action = policy.sample_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_energy_cost += info["energy_cost"]
        total_wear_penalty += info["wear_penalty"]
        total_terminal_penalty += info["terminal_penalty"]

    return {
        "reward": total_reward,
        "energy_cost": total_energy_cost,
        "wear_penalty": total_wear_penalty,
        "terminal_penalty": total_terminal_penalty,
        "final_soc": info["soc"],
        "terminal_soc_met": info["terminal_soc_met"],
    }


def evaluate_weights(env, policy, weights, seeds):
    policy.set_weights(weights)
    returns = []

    for seed in seeds:
        result = run_episode(env, policy, seed)
        returns.append(result["reward"])

    return float(np.mean(returns))


def train_quantum_policy(
    iterations=80,
    episodes_per_estimate=2,
    seed=42,
    learning_rate=0.08,
    perturbation=0.1,
):
    """
    SPSA policy optimization.

    Each iteration evaluates two perturbed parameter sets on the same
    environment seeds and estimates a reward gradient.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if episodes_per_estimate < 1:
        raise ValueError("episodes_per_estimate must be positive.")

    rng = np.random.default_rng(seed)
    env = BatteryEnv()
    policy = QuantumPolicy(seed=seed)

    weights = policy.weights.copy()
    history = []

    for iteration in range(1, iterations + 1):
        # Rademacher perturbation: each parameter is +1 or -1.
        delta = rng.choice([-1.0, 1.0], size=weights.shape)

        plus_weights = weights + perturbation * delta
        minus_weights = weights - perturbation * delta

        # Use matching daily scenarios for both perturbations.
        episode_seeds = [
        int(value)
        for value in rng.integers(
                0, 1_000_000, size=episodes_per_estimate
          )
        ]

        reward_plus = evaluate_weights(
            env, policy, plus_weights, episode_seeds
        )
        reward_minus = evaluate_weights(
            env, policy, minus_weights, episode_seeds
        )

        # Scale returns to keep SPSA updates from becoming excessively large.
        gradient_estimate = (
           (reward_plus - reward_minus)
            / (2.0 * perturbation * 1000.0)
        ) * delta

        weights = weights + learning_rate * gradient_estimate
        policy.set_weights(weights)

        # Track a separate sampled rollout for progress reporting.
        report_seed = int(rng.integers(0, 1_000_000))
        report = run_episode(env, policy, report_seed)

        history.append({
            "iteration": iteration,
            "reward_plus": reward_plus,
            "reward_minus": reward_minus,
            "gradient_estimate_norm": float(
                np.linalg.norm(gradient_estimate)
            ),
            "training_reward": report["reward"],
            "energy_cost": report["energy_cost"],
            "wear_penalty": report["wear_penalty"],
            "terminal_penalty": report["terminal_penalty"],
            "final_soc": report["final_soc"],
            "terminal_soc_met": report["terminal_soc_met"],
        })

        if iteration % 10 == 0 or iteration == 1:
            print(
                f"Iteration {iteration}/{iterations} | "
                f"Reward: {report['reward']:.2f} | "
                f"SOC: {report['final_soc']:.1f}"
            )

    env.close()

    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    np.save(models_dir / "quantum_policy_weights.npy", weights)
    pd.DataFrame(history).to_csv(
        results_dir / "quantum_training_history.csv", index=False
    )

    print("Saved quantum policy weights and training history.")
    return pd.DataFrame(history)


if __name__ == "__main__":
    train_quantum_policy()