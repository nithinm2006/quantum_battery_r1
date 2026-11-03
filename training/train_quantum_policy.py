import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.battery_env import BatteryEnv
from agents.quantum_policy import QuantumPolicy


MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

FINAL_WEIGHTS_PATH = MODELS_DIR / "quantum_policy_weights.npy"
BEST_WEIGHTS_PATH = MODELS_DIR / "quantum_policy_best_weights.npy"
CHECKPOINT_PATH = MODELS_DIR / "quantum_training_checkpoint.npz"
HISTORY_PATH = RESULTS_DIR / "quantum_training_history.csv"


def run_episode(env, policy, seed, deterministic=False):
    observation, _ = env.reset(seed=int(seed))

    total_reward = 0.0
    total_energy_cost = 0.0
    total_wear_penalty = 0.0
    total_terminal_penalty = 0.0

    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        if deterministic:
            action = policy.select_action(observation)
        else:
            action = policy.sample_action(observation)

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


def evaluate_weights(env, policy, weights, seeds):
    """Evaluate weights deterministically on a fixed set of scenarios."""
    policy.set_weights(weights)

    returns = [
        run_episode(
            env,
            policy,
            seed,
            deterministic=True,
        )["reward"]
        for seed in seeds
    ]

    return float(np.mean(returns))


def save_checkpoint(
    weights,
    iteration,
    best_weights,
    best_reward,
    rng,
    history,
):
    """Atomically save enough state to resume training."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "weights": weights.tolist(),
        "iteration": int(iteration),
        "best_weights": best_weights.tolist(),
        "best_reward": float(best_reward),
        "rng_state": rng.bit_generator.state,
        "history": history,
    }

    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")

    with temporary_path.open("wb") as file:
        np.savez(
            file,
            checkpoint_json=np.array(json.dumps(checkpoint_data)),
        )

    temporary_path.replace(CHECKPOINT_PATH)


def load_checkpoint():
    """Load saved training state."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint found at: {CHECKPOINT_PATH}"
        )

    with np.load(CHECKPOINT_PATH, allow_pickle=False) as data:
        checkpoint_data = json.loads(
            str(data["checkpoint_json"].item())
        )

    return checkpoint_data


def train_quantum_policy(
    iterations=80,
    episodes_per_estimate=2,
    seed=42,
    learning_rate=0.08,
    perturbation=0.1,
    resume=False,
):
    """
    SPSA policy optimization with checkpointing.

    When resume=True, training continues from the saved checkpoint.
    The iterations argument is the total target iteration count.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if episodes_per_estimate < 1:
        raise ValueError("episodes_per_estimate must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if perturbation <= 0:
        raise ValueError("perturbation must be positive.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = BatteryEnv()
    policy = QuantumPolicy(seed=seed)
    rng = np.random.default_rng(seed)

    # Fixed validation scenarios used to select the best policy.
    validation_seeds = [
        int(value)
        for value in np.random.default_rng(seed + 10_000).integers(
            0, 1_000_000, size=10
        )
    ]

    if resume:
        checkpoint = load_checkpoint()

        weights = np.asarray(checkpoint["weights"], dtype=float)
        best_weights = np.asarray(
            checkpoint["best_weights"], dtype=float
        )
        start_iteration = int(checkpoint["iteration"]) + 1
        best_reward = float(checkpoint["best_reward"])
        history = checkpoint["history"]

        rng.bit_generator.state = checkpoint["rng_state"]
        policy.set_weights(weights)

        print(
            f"Resuming from iteration {start_iteration - 1}. "
            f"Best validation reward: {best_reward:.2f}"
        )
    else:
        weights = policy.weights.copy()
        best_weights = weights.copy()
        start_iteration = 1
        best_reward = evaluate_weights(
            env, policy, best_weights, validation_seeds
        )
        history = []

        print(
            f"Starting fresh training. Initial validation reward: "
            f"{best_reward:.2f}"
        )

    try:
        for iteration in range(start_iteration, iterations + 1):
            # Rademacher perturbation: each parameter is +1 or -1.
            delta = rng.choice([-1.0, 1.0], size=weights.shape)

            plus_weights = weights + perturbation * delta
            minus_weights = weights - perturbation * delta

            # Matching daily scenarios for both SPSA perturbations.
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

            gradient_estimate = (
                (reward_plus - reward_minus)
                / (2.0 * perturbation * 1000.0)
            ) * delta

            weights = weights + learning_rate * gradient_estimate
            policy.set_weights(weights)

            # Evaluate the updated policy on fixed validation scenarios.
            validation_reward = evaluate_weights(
                env, policy, weights, validation_seeds
            )

            is_best = validation_reward > best_reward
            if is_best:
                best_reward = validation_reward
                best_weights = weights.copy()
                np.save(BEST_WEIGHTS_PATH, best_weights)

            # Separate stochastic rollout for training progress metrics.
            report_seed = int(rng.integers(0, 1_000_000))
            report = run_episode(
                env,
                policy,
                report_seed,
                deterministic=False,
            )

            history.append({
                "iteration": iteration,
                "reward_plus": reward_plus,
                "reward_minus": reward_minus,
                "gradient_estimate_norm": float(
                    np.linalg.norm(gradient_estimate)
                ),
                "training_reward": report["reward"],
                "validation_reward": validation_reward,
                "best_validation_reward": best_reward,
                "is_best_policy": is_best,
                "energy_cost": report["energy_cost"],
                "wear_penalty": report["wear_penalty"],
                "terminal_penalty": report["terminal_penalty"],
                "final_soc": report["final_soc"],
                "terminal_soc_met": report["terminal_soc_met"],
            })

            # Save progress after every iteration.
            save_checkpoint(
                weights=weights,
                iteration=iteration,
                best_weights=best_weights,
                best_reward=best_reward,
                rng=rng,
                history=history,
            )

            if iteration % 10 == 0 or iteration == start_iteration:
                print(
                    f"Iteration {iteration}/{iterations} | "
                    f"Reward: {report['reward']:.2f} | "
                    f"Validation: {validation_reward:.2f} | "
                    f"Best: {best_reward:.2f} | "
                    f"SOC: {report['final_soc']:.1f}"
                )

    finally:
        env.close()

    # Save final policy and complete training history.
    np.save(FINAL_WEIGHTS_PATH, weights)
    np.save(BEST_WEIGHTS_PATH, best_weights)

    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False)

    print("\nTraining completed.")
    print(f"Final weights: {FINAL_WEIGHTS_PATH}")
    print(f"Best weights: {BEST_WEIGHTS_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Training history: {HISTORY_PATH}")
    print(f"Best validation reward: {best_reward:.2f}")

    return history_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the hybrid quantum battery policy."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=80,
        help="Total target number of SPSA iterations.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the saved checkpoint.",
    )
    args = parser.parse_args()

    train_quantum_policy(
        iterations=args.iterations,
        resume=args.resume,
    )