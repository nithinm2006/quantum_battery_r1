import os
import sys

import numpy as np
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from env.battery_env import BatteryEnv
from agents.q_learning_agent import QLearningAgent


def train_agent(num_episodes=500, seed=42):
    env = BatteryEnv()

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        state_bins=10,
        seed=seed
    )

    training_records = []

    for episode in range(num_episodes):
        episode_seed = seed + episode
        observation, info = env.reset(seed=episode_seed)

        terminated = False
        truncated = False

        total_reward = 0.0
        total_energy_cost = 0.0
        total_wear_penalty = 0.0
        total_terminal_penalty = 0.0

        while not (terminated or truncated):
            action = agent.select_action(
                observation,
                training=True
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            agent.update(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated
            )

            observation = next_observation

            total_reward += reward
            total_energy_cost += info["energy_cost"]
            total_wear_penalty += info["wear_penalty"]
            total_terminal_penalty += info["terminal_penalty"]

        agent.decay_epsilon()

        training_records.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "energy_cost": total_energy_cost,
            "wear_penalty": total_wear_penalty,
            "terminal_penalty": total_terminal_penalty,
            "final_soc_kwh": env.soc,
            "terminal_soc_met": info["terminal_soc_met"],
            "epsilon": agent.epsilon
        })

        if (episode + 1) % 50 == 0:
            recent_rewards = [
                record["total_reward"]
                for record in training_records[-50:]
            ]

            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Mean Reward (last 50): "
                f"{np.mean(recent_rewards):.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    env.close()

    return agent, pd.DataFrame(training_records)


def main():
    results_dir = os.path.join(PROJECT_ROOT, "results")
    models_dir = os.path.join(PROJECT_ROOT, "models")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    agent, history_df = train_agent(
        num_episodes=500,
        seed=42
    )

    # Save the trained Q-table
    q_table_path = os.path.join(
        models_dir,
        "q_learning_table.npy"
    )
    np.save(q_table_path, agent.q_table)

    # Save training history
    history_path = os.path.join(
        results_dir,
        "q_learning_training_history.csv"
    )
    history_df.to_csv(history_path, index=False)

    print("\n===== Q-Learning Training Complete =====")
    print(f"Episodes trained: {len(history_df)}")
    print(
        "Mean reward (last 50 episodes): "
        f"{history_df['total_reward'].tail(50).mean():.2f}"
    )
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Q-table saved to: {q_table_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()