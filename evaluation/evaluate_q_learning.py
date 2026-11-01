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
from agents.rule_based_agent import RuleBasedAgent


def evaluate_agent(agent, agent_name, num_episodes=30, seed=1000):
    records = []

    for episode in range(num_episodes):
        env = BatteryEnv()
        observation, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        total_reward = 0.0
        total_energy_cost = 0.0
        total_wear_penalty = 0.0
        total_terminal_penalty = 0.0

        while not (terminated or truncated):
            if agent_name == "Q-Learning":
                action = agent.select_action(
                    observation,
                    training=False
                )
            else:
                action = agent.select_action(observation)

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            total_reward += reward
            total_energy_cost += info["energy_cost"]
            total_wear_penalty += info["wear_penalty"]
            total_terminal_penalty += info["terminal_penalty"]

            observation = next_observation

        records.append({
            "agent": agent_name,
            "episode": episode + 1,
            "total_reward": total_reward,
            "energy_cost": total_energy_cost,
            "wear_penalty": total_wear_penalty,
            "terminal_penalty": total_terminal_penalty,
            "final_soc_kwh": env.soc,
            "terminal_soc_met": info["terminal_soc_met"]
        })

        env.close()

    return records


def main():
    models_dir = os.path.join(PROJECT_ROOT, "models")
    results_dir = os.path.join(PROJECT_ROOT, "results")

    q_table_path = os.path.join(
        models_dir,
        "q_learning_table.npy"
    )

    if not os.path.exists(q_table_path):
        raise FileNotFoundError(
            "Trained Q-table not found. "
            "Run training/train_q_learning.py first."
        )

    os.makedirs(results_dir, exist_ok=True)

    # Create Q-learning agent and load its trained Q-table
    q_agent = QLearningAgent(seed=42)
    q_agent.q_table = np.load(q_table_path)

    # Create rule-based baseline agent
    rule_agent = RuleBasedAgent()

    num_episodes = 30
    evaluation_seed = 1000

    print("Evaluating Q-learning agent...")
    q_results = evaluate_agent(
        q_agent,
        "Q-Learning",
        num_episodes,
        evaluation_seed
    )

    print("Evaluating rule-based agent...")
    rule_results = evaluate_agent(
        rule_agent,
        "Rule-Based",
        num_episodes,
        evaluation_seed
    )

    # Combine results
    results_df = pd.DataFrame(q_results + rule_results)

    # Save per-episode results
    detailed_path = os.path.join(
        results_dir,
        "agent_comparison_details.csv"
    )
    results_df.to_csv(detailed_path, index=False)

    # Calculate average metrics for each agent
    summary_df = (
        results_df
        .groupby("agent")
        .agg(
            mean_reward=("total_reward", "mean"),
            mean_energy_cost=("energy_cost", "mean"),
            mean_wear_penalty=("wear_penalty", "mean"),
            mean_terminal_penalty=("terminal_penalty", "mean"),
            mean_final_soc_kwh=("final_soc_kwh", "mean"),
            terminal_soc_success_rate=("terminal_soc_met", "mean")
        )
        .reset_index()
    )

    summary_df["terminal_soc_success_rate"] *= 100

    summary_path = os.path.join(
        results_dir,
        "agent_comparison_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    print("\n===== Agent Comparison =====")
    print(summary_df.to_string(index=False, float_format="%.2f"))

    print(f"\nDetailed results saved to: {detailed_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()