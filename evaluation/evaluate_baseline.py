import os
import sys

# Add the project root to Python's import path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from env.battery_env import BatteryEnv
from agents.rule_based_agent import RuleBasedAgent


def evaluate_agent(num_episodes=30):
    results = []

    for episode in range(num_episodes):
        env = BatteryEnv()
        agent = RuleBasedAgent()

        observation, info = env.reset(seed=episode)

        terminated = False
        truncated = False

        total_reward = 0.0
        total_energy_cost = 0.0
        total_wear_penalty = 0.0
        total_terminal_penalty = 0.0

        while not (terminated or truncated):
            action = agent.select_action(observation)

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward
            total_energy_cost += info["energy_cost"]
            total_wear_penalty += info["wear_penalty"]
            total_terminal_penalty += info["terminal_penalty"]

        results.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "energy_cost": total_energy_cost,
            "wear_penalty": total_wear_penalty,
            "terminal_penalty": total_terminal_penalty,
            "final_soc_kwh": env.soc,
            "terminal_soc_met": info["terminal_soc_met"]
        })

        env.close()

    return pd.DataFrame(results)


def main():
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_df = evaluate_agent(num_episodes=30)

    output_path = os.path.join(
        results_dir,
        "baseline_evaluation.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\n===== Rule-Based Baseline Evaluation =====")
    print(f"Episodes evaluated: {len(results_df)}")
    print(f"Average reward: {results_df['total_reward'].mean():.2f}")
    print(f"Average energy cost: {results_df['energy_cost'].mean():.2f}")
    print(f"Average wear penalty: {results_df['wear_penalty'].mean():.2f}")
    print(
        "Average terminal penalty: "
        f"{results_df['terminal_penalty'].mean():.2f}"
    )
    print(
        "Average final SOC: "
        f"{results_df['final_soc_kwh'].mean():.2f} kWh"
    )
    print(
        "Terminal SOC target met: "
        f"{results_df['terminal_soc_met'].sum()}/"
        f"{len(results_df)} episodes"
    )

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()