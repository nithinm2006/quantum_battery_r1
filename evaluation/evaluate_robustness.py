import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.battery_env import BatteryEnv
from agents.rule_based_agent import RuleBasedAgent
from config.experiment_config import EVALUATION_SEED, EXPERIMENT_NAME

from evaluation.evaluate_quantum_policy import (
    load_quantum_policy,
    load_q_learning_agent,
    run_episode,
)


def evaluate_robustness(num_episodes=100, seed=EVALUATION_SEED):
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")

    env = BatteryEnv()

    try:
        agents = [
            ("Quantum Policy", load_quantum_policy(), "quantum"),
            ("Rule-Based", RuleBasedAgent(), "rule"),
            ("Tabular Q-Learning", load_q_learning_agent(), "q_learning"),
        ]

        scenario_seeds = [
            int(seed + i) for i in range(num_episodes)
        ]

        records = []

        for agent_name, agent, agent_type in agents:
            print(f"Robustness evaluation: {agent_name}")

            for episode, scenario_seed in enumerate(
                scenario_seeds, start=1
            ):
                metrics = run_episode(
                    env,
                    agent,
                    scenario_seed,
                    agent_type,
                )

                records.append({
                    "experiment": EXPERIMENT_NAME,
                    "agent": agent_name,
                    "episode": episode,
                    "seed": scenario_seed,
                    **metrics,
                })

        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        details = pd.DataFrame(records)
        details_path = results_dir / "robustness_evaluation_details.csv"
        details.to_csv(details_path, index=False)

        def lower_10_percentile(series):
            return float(series.quantile(0.10))

        summary = (
            details.groupby("agent")
            .agg(
                mean_reward=("total_reward", "mean"),
                std_reward=("total_reward", "std"),
                median_reward=("total_reward", "median"),
                worst_10_percentile_reward=(
                    "total_reward",
                    lower_10_percentile,
                ),
                min_reward=("total_reward", "min"),
                mean_energy_cost=("energy_cost", "mean"),
                mean_wear_penalty=("wear_penalty", "mean"),
                mean_terminal_penalty=("terminal_penalty", "mean"),
                mean_final_soc=("final_soc", "mean"),
                terminal_soc_success_rate=("terminal_soc_met", "mean"),
            )
            .reset_index()
        )

        summary["terminal_soc_success_rate"] *= 100

        summary_path = results_dir / "robustness_evaluation_summary.csv"
        summary.to_csv(summary_path, index=False)

        print("\nRobustness summary:")
        print(summary.to_string(index=False))

        print("\nSaved:")
        print(details_path)
        print(summary_path)

        return details, summary

    finally:
        env.close()


if __name__ == "__main__":
    evaluate_robustness()