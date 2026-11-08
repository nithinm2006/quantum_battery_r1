
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.battery_env import BatteryEnv
from agents.quantum_policy import QuantumPolicy
from agents.rule_based_agent import RuleBasedAgent
from agents.q_learning_agent import QLearningAgent

from config.experiment_config import (
    EVALUATION_SEED,
    EXPERIMENT_NAME,
)

RESULTS_DIR = PROJECT_ROOT / "results"

ROBUSTNESS_EPISODES = 30

# Each scenario specifies initial SOC and noise levels.
SCENARIOS = {
    "normal": {
        "initial_soc": 0.5,
        "price_noise_std": 0.4,
        "demand_noise_std": 2.0,
    },
    "unseen_scenarios": {
        "initial_soc": 0.5,
        "price_noise_std": 0.4,
        "demand_noise_std": 2.0,
    },
    "high_price_volatility": {
        "initial_soc": 0.5,
        "price_noise_std": 1.5,
        "demand_noise_std": 2.0,
    },
    "high_demand_volatility": {
        "initial_soc": 0.5,
        "price_noise_std": 0.4,
        "demand_noise_std": 8.0,
    },
    "low_initial_soc": {
        "initial_soc": 0.2,
        "price_noise_std": 0.4,
        "demand_noise_std": 2.0,
    },
    "high_initial_soc": {
        "initial_soc": 0.8,
        "price_noise_std": 0.4,
        "demand_noise_std": 2.0,
    },
}


def load_agents():
    weights_path = (
        PROJECT_ROOT / "models" / "quantum_policy_best_weights.npy"
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            "Best quantum policy weights not found. "
            "Train the quantum policy first."
        )

    quantum_agent = QuantumPolicy()
    quantum_agent.set_weights(np.load(weights_path))

    q_table_path = PROJECT_ROOT / "models" / "q_learning_table.npy"

    if not q_table_path.exists():
        raise FileNotFoundError(
            "Q-learning table not found. Train the Q-learning agent first."
        )

    q_learning_agent = QLearningAgent()
    q_learning_agent.q_table = np.load(q_table_path)

    return [
        ("Quantum Policy", quantum_agent, "quantum"),
        ("Rule-Based", RuleBasedAgent(), "rule"),
        ("Tabular Q-Learning", q_learning_agent, "q_learning"),
    ]


def run_episode(env, agent, agent_type, seed):
    observation, _ = env.reset(seed=int(seed))

    total_reward = 0.0
    total_energy_cost = 0.0
    total_wear_penalty = 0.0
    total_terminal_penalty = 0.0

    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        if agent_type in ("quantum", "rule"):
            action = agent.select_action(observation)

        elif agent_type == "q_learning":
            action = agent.select_action(
                observation,
                training=False,
            )

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward
        total_energy_cost += info["energy_cost"]
        total_wear_penalty += info["wear_penalty"]
        total_terminal_penalty += info["terminal_penalty"]

    return {
        "total_reward": float(total_reward),
        "energy_cost": float(total_energy_cost),
        "wear_penalty": float(total_wear_penalty),
        "terminal_penalty": float(total_terminal_penalty),
        "final_soc": float(info["soc"]),
        "terminal_soc_met": bool(info["terminal_soc_met"]),
    }


def evaluate_robustness():
    agents = load_agents()
    records = []

    for scenario_index, (scenario_name, settings) in enumerate(
        SCENARIOS.items()
    ):
        print(f"\nScenario: {scenario_name}")

        env = BatteryEnv(**settings)

        # Separate seed range for each scenario.
        # The unseen scenario uses seeds not used by the original
        # evaluation set (1000-1029).
        scenario_seed_start = (
            EVALUATION_SEED
            + 10_000
            + scenario_index * 1_000
        )

        if scenario_name == "unseen_scenarios":
            scenario_seed_start = EVALUATION_SEED + 50_000

        scenario_seeds = [
            scenario_seed_start + i
            for i in range(ROBUSTNESS_EPISODES)
        ]

        try:
            for agent_name, agent, agent_type in agents:
                print(f"  Evaluating {agent_name}...")

                for episode, seed in enumerate(
                    scenario_seeds,
                    start=1,
                ):
                    metrics = run_episode(
                        env,
                        agent,
                        agent_type,
                        seed,
                    )

                    records.append({
                        "experiment": EXPERIMENT_NAME,
                        "scenario": scenario_name,
                        "agent": agent_name,
                        "episode": episode,
                        "seed": seed,
                        "initial_soc": settings["initial_soc"],
                        "price_noise_std": settings["price_noise_std"],
                        "demand_noise_std": settings["demand_noise_std"],
                        **metrics,
                    })

        finally:
            env.close()

    details = pd.DataFrame(records)

    # Calculate metrics for each agent under each condition.
    summary = (
        details.groupby(["scenario", "agent"])
        .agg(
            episodes=("total_reward", "count"),
            mean_reward=("total_reward", "mean"),
            std_reward=("total_reward", "std"),
            mean_energy_cost=("energy_cost", "mean"),
            mean_wear_penalty=("wear_penalty", "mean"),
            mean_terminal_penalty=("terminal_penalty", "mean"),
            mean_final_soc=("final_soc", "mean"),
            terminal_soc_success_rate=("terminal_soc_met", "mean"),
        )
        .reset_index()
    )

    # Compare each agent against its own normal-condition reward.
    normal_rewards = (
        summary[summary["scenario"] == "normal"]
        [["agent", "mean_reward"]]
        .rename(columns={"mean_reward": "normal_mean_reward"})
    )

    summary = summary.merge(
        normal_rewards,
        on="agent",
        how="left",
        validate="many_to_one",
    )

    summary["reward_change"] = (
        summary["mean_reward"] - summary["normal_mean_reward"]
    )

    summary["reward_change_percent"] = np.where(
        summary["normal_mean_reward"].abs() > 1e-9,
        (
            summary["reward_change"]
            / summary["normal_mean_reward"].abs()
        ) * 100,
        np.nan,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    details_path = (
        RESULTS_DIR / "quantum_robustness_details.csv"
    )

    summary_path = (
        RESULTS_DIR / "quantum_robustness_summary.csv"
    )

    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== Robustness Summary ===")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(details_path)
    print(summary_path)

    return details, summary


if __name__ == "__main__":
    evaluate_robustness()
