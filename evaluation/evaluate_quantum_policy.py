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


def load_quantum_policy():
    # Evaluate the best policy selected during checkpointed training.
    weights_path = (
        PROJECT_ROOT / "models" / "quantum_policy_best_weights.npy"
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            "Best quantum policy weights not found. "
            "Run quantum training with checkpointing first."
        )

    policy = QuantumPolicy()
    weights = np.load(weights_path)
    policy.set_weights(weights)

    print(f"Loaded best quantum policy from: {weights_path}")
    return policy


def load_q_learning_agent():
    table_path = PROJECT_ROOT / "models" / "q_learning_table.npy"

    if not table_path.exists():
        raise FileNotFoundError(
            "Q-learning table not found. Train the Q-learning agent first."
        )

    agent = QLearningAgent()
    agent.q_table = np.load(table_path)
    return agent


def run_episode(env, agent, seed, agent_type):
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
            # Greedy evaluation: disable training exploration.
            action = agent.select_action(observation, training=False)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        observation, reward, terminated, truncated, info = env.step(action)

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


def evaluate_agents(num_episodes=30, seed=1000):
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")

    env = BatteryEnv()

    try:
        quantum_agent = load_quantum_policy()
        rule_agent = RuleBasedAgent()
        q_learning_agent = load_q_learning_agent()

        agents = [
            ("Quantum Policy", quantum_agent, "quantum"),
            ("Rule-Based", rule_agent, "rule"),
            ("Tabular Q-Learning", q_learning_agent, "q_learning"),
        ]

        records = []
        scenario_seeds = [
            int(seed + i) for i in range(num_episodes)
        ]

        for agent_name, agent, agent_type in agents:
            print(f"Evaluating {agent_name}...")

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
                    "agent": agent_name,
                    "episode": episode,
                    "seed": scenario_seed,
                    **metrics,
                })

        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(exist_ok=True)

        details = pd.DataFrame(records)
        details.to_csv(
            results_dir / "quantum_agent_comparison_details.csv",
            index=False,
        )

        summary = (
            details.groupby("agent")
            .agg(
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

        summary.to_csv(
            results_dir / "quantum_agent_comparison_summary.csv",
            index=False,
        )

        print("\nAgent comparison summary:")
        print(summary.to_string(index=False))

        print("\nSaved comparison results:")
        print(results_dir / "quantum_agent_comparison_details.csv")
        print(results_dir / "quantum_agent_comparison_summary.csv")

        return details, summary

    finally:
        env.close()


if __name__ == "__main__":
    evaluate_agents()