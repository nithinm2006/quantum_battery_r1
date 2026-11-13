
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
    EVALUATION_EPISODES,
    EXPERIMENT_NAME,
)

RESULTS_DIR = PROJECT_ROOT / "results"
BOOTSTRAP_SAMPLES = 5000
CONFIDENCE_LEVEL = 0.95
STATISTICS_SEED = 42
TIE_TOLERANCE = 1e-9


def load_quantum_policy():
    weights_path = (
        PROJECT_ROOT / "models" / "quantum_policy_best_weights.npy"
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            "Best quantum policy weights not found. "
            "Run quantum training with checkpointing first."
        )

    policy = QuantumPolicy()
    policy.set_weights(np.load(weights_path))

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

    total_baseline_grid_cost = 0.0
    total_charge_energy = 0.0
    total_discharge_energy = 0.0
    total_battery_cost_impact = 0.0

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

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_energy_cost += info["energy_cost"]
        baseline_cost = info["demand"] * info["price"]

        total_charge_energy += info["charge_energy"]
        total_discharge_energy += info["discharge_energy"]

        total_battery_cost_impact += (
        info["energy_cost"] - baseline_cost
        )
        total_wear_penalty += info["wear_penalty"]
        total_terminal_penalty += info["terminal_penalty"]

    return {
        "total_reward": float(total_reward),
        "energy_cost": float(total_energy_cost),
        "baseline_grid_cost": float(total_baseline_grid_cost),
        "battery_cost_impact": float(total_battery_cost_impact),
        "battery_savings": float(
                 total_baseline_grid_cost - total_energy_cost
        ),
        "charge_energy": float(total_charge_energy),
        "discharge_energy": float(total_discharge_energy),  
        "wear_penalty": float(total_wear_penalty),
        "terminal_penalty": float(total_terminal_penalty),
        "final_soc": float(info["soc"]),
        "terminal_soc_met": bool(info["terminal_soc_met"]),
    }


def bootstrap_confidence_interval(
    values,
    rng,
    confidence=CONFIDENCE_LEVEL,
    n_samples=BOOTSTRAP_SAMPLES,
):
    """Calculate a percentile bootstrap CI for the sample mean."""
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return float("nan"), float("nan")

    if len(values) == 1:
        value = float(values[0])
        return value, value

    bootstrap_means = np.empty(n_samples)

    for i in range(n_samples):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1.0 - confidence

    lower = np.quantile(bootstrap_means, alpha / 2)
    upper = np.quantile(bootstrap_means, 1 - alpha / 2)

    return float(lower), float(upper)


def create_summary(details):
    """Calculate descriptive statistics and reward confidence intervals."""
    rng = np.random.default_rng(STATISTICS_SEED)

    records = []

    for agent_name, group in details.groupby("agent"):
        rewards = group["total_reward"].to_numpy(dtype=float)

        ci_lower, ci_upper = bootstrap_confidence_interval(
            rewards,
            rng,
        )

        records.append({
            "experiment": EXPERIMENT_NAME,
            "agent": agent_name,
            "episodes": len(group),
            "mean_reward": float(np.mean(rewards)),
            "median_reward": float(np.median(rewards)),
            "std_reward": float(np.std(rewards, ddof=1))
            if len(rewards) > 1 else 0.0,
            "reward_ci_95_lower": ci_lower,
            "reward_ci_95_upper": ci_upper,
            "mean_energy_cost": float(group["energy_cost"].mean()),
            "mean_wear_penalty": float(group["wear_penalty"].mean()),
            "mean_terminal_penalty": float(
                group["terminal_penalty"].mean()
            ),
            "mean_final_soc": float(group["final_soc"].mean()),
            "terminal_soc_success_rate": float(
                group["terminal_soc_met"].mean()
            ),
        })

    return pd.DataFrame(records)


def create_pairwise_comparison(details):
    """Compare quantum rewards against baselines on identical seeds."""
    quantum = details[
        details["agent"] == "Quantum Policy"
    ][["seed", "total_reward"]].rename(
        columns={"total_reward": "quantum_reward"}
    )

    baselines = [
        "Rule-Based",
        "Tabular Q-Learning",
    ]

    records = []

    for baseline_name in baselines:
        baseline = details[
            details["agent"] == baseline_name
        ][["seed", "total_reward"]].rename(
            columns={"total_reward": "baseline_reward"}
        )

        comparison = quantum.merge(
            baseline,
            on="seed",
            how="inner",
            validate="one_to_one",
        )

        if comparison.empty:
            continue

        difference = (
            comparison["quantum_reward"]
            - comparison["baseline_reward"]
        )

        wins = int((difference > TIE_TOLERANCE).sum())
        losses = int((difference < -TIE_TOLERANCE).sum())
        ties = int(len(difference) - wins - losses)

        records.append({
            "experiment": EXPERIMENT_NAME,
            "comparison": f"Quantum Policy vs {baseline_name}",
            "scenarios_compared": len(comparison),
            "quantum_wins": wins,
            "ties": ties,
            "quantum_losses": losses,
            "quantum_win_rate": wins / len(comparison),
            "tie_rate": ties / len(comparison),
            "quantum_loss_rate": losses / len(comparison),
            "mean_reward_difference": float(difference.mean()),
            "median_reward_difference": float(difference.median()),
        })

    return pd.DataFrame(records)


def evaluate_agents(
    num_episodes=EVALUATION_EPISODES,
    seed=EVALUATION_SEED,
):
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

        scenario_seeds = [
            int(seed + i) for i in range(num_episodes)
        ]

        records = []

        for agent_name, agent, agent_type in agents:
            print(f"\nEvaluating {agent_name}...")

            for episode, scenario_seed in enumerate(
                scenario_seeds,
                start=1,
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

                if episode % 10 == 0 or episode == num_episodes:
                    print(
                        f"  Episode {episode}/{num_episodes}"
                    )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        details = pd.DataFrame(records)
        summary = create_summary(details)
        pairwise = create_pairwise_comparison(details)

        details_path = (
            RESULTS_DIR / "quantum_agent_comparison_details.csv"
        )
        summary_path = (
            RESULTS_DIR / "quantum_agent_comparison_summary.csv"
        )
        pairwise_path = (
            RESULTS_DIR / "quantum_agent_pairwise_comparison.csv"
        )

        details.to_csv(details_path, index=False)
        summary.to_csv(summary_path, index=False)
        pairwise.to_csv(pairwise_path, index=False)

        print("\n=== Agent Statistical Summary ===")
        print(summary.to_string(index=False))

        print("\n=== Quantum Policy Pairwise Comparison ===")
        print(pairwise.to_string(index=False))

        print("\nSaved evaluation results:")
        print(details_path)
        print(summary_path)
        print(pairwise_path)

        return details, summary, pairwise

    finally:
        env.close()


if __name__ == "__main__":
    evaluate_agents()
