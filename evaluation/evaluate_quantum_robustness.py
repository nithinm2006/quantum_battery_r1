
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
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = EVALUATION_SEED

PAIRED_COMPARISON_PATH = (
    RESULTS_DIR / "quantum_robustness_paired_comparison.csv"
)

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

    q_table_path = (
        PROJECT_ROOT / "models" / "q_learning_table.npy"
    )

    if not q_table_path.exists():
        raise FileNotFoundError(
            "Q-learning table not found. "
            "Train the Q-learning agent first."
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
            raise ValueError(
                f"Unknown agent type: {agent_type}"
            )

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


def paired_bootstrap_comparison(
    quantum_rewards,
    comparison_rewards,
    n_bootstrap=BOOTSTRAP_SAMPLES,
    seed=BOOTSTRAP_SEED,
):
    """Compare paired episode rewards against the quantum policy.

    Positive differences favour the comparison agent.
    """

    quantum_rewards = np.asarray(
        quantum_rewards,
        dtype=float,
    )

    comparison_rewards = np.asarray(
        comparison_rewards,
        dtype=float,
    )

    if quantum_rewards.shape != comparison_rewards.shape:
        raise ValueError(
            "Quantum and comparison rewards must have "
            "the same shape."
        )

    if quantum_rewards.ndim != 1 or len(quantum_rewards) == 0:
        raise ValueError(
            "Reward arrays must be non-empty one-dimensional arrays."
        )

    if not (
        np.isfinite(quantum_rewards).all()
        and np.isfinite(comparison_rewards).all()
    ):
        raise ValueError(
            "Reward arrays must contain finite values."
        )

    if n_bootstrap < 1:
        raise ValueError(
            "n_bootstrap must be at least 1."
        )

    differences = comparison_rewards - quantum_rewards

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(n_bootstrap)

    for index in range(n_bootstrap):
        sampled = rng.choice(
            differences,
            size=len(differences),
            replace=True,
        )

        bootstrap_means[index] = np.mean(sampled)

    ci_lower, ci_upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    wins = int(np.sum(differences > 1e-9))
    losses = int(np.sum(differences < -1e-9))
    ties = int(np.sum(np.isclose(
        differences,
        0.0,
        atol=1e-9,
    )))

    return {
        "scenarios_compared": len(differences),
        "comparison_agent_wins": wins,
        "quantum_wins": losses,
        "ties": ties,
        "comparison_agent_win_rate": wins / len(differences),
        "mean_reward_difference_vs_quantum": float(
            np.mean(differences)
        ),
        "median_reward_difference_vs_quantum": float(
            np.median(differences)
        ),
        "reward_difference_ci_95_lower": float(ci_lower),
        "reward_difference_ci_95_upper": float(ci_upper),
    }


def build_paired_comparisons(details):
    """Build paired comparisons for every scenario and baseline agent."""

    records = []

    for scenario in SCENARIOS:
        scenario_data = details[
            details["scenario"] == scenario
        ]

        quantum_data = (
            scenario_data[
                scenario_data["agent"] == "Quantum Policy"
            ]
            .set_index("seed")
            .sort_index()
        )

        if quantum_data.empty:
            raise ValueError(
                f"No quantum results found for scenario: {scenario}"
            )

        for agent_name in (
            "Rule-Based",
            "Tabular Q-Learning",
        ):
            comparison_data = (
                scenario_data[
                    scenario_data["agent"] == agent_name
                ]
                .set_index("seed")
                .sort_index()
            )

            if comparison_data.empty:
                raise ValueError(
                    f"No results found for {agent_name} "
                    f"in scenario {scenario}"
                )

            common_seeds = quantum_data.index.intersection(
                comparison_data.index
            )

            if len(common_seeds) == 0:
                raise ValueError(
                    f"No common evaluation seeds for "
                    f"{agent_name} and Quantum Policy "
                    f"in scenario {scenario}"
                )

            quantum_rewards = quantum_data.loc[
                common_seeds,
                "total_reward",
            ].to_numpy()

            comparison_rewards = comparison_data.loc[
                common_seeds,
                "total_reward",
            ].to_numpy()

            stats = paired_bootstrap_comparison(
                quantum_rewards,
                comparison_rewards,
                seed=BOOTSTRAP_SEED,
            )

            records.append({
                "scenario": scenario,
                "comparison_agent": agent_name,
                **stats,
            })

    return pd.DataFrame(records)


def evaluate_robustness():
    agents = load_agents()
    records = []

    for scenario_index, (scenario_name, settings) in enumerate(
        SCENARIOS.items()
    ):
        print(f"\nScenario: {scenario_name}")

        env = BatteryEnv(**settings)

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

    normal_rewards = (
        summary[summary["scenario"] == "normal"]
        [["agent", "mean_reward"]]
        .rename(columns={
            "mean_reward": "normal_mean_reward"
        })
    )

    summary = summary.merge(
        normal_rewards,
        on="agent",
        how="left",
        validate="many_to_one",
    )

    summary["reward_change"] = (
        summary["mean_reward"]
        - summary["normal_mean_reward"]
    )

    summary["reward_change_percent"] = np.where(
        summary["normal_mean_reward"].abs() > 1e-9,
        (
            summary["reward_change"]
            / summary["normal_mean_reward"].abs()
        ) * 100,
        np.nan,
    )

    paired_comparison = build_paired_comparisons(details)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    details_path = (
        RESULTS_DIR / "quantum_robustness_details.csv"
    )

    summary_path = (
        RESULTS_DIR / "quantum_robustness_summary.csv"
    )

    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    paired_comparison.to_csv(
        PAIRED_COMPARISON_PATH,
        index=False,
    )

    print("\n=== Robustness Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Paired Statistical Comparisons ===")
    print(paired_comparison.to_string(index=False))

    print("\nSaved:")
    print(details_path)
    print(summary_path)
    print(PAIRED_COMPARISON_PATH)

    return details, summary, paired_comparison


if __name__ == "__main__":
    evaluate_robustness()
