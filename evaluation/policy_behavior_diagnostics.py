
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.battery_env import BatteryEnv
from agents.quantum_policy import QuantumPolicy
from agents.rule_based_agent import RuleBasedAgent


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

EPISODES = 30
SEED = 1000

COLLAPSE_THRESHOLD = 95.0

ACTION_NAMES = {
    0: "Charge",
    1: "Discharge",
    2: "Hold",
}


# --------------------------------------------------
# Load Quantum Policy
# --------------------------------------------------

def load_quantum_agent():
    weights_path = MODELS_DIR / "quantum_policy_best_weights.npy"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Quantum weights not found: {weights_path}"
        )

    agent = QuantumPolicy()
    agent.set_weights(np.load(weights_path))

    return agent


# --------------------------------------------------
# Action Selection
# --------------------------------------------------

def choose_action(agent, observation, stochastic=False, rng=None):
    """
    Select an action using either deterministic or stochastic policy logic.

    Agents without sample_action() use their deterministic
    select_action() method.
    """

    if stochastic and hasattr(agent, "sample_action"):
        return int(
            agent.sample_action(observation, rng=rng)
        )

    return int(agent.select_action(observation))


# --------------------------------------------------
# Evaluate One Agent
# --------------------------------------------------

def evaluate_agent(
    agent,
    agent_name,
    stochastic=False,
    mode_name="deterministic"
):
    action_counts = {
        name: 0 for name in ACTION_NAMES.values()
    }

    total_switches = 0
    total_steps = 0

    episode_records = []

    for episode in range(EPISODES):

        episode_seed = SEED + episode

        env = BatteryEnv()

        observation, _ = env.reset(
            seed=episode_seed
        )

        # Separate reproducible RNG for action sampling.
        action_rng = np.random.default_rng(
            episode_seed
        )

        previous_action = None

        episode_switches = 0

        episode_counts = {
            name: 0 for name in ACTION_NAMES.values()
        }

        episode_reward = 0.0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action = choose_action(
                agent,
                observation,
                stochastic=stochastic,
                rng=action_rng
            )

            if action not in ACTION_NAMES:
                raise ValueError(
                    f"Invalid action returned: {action}"
                )

            action_name = ACTION_NAMES[action]

            action_counts[action_name] += 1
            episode_counts[action_name] += 1

            if (
                previous_action is not None
                and action != previous_action
            ):
                episode_switches += 1

            previous_action = action

            total_steps += 1

            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            episode_reward += reward

        env.close()

        total_switches += episode_switches

        episode_records.append({
            "agent": agent_name,
            "evaluation_mode": mode_name,
            "episode": episode + 1,
            "seed": episode_seed,
            "charge": episode_counts["Charge"],
            "discharge": episode_counts["Discharge"],
            "hold": episode_counts["Hold"],
            "action_switches": episode_switches,
            "total_reward": float(episode_reward),
            "final_soc": float(info["soc"]),
            "terminal_soc_met": bool(
                info["terminal_soc_met"]
            ),
        })

    # --------------------------------------------------
    # Aggregate action statistics
    # --------------------------------------------------

    percentages = {
        action: (
            count / total_steps * 100.0
            if total_steps else 0.0
        )
        for action, count in action_counts.items()
    }

    dominant_action = max(
        action_counts,
        key=action_counts.get
    )

    dominant_percentage = percentages[dominant_action]

    summary = {
        "agent": agent_name,
        "evaluation_mode": mode_name,
        "episodes": EPISODES,
        "total_steps": total_steps,

        "charge_count": action_counts["Charge"],
        "discharge_count": action_counts["Discharge"],
        "hold_count": action_counts["Hold"],

        "charge_percentage": percentages["Charge"],
        "discharge_percentage": percentages["Discharge"],
        "hold_percentage": percentages["Hold"],

        "total_action_switches": total_switches,

        "average_switches_per_episode": (
            total_switches / EPISODES
        ),

        "dominant_action": dominant_action,
        "dominant_action_percentage": dominant_percentage,

        "potentially_stuck": (
            dominant_percentage >= COLLAPSE_THRESHOLD
        ),

        "mean_episode_reward": float(np.mean([
            item["total_reward"]
            for item in episode_records
        ])),

        "mean_final_soc": float(np.mean([
            item["final_soc"]
            for item in episode_records
        ])),

        "terminal_soc_success_rate": float(np.mean([
            item["terminal_soc_met"]
            for item in episode_records
        ])),
    }

    return summary, episode_records


# --------------------------------------------------
# Run Diagnostics
# --------------------------------------------------

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    quantum_agent = load_quantum_agent()
    rule_based_agent = RuleBasedAgent()

    evaluation_configs = [
        (
            quantum_agent,
            "Quantum",
            False,
            "deterministic"
        ),
        (
            quantum_agent,
            "Quantum",
            True,
            "stochastic"
        ),
        (
            rule_based_agent,
            "Rule-Based",
            False,
            "deterministic"
        ),
    ]

    summaries = []
    all_episode_records = []

    for (
        agent,
        agent_name,
        stochastic,
        mode_name
    ) in evaluation_configs:

        print(
            f"\nEvaluating {agent_name} "
            f"({mode_name})..."
        )

        summary, episodes = evaluate_agent(
            agent=agent,
            agent_name=agent_name,
            stochastic=stochastic,
            mode_name=mode_name
        )

        summaries.append(summary)
        all_episode_records.extend(episodes)

        print(
            f"Action distribution: "
            f"Charge={summary['charge_percentage']:.2f}%, "
            f"Discharge={summary['discharge_percentage']:.2f}%, "
            f"Hold={summary['hold_percentage']:.2f}%"
        )

        print(
            f"Average switches per episode: "
            f"{summary['average_switches_per_episode']:.2f}"
        )

        print(
            f"Mean episode reward: "
            f"{summary['mean_episode_reward']:.2f}"
        )

        print(
            f"Dominant action: "
            f"{summary['dominant_action']} "
            f"({summary['dominant_action_percentage']:.2f}%)"
        )

        print(
            f"Potentially stuck: "
            f"{summary['potentially_stuck']}"
        )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    summary_df = pd.DataFrame(summaries)
    episode_df = pd.DataFrame(all_episode_records)

    summary_path = (
        RESULTS_DIR / "policy_behavior_summary.csv"
    )

    episode_path = (
        RESULTS_DIR / "policy_behavior_episode_details.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    episode_df.to_csv(
        episode_path,
        index=False
    )

    # --------------------------------------------------
    # Plot Action Distribution
    # --------------------------------------------------

    plot_df = summary_df.set_index(
        ["agent", "evaluation_mode"]
    )[
        [
            "charge_percentage",
            "discharge_percentage",
            "hold_percentage"
        ]
    ]

    plot_df.columns = [
        "Charge",
        "Discharge",
        "Hold"
    ]

    ax = plot_df.plot(
        kind="bar",
        figsize=(11, 6),
        ylim=(0, 100),
        rot=15
    )

    ax.set_title(
        "Action Distribution Across Evaluation Modes"
    )

    ax.set_ylabel("Action Percentage (%)")
    ax.set_xlabel("Agent and Evaluation Mode")

    ax.legend(title="Action")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    chart_path = (
        FIGURES_DIR / "policy_action_distribution.png"
    )

    plt.savefig(
        chart_path,
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # Plot Action Switches
    # --------------------------------------------------

    switch_plot = summary_df.set_index(
        ["agent", "evaluation_mode"]
    )["average_switches_per_episode"]

    ax = switch_plot.plot(
        kind="bar",
        figsize=(10, 5),
        rot=15
    )

    ax.set_title(
        "Average Action Switches per Episode"
    )

    ax.set_ylabel("Average Action Switches")
    ax.set_xlabel("Agent and Evaluation Mode")

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    switch_chart_path = (
        FIGURES_DIR / "policy_action_switches.png"
    )

    plt.savefig(
        switch_chart_path,
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # Final Report
    # --------------------------------------------------

    print("\nPOLICY BEHAVIOR DIAGNOSTICS SUMMARY")
    print("=" * 75)

    print(
        summary_df[
            [
                "agent",
                "evaluation_mode",
                "charge_percentage",
                "discharge_percentage",
                "hold_percentage",
                "average_switches_per_episode",
                "mean_episode_reward",
                "dominant_action",
                "potentially_stuck"
            ]
        ].to_string(index=False)
    )

    print("\nDiagnostics completed successfully.")

    print(f"Summary: {summary_path}")
    print(f"Episode details: {episode_path}")
    print(f"Action distribution chart: {chart_path}")
    print(f"Action switches chart: {switch_chart_path}")


if __name__ == "__main__":
    main()
