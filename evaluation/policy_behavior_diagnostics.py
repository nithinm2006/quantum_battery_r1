
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.battery_env import BatteryEnv
from agents.quantum_policy import QuantumPolicy
from agents.rule_based_agent import RuleBasedAgent



MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

EPISODES = 30
SEED = 1000

ACTION_NAMES = {
    0: "Charge",
    1: "Discharge",
    2: "Hold",
}


def load_quantum_agent():
    weights_path = MODELS_DIR / "quantum_policy_best_weights.npy"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Quantum weights not found: {weights_path}"
        )

    agent = QuantumPolicy()
    agent.set_weights(np.load(weights_path))
    return agent


def choose_action(agent, observation):
    try:
        return int(agent.select_action(observation, training=False))
    except TypeError:
        return int(agent.select_action(observation))


def evaluate_agent(agent, agent_name):
    action_counts = {name: 0 for name in ACTION_NAMES.values()}
    total_switches = 0
    total_steps = 0
    episode_records = []

    for episode in range(EPISODES):
        env = BatteryEnv()
        observation, _ = env.reset(seed=SEED + episode)

        previous_action = None
        episode_switches = 0
        episode_counts = {name: 0 for name in ACTION_NAMES.values()}

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = choose_action(agent, observation)

            if action not in ACTION_NAMES:
                raise ValueError(f"Invalid action returned: {action}")

            action_name = ACTION_NAMES[action]
            action_counts[action_name] += 1
            episode_counts[action_name] += 1

            if previous_action is not None and action != previous_action:
                episode_switches += 1

            previous_action = action
            total_steps += 1

            observation, _, terminated, truncated, _ = env.step(action)

        env.close()

        total_switches += episode_switches

        episode_records.append({
            "agent": agent_name,
            "episode": episode + 1,
            "charge": episode_counts["Charge"],
            "discharge": episode_counts["Discharge"],
            "hold": episode_counts["Hold"],
            "action_switches": episode_switches,
        })

    percentages = {
        action: (count / total_steps) * 100
        for action, count in action_counts.items()
    }

    dominant_action = max(action_counts, key=action_counts.get)
    dominant_percentage = percentages[dominant_action]

    summary = {
        "agent": agent_name,
        "total_steps": total_steps,
        "charge_count": action_counts["Charge"],
        "discharge_count": action_counts["Discharge"],
        "hold_count": action_counts["Hold"],
        "charge_percentage": percentages["Charge"],
        "discharge_percentage": percentages["Discharge"],
        "hold_percentage": percentages["Hold"],
        "total_action_switches": total_switches,
        "average_switches_per_episode": total_switches / EPISODES,
        "dominant_action": dominant_action,
        "dominant_action_percentage": dominant_percentage,
        "potentially_stuck": dominant_percentage >= 95.0,
    }

    return summary, episode_records


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    agents = [
        (load_quantum_agent(), "Quantum"),
        (RuleBasedAgent(), "Rule-Based"),
    ]

    summaries = []
    all_episode_records = []

    for agent, name in agents:
        print(f"\nEvaluating {name} agent...")

        summary, episodes = evaluate_agent(agent, name)

        summaries.append(summary)
        all_episode_records.extend(episodes)

        print(f"Dominant action: {summary['dominant_action']}")
        print(
            f"Action distribution: "
            f"Charge={summary['charge_percentage']:.2f}%, "
            f"Discharge={summary['discharge_percentage']:.2f}%, "
            f"Hold={summary['hold_percentage']:.2f}%"
        )
        print(
            f"Average action switches per episode: "
            f"{summary['average_switches_per_episode']:.2f}"
        )
        print(
            f"Potentially stuck: {summary['potentially_stuck']}"
        )

    summary_df = pd.DataFrame(summaries)
    episode_df = pd.DataFrame(all_episode_records)

    summary_df.to_csv(
        RESULTS_DIR / "policy_behavior_summary.csv",
        index=False
    )

    episode_df.to_csv(
        RESULTS_DIR / "policy_behavior_episode_details.csv",
        index=False
    )

    # Plot action distribution
    plot_df = summary_df.set_index("agent")[
        ["charge_percentage", "discharge_percentage", "hold_percentage"]
    ]

    plot_df.columns = ["Charge", "Discharge", "Hold"]

    ax = plot_df.plot(
        kind="bar",
        figsize=(9, 5),
        ylim=(0, 100),
        rot=0
    )

    ax.set_title("Agent Action Distribution")
    ax.set_ylabel("Action Percentage (%)")
    ax.set_xlabel("Agent")
    ax.legend(title="Action")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "policy_action_distribution.png",
        dpi=300
    )
    plt.close()

    print("\nDiagnostics completed.")
    print(f"Summary: {RESULTS_DIR / 'policy_behavior_summary.csv'}")
    print(
        f"Episode details: "
        f"{RESULTS_DIR / 'policy_behavior_episode_details.csv'}"
    )
    print(
        f"Chart: "
        f"{FIGURES_DIR / 'policy_action_distribution.png'}"
    )


if __name__ == "__main__":
    main()
