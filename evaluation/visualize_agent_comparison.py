
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SUMMARY_PATH = RESULTS_DIR / "quantum_agent_comparison_summary.csv"
DETAILS_PATH = RESULTS_DIR / "quantum_agent_comparison_details.csv"
PAIRWISE_PATH = RESULTS_DIR / "quantum_agent_pairwise_comparison.csv"


def save_figure(fig, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def load_results():
    required_files = [
        SUMMARY_PATH,
        DETAILS_PATH,
        PAIRWISE_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required evaluation file not found: {path}\n"
                "Run evaluation/evaluate_quantum_policy.py first."
            )

    summary = pd.read_csv(SUMMARY_PATH)
    details = pd.read_csv(DETAILS_PATH)
    pairwise = pd.read_csv(PAIRWISE_PATH)

    required_summary = {
        "agent",
        "mean_reward",
        "reward_ci_95_lower",
        "reward_ci_95_upper",
        "mean_energy_cost",
        "mean_final_soc",
        "terminal_soc_success_rate",
    }

    required_details = {"agent", "total_reward"}

    required_pairwise = {
        "comparison",
        "quantum_win_rate",
    }

    for name, dataframe, required in [
        ("Summary", summary, required_summary),
        ("Details", details, required_details),
        ("Pairwise", pairwise, required_pairwise),
    ]:
        missing = required - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{name} CSV is missing columns: {sorted(missing)}"
            )

    return summary, details, pairwise


def plot_mean_reward(summary):
    agents = summary["agent"]
    means = summary["mean_reward"].to_numpy()

    lower = summary["reward_ci_95_lower"].to_numpy()
    upper = summary["reward_ci_95_upper"].to_numpy()

    errors = np.vstack([
        means - lower,
        upper - means,
    ])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(agents, means, yerr=errors, capsize=6)

    ax.set_title("Mean Episode Reward with 95% Confidence Intervals")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Mean Reward")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    save_figure(fig, "agent_mean_reward.png")


def plot_reward_distribution(details):
    fig, ax = plt.subplots(figsize=(10, 6))

    agent_names = details["agent"].drop_duplicates().tolist()

    reward_groups = [
        details.loc[
            details["agent"] == agent,
            "total_reward",
        ].dropna().to_numpy()
        for agent in agent_names
    ]

    ax.boxplot(
        reward_groups,
        tick_labels=agent_names,
        showmeans=True,
    )

    ax.set_title("Episode Reward Distribution")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Episode Reward")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    save_figure(fig, "agent_reward_distribution.png")


def plot_energy_cost(summary):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        summary["agent"],
        summary["mean_energy_cost"],
    )

    ax.set_title("Average Energy Cost Comparison")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Mean Energy Cost")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    save_figure(fig, "agent_energy_cost.png")


def plot_final_soc(summary):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        summary["agent"],
        summary["mean_final_soc"],
    )

    ax.set_title("Average Final Battery State of Charge")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Final SOC")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    save_figure(fig, "agent_final_soc.png")


def plot_terminal_success(summary):
    fig, ax = plt.subplots(figsize=(10, 6))

    success_rates = (
        summary["terminal_soc_success_rate"] * 100
    )

    ax.bar(
        summary["agent"],
        success_rates,
    )

    ax.set_title("Terminal SOC Success Rate")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    save_figure(fig, "agent_terminal_soc_success.png")


def plot_quantum_win_rates(pairwise):
    if pairwise.empty:
        print("No pairwise comparison data available.")
        return

    comparisons = pairwise["comparison"]
    win_rates = pairwise["quantum_win_rate"] * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(comparisons, win_rates)

    ax.set_title("Quantum Policy Win Rate Against Baselines")
    ax.set_xlabel("Comparison")
    ax.set_ylabel("Quantum Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.setp(
        ax.get_xticklabels(),
        rotation=10,
        ha="right",
    )

    save_figure(fig, "quantum_policy_win_rate.png")


def main():
    print("Loading evaluation results...")

    summary, details, pairwise = load_results()

    print("\nGenerating agent comparison dashboard...")

    plot_mean_reward(summary)
    plot_reward_distribution(details)
    plot_energy_cost(summary)
    plot_final_soc(summary)
    plot_terminal_success(summary)
    plot_quantum_win_rates(pairwise)

    print("\nDashboard generation completed.")
    print(f"All figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
