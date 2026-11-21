
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SUMMARY_PATH = RESULTS_DIR / "quantum_robustness_summary.csv"
PAIRED_PATH = RESULTS_DIR / "quantum_robustness_paired_comparison.csv"

SCENARIO_ORDER = [
    "normal",
    "unseen_scenarios",
    "high_price_volatility",
    "high_demand_volatility",
    "low_initial_soc",
    "high_initial_soc",
]

AGENT_ORDER = [
    "Quantum Policy",
    "Rule-Based",
    "Tabular Q-Learning",
]

AGENT_COLORS = {
    "Quantum Policy": "#4169E1",
    "Rule-Based": "#2E8B57",
    "Tabular Q-Learning": "#DA8B24",
}

SCENARIO_LABELS = {
    "normal": "Normal",
    "unseen_scenarios": "Unseen\nScenarios",
    "high_price_volatility": "High Price\nVolatility",
    "high_demand_volatility": "High Demand\nVolatility",
    "low_initial_soc": "Low Initial\nSOC",
    "high_initial_soc": "High Initial\nSOC",
}


def load_data():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing robustness summary: {SUMMARY_PATH}"
        )

    if not PAIRED_PATH.exists():
        raise FileNotFoundError(
            f"Missing paired comparison: {PAIRED_PATH}"
        )

    summary = pd.read_csv(SUMMARY_PATH)
    paired = pd.read_csv(PAIRED_PATH)

    required_summary = {
        "scenario",
        "agent",
        "mean_reward",
        "std_reward",
        "mean_energy_cost",
        "mean_final_soc",
        "terminal_soc_success_rate",
        "reward_change_percent",
    }

    required_paired = {
        "scenario",
        "comparison_agent",
        "mean_reward_difference_vs_quantum",
        "reward_difference_ci_95_lower",
        "reward_difference_ci_95_upper",
    }

    missing_summary = required_summary - set(summary.columns)
    missing_paired = required_paired - set(paired.columns)

    if missing_summary:
        raise ValueError(
            f"Missing summary columns: {sorted(missing_summary)}"
        )

    if missing_paired:
        raise ValueError(
            f"Missing paired comparison columns: {sorted(missing_paired)}"
        )

    if summary[list(required_summary - {"scenario", "agent"})].isna().any().any():
        raise ValueError("Summary contains missing metric values.")

    if paired[list(required_paired - {"scenario", "comparison_agent"})].isna().any().any():
        raise ValueError("Paired comparison contains missing values.")

    unknown_agents = set(summary["agent"]) - set(AGENT_ORDER)
    unknown_scenarios = set(summary["scenario"]) - set(SCENARIO_ORDER)

    if unknown_agents or unknown_scenarios:
        raise ValueError(
            f"Unexpected agents: {unknown_agents}; "
            f"unexpected scenarios: {unknown_scenarios}"
        )

    for scenario in SCENARIO_ORDER:
        scenario_agents = set(
            summary.loc[summary["scenario"] == scenario, "agent"]
        )

        if scenario_agents != set(AGENT_ORDER):
            raise ValueError(
                f"Incomplete agent results for scenario: {scenario}"
            )

    return summary, paired


def prepare_output():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename):
    output_path = FIGURES_DIR / filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Saved: {output_path}")


def ordered_summary(summary):
    result = summary.copy()

    result["scenario"] = pd.Categorical(
        result["scenario"],
        categories=SCENARIO_ORDER,
        ordered=True,
    )

    result["agent"] = pd.Categorical(
        result["agent"],
        categories=AGENT_ORDER,
        ordered=True,
    )

    return result.sort_values(["scenario", "agent"])


def plot_mean_reward(summary):
    summary = ordered_summary(summary)

    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(SCENARIO_ORDER))
    width = 0.24

    for index, agent in enumerate(AGENT_ORDER):
        data = (
            summary[summary["agent"] == agent]
            .set_index("scenario")
            .reindex(SCENARIO_ORDER)
        )

        positions = x + (index - 1) * width

        ax.bar(
            positions,
            data["mean_reward"],
            width,
            label=agent,
            color=AGENT_COLORS[agent],
            yerr=data["std_reward"],
            capsize=3,
            error_kw={"elinewidth": 0.8},
        )

    ax.set_title(
        "Mean Episode Reward Across Robustness Scenarios",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Mean Episode Reward")
    ax.set_xlabel("Evaluation Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]
    )

    ax.axhline(0, linewidth=0.8, color="black")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    save_figure(fig, "robustness_mean_reward.png")


def plot_terminal_soc_success(summary):
    summary = ordered_summary(summary)

    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(SCENARIO_ORDER))
    width = 0.24

    for index, agent in enumerate(AGENT_ORDER):
        data = (
            summary[summary["agent"] == agent]
            .set_index("scenario")
            .reindex(SCENARIO_ORDER)
        )

        positions = x + (index - 1) * width

        values = data["terminal_soc_success_rate"] * 100

        ax.bar(
            positions,
            values,
            width,
            label=agent,
            color=AGENT_COLORS[agent],
        )

    ax.set_title(
        "Terminal State-of-Charge Success Rate",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Successful Episodes (%)")
    ax.set_xlabel("Evaluation Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]
    )
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "robustness_terminal_soc_success.png")


def plot_energy_cost(summary):
    summary = ordered_summary(summary)

    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(SCENARIO_ORDER))
    width = 0.24

    for index, agent in enumerate(AGENT_ORDER):
        data = (
            summary[summary["agent"] == agent]
            .set_index("scenario")
            .reindex(SCENARIO_ORDER)
        )

        positions = x + (index - 1) * width

        ax.bar(
            positions,
            data["mean_energy_cost"],
            width,
            label=agent,
            color=AGENT_COLORS[agent],
        )

    ax.set_title(
        "Mean Energy Cost Across Robustness Scenarios",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Mean Energy Cost")
    ax.set_xlabel("Evaluation Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "robustness_energy_cost.png")


def plot_reward_change(summary):
    summary = ordered_summary(summary)

    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(SCENARIO_ORDER))
    width = 0.24

    for index, agent in enumerate(AGENT_ORDER):
        data = (
            summary[summary["agent"] == agent]
            .set_index("scenario")
            .reindex(SCENARIO_ORDER)
        )

        positions = x + (index - 1) * width

        ax.bar(
            positions,
            data["reward_change_percent"],
            width,
            label=agent,
            color=AGENT_COLORS[agent],
        )

    ax.axhline(0, color="black", linewidth=1)

    ax.set_title(
        "Reward Change Relative to Each Agent's Normal Scenario",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Reward Change (%)")
    ax.set_xlabel("Evaluation Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "robustness_reward_change.png")


def plot_paired_confidence_intervals(paired):
    fig, ax = plt.subplots(figsize=(12, 8))

    comparisons = [
        ("Rule-Based", "Rule-Based"),
        ("Tabular Q-Learning", "Tabular Q-Learning"),
    ]

    positions = []
    labels = []
    means = []
    lower_errors = []
    upper_errors = []
    colors = []

    position = 0

    for scenario in SCENARIO_ORDER:
        for agent, label in comparisons:
            row = paired[
                (paired["scenario"] == scenario)
                & (paired["comparison_agent"] == agent)
            ]

            if len(row) != 1:
                raise ValueError(
                    f"Expected one paired result for {scenario}, {agent}"
                )

            result = row.iloc[0]

            mean = result["mean_reward_difference_vs_quantum"]
            lower = result["reward_difference_ci_95_lower"]
            upper = result["reward_difference_ci_95_upper"]

            if lower > upper or not lower <= mean <= upper:
                raise ValueError(
                    f"Invalid confidence interval for {scenario}, {agent}"
                )

            positions.append(position)
            labels.append(f"{SCENARIO_LABELS[scenario].replace(chr(10), ' ')}\n{label}")
            means.append(mean)
            lower_errors.append(mean - lower)
            upper_errors.append(upper - mean)
            colors.append(AGENT_COLORS[agent])

            position += 1

        position += 0.6

    y = np.asarray(positions)
    means = np.asarray(means)

    errors = np.vstack([
        lower_errors,
        upper_errors,
    ])

    for index in range(len(y)):
        ax.errorbar(
            means[index],
            y[index],
            xerr=errors[:, index:index + 1],
            fmt="o",
            color=colors[index],
            capsize=4,
            markersize=6,
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        label="No reward difference",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_title(
        "Paired Reward Difference vs Quantum Policy (95% Bootstrap CI)",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Comparison Agent Reward − Quantum Policy Reward"
    )

    ax.grid(axis="x", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "robustness_paired_confidence_intervals.png")


def main():
    print("Loading robustness experiment results...")

    summary, paired = load_data()
    prepare_output()

    print("\nGenerating robustness visualizations...")

    plot_mean_reward(summary)
    plot_terminal_soc_success(summary)
    plot_energy_cost(summary)
    plot_reward_change(summary)
    plot_paired_confidence_intervals(paired)

    print("\nDashboard generation completed.")
    print(f"Figures directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
