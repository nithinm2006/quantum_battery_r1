from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

DETAILS_PATH = RESULTS_DIR / "quantum_agent_comparison_details.csv"
SUMMARY_PATH = RESULTS_DIR / "quantum_agent_comparison_summary.csv"

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load evaluation results
# --------------------------------------------------

if not DETAILS_PATH.exists():
    raise FileNotFoundError(f"Missing evaluation file: {DETAILS_PATH}")

if not SUMMARY_PATH.exists():
    raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")

details = pd.read_csv(DETAILS_PATH)
summary = pd.read_csv(SUMMARY_PATH)

print("Detailed results columns:")
print(details.columns.tolist())

print("\nSummary results:")
print(summary.to_string(index=False))


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def find_column(dataframe, candidates):
    """Find a column using case-insensitive candidate names."""
    normalized = {
        column.lower().replace(" ", "_"): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]

    return None


def save_bar_chart(data, x_column, y_column, title, ylabel, filename,
                   percentage=False):
    """Create and save a labeled bar chart."""
    if x_column is None or y_column is None:
        print(f"Skipping '{title}': required columns were not found.")
        return

    plot_data = data[[x_column, y_column]].copy()
    plot_data[y_column] = pd.to_numeric(
        plot_data[y_column], errors="coerce"
    )
    plot_data = plot_data.dropna()

    if plot_data.empty:
        print(f"Skipping '{title}': no valid data.")
        return

    if percentage:
        # Convert proportions to percentages if values are in [0, 1].
        if plot_data[y_column].max() <= 1:
            plot_data[y_column] *= 100

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        plot_data[x_column].astype(str),
        plot_data[y_column]
    )

    ax.set_title(title)
    ax.set_xlabel("Agent")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, plot_data[y_column]):
        label = f"{value:.1f}%" if percentage else f"{value:.2f}"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Identify summary columns
# --------------------------------------------------

agent_column = find_column(
    summary,
    ["agent", "agent_name", "policy", "model"]
)

if agent_column is None:
    raise ValueError(
        "Could not identify the agent column in the summary CSV. "
        f"Available columns: {summary.columns.tolist()}"
    )

reward_column = find_column(
    summary,
    ["mean_reward", "average_reward", "avg_reward", "reward_mean"]
)

cost_column = find_column(
    summary,
    ["mean_energy_cost", "average_energy_cost", "avg_energy_cost",
     "energy_cost_mean"]
)

soc_column = find_column(
    summary,
    ["mean_final_soc", "average_final_soc", "avg_final_soc",
     "final_soc_mean"]
)

success_column = find_column(
    summary,
    ["terminal_soc_success_rate", "soc_success_rate",
     "terminal_success_rate", "success_rate"]
)


# --------------------------------------------------
# Generate comparison charts
# --------------------------------------------------

save_bar_chart(
    summary,
    agent_column,
    reward_column,
    title="Average Episode Reward by Agent",
    ylabel="Mean Episode Reward",
    filename="agent_mean_reward.png"
)

save_bar_chart(
    summary,
    agent_column,
    cost_column,
    title="Average Energy Cost by Agent",
    ylabel="Mean Energy Cost",
    filename="agent_energy_cost.png"
)

save_bar_chart(
    summary,
    agent_column,
    soc_column,
    title="Average Final State of Charge by Agent",
    ylabel="Final SOC",
    filename="agent_final_soc.png"
)

save_bar_chart(
    summary,
    agent_column,
    success_column,
    title="Terminal SOC Target Success Rate",
    ylabel="Success Rate (%)",
    filename="agent_terminal_soc_success_rate.png",
    percentage=True
)


# --------------------------------------------------
# Optional: reward distribution from detailed results
# --------------------------------------------------

detail_agent_column = find_column(
    details,
    ["agent", "agent_name", "policy", "model"]
)

detail_reward_column = find_column(
    details,
    ["episode_reward", "total_reward", "reward"]
)

if detail_agent_column and detail_reward_column:
    plot_data = details[
        [detail_agent_column, detail_reward_column]
    ].copy()

    plot_data[detail_reward_column] = pd.to_numeric(
        plot_data[detail_reward_column], errors="coerce"
    )
    plot_data = plot_data.dropna()

    if not plot_data.empty:
        agents = plot_data[detail_agent_column].unique()
        reward_groups = [
            plot_data.loc[
                plot_data[detail_agent_column] == agent,
                detail_reward_column
            ].values
            for agent in agents
        ]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot(reward_groups, tick_labels=agents)

        ax.set_title("Episode Reward Distribution by Agent")
        ax.set_xlabel("Agent")
        ax.set_ylabel("Episode Reward")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        fig.tight_layout()

        output_path = FIGURES_DIR / "agent_reward_distribution.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_path}")

print("\nVisualization completed.")