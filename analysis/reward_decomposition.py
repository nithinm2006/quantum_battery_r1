
"""Analyse energy, battery wear and terminal penalty contributions.

Reads existing classical-versus-quantum benchmark detail CSVs.
Writes aggregate analysis CSVs and a visualization.
Does not modify original experiment results or model checkpoints.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_FILES = {
    "deterministic": (
        RESULTS_DIR / "classical_quantum_benchmark_details.csv"
    ),
    "stochastic": (
        RESULTS_DIR / "classical_quantum_stochastic_details.csv"
    ),
}

COMPONENTS = [
    "energy_cost",
    "wear_penalty",
    "terminal_penalty",
]

REQUIRED_COLUMNS = {
    "agent",
    "evaluation_mode",
    "episode",
    "seed",
    "reward",
    "energy_cost",
    "wear_penalty",
    "terminal_penalty",
    "final_soc",
    "terminal_soc_met",
}

OUTPUT_SUMMARY = ANALYSIS_DIR / "reward_decomposition_summary.csv"
OUTPUT_DETAILS = ANALYSIS_DIR / "reward_decomposition_details.csv"
OUTPUT_FIGURE = FIGURES_DIR / "reward_decomposition.png"


def load_experiment_data():
    """Load and validate deterministic and stochastic episode results."""

    frames = []

    for mode, path in INPUT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required experiment file not found: {path}"
            )

        dataframe = pd.read_csv(path)

        missing = REQUIRED_COLUMNS - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )

        dataframe["evaluation_mode"] = mode
        frames.append(dataframe)

    combined = pd.concat(frames, ignore_index=True)

    numeric_columns = [
        "episode",
        "seed",
        "reward",
        "energy_cost",
        "wear_penalty",
        "terminal_penalty",
        "final_soc",
    ]

    for column in numeric_columns:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce",
        )

    if combined[numeric_columns].isna().any().any():
        raise ValueError(
            "Experiment data contains missing or invalid numeric values."
        )

    if not np.isfinite(
        combined[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Experiment data contains infinite numeric values."
        )

    return combined


def calculate_reward_consistency(dataframe):
    """Check whether reward equals the negative sum of its components."""

    dataframe = dataframe.copy()

    dataframe["reconstructed_reward"] = -dataframe[
        COMPONENTS
    ].sum(axis=1)

    dataframe["reward_reconstruction_error"] = (
        dataframe["reward"] - dataframe["reconstructed_reward"]
    )

    dataframe["reward_consistent"] = np.isclose(
        dataframe["reward"],
        dataframe["reconstructed_reward"],
        atol=1e-6,
    )

    return dataframe


def calculate_decomposition(dataframe):
    """Aggregate reward components and calculate their cost shares."""

    grouping = ["evaluation_mode", "agent"]

    summary = (
        dataframe.groupby(grouping)
        .agg(
            episodes=("episode", "count"),
            mean_reward=("reward", "mean"),
            std_reward=("reward", "std"),
            total_energy_cost=("energy_cost", "sum"),
            total_wear_penalty=("wear_penalty", "sum"),
            total_terminal_penalty=("terminal_penalty", "sum"),
            mean_final_soc=("final_soc", "mean"),
            terminal_soc_success_rate=(
                "terminal_soc_met",
                "mean",
            ),
            reward_consistency_rate=(
                "reward_consistent",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["total_cost"] = (
        summary["total_energy_cost"]
        + summary["total_wear_penalty"]
        + summary["total_terminal_penalty"]
    )

    for component in COMPONENTS:
        total_column = f"total_{component}"

        share_column = (
            component.replace("_cost", "")
            .replace("_penalty", "")
            + "_cost_share_percent"
        )

        summary[share_column] = np.where(
            summary["total_cost"] > 0,
            summary[total_column] / summary["total_cost"] * 100,
            0.0,
        )

    return summary


def print_findings(summary):
    """Print a readable interpretation of the decomposition."""

    print("\nREWARD DECOMPOSITION FINDINGS")
    print("=" * 65)

    for _, row in summary.iterrows():
        print(
            f"\n{row['agent']} | "
            f"{row['evaluation_mode'].upper()}"
        )

        print(f"  Episodes: {int(row['episodes'])}")
        print(f"  Mean reward: {row['mean_reward']:.3f}")
        print(f"  Total cost: {row['total_cost']:.3f}")

        print(
            f"  Energy cost: "
            f"{row['total_energy_cost']:.3f} "
            f"({row['energy_cost_share_percent']:.2f}%)"
        )

        print(
            f"  Wear penalty: "
            f"{row['total_wear_penalty']:.3f} "
            f"({row['wear_cost_share_percent']:.2f}%)"
        )

        print(
            f"  Terminal penalty: "
            f"{row['total_terminal_penalty']:.3f} "
            f"({row['terminal_cost_share_percent']:.2f}%)"
        )

        print(
            f"  Terminal SOC success: "
            f"{row['terminal_soc_success_rate'] * 100:.2f}%"
        )

        print(
            f"  Reward consistency: "
            f"{row['reward_consistency_rate'] * 100:.2f}%"
        )


def create_visualization(summary):
    """Create stacked cost-component comparison charts."""

    agents = summary["agent"].unique()
    modes = ["deterministic", "stochastic"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    labels = [
        "Energy cost",
        "Wear penalty",
        "Terminal penalty",
    ]

    columns = [
        "total_energy_cost",
        "total_wear_penalty",
        "total_terminal_penalty",
    ]

    for ax, mode in zip(axes, modes):
        subset = summary[
            summary["evaluation_mode"] == mode
        ].set_index("agent")

        bottoms = np.zeros(len(agents))

        for column, label in zip(columns, labels):
            values = np.array([
                subset.loc[agent, column]
                if agent in subset.index else 0.0
                for agent in agents
            ])

            ax.bar(
                agents,
                values,
                bottom=bottoms,
                label=label,
            )

            bottoms += values

        ax.set_title(f"{mode.title()} Evaluation")
        ax.set_ylabel("Aggregate cost across episodes")
        ax.tick_params(axis="x", rotation=15)
        ax.legend()

    fig.suptitle("Reward Component Decomposition")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 65)
    print("QUANTUM BATTERY REWARD DECOMPOSITION")
    print("=" * 65)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_experiment_data()
    dataframe = calculate_reward_consistency(dataframe)

    summary = calculate_decomposition(dataframe)

    dataframe.to_csv(OUTPUT_DETAILS, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    create_visualization(summary)
    print_findings(summary)

    inconsistent = int(
        (~dataframe["reward_consistent"]).sum()
    )

    print("\nOUTPUT FILES")
    print(f"Summary: {OUTPUT_SUMMARY}")
    print(f"Details: {OUTPUT_DETAILS}")
    print(f"Figure: {OUTPUT_FIGURE}")

    print(
        f"\nReward accounting mismatches: {inconsistent}"
    )

    if inconsistent:
        print(
            "WARNING: Some recorded rewards do not match "
            "their component totals."
        )
        return 1

    print("\nREWARD DECOMPOSITION COMPLETED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
