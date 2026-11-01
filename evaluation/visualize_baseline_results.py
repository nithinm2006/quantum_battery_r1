import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)


def main():
    results_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "baseline_evaluation.csv"
    )

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            "Baseline evaluation CSV not found. "
            "Run evaluation/evaluate_baseline.py first."
        )

    df = pd.read_csv(results_path)

    required_columns = [
        "episode",
        "total_reward",
        "energy_cost",
        "wear_penalty",
        "terminal_penalty",
        "final_soc_kwh",
        "terminal_soc_met"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in evaluation CSV: {missing_columns}"
        )

    os.makedirs(
        os.path.join(PROJECT_ROOT, "results"),
        exist_ok=True
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 10)
    )

    # 1. Total reward per episode
    axes[0, 0].plot(
        df["episode"],
        df["total_reward"],
        marker="o"
    )
    axes[0, 0].set_title("Total Reward per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].grid(True)

    # 2. Energy cost per episode
    axes[0, 1].plot(
        df["episode"],
        df["energy_cost"],
        marker="o"
    )
    axes[0, 1].set_title("Energy Cost per Episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Energy Cost")
    axes[0, 1].grid(True)

    # 3. Final battery SOC
    axes[1, 0].plot(
        df["episode"],
        df["final_soc_kwh"],
        marker="o"
    )
    axes[1, 0].axhline(
        y=50.0,
        linestyle="--",
        label="Initial SOC Target"
    )
    axes[1, 0].set_title("Final Battery SOC")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("SOC (kWh)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. Cost breakdown averages
    cost_categories = [
        "energy_cost",
        "wear_penalty",
        "terminal_penalty"
    ]

    average_costs = [
        df[column].mean()
        for column in cost_categories
    ]

    axes[1, 1].bar(
        ["Energy", "Wear", "Terminal"],
        average_costs
    )
    axes[1, 1].set_title("Average Cost Breakdown")
    axes[1, 1].set_ylabel("Average Cost")
    axes[1, 1].grid(axis="y")

    plt.suptitle(
        "Rule-Based Battery Agent Evaluation",
        fontsize=16
    )
    plt.tight_layout()

    output_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "baseline_results_visualization.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.show()

    print("\n===== Baseline Summary =====")
    print(f"Mean reward: {df['total_reward'].mean():.2f}")
    print(f"Mean energy cost: {df['energy_cost'].mean():.2f}")
    print(f"Mean final SOC: {df['final_soc_kwh'].mean():.2f} kWh")
    print(
        "Terminal SOC target met: "
        f"{df['terminal_soc_met'].sum()}/{len(df)} episodes"
    )
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()