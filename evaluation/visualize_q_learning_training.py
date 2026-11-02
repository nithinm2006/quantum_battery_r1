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
    history_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "q_learning_training_history.csv"
    )

    if not os.path.exists(history_path):
        raise FileNotFoundError(
            "Training history not found. "
            "Run training/train_q_learning.py first."
        )

    df = pd.read_csv(history_path)

    required_columns = [
        "episode",
        "total_reward",
        "epsilon",
        "final_soc_kwh"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in training history: {missing}"
        )

    # Calculate a rolling reward average
    window = min(25, len(df))
    df["rolling_reward"] = (
        df["total_reward"]
        .rolling(window=window, min_periods=1)
        .mean()
    )

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 12),
        sharex=True
    )

    # Episode rewards and rolling average
    axes[0].plot(
        df["episode"],
        df["total_reward"],
        alpha=0.35,
        label="Episode Reward"
    )
    axes[0].plot(
        df["episode"],
        df["rolling_reward"],
        linewidth=2,
        label=f"{window}-Episode Moving Average"
    )
    axes[0].set_title("Q-Learning Reward During Training")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()
    axes[0].grid(True)

    # Exploration decay
    axes[1].plot(
        df["episode"],
        df["epsilon"]
    )
    axes[1].set_title("Exploration Rate (Epsilon)")
    axes[1].set_ylabel("Epsilon")
    axes[1].grid(True)

    # Final SOC over training
    axes[2].plot(
        df["episode"],
        df["final_soc_kwh"],
        alpha=0.7
    )
    axes[2].axhline(
        y=50.0,
        linestyle="--",
        label="Initial SOC Target"
    )
    axes[2].set_title("Final Battery SOC per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Final SOC (kWh)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    output_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "q_learning_training_visualization.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.show()

    print("\n===== Training Summary =====")
    print(f"Episodes: {len(df)}")
    print(
        "Mean reward (last 50 episodes): "
        f"{df['total_reward'].tail(50).mean():.2f}"
    )
    print(f"Final epsilon: {df['epsilon'].iloc[-1]:.3f}")
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()