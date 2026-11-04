from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
HISTORY_PATH = RESULTS_DIR / "quantum_training_history.csv"

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(
    x,
    y_series,
    title,
    xlabel,
    ylabel,
    filename,
):
    """Plot one or more series and save the figure."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, values in y_series.items():
        ax.plot(x, values, label=label, linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_PATH}"
        )

    history = pd.read_csv(HISTORY_PATH)

    required_columns = [
        "iteration",
        "training_reward",
        "validation_reward",
        "best_validation_reward",
        "gradient_estimate_norm",
        "energy_cost",
        "final_soc",
    ]

    missing = [
        column for column in required_columns
        if column not in history.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in training history: {missing}"
        )

    history = history.sort_values("iteration")

    iterations = history["iteration"]

    # 1. Training reward
    save_plot(
        iterations,
        {"Training Reward": history["training_reward"]},
        "Quantum Policy Training Reward",
        "Iteration",
        "Episode Reward",
        "quantum_training_reward.png",
    )

    # 2. Validation reward and best-so-far reward
    save_plot(
        iterations,
        {
            "Validation Reward": history["validation_reward"],
            "Best Validation Reward": history["best_validation_reward"],
        },
        "Quantum Policy Validation Performance",
        "Iteration",
        "Mean Validation Reward",
        "quantum_validation_reward.png",
    )

    # 3. SPSA gradient estimate norm
    save_plot(
        iterations,
        {
            "Gradient Estimate Norm":
                history["gradient_estimate_norm"]
        },
        "SPSA Gradient Estimate Norm",
        "Iteration",
        "Gradient Norm",
        "quantum_gradient_norm.png",
    )

    # 4. Energy cost
    save_plot(
        iterations,
        {"Energy Cost": history["energy_cost"]},
        "Energy Cost During Quantum Training",
        "Iteration",
        "Energy Cost",
        "quantum_energy_cost.png",
    )

    # 5. Final SOC
    save_plot(
        iterations,
        {"Final SOC": history["final_soc"]},
        "Final Battery SOC During Quantum Training",
        "Iteration",
        "Final SOC",
        "quantum_final_soc.png",
    )

    # 6. Terminal SOC success rate using a rolling window
    if "terminal_soc_met" in history.columns:
        success = history["terminal_soc_met"].astype(float)
        rolling_success = success.rolling(
            window=min(10, len(success)),
            min_periods=1,
        ).mean() * 100

        save_plot(
            iterations,
            {"Rolling Terminal SOC Success Rate": rolling_success},
            "Terminal SOC Success Rate During Training",
            "Iteration",
            "Success Rate (%)",
            "quantum_terminal_soc_success.png",
        )

    print("\nQuantum training visualizations completed.")


if __name__ == "__main__":
    main()