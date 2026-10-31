import os
import matplotlib.pyplot as plt

from env.battery_env import BatteryEnv
from agents.rule_based_agent import RuleBasedAgent

def demo_policy(price, soc_percent):
    """
    Simple threshold policy for visualization.
    This is not a trained reinforcement learning agent.
    """
    if price < 6.5 and soc_percent < 80:
        return 0  # Charge

    if price > 9.0 and soc_percent > 50:
        return 1  # Discharge

    return 2  # Hold


def main():
    env = BatteryEnv()
    agent = RuleBasedAgent()
    observation, info = env.reset(seed=42)

    hours = []
    prices = []
    demands = []
    soc_values = []
    actions = []

    action_names = {
        0: "Charge",
        1: "Discharge",
        2: "Hold"
    }

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Read current state from the observation vector
        price = float(observation[1] * 15.0)
        demand = float(observation[2] * 100.0)
        soc_percent = float(observation[0] * 100.0)

        action = agent.select_action(observation)

        # Execute the selected action
        observation, reward, terminated, truncated, info = env.step(action)

        # Store data for this hour
        hours.append(info["hour"])
        prices.append(info["price"])
        demands.append(info["demand"])
        soc_values.append(info["soc"])
        actions.append(action_names[action])

    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(
        4, 1,
        figsize=(12, 13),
        sharex=True
    )

    # Electricity price
    axes[0].plot(hours, prices, marker="o")
    axes[0].set_ylabel("Price")
    axes[0].set_title("Electricity Price")
    axes[0].grid(True)

    # Electricity demand
    axes[1].plot(hours, demands, marker="o")
    axes[1].set_ylabel("Demand (kWh)")
    axes[1].set_title("Energy Demand")
    axes[1].grid(True)

    # Battery state of charge
    axes[2].plot(hours, soc_values, marker="o", label="SOC")
    axes[2].axhline(
        y=env.initial_soc_kwh,
        linestyle="--",
        label="Initial SOC"
    )
    axes[2].set_ylabel("SOC (kWh)")
    axes[2].set_title("Battery State of Charge")
    axes[2].legend()
    axes[2].grid(True)

    # Agent actions
    action_values = [
        {"Charge": 1, "Discharge": -1, "Hold": 0}[action]
        for action in actions
    ]

    axes[3].step(hours, action_values, where="mid")
    axes[3].set_yticks([-1, 0, 1])
    axes[3].set_yticklabels(["Discharge", "Hold", "Charge"])
    axes[3].set_xlabel("Hour")
    axes[3].set_ylabel("Action")
    axes[3].set_title("Battery Actions")
    axes[3].grid(True)

    plt.tight_layout()

    output_path = os.path.join(
        "results",
        "environment_visualization.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Visualization saved to: {output_path}")
    print(f"Final SOC: {env.soc:.2f} kWh")

    env.close()


if __name__ == "__main__":
    main()