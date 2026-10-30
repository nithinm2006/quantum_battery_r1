
from env.battery_env import BatteryEnv


def run_strategy(name, action, seed=42):

    env = BatteryEnv()

    state, info = env.reset(seed=seed)

    total_reward = 0.0
    total_energy_cost = 0.0
    total_wear = 0.0
    total_terminal_penalty = 0.0

    for hour in range(24):

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_energy_cost += info["energy_cost"]
        total_wear += info["wear_penalty"]
        total_terminal_penalty += info["terminal_penalty"]

        # Check battery capacity constraints
        assert 0 <= info["soc"] <= env.capacity

        if terminated or truncated:
            break

    print("\n" + "=" * 50)
    print("STRATEGY:", name)
    print("=" * 50)

    print("Starting SOC:", info["terminal_soc_target"], "kWh")
    print("Final SOC:", round(info["soc"], 2), "kWh")

    print("Energy Cost:", round(total_energy_cost, 2))
    print("Battery Wear:", round(total_wear, 2))
    print("Terminal Penalty:", round(total_terminal_penalty, 2))

    print("Total Reward:", round(total_reward, 2))
    print("Terminal SOC Met:", info["terminal_soc_met"])

    # Verify reward calculation
    expected_reward = (
        -total_energy_cost
        - total_wear
        - total_terminal_penalty
    )

    assert np.isclose(total_reward, expected_reward)

    env.close()

    return total_reward, info


import numpy as np


# Strategy A: Always Hold
hold_reward, hold_info = run_strategy(
    "Always Hold", 2
)

# Strategy B: Always Discharge
discharge_reward, discharge_info = run_strategy(
    "Always Discharge", 1
)
