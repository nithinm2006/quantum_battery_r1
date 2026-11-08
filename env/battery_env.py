
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BatteryEnv(gym.Env):

    def __init__(
        self,
        capacity=100.0,
        max_power=25.0,
        initial_soc=0.5,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        wear_cost=0.05,
        terminal_penalty_rate=20.0,
        price_noise_std=0.4,
        demand_noise_std=2.0,
    ):
        super().__init__()

        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if max_power <= 0:
            raise ValueError("max_power must be positive.")
        if not 0.0 <= initial_soc <= 1.0:
            raise ValueError("initial_soc must be between 0 and 1.")
        if not 0 < charge_efficiency <= 1:
            raise ValueError("charge_efficiency must be in (0, 1].")
        if not 0 < discharge_efficiency <= 1:
            raise ValueError("discharge_efficiency must be in (0, 1].")
        if price_noise_std < 0 or demand_noise_std < 0:
            raise ValueError("Noise standard deviations cannot be negative.")

        self.capacity = capacity
        self.max_power = max_power
        self.initial_soc = initial_soc

        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency

        self.wear_cost = wear_cost
        self.terminal_penalty_rate = terminal_penalty_rate

        self.price_noise_std = price_noise_std
        self.demand_noise_std = demand_noise_std

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        self.current_step = 0
        self.soc = 0.0
        self.initial_soc_kwh = 0.0

        self.prices = None
        self.demands = None

    def generate_daily_data(self):
        hours = np.arange(24)

        morning_peak = 2.0 * np.exp(
            -((hours - 8) / 2) ** 2
        )

        evening_peak = 4.0 * np.exp(
            -((hours - 19) / 3) ** 2
        )

        midday_discount = 1.5 * np.exp(
            -((hours - 13) / 3) ** 2
        )

        self.prices = (
            7.0
            + morning_peak
            + evening_peak
            - midday_discount
            + self.np_random.normal(
                0,
                self.price_noise_std,
                24
            )
        )

        self.prices = np.clip(self.prices, 3, 15)

        self.demands = (
            35
            + 8 * np.exp(-((hours - 8) / 2) ** 2)
            + 18 * np.exp(-((hours - 19) / 3) ** 2)
            + self.np_random.normal(
                0,
                self.demand_noise_std,
                24
            )
        )

        self.demands = np.clip(self.demands, 5, 100)

    def get_observation(self):
        hour = min(self.current_step, 23)

        observation = np.array([
            self.soc / self.capacity,
            self.prices[hour] / 15.0,
            self.demands[hour] / 100.0,
            hour / 23.0
        ], dtype=np.float32)

        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.soc = self.capacity * self.initial_soc
        self.initial_soc_kwh = self.soc

        self.generate_daily_data()

        observation = self.get_observation()

        info = {
            "soc": self.soc
        }

        return observation, info

    def step(self, action):
        if self.current_step >= 24:
            raise RuntimeError(
                "Episode has ended. Call reset() before step()."
            )

        if not self.action_space.contains(action):
            raise ValueError("Invalid action. Use 0, 1 or 2.")

        hour = self.current_step

        price = self.prices[hour]
        demand = self.demands[hour]

        charge_energy = 0.0
        discharge_energy = 0.0
        throughput = 0.0

        if action == 0:
            available_space = self.capacity - self.soc

            charge_energy = min(
                self.max_power,
                available_space / self.charge_efficiency
            )

            stored_energy = (
                charge_energy * self.charge_efficiency
            )

            self.soc += stored_energy
            throughput = stored_energy

        elif action == 1:
            discharge_energy = min(
                self.max_power,
                self.soc * self.discharge_efficiency
            )

            battery_energy_used = (
                discharge_energy / self.discharge_efficiency
            )

            self.soc -= battery_energy_used
            throughput = battery_energy_used

        grid_energy = (
            demand + charge_energy - discharge_energy
        )

        energy_cost = grid_energy * price
        wear_penalty = throughput * self.wear_cost

        self.current_step += 1

        terminated = self.current_step >= 24
        truncated = False

        terminal_shortfall = 0.0
        terminal_penalty = 0.0

        if terminated:
            terminal_shortfall = max(
                0.0,
                self.initial_soc_kwh - self.soc
            )

            terminal_penalty = (
                terminal_shortfall * self.terminal_penalty_rate
            )

        reward = (
            -energy_cost
            - wear_penalty
            - terminal_penalty
        )

        observation = self.get_observation()

        info = {
            "hour": hour,
            "action": action,
            "price": price,
            "demand": demand,
            "soc": self.soc,
            "grid_energy": grid_energy,
            "energy_cost": energy_cost,
            "wear_penalty": wear_penalty,
            "terminal_shortfall": terminal_shortfall,
            "terminal_penalty": terminal_penalty,
            "terminal_soc_target": self.initial_soc_kwh,
            "terminal_soc_met": (
                self.soc >= self.initial_soc_kwh - 1e-6
            )
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    def close(self):
        pass
