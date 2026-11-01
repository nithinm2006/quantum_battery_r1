import numpy as np


class QLearningAgent:
    """
    Tabular Q-learning agent for the BatteryEnv.

    Actions:
        0 = Charge
        1 = Discharge
        2 = Hold

    Observation:
        [normalized SOC, normalized price,
         normalized demand, normalized hour]
    """

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        state_bins=10,
        seed=None
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.state_bins = state_bins
        self.rng = np.random.default_rng(seed)

        # SOC, price, demand, hour
        # Hour uses 24 discrete values: 0 through 23
        self.state_shape = (
            state_bins,
            state_bins,
            state_bins,
            24
        )

        # Three possible actions: charge, discharge, hold
        self.q_table = np.zeros(
            self.state_shape + (3,),
            dtype=np.float32
        )

    def discretize_state(self, observation):
        """
        Convert the continuous observation into a discrete
        state index suitable for the Q-table.
        """
        soc, price, demand, hour = observation

        soc_bin = min(
            int(soc * self.state_bins),
            self.state_bins - 1
        )

        price_bin = min(
            int(price * self.state_bins),
            self.state_bins - 1
        )

        demand_bin = min(
            int(demand * self.state_bins),
            self.state_bins - 1
        )

        hour_bin = int(round(hour * 23))
        hour_bin = int(np.clip(hour_bin, 0, 23))

        return (
            soc_bin,
            price_bin,
            demand_bin,
            hour_bin
        )

    def select_action(self, observation, training=True):
        """
        Select an action using epsilon-greedy exploration.
        """
        state = self.discretize_state(observation)

        # Exploration: choose a random action
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 3))

        # Exploitation: choose the action with highest Q-value
        q_values = self.q_table[state]
        best_actions = np.flatnonzero(
            q_values == np.max(q_values)
        )

        # Randomly break ties between equally valued actions
        return int(self.rng.choice(best_actions))

    def update(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminated
    ):
        """
        Update the Q-value using the Q-learning equation:

        Q(s,a) = Q(s,a) + alpha *
                 [r + gamma * max Q(s',a') - Q(s,a)]
        """
        state = self.discretize_state(observation)
        next_state = self.discretize_state(next_observation)

        current_q = self.q_table[state][action]

        if terminated:
            target = reward
        else:
            next_max_q = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * next_max_q

        self.q_table[state][action] = (
            current_q
            + self.learning_rate * (target - current_q)
        )

    def decay_epsilon(self):
        """
        Reduce exploration gradually after each episode.
        """
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )