from env.battery_env import BatteryEnv
from agents.q_learning_agent import QLearningAgent


def main():
    env = BatteryEnv()
    agent = QLearningAgent(seed=42)

    observation, info = env.reset(seed=42)

    action = agent.select_action(observation)
    print("Initial state:", agent.discretize_state(observation))
    print("Selected action:", action)

    next_observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    agent.update(
        observation,
        action,
        reward,
        next_observation,
        terminated
    )

    print("Q-table shape:", agent.q_table.shape)
    print("Updated Q-value:", agent.q_table[
        agent.discretize_state(observation)
    ][action])
    print("Q-learning agent test passed.")

    env.close()


if __name__ == "__main__":
    main()