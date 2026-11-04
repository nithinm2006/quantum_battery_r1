"""Shared settings for reproducible quantum battery experiments."""

# Random seeds
TRAINING_SEED = 42
EVALUATION_SEED = 1000
VALIDATION_SEED_OFFSET = 10_000

# Quantum policy training (SPSA)
TRAINING_ITERATIONS = 80
EPISODES_PER_ESTIMATE = 2
SPSA_LEARNING_RATE = 0.08
SPSA_PERTURBATION = 0.1
VALIDATION_EPISODES = 10

# Agent comparison
EVALUATION_EPISODES = 30

# Environment / experiment metadata
EXPERIMENT_NAME = "hybrid_quantum_battery_rl"