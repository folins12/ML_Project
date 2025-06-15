ENV_CONFIG = {
    "map_name": "8x8",          # "4x4" or "8x8"
    "is_slippery": True         # True or False (deterministic)
}

AGENT_PARAMS = {
    "alpha_start": 0.9,
    "alpha_end": 0.1,
    "gamma": 0.9,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00003    # Linear Decay
    #"epsilon_decay": 0.99995   # Exponential Decay
}

TRAINING_PARAMS = {
    "episodes": 50000
}

TESTING_PARAMS = {
    "test_episodes": 1000
}