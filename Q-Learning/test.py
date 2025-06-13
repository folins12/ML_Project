# test.py

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from agent import QLearningAgent

def test(map_name, is_slippery, test_episodes):
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)
    agent = QLearningAgent(env)
    q_file = f"q_{map_name}_{'slippery' if is_slippery else 'deterministic'}.pkl"
    agent.load(q_file)

    successes = np.zeros(test_episodes)

    for i in range(test_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(agent.q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        successes[i] = reward

    success_rate = successes.mean() * 100
    print(f"Success rate: {success_rate:.2f}%")

    # Plot
    sum_test = [sum(successes[max(0, t-100): t+1]) for t in range(test_episodes)]
    plt.figure(figsize=(8,4))
    plt.plot(sum_test)
    plt.axhline(y=success_rate, linestyle='--', label=f"Average {success_rate:.1f}%")
    plt.title(f"Test: successes in last 100 episodes ({'slippery' if is_slippery else 'deterministic'})")
    plt.xlabel("Test Episodes")
    plt.ylabel("Successes (last 100)")
    plt.legend()
    plt.grid(True)
    test_filename = f"test_{map_name}_{'slippery' if is_slippery else 'deterministic'}.png"
    plt.savefig(test_filename)
    plt.close()
    print(f"Test plot saved as '{test_filename}'")

    env.close()
