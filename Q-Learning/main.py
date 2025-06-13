# main.py

from config import ENV_CONFIG, TRAINING_PARAMS, TESTING_PARAMS
from train import train
from test import test
import gymnasium as gym
from map_plot import plot_frozenlake_map
import os

if __name__ == '__main__':
    map_name = ENV_CONFIG["map_name"]
    is_slippery = ENV_CONFIG["is_slippery"]

    # Genera mappa solo se non esiste
    plot_filename = f"map_{map_name}.png"
    if not os.path.exists(plot_filename):
        print(f"--- GENERATING MAP for {map_name} ---")
        env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False)  # usa deterministic
        plot_frozenlake_map(env, plot_filename)
        env.close()
    else:
        print(f"Map for {map_name} already exists as '{plot_filename}'")

    print(f"--- TRAINING on {map_name}, slippery={is_slippery} ---")
    train(map_name=map_name, is_slippery=is_slippery, episodes=TRAINING_PARAMS["episodes"])

    print(f"\n--- TESTING on {map_name}, slippery={is_slippery} ---")
    test(map_name=map_name, is_slippery=is_slippery, test_episodes=TESTING_PARAMS["test_episodes"])
