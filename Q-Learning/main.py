from config import ENV_CONFIG, TRAINING_PARAMS, TESTING_PARAMS
from train import train
from test import test
import gymnasium as gym
from map_plot import plot_frozenlake_map
import os

if __name__ == '__main__':
    map_name = ENV_CONFIG["map_name"]
    is_slippery = ENV_CONFIG["is_slippery"]

    # Map 4x4 or 8x8
    plot_filename = f"map_{map_name}.png"
    if not os.path.exists(plot_filename):
        env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False)
        plot_frozenlake_map(env, plot_filename)
        env.close()
    
    print(f"--- TRAINING on {map_name}, slippery={is_slippery} ---")
    train(map_name=map_name, is_slippery=is_slippery, episodes=TRAINING_PARAMS["episodes"])

    print(f"\n--- TESTING on {map_name}, slippery={is_slippery} ---")
    test(map_name=map_name, is_slippery=is_slippery, test_episodes=TESTING_PARAMS["test_episodes"])