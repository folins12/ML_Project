from config import get_config
from train import train
from test import test
import os

if __name__ == "__main__":
    config = get_config()
    env_name = f"FrozenLake{config['map_name']}-v1"

    model_path = f"frozenlake_{config['map_name'].lower()}_{'slippery' if config['slippery'] else 'deterministic'}.pt"

    print(f"--- TRAINING on {config['map_name']}, slippery={config['slippery']} ---")
    train(env_name, config['slippery'], config)

    test_ep = config['test_ep_slippery'] if config['slippery'] else config['test_ep_nonslippery']

    print(f"\n--- TESTING on {config['map_name']}, slippery={config['slippery']} ---")
    if os.path.exists(model_path):
        test(env_name, config['slippery'], model_path, test_ep)