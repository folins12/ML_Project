import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

# DQN Model (same as in training)
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def state_to_input(state, num_states):
    return torch.FloatTensor([1 if i == state else 0 for i in range(num_states)])

# Testing
def test_model(episodes=20, model_path="frozenlake_slippery_true.pt"):
    env = gym.make('FrozenLake8x8-v1', is_slippery=True, render_mode='human')
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    model = DQN(num_states, 128, num_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    successes = 0
    rewards = []
    
    for _ in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not terminated and not truncated:
            with torch.no_grad():
                action = model(state_to_input(state, num_states)).argmax().item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
        if reward == 1:
            successes += 1
    
    env.close()
    print(f"Success Rate: {successes/episodes*100:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Rewards per Test Episode")
    plt.subplot(1, 2, 2)
    plt.bar(['Success', 'Failure'], [successes, episodes-successes])
    plt.title("Success vs Failure")
    plt.savefig('testing_slippery_true.png')

if __name__ == '__main__':
    test_model()