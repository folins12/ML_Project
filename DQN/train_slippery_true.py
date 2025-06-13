import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

# Hyperparameters for slippery=True
learning_rate = 0.001
discount_factor = 0.9
network_sync_rate = 500
replay_memory_size = 10000
mini_batch_size = 128
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995
episodes = 5000
hidden_nodes = 128

# DQN Model
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

# Experience Replay
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

def state_to_input(state, num_states):
    return torch.FloatTensor([1 if i == state else 0 for i in range(num_states)])

def optimize(model, target_model, memory, optimizer, loss_fn, mini_batch_size, discount_factor):
    if len(memory) < mini_batch_size:
        return
    
    mini_batch = memory.sample(mini_batch_size)
    current_q_list = []
    target_q_list = []

    for state, action, new_state, reward, terminated in mini_batch:
        if terminated:
            target = torch.FloatTensor([reward])
        else:
            with torch.no_grad():
                target = reward + discount_factor * target_model(state_to_input(new_state, num_states)).max()
        
        current_q = model(state_to_input(state, num_states))
        target_q = target_model(state_to_input(state, num_states))
        target_q[action] = target

        current_q_list.append(current_q)
        target_q_list.append(target_q)

    loss = loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training
env = gym.make('FrozenLake8x8-v1', is_slippery=True)
num_states = env.observation_space.n
num_actions = env.action_space.n

policy_net = DQN(num_states, hidden_nodes, num_actions)
target_net = DQN(num_states, hidden_nodes, num_actions)
target_net.load_state_dict(policy_net.state_dict())

memory = ReplayMemory(replay_memory_size)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

epsilon = epsilon_start
rewards = np.zeros(episodes)
epsilon_history = []
step_count = 0

for episode in range(episodes):
    state = env.reset()[0]
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state_to_input(state, num_states)).argmax().item()

        new_state, reward, terminated, truncated, _ = env.step(action)
        memory.append((state, action, new_state, reward, terminated))
        state = new_state
        step_count += 1

    rewards[episode] = reward

    # Training
    optimize(policy_net, target_net, memory, optimizer, loss_fn, mini_batch_size, discount_factor)
    
    # Linear epsilon decay
    epsilon = max(epsilon_min, epsilon - (epsilon_start - epsilon_min)/episodes)
    epsilon_history.append(epsilon)
    
    # Sync networks
    if step_count >= network_sync_rate:
        target_net.load_state_dict(policy_net.state_dict())
        step_count = 0

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}, Epsilon: {epsilon:.3f}, Success Rate: {np.mean(rewards[max(0, episode-99):episode+1]):.2f}")

env.close()
torch.save(policy_net.state_dict(), "frozenlake_slippery_true.pt")

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
plt.title("Rewards per Episode (Smoothed)")
plt.subplot(1, 2, 2)
plt.plot(epsilon_history)
plt.title("Linear Epsilon Decay")
plt.savefig('training_slippery_true.png')