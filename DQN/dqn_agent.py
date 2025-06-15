import torch
from torch import nn
import torch.nn.functional as F
import random
from collections import deque

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

def optimize(model, target_model, memory, optimizer, loss_fn, mini_batch_size, discount_factor, num_states):
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

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()