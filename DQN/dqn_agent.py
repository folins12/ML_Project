import torch
from torch import nn
import torch.nn.functional as F
import random
from collections import deque

# Initialization of the DQN
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


# Memory management
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


#def one_hot_encode(state, num_states):
#    return torch.FloatTensor([1 if i == state else 0 for i in range(num_states)])

def one_hot_encode(state, num_states):
    input_vector = []
    for i in range(num_states):
        if i == state:
            input_vector.append(1)
        else:
            input_vector.append(0)

    return torch.FloatTensor(input_vector)

