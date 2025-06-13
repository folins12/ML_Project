# agent.py

import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, env, alpha_start=0.9, alpha_end=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.00005, rng_seed=None):
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(rng_seed)

    def choose_action(self, state):
        if self.rng.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def update_q(self, state, action, reward, next_state, alpha):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state, action] += alpha * (reward + self.gamma * best_next - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
