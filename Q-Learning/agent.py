import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, env, alpha_start, alpha_end, gamma, epsilon, epsilon_min, epsilon_decay):
        self.env = env
        
        # Initialization of the Q table
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions))

        # Hyperparameters
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng()

    # Act according to the Epsilon Greedy Search
    def epsilon_greedy_search(self, state):
        if self.rng.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        return action

    # Updating rule
    def update_q(self, state, action, reward, next_state, alpha):
        self.q_table[state, action] += alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])

    # Decrease epsilon
    def decay_epsilon(self):
        # Linear Decay
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        # Exponential Decay
        #self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)