import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQN, state_to_input
from dqn_agent import ReplayMemory


def train(env_name, is_slippery, config):
    env = gym.make(env_name, is_slippery=is_slippery)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    policy_net = DQN(num_states, config['hidden_nodes'], num_actions)
    target_net = DQN(num_states, config['hidden_nodes'], num_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config['learning_rate'])
    loss_fn = torch.nn.SmoothL1Loss()
    memory = ReplayMemory(config['replay_memory_size'])

    epsilon = config['epsilon_start']
    epsilon_history = []
    rewards = np.zeros(config['episodes'])
    step_count = 0

    for episode in range(config['episodes']):
        state = env.reset()[0]
        terminated = truncated = False
        while not terminated and not truncated:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(state_to_input(state, num_states)).argmax().item()

            new_state, reward, terminated, truncated, _ = env.step(action)
            memory.append((state, action, new_state, reward, terminated))
            state = new_state
            step_count += 1

        rewards[episode] = reward

        if len(memory) >= config['mini_batch_size']:
            batch = memory.sample(config['mini_batch_size'])
            current_qs, target_qs = [], []

            for state, action, new_state, reward, done in batch:
                target = torch.tensor([reward]) if done else reward + config['discount_factor'] * target_net(state_to_input(new_state, num_states)).max()
                current_q = policy_net(state_to_input(state, num_states))
                target_q = current_q.clone().detach()
                target_q[action] = target

                current_qs.append(current_q)
                target_qs.append(target_q)

            loss = loss_fn(torch.stack(current_qs), torch.stack(target_qs))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (is_slippery):
            epsilon = max(config['epsilon_min'], epsilon * config['epsilon_decay'])
        elif(not is_slippery):
            epsilon = max(config['epsilon_min'], epsilon - config['epsilon_decay'])
        
        epsilon_history.append(epsilon)

        if step_count >= config['network_sync_rate']:
            target_net.load_state_dict(policy_net.state_dict())
            step_count = 0

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}, Epsilon: {epsilon:.3f}, Success Rate: {np.mean(rewards[max(0, episode-99):episode+1]):.2f}")

    env.close()

    model_path = f"frozenlake_{config['map_name'].lower()}_{'slippery' if is_slippery else 'deterministic'}.pt"
    torch.save(policy_net.state_dict(), model_path)

    # Plot rewards (successes in last 100 episodes)
    sum_rewards = [sum(rewards[max(0, t - 100): t + 1]) for t in range(len(rewards))]
    plt.figure(figsize=(8, 4))
    plt.plot(sum_rewards)
    plt.title(f"Training: successes in last 100 episodes ({'slippery' if is_slippery else 'deterministic'})")
    plt.xlabel("Episodes")
    plt.ylabel("Successes (last 100)")
    plt.grid(True)
    plt.savefig(f"train_{config['map_name'].lower()}_{'slippery' if is_slippery else 'deterministic'}.png")
    plt.close()

    # Plot Epsilon Decay
    plt.figure(figsize=(8, 4))
    plt.plot(epsilon_history, color='orange')
    plt.title("Decay of Epsilon over Training")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.savefig(f"epsilon_decay_{config['map_name'].lower()}_{'slippery' if is_slippery else 'deterministic'}.png")
    plt.close()