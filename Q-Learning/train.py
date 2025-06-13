# train.py

import matplotlib.pyplot as plt
import gymnasium as gym
from agent import QLearningAgent

def train(map_name, is_slippery, episodes):
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)
    agent = QLearningAgent(env)
    rewards = []

    for i in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        alpha = agent.alpha_end + (agent.alpha_start - agent.alpha_end) * (1 - i / episodes)

        while not done:
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update_q(state, action, reward, new_state, alpha)
            state = new_state
            ep_reward += reward

        rewards.append(ep_reward)
        agent.decay_epsilon()

        if (i + 1) % 5000 == 0:
            print(f"Episode {i+1}/{episodes}, avg last 100: {sum(rewards[-100:])/100:.3f}, epsilon: {agent.epsilon:.3f}")

    # Plot
    sum_rewards = [sum(rewards[max(0, t - 100): t + 1]) for t in range(len(rewards))]
    plt.figure(figsize=(8, 4))
    plt.plot(sum_rewards)
    plt.title(f"Training: successes in last 100 episodes ({'slippery' if is_slippery else 'deterministic'})")
    plt.xlabel("Episodes")
    plt.ylabel("Successes (last 100)")
    plt.grid(True)
    plot_filename = f"train_{map_name}_{'slippery' if is_slippery else 'deterministic'}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Training plot saved as '{plot_filename}'")

    # Save Q-table
    q_file = f"q_{map_name}_{'slippery' if is_slippery else 'deterministic'}.pkl"
    agent.save(q_file)
    print(f"Training complete. Q-table saved as '{q_file}'")

    env.close()
