import matplotlib.pyplot as plt
import gymnasium as gym
from agent import QLearningAgent
from config import AGENT_PARAMS
from map_plot import plot_q_table

def train(map_name, is_slippery, episodes):
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)
    
    agent = QLearningAgent(
        env,
        alpha_start=AGENT_PARAMS["alpha_start"],
        alpha_end=AGENT_PARAMS["alpha_end"],
        gamma=AGENT_PARAMS["gamma"],
        epsilon=AGENT_PARAMS["epsilon"],
        epsilon_min=AGENT_PARAMS["epsilon_min"],
        epsilon_decay=AGENT_PARAMS["epsilon_decay"]
    )
    
    rewards = []
    epsilons = []

    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0
        alpha = agent.alpha_end + (agent.alpha_start - agent.alpha_end) * (1 - i / episodes)

        while (terminated == False and truncated == False):
            action = agent.epsilon_greedy_search(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.update_q(state, action, reward, new_state, alpha)
            state = new_state
            
            ep_reward += reward

        rewards.append(ep_reward)
        agent.decay_epsilon()
        epsilons.append(agent.epsilon)

        if (i + 1) % 5000 == 0:
            print(f"Episode {i+1}/{episodes}, avg last 100: {sum(rewards[-100:])/100:.3f}, epsilon: {agent.epsilon:.3f}")

    # Plot Rewards
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

    # Plot Epsilon Decay
    plt.figure(figsize=(8, 4))
    plt.plot(epsilons, color='orange')
    plt.title("Decay of Epsilon over Training")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.grid(True)
    epsilon_plot_filename = f"epsilon_decay_{map_name}_{'slippery' if is_slippery else 'deterministic'}.png"
    plt.savefig(epsilon_plot_filename)
    plt.close()

    # Plot Q table
    plot_q_table(env, agent.q_table, map_name, is_slippery)


    # Save Q-table
    q_file = f"q_{map_name}_{'slippery' if is_slippery else 'deterministic'}.pkl"
    agent.save(q_file)
    print(f"Training complete. Q-table saved as '{q_file}'")

    env.close()