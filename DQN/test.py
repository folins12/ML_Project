import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from dqn_agent import DQN, state_to_input


def test(env_name, is_slippery, model_path, episodes):
    env = gym.make(env_name, is_slippery=is_slippery)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    model = DQN(num_states, 128, num_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    success_history = []

    for _ in range(episodes):
        state = env.reset()[0]
        terminated = truncated = False

        while not terminated and not truncated:
            with torch.no_grad():
                action = model(state_to_input(state, num_states)).argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)

        success_history.append(1 if reward == 1 else 0)

    env.close()

    success_rate = sum(success_history) / episodes * 100
    print(f"Success Rate: {success_rate:.2f}%")


    sum_test = [sum(success_history[max(0, t - 100): t + 1]) for t in range(episodes)]

    plt.figure(figsize=(8, 4))
    plt.plot(sum_test)
    plt.axhline(y=success_rate, linestyle='--', color='orange', label=f"Average {success_rate:.1f}%")
    plt.title(f"Test: successes in last 100 episodes ({'slippery' if is_slippery else 'deterministic'})")
    plt.xlabel("Test Episodes")
    plt.ylabel("Successes (last 100)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"test_{env_name.split('-')[0].lower()}_{'slippery' if is_slippery else 'deterministic'}.png")
    plt.close()
