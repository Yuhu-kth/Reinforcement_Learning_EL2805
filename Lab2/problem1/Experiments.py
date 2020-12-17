##### SECTION f and g Experiments #####
import gym
import torch
import numpy as np
import Lab2.problem1.DQN_agent as DQN_agent
import matplotlib.pyplot as plt
from tqdm import trange

# Load model
try:
    model = torch.load('neural-network-1.pth')
    print('Network model: {}'.format(model))
except:
    print('File neural-network-1.pth not found!')
    exit(-1)


def main_experiment_g():
    random_rewards = run_game(random=True)
    best_model_rewards = run_game(random=False)
    x = np.arange(len(random_rewards))
    plt.title("Episodic Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(x, random_rewards, label="Random Agent")
    plt.plot(x, best_model_rewards, label="Best Q-Network")
    plt.legend()
    plt.show()


def run_game(random=False):
    # Import and initialize Mountain Car Environment
    env = gym.make('LunarLander-v2')
    env.reset()

    random_agent = DQN_agent.RandomAgent(4)

    # Parameters
    N_EPISODES = 50  # Number of episodes to run for trainings

    # Reward
    episode_reward_list = []  # Used to store episodes reward

    # Simulate episodes
    print('Checking solution...')
    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
    for i in EPISODES:
        EPISODES.set_description("Episode {}".format(i))
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        while not done:
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise

            if random:
                action = random_agent.forward(state)
                next_state, reward, done, _ = env.step(action)
            else:
                q_values = model(torch.tensor([state]))
                _, action = torch.max(q_values, axis=1)
                next_state, reward, done, _ = env.step(action.item())

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()

    return episode_reward_list


def main_experiment_f():
    y = np.linspace(0, 1.5)
    w = np.linspace(-np.pi, np.pi)
    # plot_max_Q_values(y, w)
    plot_argmax_Q_values(y, w)


def Q(state):
    states_tensor = torch.tensor(state, requires_grad=False, dtype=torch.float32)
    target_values = model(states_tensor)
    return target_values.max(0)[0].detach().numpy()


def arg_Q(state):
    states_tensor = torch.tensor(state, requires_grad=False, dtype=torch.float32)
    target_values = model(states_tensor)
    return torch.argmax(target_values).item()


def s(y, w):
    return np.array([0, y, 0, 0, w, 0, 0, 0])


def plot_max_Q_values(y, w):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    Y, W = np.meshgrid(y, w)
    q = np.array([Q(s(y, w)) for y, w in zip(np.ravel(Y), np.ravel(W))])
    Z = q.reshape(np.shape(Y))

    ax.plot_surface(Y, W, Z, cmap='viridis', edgecolor='none')

    ax.view_init(20, 35)

    ax.set_title('Max Q-values')
    ax.set_xlabel('y')
    ax.set_ylabel('w')
    ax.set_zlabel('Max Q')

    plt.show()


def plot_argmax_Q_values(y, w):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    Y, W = np.meshgrid(y, w)
    q = np.array([arg_Q(s(y, w)) for y, w in zip(np.ravel(Y), np.ravel(W))])
    Z = q.reshape(np.shape(Y))

    ax.plot_surface(Y, W, Z, cmap='viridis', edgecolor='none')

    ax.view_init(20, 35)

    ax.set_title('Argmax Q Actions')
    ax.set_xlabel('y')
    ax.set_ylabel('w')
    ax.set_zlabel('Action')

    plt.show()


if __name__ == '__main__':
    # main_experiment_f()
    main_experiment_g()
