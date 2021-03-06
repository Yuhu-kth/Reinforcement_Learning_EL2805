import matplotlib.pyplot as plt
import numpy as np


def plot_reward_and_steps(N_episodes, episode_reward_list, episode_number_of_steps, running_average,
                          n_ep_running_average):
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, N_episodes + 1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, N_episodes + 1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, N_episodes + 1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, N_episodes + 1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()


def print_loss(episode, loss):
    print("Loss for episode " + str(episode) + ": ", loss)


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def decay_linear(eps_min, eps_max, k, Z):
    value_2 = eps_max - (eps_max - eps_min) * (k - 1) / (Z - 1)
    return np.max((eps_min, value_2))


def decay_exp(eps_min, eps_max, k, Z):
    value_2 = eps_max * (eps_min / eps_max) ** ((k - 1) / (Z - 1))
    return np.max((eps_min, value_2))
