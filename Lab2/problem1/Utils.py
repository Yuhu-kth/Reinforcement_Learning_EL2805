import matplotlib.pyplot as plt

STATE_LAYOUT = ["x-pos", "y-pos", "x-velocity", "y-velocity", "lander-angle", "angular-velocity",
        "left-contact point bool", "right-contact point bool"]

ACTIONS = ["Do nothing", "fire left orientation engine", "fire main engine", "fire right orientation engine"]

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

def print_SARSD(state, action, next_state, reward, done):
    print("State layout: ", STATE_LAYOUT)
    print("State: ", state)
    print("Action taken: ", ACTIONS[action])
    print("Next state: ", next_state)
    print("Reward: ", reward)
    print("Is done: ", done)
    print("--------------------------------------")
