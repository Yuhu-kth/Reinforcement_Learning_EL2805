# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym.envs.box2d.lunar_lander
import gym
import torch
import torch.nn.functional as functional
import torch.nn as nn
import copy
from tqdm import trange
from Lab2.problem1.ExperienceReplayBuffer import ExperienceReplayBuffer, Experience
from Lab2.problem1.DQN_agent import RandomAgent
from Lab2.problem1.DQN_agent import DQNAgentHidden1
from Lab2.problem1.DQN_agent import DQNAgentHidden2
from Lab2.problem1 import Utils as Utils

#############################################
### Used for 2 Hidden, 64 Neurons ###

# Environmental Parameters
N_ACTIONS = 4  # Number of available actions
DIM_STATES = 8  # State dimensionality

#  Hyper parameters
DISCOUNT = 0.1
BUFFER_SIZE = 30000  # Should be 5000-30000
BUFFER_EXP_START = 10000
N_EPISODES = 3000  # Should be 100-1000
Z = N_EPISODES * 0.95  # Z is usually 90 − 95% of the total number of episodes
BATCH_SIZE_N = 16  # Should 4-128
C = BUFFER_SIZE / BATCH_SIZE_N  # Update frequency of the target neural network
DECAY_MAX = 0.5
DECAY_MIN = 0.1
EPS_LINEAR = True

# Hyper parameters, Neural Network
LEARNING_RATE = 10e-4  # Should be between 10e-3 and 10e-4
CLIPPING_VALUE = 0.5  # 0.5 and 2

# Training Procedure
N_EP_RUNNING_AVERAGE = 50
EARLY_STOPPING_THRESHOLD = 1


##############################################


def main():
    model_url = 'neural-network-1.pth'

    # Random agent initialization
    # agent = RandomAgent(N_ACTIONS)

    # agent = DQNAgentHidden1(DIM_STATES, N_ACTIONS)
    # target = DQNAgentHidden1(DIM_STATES, N_ACTIONS)
    agent = DQNAgentHidden2(DIM_STATES, N_ACTIONS)
    target = DQNAgentHidden2(DIM_STATES, N_ACTIONS)
    training_DQN(agent, target, model_url)

    # load_model(agent, model_url)

    # trial_run(agent, random=False)


def trial_run(agent, random=False):
    # Import and initialize the discrete Lunar Laner Environment
    env = gym.make('LunarLander-v2')
    env.reset()

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []  # this list contains the total reward per episode
    episode_number_of_steps = []  # this list contains the number of steps per episode

    ### Training process

    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()  # State shape (x-pos, y-pos, x-velocity, y-velocity, lander-angle, angular-velocity,
        # left-contact point bool, right-contact point bool)
        total_episode_reward = 0.
        t = 0
        while not done:
            env.render()

            if random:
                # Take a random action
                action = agent.forward(state)
            else:
                state_tensor = torch.tensor(state, requires_grad=False, dtype=torch.float32)
                # Returns the argmax of the network
                action = torch.argmax(agent(state_tensor)).item()

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)

            # Utils.print_SARSD(state, action, next_state, reward, done)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward, t,
                Utils.running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1],
                Utils.running_average(episode_number_of_steps, N_EP_RUNNING_AVERAGE)[-1]))

    Utils.plot_reward_and_steps(N_EPISODES, episode_reward_list, episode_number_of_steps, Utils.running_average,
                                N_EP_RUNNING_AVERAGE)


def training_DQN(main_network, target_network, URL):
    # Import and initialize the discrete Lunar Laner Environment
    env = gym.make('LunarLander-v2')
    env.reset()

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []  # this list contains the total reward per episode
    episode_number_of_steps = []  # this list contains the number of steps per episode

    buffer = init_buffer()

    # optimizer = torch.optim.SGD(main_network.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.Adam(main_network.parameters(), lr=LEARNING_RATE)

    ### Training process

    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)

    n_episodes_so_far = 0

    c = 0

    for k in EPISODES:
        # Reset environment data and initialize variables
        done = False
        state = env.reset()  # State shape (x-pos, y-pos, x-velocity, y-velocity, lander-angle, angular-velocity,
        # left-contact point bool, right-contact point bool)
        total_episode_reward = 0.
        t = 0
        while not done:
            # env.render()

            next_state, reward, done, target_network = q_step(k, main_network, state, env, buffer, target_network, optimizer, c)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1
            c += 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        running_average_reward = Utils.running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1]
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                k, total_episode_reward, t,
                running_average_reward,
                Utils.running_average(episode_number_of_steps, N_EP_RUNNING_AVERAGE)[-1]))

        # Early stopping
        n_episodes_so_far = k + 1
        if running_average_reward > EARLY_STOPPING_THRESHOLD:
            break

    Utils.plot_reward_and_steps(n_episodes_so_far, episode_reward_list, episode_number_of_steps, Utils.running_average,
                                N_EP_RUNNING_AVERAGE)

    save_model(main_network, URL)


def save_model(model, URL):
    torch.save(model.state_dict(), URL)


def load_model(model, URL):
    ### Load model ###
    model.load_state_dict(torch.load(URL))


def q_step(k, main_network, state, env, buffer, target_network, optimizer, c):
    optimizer.zero_grad()  # Necessery to reset the gradients since pytorch accumulates them by default

    action = eps_greedy(k, main_network, state)
    next_state, reward, done, _ = env.step(action)
    # Append experience to the buffer
    exp_z = Experience(state, action, reward, next_state, done)
    buffer.append(exp_z)
    print(len(buffer))

    # Utils.print_SARSD(state, action, next_state, reward, done)

    states, actions, rewards, next_states, dones = buffer.sample_batch(n=BATCH_SIZE_N)
    y = target_values_y(rewards, target_network, next_states, done)
    y_tensor = torch.tensor(y, requires_grad=False, dtype=torch.float32)

    states_tensor = torch.tensor(states, requires_grad=False, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, requires_grad=False, dtype=torch.int64)
    predicted_Q_values = main_network(states_tensor)

    actions_tensor = torch.unsqueeze(actions_tensor, -1)
    # We only want to perform backprop the q-value for the taken action
    predicted_Q_values_for_taken_actions = torch.gather(predicted_Q_values, -1, actions_tensor)

    # Old way of 'removing' irrelevant action outputs
    '''y = np.array([y, ] * N_ACTIONS).transpose()

    actions_mask = np.zeros((BATCH_SIZE_N, N_ACTIONS))
    actions_mask[np.arange(BATCH_SIZE_N), actions] = 1

    y *= actions_mask
    states_tensor = torch.tensor(states, requires_grad=True, dtype=torch.float32)
    action_mask_tensor = torch.tensor(actions_mask, requires_grad=False, dtype=torch.float32)

    predicted_actions = agent(states_tensor, action_mask_tensor)
    y_tensor = torch.tensor(y, requires_grad=False, dtype=torch.float32)
    '''

    # Compute loss function
    loss = functional.mse_loss(predicted_Q_values_for_taken_actions, torch.unsqueeze(y_tensor, -1))

    # Compute gradient
    loss.backward()

    # Clip gradient norm to CLIPPING_VALUE
    nn.utils.clip_grad_norm_(main_network.parameters(), max_norm=CLIPPING_VALUE)

    # Perform backward pass (backpropagation)
    optimizer.step()

    if c % C == 0 and c != 0:
        target_network = copy.deepcopy(main_network)

    return next_state, reward, done, target_network


def eps_greedy(k, agent, state):
    if EPS_LINEAR:
        eps_k = Utils.decay_linear(DECAY_MIN, DECAY_MAX, k, Z)
    else:
        eps_k = Utils.decay_exp(DECAY_MIN, DECAY_MAX, k, Z)

    p = np.random.uniform(0, 1)
    if p < eps_k:
        return np.random.choice(N_ACTIONS)
    else:
        state_tensor = torch.tensor(state, requires_grad=False, dtype=torch.float32)
        # no_mask = torch.tensor(np.ones(N_ACTIONS), requires_grad=False, dtype=torch.float32)
        return torch.argmax(agent(state_tensor)).item()  # Returns the argmax of the network # Check this???


"""
def target_values_y(rewards, target, next_states, done):
    if not done:
        states_tensor = torch.tensor(next_states, requires_grad=False, dtype=torch.float32)
        action_mask_tensor = create_no_mask_tensor()
        Q_theta_prime = target(states_tensor, action_mask_tensor).max(1)[0].detach().numpy()
        return rewards + DISCOUNT * Q_theta_prime
    else:
        return rewards
"""


def target_values_y(rewards, target_network, next_states, done):
    if not done:
        states_tensor = torch.tensor(next_states, requires_grad=False, dtype=torch.float32)
        Q_theta_prime = target_network(states_tensor).max(1)[0].detach().numpy()
        return rewards + DISCOUNT * Q_theta_prime
    else:
        return rewards


# def create_no_mask_tensor():
#    return torch.tensor(np.ones((BATCH_SIZE_N, N_ACTIONS)), requires_grad=False, dtype=torch.float32)


def init_buffer():
    print("### Creating and filling buffer")

    buffer = ExperienceReplayBuffer(maximum_length=BUFFER_SIZE)
    env = gym.make('LunarLander-v2')
    env.reset()

    agent = RandomAgent(N_ACTIONS)

    for i in range(BUFFER_EXP_START):
        # Reset environment data and initialize variables
        done = False
        state = env.reset()  # State shape (x-pos, y-pos, x-velocity, y-velocity, lander-angle, angular-velocity,
        # left-contact point bool, right-contact point bool)
        total_episode_reward = 0.
        t = 0
        while not done:
            # Take a random action
            action = agent.forward(state)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)

            # Update episode reward
            total_episode_reward += reward

            # Append experience to the buffer
            exp = Experience(state, action, reward, next_state, done)
            buffer.append(exp)

            # Update state for next iteration
            state = next_state
            t += 1

        # Close environment
        env.close()

    return buffer


# TODO Write a load and running routine


if __name__ == '__main__':
    main()
