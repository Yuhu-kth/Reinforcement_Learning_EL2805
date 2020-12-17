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
from Lab2.problem1.DQN_agent import DQNAgentHidden1, DQNAgentHidden2, DQNAgentHidden3
from Lab2.problem1 import Utils as Utils

#############################################
### Used for 2 Hidden, 64 Neurons ###

# Environmental Parameters
N_ACTIONS = 4  # Number of available actions
DIM_STATES = 8  # State dimensionality

#  Hyper parameters
DISCOUNT = 0.99
BUFFER_SIZE = 10000  # Should be 5000-30000
BUFFER_EXP_START = 5000  # Typically filling it half-full
N_EPISODES = 600  # Should be 100-1000
Z = N_EPISODES * 0.95  # Z is usually 90 − 95% of the total number of episodes
BATCH_SIZE_N = 8  # Should 4-128
C = int(BUFFER_SIZE / BATCH_SIZE_N)  # Update frequency of the target neural network
DECAY_MAX = 0.99
DECAY_MIN = 0.05
EPS_LINEAR = True

# Hyper parameters, Neural Network
LEARNING_RATE = 2 * (10e-4)  # Should be between 10e-3 and 10e-4
CLIPPING_VALUE = 1  # 0.5 and 2
HIDDEN_SIZE = 64  # Nodes per hidden layer
N_HIDDEN = 2  # Number of hidden layer

# Training Procedure
N_EP_RUNNING_AVERAGE = 50
EARLY_STOPPING_THRESHOLD = 50  # After the average reward reaches this value, we stop


##############################################


def main():
    model_url = 'neural-network-trash.pth'
    training_DQN(model_url)


def training_DQN(URL):
    # Import and initialize the discrete Lunar Laner Environment
    env = gym.make('LunarLander-v2')
    env.reset()

    if N_HIDDEN == 1:
        print("Init network 1")
        main_network = DQNAgentHidden1(DIM_STATES, N_ACTIONS, HIDDEN_SIZE)
        target_network = copy.deepcopy(main_network)
    elif N_HIDDEN == 2:
        print("Init network 2")
        main_network = DQNAgentHidden2(DIM_STATES, N_ACTIONS, HIDDEN_SIZE)
        target_network = copy.deepcopy(main_network)
    else:
        print("Init network 3")
        main_network = DQNAgentHidden3(DIM_STATES, N_ACTIONS, HIDDEN_SIZE)
        target_network = copy.deepcopy(main_network)

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []  # this list contains the total reward per episode
    episode_number_of_steps = []  # this list contains the number of steps per episode

    buffer = init_buffer()

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
            next_state, reward, done = q_step(k, main_network, state, env, buffer, target_network, optimizer)

            if c == C:
                target_network = copy.deepcopy(main_network)
                print(C)
                c = 0
                print("Updating target network!")

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
    torch.save(model, URL)


def load_model(URL):
    ### Load model ###
    return torch.load(URL)


def q_step(k, main_network, state, env, buffer, target_network, optimizer):
    optimizer.zero_grad()  # Necessery to reset the gradients since pytorch accumulates them by default

    main_network.eval()
    with torch.no_grad():
        action = eps_greedy(k, main_network, state)
    next_state, reward, done, _ = env.step(action)
    # Append experience to the buffer
    exp_z = Experience(state, action, reward, next_state, done)
    buffer.append(exp_z)

    main_network.train()
    states, actions, rewards, next_states, dones = buffer.sample_batch(n=BATCH_SIZE_N)
    y = target_values_y(rewards, target_network, next_states, done)
    y_tensor = torch.tensor(y, requires_grad=False, dtype=torch.float32)

    states_tensor = torch.tensor(states, requires_grad=False, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, requires_grad=False, dtype=torch.int64)
    predicted_Q_values = main_network.forward(states_tensor)

    actions_tensor = torch.unsqueeze(actions_tensor, -1)
    # We only want to perform backprop the q-value for the taken action
    predicted_Q_values_for_taken_actions = torch.gather(predicted_Q_values, -1, actions_tensor)

    # Compute loss function
    loss = functional.mse_loss(predicted_Q_values_for_taken_actions, torch.unsqueeze(y_tensor, -1))

    # Compute gradient
    loss.backward()

    # Clip gradient norm to CLIPPING_VALUE
    nn.utils.clip_grad_norm_(main_network.parameters(), max_norm=CLIPPING_VALUE)

    # Perform backward pass (backpropagation)
    optimizer.step()

    return next_state, reward, done


def eps_greedy(k, agent, state):
    if EPS_LINEAR:
        eps_k = Utils.decay_linear(DECAY_MIN, DECAY_MAX, k, Z)
    else:
        eps_k = Utils.decay_exp(DECAY_MIN, DECAY_MAX, k, Z)

    p = np.random.random()
    if p < eps_k:
        action = np.random.choice(N_ACTIONS)
        return action
    else:
        state_tensor = torch.tensor(state, requires_grad=False, dtype=torch.float32)
        evaluation = agent(state_tensor)
        argmax_value = torch.argmax(evaluation).item()
        return argmax_value


def target_values_y(rewards, target_network, next_states, done):
    if not done:
        states_tensor = torch.tensor(next_states, requires_grad=False, dtype=torch.float32)
        target_values = target_network(states_tensor)
        Q_theta_prime = target_values.max(1)[0].detach().numpy()
        target_value = rewards + DISCOUNT * Q_theta_prime
        return target_value
    else:
        return rewards


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
