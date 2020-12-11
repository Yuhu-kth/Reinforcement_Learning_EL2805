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
from tqdm import trange
from Lab2.problem1.DQN_agent import RandomAgent
from Lab2.problem1 import Utils as Utils

#############################################

#  Hyper parameters
DISCOUNT = 0.1
BUFFER_SIZE = 5000  # Should be 5000-30000
N_EPISODES = 100  # Should be 100-1000
BATCH_SIZE = 4  # Should 4-128
C = BUFFER_SIZE / BATCH_SIZE  # Update frequency of the target neural network

# Hyper parameters, Neural Network
LEARNING_RATE = 10e-4  # Should be between 10e-3 and 10e-4
OPTIMIZER = torch.optim.adam
CLIPPING_VALUE = 0.5  # 0.5 and 2
N_HIDDEN_LAYERS = 1  # Should not be more than 2
HIDDEN_SIZE = 8  # Should be between 8-128

# Training Procedure
N_EPISODE_AVERAGE = 50


##############################################

def training():
    # Import and initialize the discrete Lunar Laner Environment
    env = gym.make('LunarLander-v2')
    env.reset()

    # Parameters
    N_episodes = 100  # Number of episodes
    discount_factor = 0.95  # Value of the discount factor
    n_ep_running_average = 50  # Running average of 50 episodes
    n_actions = env.action_space.n  # Number of available actions
    dim_state = len(env.observation_space.high)  # State dimensionality

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []  # this list contains the total reward per episode
    episode_number_of_steps = []  # this list contains the number of steps per episode

    # Random agent initialization
    agent = RandomAgent(n_actions)

    ### Training process

    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()  # State shape (x-pos, y-pos, x-velocity, y-velocity, lander-angle, angular-velocity,
        # left-contact point bool, right-contact point bool)
        total_episode_reward = 0.
        t = 0
        while not done:
            env.render()

            # Take a random action
            # action = agent.forward(state)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)

            Utils.print_SARSD(state, action, next_state, reward, done)

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
                running_average(episode_reward_list, n_ep_running_average)[-1],
                running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    Utils.plot_reward_and_steps(N_episodes, episode_reward_list, episode_number_of_steps, running_average,
                                n_ep_running_average)


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


if __name__ == '__main__':
    training()
