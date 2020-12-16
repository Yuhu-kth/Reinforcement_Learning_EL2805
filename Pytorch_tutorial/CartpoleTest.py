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
# Course: EL2805 - Reinforcement Learning - Lab 0 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 17th October 2020, by alessior@kth.se

### IMPORT PACKAGES ###
# numpy for numerical/random operations
# gym for the Reinforcement Learning environment
import numpy as np
import gym
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
import matplotlib.pyplot as plt

MODEL_URL = "./Model1Hidden8Neurons"
SLOW_DOWN_RATE = 1000000

### Experience class ###

# namedtuple is used to create a special type of tuple object. Namedtuples
# always have a specific name (like a class) and specific fields.
# In this case I will create a namedtuple 'Experience',
# with fields: state, action, reward,  next_state, done.
# Usage: for some given variables s, a, r, s, d you can write for example
# exp = Experience(s, a, r, s, d). Then you can access the reward
# field by  typing exp.reward
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """

    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')
        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)


### Neural Network ###
class MyNetwork(nn.Module):
    """ Create a feedforward neural network """

    def __init__(self, input_size, output_size):
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, 8)
        self.input_layer_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(8, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        # Compute output layer
        out = self.output_layer(l1)
        return out


def train_model(episodes):
    ### CREATE RL ENVIRONMENT ###
    env = gym.make('CartPole-v0')  # Create a CartPole environment

    state_size = len(env.observation_space.low)  # State space dimensionality
    n_actions = env.action_space.n  # Number of actions

    ### Create Experience replay buffer ###
    buffer = ExperienceReplayBuffer(maximum_length=1000)

    ### Create network ###
    network = MyNetwork(input_size=state_size, output_size=n_actions)

    ### Create optimizer ###
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    losses = []

    ### PLAY ENVIRONMENT ###
    # The next while loop plays 5 episode of the environment
    for episode in range(episodes):
        episode_loss = 0
        state = env.reset()  # Reset environment, returns initial state
        done = False  # Boolean variable used to indicate if an episode terminated

        while not done:

            env.render()  # Render the environment, remove this line if you run on Google Colab
            # Create state tensor, remember to use single precision (torch.float32)
            state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)

            # Compute output of the network
            values = network(state_tensor)

            # Pick the action with greatest value
            # .max(1) picks the action with maximum value along the first dimension
            # [1] picks the argmax
            # .item() is used to cast the tensor to a real value
            action = values.max(1)[1].item()

            # The next line takes permits you to take an action in the RL environment
            # env.step(action) returns 4 variables:
            # (1) next state; (2) reward; (3) done variable; (4) additional stuff
            next_state, reward, done, _ = env.step(action)
            print(reward)

            # Append experience to the buffer
            exp = Experience(state, action, reward, next_state, done)
            buffer.append(exp)

            ### TRAINING ###
            # Perform training only if we have more than 3 elements in the buffer
            if len(buffer) >= 3:
                # Sample a batch of 3 elements
                states, actions, rewards, next_states, dones = buffer.sample_batch(n=3)

                # Training process, set gradients to 0
                optimizer.zero_grad()

                # Compute output of the network given the states batch
                values = network(torch.tensor(states, requires_grad=True, dtype=torch.float32))

                # Compute loss function
                loss = nn.functional.mse_loss(values, torch.ones_like(values, requires_grad=False))

                # Add loss
                episode_loss += loss.item()

                print("Loss for episode " + str(episode) + ": ", loss.item())

                # Compute gradient
                loss.backward()

                # Clip gradient norm to 1
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.)

                # Perform backward pass (backpropagation)
                optimizer.step()
        losses.append(episode_loss)

    # Close all the windows
    env.close()

    save_model(network)
    return losses


def save_model(model):
    torch.save(model.state_dict(), MODEL_URL)


def apply_slow_down_game():
    i = 0
    while i < SLOW_DOWN_RATE:
        i += 1


def plot_prediction(losses):
    plt.plot(np.arange(len(losses)), losses)
    plt.title("Loss per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.show()


def load_and_run_model():
    ### CREATE RL ENVIRONMENT ###
    env = gym.make('CartPole-v0')  # Create a CartPole environment

    state_size = len(env.observation_space.low)  # State space dimensionality
    n_actions = env.action_space.n  # Number of actions

    ### Create network ###
    network = MyNetwork(state_size, n_actions)

    ### Load model ###
    network.load_state_dict(torch.load(MODEL_URL))

    for i in range(100):
        state = env.reset()  # Reset environment, returns initial state
        done = False

        while not done:
            env.render()  # Render the environment, remove this line if you run on Google Colab
            # Create state tensor, remember to use single precision (torch.float32)
            state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)

            # Compute output of the network
            values = network(state_tensor)

            # Pick the action with greatest value
            # .max(1) picks the action with maximum value along the first dimension
            # [1] picks the argmax
            # .item() is used to cast the tensor to a real value
            action = values.max(1)[1].item()

            # The next line takes permits you to take an action in the RL environment
            # env.step(action) returns 4 variables:
            # (1) next state; (2) reward; (3) done variable; (4) additional stuff
            next_state, reward, done, _ = env.step(action)
            print(done)
            apply_slow_down_game()

    env.close()

if __name__ == '__main__':
    losses = train_model(episodes=1000)  # Seems to have converged after 400 episodes
    plot_prediction(losses)
    load_and_run_model()
