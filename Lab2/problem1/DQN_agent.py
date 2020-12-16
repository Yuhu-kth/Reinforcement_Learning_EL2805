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
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim

N_HIDDEN_LAYERS = 1  # Should not be more than 2
HIDDEN_SIZE = 8  # Should be between 8-128


class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


###########################################

# Architecture for a two hidden layer network

###########################################

# Will also be used for Target Network
class DQNAgentHidden2(nn.Module):
    def __init__(self, state_space_size: int, n_actions: int, hidden_size: int):
        super(DQNAgentHidden2, self).__init__()
        # N hidden layers should not be more than 2

        self.hidden_size = hidden_size  # Should be between 8-128

        self.input_layer = nn.Linear(state_space_size, self.hidden_size)
        self.input_layer_activation = nn.ReLU()
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_layer_activation = nn.ReLU()

        self.output_layer = nn.Linear(self.hidden_size, n_actions)

    """
    # State is 8-dim array representation of the state
    # Action is a one-hot encoded array during training indicating what action is relevant for training
    def forward(self, state, action):
        # Compute first layer

        l1 = self.input_layer(state)
        l1 = self.input_layer_activation(l1)
        l2 = self.hidden_layer(l1)
        l2 = self.hidden_layer(l2)

        # Compute output layer
        out = self.output_layer(l2)
        return out * action
    """

        # State is 8-dim array representation of the state
        # Action is a one-hot encoded array during training indicating what action is relevant for training
    def forward(self, state):
        # Compute first layer
        l1 = self.input_layer(state)
        l1 = self.input_layer_activation(l1)
        l2 = self.hidden_layer(l1)
        l2 = self.hidden_layer_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)
        return out


###########################################

# Architecture for a single hidden layer network

###########################################

# Will also be used for Target Network
class DQNAgentHidden1(nn.Module):
    def __init__(self, state_space_size: int, n_actions: int, hidden_size):
        super(DQNAgentHidden1, self).__init__()
        # N hidden layers should not be more than 2

        self.hidden_size = hidden_size  # Should be between 8-128
        self.input_layer = nn.Linear(state_space_size, self.hidden_size)
        self.input_layer_activation = nn.ReLU()
        self.output_layer = nn.Linear(self.hidden_size, n_actions)

    # State is 8-dim array representation of the state
    # Action is a one-hot encoded array during training indicating what action is relevant for training
    def forward(self, state):
        # Compute first layer
        l1 = self.input_layer(state)
        l1 = self.input_layer_activation(l1)

        # Compute output layer
        out = self.output_layer(l1)
        return out


    """
        # State is 8-dim array representation of the state
        # Action is a one-hot encoded array during training indicating what action is relevant for training
        def forward(self, state, action):
            # Compute first layer
            l1 = self.input_layer(state)
            l1 = self.input_layer_activation(l1)

            # Compute output layer
            out = self.output_layer(l1)
            return out * action

    """

###########################################

# Architecture for a two hidden layer network

###########################################

# Will also be used for Target Network
class DQNAgentHidden3(nn.Module):
    def __init__(self, state_space_size: int, n_actions: int, hidden_size: int):
        super(DQNAgentHidden3, self).__init__()
        # N hidden layers should not be more than 2

        self.hidden_size = hidden_size  # Should be between 8-128

        self.input_layer = nn.Linear(state_space_size, self.hidden_size)
        self.input_layer_activation = nn.ReLU()
        self.hidden_layer_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_layer_activation_1 = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_layer_activation_2 = nn.ReLU()

        self.output_layer = nn.Linear(self.hidden_size, n_actions)

    """
    # State is 8-dim array representation of the state
    # Action is a one-hot encoded array during training indicating what action is relevant for training
    def forward(self, state, action):
        # Compute first layer

        l1 = self.input_layer(state)
        l1 = self.input_layer_activation(l1)
        l2 = self.hidden_layer(l1)
        l2 = self.hidden_layer(l2)

        # Compute output layer
        out = self.output_layer(l2)
        return out * action
    """

        # State is 8-dim array representation of the state
        # Action is a one-hot encoded array during training indicating what action is relevant for training
    def forward(self, state):
        # Compute first layer
        l1 = self.input_layer(state)
        l1 = self.input_layer_activation(l1)
        l2 = self.hidden_layer_1(l1)
        l2 = self.hidden_layer_activation_1(l2)
        l3 = self.hidden_layer_2(l2)
        l3 = self.hidden_layer_activation_2(l3)

        # Compute output layer
        out = self.output_layer(l3)
        return out
