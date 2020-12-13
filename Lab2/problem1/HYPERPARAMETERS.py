### Used for 1 Hidden, 8 Neurons ###

# Environmental Parameters
N_ACTIONS = 4  # Number of available actions
DIM_STATES = 8  # State dimensionality

#  Hyper parameters
DISCOUNT = 0.1
BUFFER_SIZE = 5000  # Should be 5000-30000
BUFFER_EXP_START = 5
N_EPISODES = 1000  # Should be 100-1000
Z = N_EPISODES * 0.95  # Z is usually 90 − 95% of the total number of episodes
BATCH_SIZE_N = 4  # Should 4-128
C = BUFFER_SIZE / BATCH_SIZE_N  # Update frequency of the target neural network
DECAY_MAX = 0.99
DECAY_MIN = 0.05

# Hyper parameters, Neural Network
LEARNING_RATE = 10e-4  # Should be between 10e-3 and 10e-4
CLIPPING_VALUE = 0.5  # 0.5 and 2

# Training Procedure
N_EP_RUNNING_AVERAGE = 50

##############################################

