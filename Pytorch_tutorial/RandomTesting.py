import numpy as np
import Lab2.problem1.Utils as Utils
import matplotlib.pyplot as plt


DECAY_MAX = 0.5
DECAY_MIN = 0.05
N_EPISODES = 1000  # Should be 100-1000
Z = N_EPISODES * 0.7  # Z is usually 90 − 95% of the total number of episodes


if __name__ == '__main__':
    x = np.arange(N_EPISODES)
    y_linear = []
    y_exp = []

    for i in x:
        y_linear.append(Utils.decay_linear(DECAY_MIN, DECAY_MAX, i, Z))
        y_exp.append(Utils.decay_exp(DECAY_MIN, DECAY_MAX, i, Z))

    plt.plot(x, np.array(y_linear))
    plt.show()

    plt.plot(x, np.array(y_exp))
    plt.show()