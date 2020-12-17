##### SECTION f Experiments #####

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load model
try:
    model = torch.load('neural-network-3.pth')
    print('Network model: {}'.format(model))
except:
    print('File neural-network-1.pth not found!')
    exit(-1)


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
    print(target_values)
    print(torch.argmax(target_values).item())
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
    main_experiment_f()
