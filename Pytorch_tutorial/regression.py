import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim

# Create Tensors

input_data = torch.tensor(
    [[73, 67, 43],
     [91, 88, 64],
     [87, 135, 58],
     [102, 43, 37],
     [69, 96, 70]], dtype=torch.float32, requires_grad=False
)

target_data = torch.tensor(
    [[56, 70],
     [81, 101],
     [119, 133],
     [22, 37],
     [103, 119]], dtype=torch.float32, requires_grad=False
)

# Neural network 2 layers (input and output)
# The number of neurons is a parameter
# Inner layer used ReLU activation

class MyNet(nn.Module):
    def __init__(self, neurons):
        super().__init__()
        self.input_layer = nn.Linear(3, neurons)
        self.output_layer = nn.Linear(neurons, 2)
        self.input_layer_activation = nn.ReLU()

    def forward(self, x):
        out_l1 = self.input_layer(x)
        out_l1_a = self.input_layer_activation(out_l1)
        output = self.output_layer(out_l1_a)
        return output

if __name__ == '__main__':
    network = MyNet(neurons=100)
    optim = optim.SGD(network.parameters(), lr=1e-6)

    N = 1000
    losses = []
    network_output = 0
    for epoch in range(N):
        optim.zero_grad()
        # Compute output of the network given the input_data tensor
        network_output = network(input_data)

        # Compute Loss
        loss = nn.functional.mse_loss(network_output, target_data)
        losses.append(loss.item())

        # Compute gradient
        loss.backward()

        # Perform backpropagation
        optim.step()

    print(network_output)

    plt.plot(losses)
    plt.show()
