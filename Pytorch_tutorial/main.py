import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

input_data = torch.tensor(
    [[73, 67, 43],
     [91, 88, 64],
     [87, 134, 58],
     [102, 43, 37],
     [69, 96, 70]],
    dtype=torch.float32,
    requires_grad=False
)

target_data = torch.tensor(
    [[56, 70],
     [81, 101],
     [119, 133],
     [22, 37],
     [103, 119]],
    dtype=torch.float32,
    requires_grad=False
)


# Neural network 2 layers (input and output)
# The number of neurons is a parameter
# Output layer has no activation function
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


network = MyNet(neurons=6)
optim = optim.SGD(network.parameters(), lr=1e-6)

N = 1000
losses = []
network_output = 0

for epoch in range(N):
    optim.zero_grad()
    # Compute output of the network given the input_data tensor
    network_output = network(input_data)

    # Compute loss
    loss = nn.functional.mse_loss(network_output, target_data)
    losses.append(loss)

    # Compute gradient
    loss.backward()

    # Perform backpropagation
    optim.step()

print(network_output)
plt.plot(losses)
plt.show()

#Output by Network
"""tensor([[ 59.8557,  70.5090],
        [ 84.1487,  99.2270],
        [115.3456, 136.1236],
        [ 31.9358,  37.6483],
        [ 97.3693, 114.8508]], grad_fn=<AddmmBackward>)"""

#Expected Output
"""target_data = torch.tensor(
    [[56, 70],
     [81, 101],
     [119, 133],
     [22, 37],
     [103, 119]],
    dtype=torch.float32,
    requires_grad=False
)"""
