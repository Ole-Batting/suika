import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Pyramid(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, num_layers: int):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        num_per_layer = np.linspace(self.num_inputs, self.num_outputs, self.num_layers)

        layers = []
        for i in range(self.num_layers - 1):
            layers.append(nn.Linear(int(num_per_layer[i]), int(num_per_layer[i + 1])))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.net(x)

    def pi(self, x, softmax_dim):
        return F.softmax(self.net(x), dim=softmax_dim)

