import torch.nn as nn


class shardedMLP(nn.Module):
    def __init__(self, rank, world_size, dim, num_layers):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.rank = rank
        self.world_size = world_size

        layers = []

        for _ in range(num_layers // world_size):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())

        if self.rank == world_size - 1:
            layers.append(nn.Linear(dim, 2))
            self.loss_fn = nn.CrossEntropyLoss()

        self.model = nn.Sequential(*layers)

    def forward(self, x, y=None):

        z = self.model(x)
        if self.rank == self.world_size - 1 and y is not None:
            z = self.loss_fn(z, y)

        return z
