import torch.nn as nn

class dualpipevMLP(nn.Module):
    def __init__(self, dim, rank, world_size, num_layers):
        super().__init__()

        self.rank = rank
        self.dim = dim
        self.num_layers = num_layers

        num_stages = world_size * 2
        layers_per_stage = num_layers // num_stages


        def build_stage(idx):

            layers = []

            for _ in range(layers_per_stage):
                layers.append(nn.Linear(dim, dim))
                layers.append(nn.ReLU())

            if idx == num_stages - 1:
                layers.append(nn.Linear(dim, 2))

            return nn.Sequential(*layers)

        self.down = build_stage(rank)
        self.up = build_stage(num_stages - rank - 1)

        if rank == 0:
            self.loss_fn = nn.CrossEntropyLoss()


    def forward_down(self, x):
        return self.down(x)

    def forward_up(self, x, targets=None):

        l = self.up(x)

        if self.rank != 0 and targets is not None:
            l = self.loss_fn(x, targets)
        return l
