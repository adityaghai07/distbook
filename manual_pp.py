import time

import torch
import torch.nn as nn

BATCH_SIZE = 32
STEPS = 50
HIDDEN_DIM = 128
LAYERS = 12


class Part1(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth // 2):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Part2(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth // 2):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, 2))

        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        return loss


cuda = False

if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    cuda = True

x = torch.rand(BATCH_SIZE, HIDDEN_DIM)
y = torch.randint(2, (BATCH_SIZE,))

if cuda:
    x = x.cuda(0)
    y = y.cuda(1)


if cuda:
    model1 = Part1(HIDDEN_DIM, LAYERS).cuda(0)
    model2 = Part2(HIDDEN_DIM, LAYERS).cuda(1)

else:
    model1 = Part1(HIDDEN_DIM, LAYERS)
    model2 = Part2(HIDDEN_DIM, LAYERS)

# training time!
optimizer = torch.optim.Adam(
    list(model1.parameters()) + list(model2.parameters()), lr=0.001
)

start_time = time.time()
for step in range(STEPS):
    optimizer.zero_grad()
    hidden = model1(x)
    if cuda:
        hidden.to(torch.cuda.device(1))
    loss = model2(hidden, y)
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"step : {step}, loss : {loss.item():4f}")

end_time = time.time()
print(f"Total Time : {(end_time - start_time):3f}")
