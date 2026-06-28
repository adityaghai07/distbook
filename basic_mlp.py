import time

import torch
import torch.nn as nn

NUM_LAYERS = 12
BATCH_SIZE = 32
STEPS = 50
HIDDEN_DIM = 128


class basicMLP(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, 2))
        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        logits = self.model(x)
        loss = self.loss_fn(logits, targets)
        return loss


torch.manual_seed(42)

my_model = basicMLP(HIDDEN_DIM, NUM_LAYERS)
x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
y = torch.randint(0, 2, (BATCH_SIZE,))

optimizer = torch.optim.Adam(my_model.parameters(), lr=0.01)

# training
start_time = time.time()
for step in range(STEPS):
    loss = my_model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"step {step}, loss : {loss.item():.5f}")

end_time = time.time()
print(f"training time : {(end_time - start_time):.4f}")
