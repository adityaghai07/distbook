import torch
from baseline_comms import PipelineComms, init_distributed
from pp_schedulers import gpipe_pipeline_step, naive_pp_step, onef_oneb_pipeline_step, zb1p_pipeline_step
from sharded_mlp import shardedMLP

BATCH_SIZE = 32
HIDDEN_DIM = 128
LAYERS = 16
STEPS = 50
chunks = 4

# setup distributed
rank, world_size, device = init_distributed()
comms = PipelineComms(world_size=world_size, rank=rank)

# torch.manual_seed(42)

if rank == 0:
    print("we are starting!!")

# initalize model
model = shardedMLP(rank, world_size, HIDDEN_DIM, LAYERS).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# load data
if rank == 0:
    inputs = torch.rand(BATCH_SIZE, HIDDEN_DIM).to(device)
else:
    inputs = BATCH_SIZE

if rank == world_size - 1:
    y = torch.randint(0, 2, (BATCH_SIZE,)).to(device)
else:
    y = None

# training
model.train()
for step in range(STEPS):
    optimizer.zero_grad()
    if rank == world_size - 1:
        # loss = naive_pp_step(model, comms, inputs, y, HIDDEN_DIM, device)
        # loss = gpipe_pipeline_step(model, comms, inputs, y, HIDDEN_DIM, chunks, device)
        # loss = onef_oneb_pipeline_step(model, comms, inputs, y, HIDDEN_DIM, chunks, device)
        loss = zb1p_pipeline_step(model, comms, inputs, y, HIDDEN_DIM, chunks, device)

    else:
        # naive_pp_step(model, comms, inputs, y, HIDDEN_DIM, device)
        # gpipe_pipeline_step(model, comms, inputs, y, HIDDEN_DIM, chunks, device)
        # onef_oneb_pipeline_step(model, comms, inputs, y, HIDDEN_DIM, chunks, device)
        zb1p_pipeline_step(model, comms, inputs, y, HIDDEN_DIM, chunks, device)

    optimizer.step()

    if rank == world_size - 1 and step % 10 == 0:
        print(f"step : {step} , loss : {loss.item():.4f}")

torch.distributed.destroy_process_group()
