import torch
# from comms.baseline_comms import PipelineComms, init_distributed
from comms.dualpipev_comms import dualpipevComms, init_distributed
# from scheduler_scripts.pp_schedulers import gpipe_pipeline_step, naive_pp_step, onef_oneb_pipeline_step, zb1p_pipeline_step
from scheduler_scripts.dualpipev_scheduler import dualpipev_pipeline_step
from models.dualpipev_mlp import dualpipevMLP
# from models.sharded_mlp import shardedMLP

BATCH_SIZE = 32
HIDDEN_DIM = 128
LAYERS = 16
STEPS = 50
chunks = 8  # dualpipeV requires chunks >= 2 * world_size

# setup distributed
rank, world_size, device = init_distributed()
# comms = PipelineComms(world_size=world_size, rank=rank)
comms = dualpipevComms(world_size=world_size, rank=rank)

# torch.manual_seed(42)

if rank == 0:
    print("we are starting!!")

# initalize model
# model = shardedMLP(rank, world_size, HIDDEN_DIM, LAYERS).to(device)
model = dualpipevMLP(dim=HIDDEN_DIM, rank=rank, world_size=world_size, num_layers=LAYERS).to(device)


# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# load data
# in dualpipeV rank 0 holds both the first stage and the last stage, so it owns
# the inputs and the loss. Other ranks only need the batch size.
if rank == 0:
    inputs = torch.rand(BATCH_SIZE, HIDDEN_DIM).to(device)
    y = torch.randint(0, 2, (BATCH_SIZE,)).to(device)
else:
    inputs = BATCH_SIZE
    y = None

# training
model.train()
for step in range(STEPS):
    optimizer.zero_grad()

    loss = dualpipev_pipeline_step(model, comms, inputs, y, HIDDEN_DIM, chunks, device)

    optimizer.step()

    if rank == 0 and step % 10 == 0:
        print(f"step : {step} , loss : {loss.item():.4f}")

torch.distributed.destroy_process_group()
