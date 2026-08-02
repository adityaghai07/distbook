import torch
from baseline_comms import PipelineComms, init_distributed


def ping_pong():

    rank, world_size, device = init_distributed()
    comms = PipelineComms(world_size, rank)
    print(f"rank: {rank}, world_size: {world_size}, device = {device}")

    if rank == 0:
        tensor = torch.rand(3).to(device)
        print("sending tensor \n")
        comms.send_forward(tensor)

    if rank == 1:
        comms.recv_forward(3, device=device, dtype=torch.float32)
        print("recieved!!")


if __name__ == "__main__":
    ping_pong()
