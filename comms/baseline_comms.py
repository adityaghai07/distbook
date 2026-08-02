import os

import torch
import torch.distributed as dist


def init_distributed():
    """initialize the distributed environment."""

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = None

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if torch.cuda.is_available():
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size, device_id=device
        )
    else:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    return rank, world_size, device


class PipelineComms:
    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank

        self.next_rank = rank + 1 if rank < world_size - 1 else None
        self.prev_rank = rank - 1 if rank > 0 else None

    def send_forward(self, tensor):
        dist.send(tensor, dst=self.next_rank)

    def isend_forward(self, tensor):
        return dist.isend(tensor, dst=self.next_rank)

    def recv_forward(self, shape, device, dtype):

        tensor = torch.zeros(shape, device=device, dtype=torch.float32)
        dist.recv(tensor, src=self.prev_rank)
        return tensor

    def send_backward(self, tensor):
        dist.send(tensor, dst=self.prev_rank)

    def recv_backward(self, shape, device, dtype):
        tensor = torch.zeros(shape, device=device, dtype=torch.float32)
        dist.recv(tensor, src=self.next_rank)
        return tensor
