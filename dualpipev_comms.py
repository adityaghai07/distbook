import os

import torch
import torch.distributed as dist


def init_distributed():

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size, device_id=device
        )
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    return rank, world_size, device


class dualpipevComms:
    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank

        self.next_rank = rank + 1 if rank < world_size - 1 else None
        self.prev_rank = rank -1 if rank > 0 else None

    def _downstream(self, phase):
        return self.next_rank if phase == 0 else self.prev_rank

    def _upstream(self, phase):
        return self.prev_rank if phase == 0 else self.next_rank

    def send_forward(self, tensor, phase):
        dist.send(tensor, dst=self._downstream(phase), tag=phase)

    def isend_forward(self, tensor, phase):
        return dist.isend(tensor, dst=self._downstream(phase), tag=phase)

    def recv_forward(self, shape, device, dtype, phase):

        tensor = torch.zeros(shape, device=device, dtype=torch.float32)
        dist.recv(tensor, src=self._upstream(phase), tag=phase)
        return tensor

    def send_backward(self, tensor, phase):
        dist.send(tensor, dst=self._upstream(phase), tag = 10 + phase)

    def isend_backward(self, tensor, phase):
        return dist.send(tensor, dst=self._upstream(phase), tag = 10 + phase)

    def recv_backward(self, shape, device, dtype, phase):

        tensor = torch.zeros(shape, device=device, dtype=torch.float32)
        dist.recv(tensor, src=self._downstream(phase),tag= 10 + phase)
        return tensor
