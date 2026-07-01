import torch
from baseline_comms import PipelineComms
from sharded_mlp import shardedMLP


def naive_pp_step(model, comms, batch, targets, hidden_dim, device):
    is_first = comms.rank == 0
    is_last = comms.rank == comms.world_size - 1

    if is_first:
        inputs = batch
    else:
        inputs = comms.recv_forward((batch, hidden_dim), device, dtype=torch.float32)
        inputs.requires_grad = True

    out = model(inputs, targets)

    if not is_last:
        comms.send_forward(out.detach())

    if is_last:
        loss = out
        loss.backward()
    else:
        grads = comms.recv_backward(out.shape, device, dtype=torch.float32)
        out.backward(grads)

    if not is_first:
        comms.send_backward(inputs.grad)

    if is_last:
        return loss
