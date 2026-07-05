import torch
from baseline_comms import PipelineComms
from sharded_mlp import shardedMLP


def naive_pp_step(
    model: shardedMLP, comms: PipelineComms, batch, targets, hidden_dim, device
):
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


def gpipe_pipeline_step(
    model: shardedMLP, comms: PipelineComms, batch, targets, hidden_dim, chunks, device
):

    is_first = comms.rank == 0
    is_last = comms.rank == comms.world_size - 1

    if is_first:
        microbatches = torch.chunk(batch, chunks)
    if is_last:
        microtargets = torch.chunk(targets, chunks)

    input_buffer = []
    output_buffer = []

    for i in range(chunks):
        if is_first:
            inputs = microbatches[i]
        else:
            # mini_bs = microbatches[i].shape[0]
            inputs = comms.recv_forward(
                (batch // chunks, hidden_dim), device, dtype=torch.float32
            )
            inputs.requires_grad = True
        if is_last:
            outs = model(inputs, microtargets[i])
        else:
            outs = model(inputs)

        if not is_last:
            comms.send_forward(outs.detach())

        input_buffer.append(inputs)
        output_buffer.append(outs)

    total_loss = torch.zeros(1, device=device)

    for i in range(chunks):
        inputs = input_buffer[i]
        outs = output_buffer[i]

        if is_last:
            loss = outs / chunks
            loss.backward()
            total_loss += loss
        else:
            shape = outs.shape
            grads = comms.recv_backward(shape, device, dtype=torch.float32)
            outs.backward(grads)
        if not is_first:
            comms.send_backward(inputs.grad)

    if is_last:
        return total_loss


def onef_oneb_pipeline_step(
    model: shardedMLP, comms: PipelineComms, batch, targets, hidden_dim, chunks, device
):

    is_first = comms.rank == 0
    is_last = comms.rank == comms.world_size - 1

    warmups = comms.world_size - comms.rank - 1
    onefoneb_steps = chunks - warmups

    input_buffer = [None] * chunks
    output_buffer = [None] * chunks
    async_reqs = []

    if is_first:
        micro_batches = torch.chunk(batch, chunks)
    if is_last:
        micro_targets = torch.chunk(targets, chunks)
        total_loss = torch.zeros(1, device=device)

    def forward(idx):
        if is_first:
            inputs = micro_batches[idx]
        else:
            shape = (batch // chunks, hidden_dim)
            inputs = comms.recv_forward(shape, device, dtype=torch.float32)
            inputs.requires_grad = True

        if is_last:
            outs = model(inputs, micro_targets[idx])
        else:
            outs = model(inputs)
            req = comms.isend_forward(outs.detach())
            async_reqs.append(req)

        input_buffer[idx] = inputs
        output_buffer[idx] = outs

    def back(idx):

        inputs = input_buffer[idx]
        outs = output_buffer[idx]

        if is_last:
            loss = outs / chunks
            loss.backward()
        else:
            shape = outs.shape
            grads = comms.recv_backward(shape, device, dtype=torch.float32)
            outs.backward(grads)
        if not is_first:
            comms.send_backward(inputs.grad)

        if is_last:
            return loss

    for i in range(warmups):
        forward(i)

    for i in range(onefoneb_steps):
        forward(i + warmups)
        res = back(i)
        if is_last:
            total_loss += res

    for i in range(warmups):
        res = back(onefoneb_steps + i)

    for r in async_reqs:
        r.wait()

    if is_last:
        return total_loss
