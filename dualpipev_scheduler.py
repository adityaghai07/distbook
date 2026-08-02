from collections import deque
import torch


def dualpipev_pipeline_step(model, comms, batch, targets, hidden_dim, chunks, device):
    rank = comms.rank
    num_ranks = comms.world_size
    is_first = rank == 0
    is_last = rank == num_ranks - 1

    # plz add this in your code, was fed up with this bug :)
    assert chunks >= num_ranks * 2, (
        f"dualpipeV needs chunks >= 2 * world_size ({num_ranks * 2}), got {chunks}"
    )

    micro_batch_size = (batch.shape[0] if torch.is_tensor(batch) else batch) // chunks
    shape = (micro_batch_size, hidden_dim)

    # rank 0 owns both the first stage (down) and the last stage (up),
    # so both the data and the targets/loss live there.
    if is_first:
        micro_batches = torch.chunk(batch, chunks)
        micro_targets = torch.chunk(targets, chunks)
        total_loss = torch.zeros(1, device=device)

    input_buffer = [[None] * chunks, [None] * chunks]
    output_buffer = [[None] * chunks, [None] * chunks]

    f_next_id = [0, 0]
    b_next_id = [0, 0]

    fold_fwd = deque()
    fold_bwd = deque()

    async_reqs = []

    w_queue = deque()

    def forward(phase):
        idx = f_next_id[phase]
        f_next_id[phase] += 1

        if phase == 0:
            if is_first:
                inputs = micro_batches[idx]
            else:
                inputs = comms.recv_forward(shape, device, torch.float32, phase=0)
                inputs.requires_grad = True

            outs = model.forward_down(inputs)

            if is_last:
                fold_fwd.append(outs.detach().requires_grad_())
            else:
                sent = outs.detach()
                async_reqs.append((comms.isend_forward(sent, phase=0), sent))

        else:  # phase == 1
            if is_last:

                inputs = fold_fwd.popleft()
            else:
                inputs = comms.recv_forward(shape, device, torch.float32, phase=1)
                inputs.requires_grad = True

            if is_first:

                outs = model.forward_up(inputs, micro_targets[idx])
            else:
                outs = model.forward_up(inputs)
                sent = outs.detach()
                async_reqs.append((comms.isend_forward(sent, phase=1), sent))

        input_buffer[phase][idx] = inputs
        output_buffer[phase][idx] = outs

    def back_input(phase):

        idx = b_next_id[phase]
        b_next_id[phase] += 1
        inputs = input_buffer[phase][idx]
        outs = output_buffer[phase][idx]

        is_loss_stage = phase == 1 and is_first
        is_first_stage = phase == 0 and is_first

        if is_loss_stage:
            loss = outs / chunks
            (input_grad,) = torch.autograd.grad(loss, inputs, retain_graph=True)
            grad_out, node, result = None, loss, loss

        else:
            if phase == 0 and is_last:
                grad_out = fold_bwd.popleft()
            else:
                grad_out = comms.recv_backward(outs.shape, device, torch.float32, phase=phase)

            node, result = outs, None

            if not is_first_stage:

                (input_grad,) = torch.autograd.grad(
                    outs, inputs, grad_outputs=grad_out, retain_graph=True
                )

        if phase == 1 and is_last:

            fold_bwd.append(input_grad)
        elif is_first_stage:
            pass
        else:
            async_reqs.append((comms.isend_backward(input_grad, phase=phase), input_grad))


        w_queue.append((node, grad_out))
        return result

    def back_weight():
        if not w_queue:
            return
        node, grad_out = w_queue.popleft()
        params = [p for p in model.parameters() if p.requires_grad]

        w_grads = torch.autograd.grad(
            node, params, grad_outputs=grad_out, retain_graph=False, allow_unused=True
        )
        for p, g in zip(params, w_grads):
            if g is None:

                continue
            p.grad = g if p.grad is None else p.grad + g

    def forward_backward(fphase, bphase):

        forward(fphase)
        res = back_input(bphase)
        back_weight()
        return res

    def maybe_accumulate(res):
        nonlocal total_loss
        if is_first and res is not None:
            total_loss += res.detach()

    for _ in range((num_ranks - rank - 1) * 2):
        forward(0)

    for _ in range(rank + 1):
        forward(0)
        forward(1)

    for _ in range(num_ranks - rank - 1):
        maybe_accumulate(back_input(1))
        back_weight()
        forward(1)

    for _ in range(chunks - num_ranks * 2 + rank + 1):
        maybe_accumulate(forward_backward(0, 1))
        maybe_accumulate(forward_backward(1, 0))

    for _ in range(num_ranks - rank - 1):
        maybe_accumulate(back_input(1))
        back_weight()
        maybe_accumulate(forward_backward(1, 0))

    for _ in range(rank + 1):
        maybe_accumulate(back_input(1))
        back_weight()
        maybe_accumulate(back_input(0))
        back_weight()

    for _ in range(num_ranks - rank - 1):
        back_weight()
        maybe_accumulate(back_input(0))

    while w_queue:
        back_weight()

    for req, _tensor in async_reqs:
        req.wait()

    if is_first:
        return total_loss
