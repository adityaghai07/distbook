# distbook

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Distributed Training](https://img.shields.io/badge/Distributed-Training-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)

Scratch implementations of distributed training techniques, built from the ground up to understand how they actually work under the hood; comms, scheduling, memory tradeoffs, and all.

Inspired by and referencing [micropp](https://github.com/kiankyars/micropp) by [@kiankyars](https://github.com/kiankyars).

<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/da2225ab-6d27-438d-879f-dd29fb117f53" />


## Implemented

- **Naive Pipeline Parallelism** — the simplest possible split-and-forward baseline
- **GPipe** — micro-batching with a flush-based warmup/cooldown schedule
- **1F1B** — interleaved forward/backward steady-state scheduling to cut bubble time
- **ZB1P** — zero-bubble scheduling with decoupled input/weight backward passes
- **DualPipeV** — the V-shaped variant of DualPipe: each rank holds two stages, so the pipeline runs down the ranks and back up again

## Structure

```
distbook/
├── pp_schedulers.py         # naive, gpipe, 1f1b, zb1p schedules
├── sharded_mlp.py           # toy model sharded across pipeline stages
├── baseline_comms.py        # PipelineComms primitives (send/recv, distributed init)
├── dualpipev_scheduler.py   # DualPipeV schedule
├── dualpipev_mlp.py         # toy model with a down stage and an up stage per rank
├── dualpipev_comms.py       # send/recv for both pipeline directions
└── trainer.py               # training loop entrypoint
```

## Running

```bash
torchrun --nproc_per_node=4 trainer.py
```

Swap which scheduler runs by editing the call in `trainer.py`.

### DualPipeV notes

With `world_size` ranks there are `2 * world_size` stages. Rank `r` holds stage `r`
on the way down and stage `2 * world_size - 1 - r` on the way up, so **rank 0 holds
both the first and the last stage** — the inputs, the targets, and the loss all live
there, not on the last rank.

`chunks` must be at least `2 * world_size`. Rank 0 runs the most warmup forwards, and
with fewer micro-batches than that it runs out of them mid-warmup.

## Why

Most pipeline parallelism explanations stop at diagrams. This repo is about implementing the actual comms and scheduling logic — including the annoying bugs (deadlocks from mismatched send/recv, autograd graphs that need to stay alive across microbatches, decoupled input-grad vs weight-grad passes) that diagrams don't show you.

A schedule that runs without crashing can still compute wrong gradients, so the way to
check one is to compare against plain autograd: build the same stages as a single
`nn.Sequential`, call `.backward()`, and diff the grads. DualPipeV matches to ~1e-9.

## License

MIT
