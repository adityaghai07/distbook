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

## In progress

- **DualPipe** — bidirectional pipeline scheduling (next up)

## Structure

```
distbook/
├── pp_schedulers.py     # pipeline parallelism schedules (naive, gpipe, 1f1b, zb1p, ...)
├── sharded_mlp.py        # toy model sharded across pipeline stages
├── baseline_comms.py      # PipelineComms primitives (send/recv, distributed init)
└── trainer.py              # training loop entrypoint
```

## Running

```bash
torchrun --nproc_per_node=4 trainer.py
```

Swap which scheduler runs by editing the call in `trainer.py`.

## Why

Most pipeline parallelism explanations stop at diagrams. This repo is about implementing the actual comms and scheduling logic — including the annoying bugs (deadlocks from mismatched send/recv, autograd graphs that need to stay alive across microbatches, decoupled input-grad vs weight-grad passes) that diagrams don't show you.

## License

MIT
