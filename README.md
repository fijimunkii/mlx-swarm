# mlx-swarm

Community-scale, fault-tolerant distributed inference on heterogeneous consumer hardware, built around MLX.

`mlx-swarm` is an early research project exploring whether unreliable consumer devices can collectively serve models larger than any single participant can host, while maintaining useful interactive latency and graceful failure recovery.

## Principles

- **Do not rewrite MLX.** MLX remains the tensor/runtime layer.
- **Go owns the swarm.** Discovery, networking, scheduling, health, deadlines, hedging, and APIs live in Go.
- **Swift owns MLX execution.** The first worker uses MLX Swift and MLX Swift LM for model loading and inference.
- **The WAN is not an MLX process group.** Arbitrary peers communicate through a language-neutral swarm protocol; MLX distributed primitives may be used inside stable high-speed execution islands.
- **Consumer nodes are disposable.** The design assumes workers can slow down, sleep, disconnect, or disappear.
- **Measure before optimizing.** v0 exists to establish correctness and characterize latency/failure behavior.

## v0 architecture

```text
                         +------------------+
                         |      swarmd      |
                         |       (Go)       |
                         |                  |
                         | registry         |
                         | scheduler        |
                         | health           |
                         | transport        |
                         +--------+---------+
                                  |
                         local RPC / protocol
                                  |
                         +--------v---------+
                         |    mlx-worker    |
                         |     (Swift)      |
                         |                  |
                         | MLX Swift LM     |
                         | MLX Swift        |
                         +--------+---------+
                                  |
                              MLX core
                             C++ / Metal
```

Future NVIDIA workers should preserve the same swarm protocol and use MLX's CUDA backend where practical.

## First milestone

Prove a deterministic two-node pipeline using a small model:

1. Load complementary layer ranges on two Apple-silicon Macs.
2. Execute a forward pass across both workers.
3. Compare distributed logits against a single-node baseline.
4. Record bytes transferred, p50/p95 stage latency, TTFT, and tokens/sec.
5. Kill or pause a worker and characterize failure behavior.

Only after correctness is established do we add hedged execution, dynamic placement, WAN peers, and pooled-memory-only models.

## Repository layout

```text
cmd/swarmd/              Go daemon
internal/                Go control-plane packages
proto/                   language-neutral protocol definitions
worker/mlx/              Swift MLX worker
ARCHITECTURE.md           design boundaries and execution model
ROADMAP.md                staged experimental plan
```

## Status

Bootstrap stage. Interfaces are intentionally small and expected to change.
