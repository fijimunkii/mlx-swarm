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

## Local worker

The first worker has a zero-download health command and a real local generation path through MLX Swift LM.

```bash
cd worker/mlx
swift build
swift run --skip-build MLXWorker health
swift run --skip-build MLXWorker generate "Reply with exactly: swarm online"
```

`generate` currently uses MLX Swift LM's registered `mlx-community/SmolLM-135M-Instruct-4bit` model and downloads/caches its Hugging Face assets on first use. This tiny model is our fast single-worker reference path; larger Qwen/Gemma models will be used when we begin exercising layer-range sharding.

## First milestone

Prove a deterministic two-node pipeline using a small model:

1. Establish a real single-worker MLX Swift LM reference path.
2. Load complementary layer ranges on two Apple-silicon Macs.
3. Execute a forward pass across both workers.
4. Compare distributed logits against the single-node baseline.
5. Record bytes transferred, p50/p95 stage latency, TTFT, and tokens/sec.
6. Kill or pause a worker and characterize failure behavior.

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

M1 in progress: establishing the single-worker reference path before adding layer-range execution and two-node transport.
