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

The worker has zero-download health/capability commands, a tiny real generation path, and deterministic shard-boundary tests.

```bash
cd worker/mlx
swift build
swift run --skip-build MLXWorker health
swift run --skip-build MLXWorker capabilities
swift run --skip-build MLXWorker shard-smoke
swift run --skip-build MLXWorker generate "Reply with exactly: swarm online"
```

`generate` uses MLX Swift LM's registered `mlx-community/SmolLM-135M-Instruct-4bit` model and downloads/caches its Hugging Face assets on first use.

### Go-relayed two-process shard proof

The first true process-boundary test uses a deterministic tiny Gemma 3 model so it requires no checkpoint download. Go launches one Swift worker process for layers 0–3, captures its language-neutral tensor payload, then launches a fresh Swift worker for layers 4–7 and forwards the payload over stdin. The second process checks the result against its own full-model reference.

```bash
# From the repository root, after `cd worker/mlx && swift build && cd ../..`
go run ./cmd/swarm-local -worker worker/mlx/.build/debug/MLXWorker
```

The relay payload is exactly the shape we expect to place on the swarm transport: `shape + dtype + contiguous bytes`. JSON/base64 framing is temporary; the WAN transport will carry the existing protobuf `Tensor` fields directly.

The worker also keeps file-based `shard-produce` / `shard-finish` commands as a debugging aid, but the primary experiment does not share a file or MLX process state.

## First milestone

Prove a deterministic two-node pipeline using a small model:

1. Establish a real single-worker MLX Swift LM reference path.
2. Prove complementary layer ranges compose across separate worker processes.
3. Replace the local Go process relay with peer-to-peer swarm transport.
4. Split a real checkpoint across two Apple-silicon Macs.
5. Compare distributed logits against the single-node baseline.
6. Record bytes transferred, p50/p95 stage latency, TTFT, and tokens/sec.
7. Kill or pause a worker and characterize failure behavior.

Only after correctness is established do we add hedged execution, dynamic placement, WAN peers, and pooled-memory-only models.

## Repository layout

```text
cmd/swarmd/              Go daemon
cmd/swarm-local/         local two-worker orchestration/benchmark
internal/                Go control-plane packages
proto/                   language-neutral protocol definitions
worker/mlx/              Swift MLX worker
ARCHITECTURE.md           design boundaries and execution model
ROADMAP.md                staged experimental plan
```

## Status

M2 process-boundary proof: Go can relay a serialized hidden state between two independently initialized Swift/MLX worker processes. The next step is moving the same tensor payload over the Go swarm network between two machines.
