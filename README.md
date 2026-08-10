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

## Building the MLX worker on macOS

MLX's Metal shader library is built by Xcode, not command-line SwiftPM. Use the repository helper rather than `swift build`:

```bash
./scripts/build-mlx-worker.sh
```

The resulting executable is:

```text
worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker
```

Then run:

```bash
worker="worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker"
"$worker" health
"$worker" capabilities
"$worker" shard-smoke
"$worker" generate "Reply with exactly: swarm online"
```

`generate` uses MLX Swift LM's registered `mlx-community/SmolLM-135M-Instruct-4bit` model and downloads/caches its Hugging Face assets on first use.

### Go-relayed two-process shard proof

After building the worker:

```bash
go run ./cmd/swarm-local
```

Go launches one Swift worker for layers 0–3, captures its `WireTensor`, launches an independently initialized worker for layers 4–7, forwards the payload over stdin, and verifies the result against the second process's full-model reference.

The relay payload is exactly the shape we expect to put on the swarm transport: `shape + dtype + contiguous bytes`. JSON/base64 framing is temporary; the protobuf `Tensor` message already models the intended representation.

## First milestone

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

M1/M2 boundary proof in progress: Swift compilation is green; CI now builds via Xcode so MLX's Metal shader library is available at runtime. The next step is the Go-relayed two-process shard test and then the same tensor over the LAN between two Macs.
