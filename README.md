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

## Building the MLX worker on macOS

MLX's Metal shader library is built by Xcode, not command-line SwiftPM. Use the repository helper:

```bash
./scripts/build-mlx-worker.sh
```

The executable is produced at:

```text
worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker
```

## Local worker

```bash
worker="worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker"
"$worker" health
"$worker" capabilities
"$worker" shard-smoke
"$worker" generate "Reply with exactly: swarm online"
```

`generate` uses MLX Swift LM's registered `mlx-community/SmolLM-135M-Instruct-4bit` model and downloads/caches its Hugging Face assets on first use.

### Go-relayed two-process proof

```bash
go run ./cmd/swarm-local
```

Go launches one Swift worker for layers 0–3, captures its `WireTensor`, launches an independently initialized worker for layers 4–7, forwards the payload over stdin, and verifies the result against the second process's full-model reference.

### Two-Mac network proof

`swarmd` now has a deliberately temporary HTTP endpoint carrying the same tensor payload. This transport is **unauthenticated and unencrypted** and is only for a trusted LAN/debug experiment; do not expose it to the public Internet.

On Mac B:

```bash
./scripts/build-mlx-worker.sh
MLX_SWARM_WORKER="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker" \
SWARMD_ADDR=0.0.0.0:8080 \
go run ./cmd/swarmd
```

On Mac A:

```bash
./scripts/build-mlx-worker.sh
go run ./cmd/swarm-net \
  -peer http://MAC_B_LAN_IP:8080
```

The result records logical tensor bytes, encoded wire bytes, local producer time, network/remote time, and whether the remote completion matches the reference.

The current boundary representation is `shape + dtype + contiguous bytes`. JSON/base64 framing is temporary; `proto/swarm.proto` already models the intended raw-byte tensor message.

## First milestone

1. Establish a real single-worker MLX Swift LM reference path.
2. Prove complementary layer ranges compose across independent worker processes.
3. Prove the same boundary crosses the Go network between two machines.
4. Split a real checkpoint across the two Apple-silicon Macs.
5. Compare distributed logits against the single-node baseline.
6. Record p50/p95 stage latency, transfer bytes, TTFT, and tokens/sec.
7. Kill or pause a worker and characterize failure behavior.

Only after correctness is established do we add hedged execution, dynamic placement, public membership, and pooled-memory-only models.

## Repository layout

```text
cmd/swarmd/              Go daemon / debug network receiver
cmd/swarm-local/         local two-worker orchestration
cmd/swarm-net/           two-machine shard experiment client
internal/                Go control-plane and worker-process packages
proto/                   language-neutral protocol definitions
worker/mlx/              Swift MLX worker
ARCHITECTURE.md           design boundaries and execution model
ROADMAP.md                staged experimental plan
```

## Status

M2 network proof in progress. The process boundary and Go relay are implemented; the next physical experiment sends the same tensor between the two Macs.
