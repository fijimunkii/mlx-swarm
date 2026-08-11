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
"$worker" checkpoint-shard-smoke
"$worker" generate "Reply with exactly: swarm online"
```

`generate` uses MLX Swift LM's registered `mlx-community/SmolLM-135M-Instruct-4bit` model and downloads/caches its Hugging Face assets on first use.

`checkpoint-shard-smoke` resolves the checkpoint's `model_type` through the shard-adapter registry. The first validated adapter is Gemma 3: it downloads/caches `mlx-community/gemma-3-270m-it-4bit`, runs its 18-layer checkpoint as two independently loaded 9-layer stages, applies the upstream final norm and language head, and verifies the final logits against full-checkpoint inference at `rtol=atol=1e-4`. It also fails unless both workers stay below a 128 MiB budget that the full checkpoint exceeds.

Checkpoint resolution, boundary transport, correctness checks, memory budgeting, safetensor selection, and partial module updates are architecture-neutral. Model-family adapters own parameter paths and execution details such as embedding scaling, attention masks, caches, final normalization, and output heads.

### Persistent shard worker

`swarmd` starts and supervises one long-lived Swift worker. A shard is loaded
once, retained for repeated forwards, and explicitly unloaded after its open
sequences are closed. The worker reports its registered adapter/model type,
loaded ranges, reuse counters, sequence count, and live MLX memory.

Run the real-checkpoint lifecycle proof directly:

```bash
worker="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker"
go run ./cmd/swarm-session-smoke -worker "$worker"
```

The smoke performs 100 forwards through one retained stage, alternates two
sequence IDs, rejects incorrect shard/sequence routing, unloads the stage, and
proves clean shutdown. Direct mode also proves bounded local crash reporting;
HTTP `-peer` mode does not claim control of or crash-test the remote process.
Its default checkpoint is the current CI fixture; the protocol and loader
select arbitrary registered model-family adapters by checkpoint `model_type`.

The framed request contract is documented in
[`docs/persistent-worker-protocol.md`](docs/persistent-worker-protocol.md).

### Go-relayed two-process proof

```bash
go run ./cmd/swarm-local
```

Go launches one Swift worker for layers 0–3, captures its `WireTensor`, launches an independently initialized worker for layers 4–7, forwards the payload over stdin, and verifies the result against the second process's full-model reference.

### Two-Mac network proof

`swarmd` has a deliberately temporary HTTP endpoint carrying the real checkpoint boundary envelope. Mac A loads the embedding and layers 0–8; Mac B independently loads layers 9–17, the final norm, and the language head. This transport is **unauthenticated and unencrypted** and is only for a trusted LAN/debug experiment; do not expose it to the public Internet.

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

The result records logical tensor bytes, encoded wire bytes, local producer time, network/remote time, final-logit correctness, and measured full/producer/consumer peak memory. The current 270M checkpoint produces a 7,680-byte `bfloat16` boundary and logits shaped `[1, 6, 262144]`.

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
cmd/swarm-session-smoke/ persistent shard lifecycle and reuse proof
internal/workerproc/     supervised local and HTTP persistent-worker clients
proto/                   language-neutral protocol definitions
worker/mlx/              Swift MLX worker
  CheckpointShardRuntime generic orchestration and adapter registry
  CheckpointWeightLoader generic filtered checkpoint I/O
  WorkerCheckpointShards app-level adapter registration and defaults
  Gemma3CheckpointShard first model-family adapter
ARCHITECTURE.md           design boundaries and execution model
ROADMAP.md                staged experimental plan
```

## Status

The M2 real-checkpoint pipeline and persistent shard lifecycle are implemented and enforced in paired macOS CI runners. Remaining M2 work is decode-time KV-cache ownership and statistically useful latency/throughput measurement; see [ROADMAP.md](ROADMAP.md).
