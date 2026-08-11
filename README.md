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

Run the incremental cache proof locally with three independently retained
workers (producer shard, consumer shard, and upstream full-model reference):

```bash
go run ./cmd/swarm-cache-smoke -worker "$worker"
```

It prefills two interleaved sequences once, decodes 32 tokens per sequence,
and compares every distributed final-logit vector with the normal
full-checkpoint cached path at `rtol=atol=1e-4`. It also rejects stale,
skipped, conflicting, unknown, and closed sequence positions, reports KV
memory separately from weights and allocator cache, and proves that closing a
sequence releases its KV state without unloading the shard.

### Distributed text generation

Generate text through two complementary cached shards with one command:

```bash
go run ./cmd/swarm-generate \
  -worker "$worker" \
  -model mlx-community/gemma-3-270m-it-4bit \
  -prompt "Write a short story about two computers working together:" \
  -max-tokens 32
```

The command resolves the checkpoint's registered adapter and layer count,
tokenizes with that checkpoint's tokenizer, prefills both shards once, chooses
each next token by deterministic greedy decoding, and returns JSON containing
the prompt and generated token IDs, decoded text, model and shard plan,
sequence ID, EOS/max stop reason, KV bytes, timings, and numerical tolerance.
Add `-verify` to compare every distributed logit vector and greedy token with a
separate full-checkpoint cached reference.

Direct mode starts temporary local workers. To keep weights resident across
separate command invocations, run `swarmd` on both Macs and address both
persistent workers over the trusted-network API:

```bash
go run ./cmd/swarm-generate \
  -producer http://127.0.0.1:8080 \
  -peer http://MAC_B_LAN_IP:8080 \
  -model mlx-community/gemma-3-270m-it-4bit \
  -prompt "Write a short story about two computers working together:" \
  -max-tokens 32
```

Stable model-and-checkpoint-derived shard IDs make later commands reuse matching
loaded stages. Setup fingerprints each resolved checkpoint and rejects a
producer/consumer revision mismatch before composing their weights. Every
request receives a fresh sequence owner and closes only its own KV state on
EOS, maximum length, cancellation, or failure. The deterministic
generation smoke opens two sessions against the same retained workers and
requires the complete 32-token distributed sequence to match the cached
single-node path:

```bash
go run ./cmd/swarm-generate-smoke -worker "$worker"
```

The framed request contract is documented in
[`docs/persistent-worker-protocol.md`](docs/persistent-worker-protocol.md).

### Failure characterization

Run the deterministic process and transport fault proof without a model or
public-network dependency:

```bash
go run ./cmd/swarm-failure-smoke > failure-characterization.json
```

The harness injects a paused worker, a killed worker, a long inference delay,
bounded jitter, and a dropped loopback HTTP connection. Each inference call
has its own deadline (150 ms by default), and each scenario must terminate
within a larger bound (2 seconds by default). Its JSON records accepted and
failed tokens, the failed sequence/shard/phase/position and last accepted
token, worker restart count, sequence/KV cleanup, next-sequence recovery, and
the aggregate failed-token rate.

An inference timeout kills the local MLX process because an already-running
kernel cannot be safely canceled and may have mutated cache state. `swarmd`
starts a clean worker before returning the failure, but it never retries a
cache mutation. The active sequence fails; the next session reloads its shards
and can generate normally. This is bounded recovery, not transparent
same-sequence recovery.

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
MLX_SWARM_WORKER="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker" \
SWARMD_ADDR=127.0.0.1:8080 \
go run ./cmd/swarmd
```

Then, from another shell on Mac A:

```bash
go run ./cmd/swarm-generate \
  -producer http://127.0.0.1:8080 \
  -peer http://MAC_B_LAN_IP:8080 \
  -prompt "Write a short story about two computers working together:" \
  -max-tokens 32
go run ./cmd/swarm-cache-smoke \
  -worker "$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker" \
  -peer http://MAC_B_LAN_IP:8080
go run ./cmd/swarm-benchmark \
  -worker "$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker" \
  -peer http://MAC_B_LAN_IP:8080 \
  -hardware "Apple Silicon description" \
  -route "LAN or Tailscale route" \
  > distributed-benchmark.json
go run ./cmd/swarm-net \
  -peer http://MAC_B_LAN_IP:8080
```

The generation command uses `swarmd`'s persistent worker API for tokenizer,
prompt prefill, greedy sampling, incremental decode, and detokenization. The
cached and one-shot proofs also record logical tensor bytes, encoded wire
bytes, local producer time, network/remote time, final-logit correctness, and
measured full/producer/consumer peak memory. The current 270M checkpoint
produces a 7,680-byte `bfloat16` boundary and logits shaped `[1, 6, 262144]`.

`swarm-benchmark` separates model setup and an explicit warmup from measured
work. By default it records five fresh-sequence prefills and 100 cached decode
steps. Each distributed sample reports producer wall/compute time,
representative boundary JSON serialization, consumer round-trip/compute time,
transport overhead, end-to-end token latency, tensor/wire bytes, and memory.
The cached full-model oracle receives the identical prompt and greedy token
plan but runs outside the distributed hot-path timer. JSON contains both raw
samples and nearest-rank p50/p95 summaries; CI uploads it as an artifact and
prints a readable table without performance pass/fail thresholds.

### Physically pooled-memory generation

The MVP proof uses `mlx-community/gemma-3-text-12b-it-4bit`, a 48-layer,
7,220,708,353-byte checkpoint, across two independent workers limited to 6 GiB
of MLX memory each. A separate upstream full-model run measured a
7,416,404,295-byte inference footprint and pinned the exact 32-token greedy
output. The serving command loads only layers 0–23 on the producer and 24–47
plus the output modules on the consumer; it never starts or loads a full-model
oracle on either serving Mac.

```bash
go run ./cmd/swarm-pooled-memory \
  -producer http://127.0.0.1:8080 \
  -peer http://MAC_B_LAN_IP:8080 \
  -reference testdata/pooled-memory/gemma-3-text-12b-it-4bit.json \
  > pooled-memory.json
```

The machine-readable result inventories both Macs, verifies their configured
limits, reports load/prefill/decode peaks, checks every generated token against
the pinned full-model reference, rejects any full-range serving shard, and
proves sequence teardown. See
[`docs/pooled-memory-proof.md`](docs/pooled-memory-proof.md) for checkpoint
prefetch, worker startup, reference provenance, and the paired-runner workflow.

The current boundary representation is `shape + dtype + contiguous bytes`. JSON/base64 framing is temporary; `proto/swarm.proto` already models the intended raw-byte tensor message.

## First milestone

1. Establish a real single-worker MLX Swift LM reference path.
2. Prove complementary layer ranges compose across independent worker processes.
3. Prove the same boundary crosses the Go network between two machines.
4. Split a real checkpoint across the two Apple-silicon Macs.
5. Compare distributed logits against the single-node baseline.
6. Record p50/p95 stage latency, transfer bytes, TTFT, and tokens/sec.
7. Kill or pause a worker and characterize failure behavior.

With correctness and pooled memory established, hedged execution, dynamic
placement, and public membership remain post-MVP work.

## Repository layout

```text
cmd/swarmd/              Go daemon / debug network receiver
cmd/swarm-local/         local two-worker orchestration
cmd/swarm-net/           two-machine shard experiment client
cmd/swarm-session-smoke/ persistent shard lifecycle and reuse proof
cmd/swarm-cache-smoke/   cached prefill/decode and reference-logit proof
cmd/swarm-benchmark/     warm distributed/reference benchmark and JSON artifact
cmd/swarm-generate/      prompt-to-text distributed greedy generation
cmd/swarm-generate-smoke/ deterministic generation/reference/retention proof
cmd/swarm-failure-smoke/ deterministic process/transport failure proof
cmd/swarm-pooled-memory/ physical-memory-bounded generation/reference proof
internal/generation/     reusable model planning and generation session
internal/benchmark/      percentile, throughput, transfer, and memory summaries
internal/failureharness/ reusable fault workers, injection, and observations
internal/pooledproof/    pooled-memory evidence and acceptance validation
internal/smoke/          shared smoke request and assertion helpers
internal/tensorcheck/    reusable tensor decoding and reference comparison
internal/workerproc/     persistent targets, transports, and sequence lifecycle
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

The M2 real-checkpoint pipeline now produces tokenizer-backed text through
per-sequence KV-cached prefill/decode, with exact greedy-token parity enforced
between paired macOS CI runners and a cached full-model reference. Warm paired
runs also publish statistically useful latency, throughput, transfer, and
memory evidence. Deterministic CI now characterizes deadline, process, and
transport failures with bounded cleanup and next-sequence recovery. The
pooled-memory workflow serves a 12B checkpoint across two 6 GiB MLX budgets,
matches a separately recorded full-model reference, and emits phase memory
evidence without loading an oracle on either serving runner. Final MVP
integration and clean-machine documentation remain in issue #11; see
[ROADMAP.md](ROADMAP.md).
