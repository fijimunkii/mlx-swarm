# Architecture

## Goal

Test whether a dynamic pool of heterogeneous consumer devices can provide useful distributed LLM inference over ordinary networks without treating every participant as a reliable cluster member.

## Boundary: swarm vs. MLX

`mlx-swarm` should add distributed-systems behavior, not duplicate the ML stack.

### MLX is responsible for

- arrays and lazy graph execution
- Metal/CUDA kernels
- quantized operations
- attention and matrix operations
- device memory management
- model math exposed by MLX Swift / MLX Swift LM

### The swarm is responsible for

- peer discovery and membership
- capability and latency profiling
- model/shard placement
- WAN transport
- deadlines and cancellation
- retries and hedged execution
- replica selection
- failure detection and recovery
- topology-aware routing
- public API and, later, trust/accounting

## Processes

Each participating machine initially runs two processes:

1. `swarmd` (Go): network-facing daemon, scheduler agent, and worker supervisor.
2. `mlx-worker` (Swift): long-lived local MLX execution service.

A process boundary gives us crash isolation and lets future worker backends implement the same protocol without changing the control plane.

`swarmd` starts one `MLXWorker serve-stdio` child and exchanges framed,
request-ID-correlated messages with it. The child retains assigned checkpoint
stages until unload or shutdown. Unexpected EOF is a daemon health failure;
only an acknowledged shutdown is treated as a clean exit. The concrete v0
framing is described in `docs/persistent-worker-protocol.md`.

Inside the Swift worker, checkpoint sharding has three boundaries:

- `CheckpointShardRuntime` resolves a checkpoint, chooses an adapter from an injected registry using `model_type`, transports stage boundaries, and enforces correctness and memory budgets.
- `CheckpointWeightLoader` performs architecture-neutral safetensor selection and partial module updates.
- a `CheckpointShardAdapter` owns model-family details: parameter paths, embeddings, attention/cache semantics, normalization, and output heads. Gemma 3 is the first registered adapter, not a control-plane special case.

`WorkerCheckpointShards` is the composition root: it registers adapters and supplies experiment defaults without coupling the reusable runtime to Gemma or any future model family.

## Execution model

The first useful primitive is conceptually:

```text
Prefill(model, shard, sequence, position=0, tensor) -> tensor
Decode(model, shard, sequence, next_position, tensor) -> tensor
```

A shard is initially a contiguous transformer layer range. The worker owns its
weights while it is assigned. Opening a sequence creates an adapter-owned cache
with the correct concrete type for every layer in that shard. The worker keys
that cache by `(shardID, sequenceID)`, enforces contiguous positions, and drops
only the sequence cache on close. Model-family adapters implement cached
prefill/decode and attention-mask semantics behind an opaque cache interface;
the Go control plane never handles cache tensors.

The coordinator transfers only execution inputs/outputs and metadata during
steady-state inference; weights and KV state remain cached on workers. A
stateless `forward` operation remains as a diagnostic primitive, but generation
uses explicit `prefill` and `decode` operations.

## Network model

The global network is dynamic. A worker may disappear at any point. We therefore do not model the public swarm as one `mx.distributed.Group`.

Stable local groups may later use MLX Ring/JACCL/NCCL internally, while the swarm treats each group as an execution island.

```text
             swarm protocol
                  |
      +-----------+-----------+
      |                       |
  MLX island A             worker B
  Mac <-> Mac              RTX GPU
   JACCL                    MLX CUDA
```

## Scheduling direction

The scheduler should eventually optimize an execution plan against observed latency distributions rather than static GPU labels.

Future policy example:

```text
execute shard 12
  deadline: 25 ms
  primary: worker-a
  hedge after: 8 ms
  replica: worker-b
  accept first valid result
```

v0 does **not** implement this. It first measures the baseline needed to decide whether hedging is worthwhile.

## Correctness rule

A distributed execution path must match a single-worker reference within an explicitly defined numerical tolerance before its performance matters.

## Non-goals for v0

- cryptocurrency or payments
- permissionless public discovery
- malicious-worker verification
- iPhone background workers
- expert parallelism
- coded computation
- custom GPU kernels
- replacing MLX/MLX Swift LM
