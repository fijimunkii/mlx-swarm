# Architecture

## Goal

Test whether a dynamic pool of heterogeneous consumer devices can provide useful distributed LLM inference over ordinary networks without treating every participant as a reliable cluster member.

## v0 proof boundary

The v0 MVP is an evidence milestone, not a production serving architecture. It
proves a two-Mac, contiguous-shard pipeline with deterministic correctness,
retained KV state, measured warm performance, bounded next-sequence recovery,
and a physically pooled 12B checkpoint. The authoritative supported hardware,
checkpoint revisions, shard plans, acceptance predicates, and clean-checkout
procedure live in [`docs/mvp-runbook.md`](docs/mvp-runbook.md).

Terminal-shard sampling is included because it removes full-vocabulary logits
from the normal serving hot path. It is not evidence for the central
pooled-memory claim, but once integrated into the default path its exact-token
and payload-reduction invariants become release regression gates. Public
membership, automatic placement, replicated execution, same-sequence recovery,
and heterogeneous GPU workers remain post-MVP.

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
that cache by `(shardID, sequenceID)`, associates generation requests with a
private sequence owner, enforces contiguous positions, and drops only
owner-matching sequence state on close. Before shards are composed, the Go
coordinator also requires an exact fingerprint match for the resolved
checkpoint contents on every worker. Model-family adapters implement cached
prefill/decode and attention-mask semantics behind an opaque cache interface;
the Go control plane never handles cache tensors.

The coordinator transfers only execution inputs/outputs and metadata during
steady-state inference; weights and KV state remain cached on workers. A
stateless `forward` operation remains as a diagnostic primitive, but generation
uses explicit `prefill` and `decode` operations.

The Go generation session asks the Swift worker for architecture-neutral model
metadata and uses the checkpoint's Hugging Face tokenizer through `tokenize`
and `detokenize` commands. Go owns the autoregressive control loop and requests
the deterministic greedy policy; the output-owning Swift shard applies argmax
before crossing the network in normal serving mode:

```text
prompt -> tokenize -> producer prefill -> consumer prefill + argmax
       -> producer decode -> consumer decode + argmax -> ... -> EOS/max
       -> detokenize generated token IDs
```

The default split is derived from the adapter-reported layer count rather than
from a model-family constant. Stable model-and-checkpoint-derived shard IDs let
independent commands reuse stages already retained by `swarmd`; request-specific sequence
IDs isolate and bound KV state. A local reference worker is optional in normal
use and mandatory in the full-logit CI proof, where every distributed logit
vector must be within tolerance and every greedy token must match. Verification
mode keeps the full-logit response; the paired serving benchmark separately
requires the terminal-sampled token sequence to match it exactly.

## Deadlines and bounded recovery

Every cache-mutating or diagnostic inference request (`forward`, `prefill`, or
`decode`) must carry `deadlineUnixMillis`. The generation coordinator creates a
fresh timeout for each stage call; `swarmd` preserves the earlier of the
caller's context deadline and the wire deadline. The Swift worker checks the
absolute deadline immediately before and after inference so a queued request
cannot begin after its budget and a late result cannot be reported as timely.
Context-derived wire deadlines round up to millisecond precision while Go
retains the exact local deadline, so serialization cannot shorten the caller's
budget.

MLX inference is not synchronously cancelable once a kernel is running. If an
inference context expires, the Go process client kills that worker to discard
any possibly mutated KV state. The supervisor reaps and replaces a crashed,
disconnected, or timed-out worker before returning the triggering error.
Mutating requests are never retried: the active sequence fails, cleanup treats
state lost with the process as released, and a later session reloads its shards
and opens a fresh sequence. Health and state probes may be retried after a
restart because they do not mutate model or cache state. `/healthz` exposes the
worker restart count.

Generation errors retain the partial result and identify the sequence, shard,
phase, operation, position, and last accepted token. A deterministic fault
harness drives protocol-compatible child processes through pause, kill, delay,
jitter, and loopback HTTP-disconnect scenarios. CI requires every fault to
terminate within its bound, release sequence/KV state, and admit a clean next
sequence, then records the failed-token rate as a JSON artifact.

## Network model

The global network is dynamic. A worker may disappear at any point. We therefore do not model the public swarm as one `mx.distributed.Group`.

The implemented v0 network is narrower than that direction: `swarmd` exposes
an unauthenticated, unencrypted HTTP debug API for a private LAN or tailnet.
It provides no peer identity, authorization, confidentiality, integrity
protection, malicious-worker defense, NAT traversal, or multi-tenant resource
isolation. Binding that endpoint to a public interface is outside the v0 trust
model.

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
