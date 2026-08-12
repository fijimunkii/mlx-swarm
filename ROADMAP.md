# Roadmap

## Mission

Build a failure-aware inference fabric that pools model memory and compute across
heterogeneous consumer devices without pretending that every participant is a
reliable datacenter node.

A useful mesh does **not** put every available device in every token's critical
path. Eligible devices should contribute where the scheduler predicts positive
marginal value: hosting required model state, executing a latency-appropriate
stage, serving a replica, rebuilding failed state, handling another sequence,
or eventually hosting selected experts.

The project advances only when each capability has a reproducible correctness,
memory, performance, and failure contract. Performance claims never replace
reference parity.

## Current status — Distributed Inference Proof complete

The first proof milestone merged to `main` in PR #1 at commit
`c4bd00d22943dd8d0aa7814fdeffe6267dea16f4`.

- [x] retain complementary real-checkpoint shards in supervised workers
- [x] generate tokenizer-backed text with per-shard KV-cached prefill/decode
- [x] match a cached full-checkpoint reference at explicit tolerances
- [x] publish reproducible latency, throughput, transfer, and memory evidence
- [x] characterize bounded deadline, process, and transport failures
- [x] prove 32-token generation from a model that cannot fit on either serving worker
- [x] provide one clean-checkout, two-Mac verification runbook with machine-readable gates
- [x] finish the integrated review and merge the proof PR to `main`

The authoritative procedure and supported release boundary remain in
[`docs/distributed-inference-proof.md`](docs/distributed-inference-proof.md).
The proof establishes a correct, stateful, physically pooled two-Mac inference
pipeline. It does not yet establish automatic placement, public membership,
same-sequence recovery, heterogeneous GPU execution, or MoE expert placement.

### Proof baseline

- **Execution:** Go control plane supervising persistent Swift/MLX workers.
- **Correctness:** Gemma 3 270M split into complementary 9-layer stages, with
  cached distributed logits matching full-checkpoint inference at
  `rtol=atol=1e-4` and exact greedy-token parity.
- **Pooled memory:** Gemma 3 12B split 24/24 layers across two fresh 7 GiB
  Apple-silicon workers. Neither serving process loads a full model; both the
  checkpoint and measured full-model process footprint exceed either worker's
  physical memory.
- **State:** Per-shard, per-sequence KV caches survive repeated decode calls and
  are released without unloading resident weights.
- **Serving payload:** The terminal shard performs deterministic greedy sampling
  and returns one token instead of full-vocabulary logits.
- **Failure contract:** A timed-out or lost worker fails the active sequence,
  discards ambiguous KV state, is replaced, and admits a clean next sequence.
  Transparent continuation is not yet claimed.
- **Evidence:** CI publishes machine-readable correctness, benchmark, failure,
  hardware, memory, shard-ownership, and teardown artifacts.

## Milestone overview

| Milestone | Outcome | Tracking |
|---|---|---|
| M0–M6 | Distributed Inference Proof | Complete in PR #1 |
| M7 | Immutable `v0.1.0-proof` release baseline | [#24](https://github.com/fijimunkii/mlx-swarm/issues/24) |
| M8 | Three-or-more-node Dynamic Trusted Mesh | [#25](https://github.com/fijimunkii/mlx-swarm/issues/25) |
| M9 | Persistent binary tensor transport | [#26](https://github.com/fijimunkii/mlx-swarm/issues/26) |
| M10 | Same-sequence recovery by deterministic replay | [#27](https://github.com/fijimunkii/mlx-swarm/issues/27) |
| M11 | Mixed Apple/NVIDIA inference through a second backend | [#28](https://github.com/fijimunkii/mlx-swarm/issues/28) |
| M12 | Measured topology-aware MoE placement experiment | [#29](https://github.com/fijimunkii/mlx-swarm/issues/29) |

## Completed milestones

### M0 — bootstrap

- [x] establish the Go control-plane / worker boundary
- [x] define initial language-neutral wire messages
- [x] build `swarmd` and the MLX worker in CI
- [x] add local health and capability handshakes

### M1 — one machine, one worker

- [x] load a supported model through MLX Swift LM
- [x] expose worker capabilities and memory information
- [x] run deterministic prefill/decode through the worker command API
- [x] establish single-worker reference outputs

### M2 — retained two-Mac pipeline

- [x] load complementary contiguous layer ranges from a real checkpoint
- [x] retain assigned shards across repeated forwards
- [x] carry hidden states between independent workers
- [x] preserve isolated per-sequence KV caches
- [x] generate user-visible text with cached greedy decode
- [x] compare distributed logits and tokens with a full-checkpoint reference
- [x] measure warm p50/p95 latency, TTFT, transfer, and tokens/sec
- [x] move greedy sampling to the output-owning shard

### M3 — bounded failure characterization

- [x] propagate absolute deadlines through Go, transport, and Swift execution
- [x] inject pause, kill, delay, jitter, and connection loss
- [x] return structured partial-generation failures
- [x] discard ambiguous state after non-preemptible MLX timeouts
- [x] replace failed workers and admit a clean next sequence
- [x] publish failed-token, cleanup, restart, and recovery observations

### M4 — reusable orchestration boundaries

- [x] separate model-family adapters from generic checkpoint orchestration
- [x] centralize worker supervision, sequence ownership, and bounded cleanup
- [x] enforce checkpoint fingerprints before composing shards
- [x] retain a language-neutral control-plane boundary for future backends

### M5 — reproducible network evidence

- [x] exercise the real checkpoint boundary between paired macOS runners
- [x] record producer, serialization, transport, consumer, and end-to-end time
- [x] preserve raw benchmark samples plus p50/p95 summaries
- [x] characterize private-tailnet routing without imposing a universal latency SLO

### M6 — physically pooled-memory generation

- [x] select a checkpoint and full inference footprint larger than either worker
- [x] load only complementary checkpoint ranges on the serving workers
- [x] generate at least 32 deterministic reference-matched tokens
- [x] record load, prefill, decode, process, and MLX memory evidence
- [x] prove no serving process loads a full-model correctness oracle
- [x] release all proof-owned sequence and shard state after validation

## M7 — publish the proof baseline

Tracking: [#24](https://github.com/fijimunkii/mlx-swarm/issues/24)

Freeze the successful proof before changing membership, placement, transport,
and worker implementations.

- [ ] add an explicit open-source license
- [ ] reconcile README, roadmap, merged PR, and completed issue state
- [ ] validate one exact release revision through every documented gate
- [ ] create the immutable `v0.1.0-proof` tag
- [ ] publish release notes with supported hardware, checkpoints, tolerances,
  trust boundary, evidence, and deliberate limitations
- [ ] make benchmark, failure, and pooled-memory artifacts durable from the release

## M8 — Dynamic Trusted Mesh

Tracking: [#25](https://github.com/fijimunkii/mlx-swarm/issues/25)

Turn the fixed two-node experiment into a trusted-LAN or private-tailnet mesh
that automatically constructs plans across three or more unequal workers.

- [ ] define a versioned capability and inventory record
- [ ] register workers and expire stale membership through heartbeats
- [ ] measure or ingest compute latency, RTT, bandwidth, memory, health, and
  retained-shard state
- [ ] represent execution plans independently from a fixed two-node split
- [ ] place complementary ranges subject to memory, adapter, fingerprint, and
  output-head constraints
- [ ] score plans by observed compute, transfer cost, health, and shard reuse
- [ ] replan between sequences when workers join, leave, slow down, or fail
- [ ] prove correct pooled-memory generation across at least three unequal nodes
- [ ] emit plan decisions, topology, shard ownership, and cleanup as
  machine-readable evidence

The first scheduler should prefer a simple valid contiguous-layer plan over a
clever opaque one. A worker may be excluded from a request when using it would
make that execution plan worse.

## M9 — persistent binary tensor transport

Tracking: [#26](https://github.com/fijimunkii/mlx-swarm/issues/26)

Replace temporary JSON/base64 activation relaying before treating network
performance as representative of the architecture.

- [ ] implement versioned framed tensor messages with raw contiguous bytes
- [ ] reuse persistent multiplexed connections across prefill and decode calls
- [ ] preserve request IDs, deadlines, response modes, mutation identity,
  sequence ownership, and structured errors
- [ ] enforce frame, tensor, metadata, and in-flight request limits
- [ ] detect malformed, truncated, corrupt, and incompatible frames
- [ ] retain HTTP/JSON temporarily as a diagnostic compatibility path
- [ ] compare legacy and binary paths with identical hardware, prompt, split,
  token plan, and correctness gates
- [ ] demonstrate lower wire bytes and serialization overhead without changing
  model outputs or failure bounds

M8 and M9 may proceed in parallel after M7, but membership and transport should
share stable peer and connection abstractions.

## M10 — recover an interrupted sequence by replay

Tracking: [#27](https://github.com/fijimunkii/mlx-swarm/issues/27)

Use deterministic token history as the first recovery log instead of immediately
replicating live KV tensors.

- [ ] record prompt identity, checkpoint fingerprint, plan revision, accepted
  token IDs, and the last committed position
- [ ] distinguish committed output from in-flight or ambiguously acknowledged work
- [ ] select or load a compatible replacement for the lost shard
- [ ] replay the prompt and committed tokens to rebuild replacement KV state
- [ ] validate the reconstructed sequence position before resuming
- [ ] preserve exact greedy-token parity with an uninterrupted reference
- [ ] bound replay and overall recovery time, with structured terminal failure
  when recovery cannot complete
- [ ] inject failures during prefill, early decode, late decode, response loss,
  and replacement startup
- [ ] publish replay length, latency, replacement choice, parity, and cleanup evidence

This milestone establishes correctness-first same-sequence recovery. Replicated
KV caches and near-zero-latency failover remain later optimizations.

## M11 — second worker backend and mixed hardware

Tracking: [#28](https://github.com/fijimunkii/mlx-swarm/issues/28)

Prove that `swarmd` coordinates protocol-compatible execution services rather
than only one Swift implementation.

- [ ] select an NVIDIA-capable backend after measuring partial loading,
  quantization, cache, deterministic execution, and memory-accounting support
- [ ] implement the common lifecycle, capability, shard, prefill, decode,
  deadline, and cleanup contract
- [ ] load only the assigned checkpoint range on the new backend
- [ ] normalize checkpoint identity, tokenizer, dtype, shape, layout, and
  sampling behavior across runtimes
- [ ] place complementary stages across Apple-silicon and NVIDIA workers
- [ ] match reference logits at an explicit tolerance and at least 32 exact
  greedy tokens
- [ ] publish mixed-backend hardware, versions, placement, memory, transfer,
  correctness, performance, and failure evidence

MLX CUDA is preferred when it satisfies the proof contract, but the protocol
must not require every backend to share one local runtime implementation.

## M12 — topology-aware Mixture-of-Experts experiment

Tracking: [#29](https://github.com/fijimunkii/mlx-swarm/issues/29)

Treat MoE as a measured scheduler strategy, not as an assumed shortcut. Expert
routing can reduce active compute, but repeated fan-out and gather traffic may
make distributed experts worse than contiguous pipeline stages on ordinary
networks.

- [ ] select one real open-weight MoE checkpoint and expose shared modules,
  routers, and individually placeable experts through an adapter
- [ ] record expert activation frequency, co-activation, skew, fan-out, and
  communication volume
- [ ] place experts across at least three workers using memory, topology,
  compute, and observed routing data
- [ ] selectively replicate hot or latency-critical experts
- [ ] match router aggregation and final outputs against an independent reference
- [ ] compare MoE placement with contiguous pipeline sharding on capacity,
  latency, throughput under concurrency, memory, wire bytes, and failures
- [ ] test at least one LAN and one higher-latency private-network condition
- [ ] publish a go/no-go report defining where expert placement wins and loses

## Dependency order

```text
M7 proof release
├── M8 Dynamic Trusted Mesh
│   ├── M10 replay recovery
│   └── M11 mixed hardware
└── M9 binary transport
    ├── strengthens M10 recovery performance
    └── joins M8 as a foundation for M11

M8 + M9 + M11 -> M12 MoE placement experiment
```

Correctness work for M10 may start once M8 can select replacement workers; final
performance characterization should use M9. M11 should consume the stable
membership and transport contracts rather than adding a separate control path.

## Cross-cutting release gates

Every milestone must continue to satisfy these rules:

1. **Correctness before performance.** Distributed outputs must match an
   independent reference at a stated tolerance before speed matters.
2. **Physical claims use physical evidence.** Configured allocator thresholds
   alone do not prove that a model cannot fit on one host.
3. **State mutations have explicit semantics.** Timeouts, retries, replay,
   response loss, ownership, and cleanup must be deterministic.
4. **Evidence is machine-readable.** Commands, schemas, raw samples, hardware,
   revisions, fingerprints, and pass predicates belong in the repository or CI.
5. **The trust boundary is stated.** Trusted private networking is not public or
   multi-tenant security.
6. **No universal performance claim.** Results are tied to the measured model,
   hardware, topology, transport, and workload.
7. **The mesh may decline work.** Membership does not guarantee placement on a
   request when a node would violate correctness, memory, health, or latency
   constraints.

## Later

The following remain deliberately deferred until the Dynamic Trusted Mesh has
measured data behind it:

- peer authentication, encryption, authorization, and multi-tenant isolation
- public membership, NAT traversal, and malicious-worker verification
- replicated shards, delayed hedging, and first-valid-result selection
- continuous batching and concurrent-request scheduling
- speculative decoding and draft-model placement
- activation quantization or compression across shard boundaries
- advanced execution islands using stable high-speed collective groups
- mobile and opportunistic background workers
- compute credits, accounting, and payments
