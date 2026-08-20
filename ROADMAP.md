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

The resident mesh is the baseline, not the final differentiator. The project is
now focused on two harder outcomes:

1. an admitted sequence can continue after a selected worker disappears; and
2. a bounded physical mesh can trade latency for capacity by staging a model
   whose checkpoint and resident footprint exceed aggregate active memory.

`mlx-swarm` is not trying to become a general local-cluster product. Broad model
catalogs, dashboards, API compatibility, RDMA tuning, continuous batching, and
ordinary tensor-parallel feature parity are useful elsewhere, but they are not
roadmap goals unless they directly strengthen elastic recovery or bounded
out-of-core execution.

## Current status — resident mesh proof complete

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

The Dynamic Trusted Mesh implementation is complete on the `mesh/v0`
integration branch in PR #36. It adds arbitrary N-stage execution, a real
five-Mac/Linux-coordinator proof, automatic placement from fresh membership,
between-sequence replanning, and deterministic 32-member control-plane churn.
M8 becomes part of the default branch when PR #36 merges; same-sequence recovery
and out-of-core execution remain unproven.

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
| M7 | Immutable Distributed Inference Proof release | [#24](https://github.com/fijimunkii/mlx-swarm/issues/24) |
| M8 | Dynamic Trusted Mesh: N-stage execution, five-Mac scale, automatic placement, and control-plane churn | [#25](https://github.com/fijimunkii/mlx-swarm/issues/25), [#31](https://github.com/fijimunkii/mlx-swarm/issues/31), [#32](https://github.com/fijimunkii/mlx-swarm/issues/32), [#35](https://github.com/fijimunkii/mlx-swarm/issues/35), [#34](https://github.com/fijimunkii/mlx-swarm/issues/34) |
| M9 | Continue an interrupted sequence by deterministic replay | [#27](https://github.com/fijimunkii/mlx-swarm/issues/27) |
| M10 | Stage verified checkpoint ranges without full-model downloads | [#47](https://github.com/fijimunkii/mlx-swarm/issues/47) |
| M11 | Distributed out-of-core inference beyond aggregate active memory | [#48](https://github.com/fijimunkii/mlx-swarm/issues/48) |
| M12 | Resumable binary tensor and sequence-state transport | [#26](https://github.com/fijimunkii/mlx-swarm/issues/26) |
| M13 | Heterogeneous replacement capacity: Linux CPU, then CUDA/NVIDIA | [#33](https://github.com/fijimunkii/mlx-swarm/issues/33) → [#28](https://github.com/fijimunkii/mlx-swarm/issues/28) |
| M14 | Failure-aware expert paging experiment | [#29](https://github.com/fijimunkii/mlx-swarm/issues/29) |

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

Publish the successful proof as an immutable named release while later work
continues on separate integration branches.

- [ ] add an explicit open-source license
- [ ] reconcile README, roadmap, merged PR, and completed issue state
- [ ] validate one exact release revision through every documented gate
- [ ] create an immutable, non-`v0` proof tag from the validated revision
- [ ] publish release notes with supported hardware, checkpoints, tolerances,
  trust boundary, evidence, and deliberate limitations
- [ ] make benchmark, failure, and pooled-memory artifacts durable from the release

## M8 — Dynamic Trusted Mesh

Parent milestone: [#25](https://github.com/fijimunkii/mlx-swarm/issues/25)

Turn the fixed two-node experiment into a trusted-LAN or private-tailnet mesh
that can use all available Apple-silicon inference slots, automatically choose a
valid plan, and remain understandable under larger membership sets.

M8 is intentionally split into focused deliverables. The roadmap PR defines the
sequence; implementation should land through a dedicated draft **Dynamic
Trusted Mesh** integration PR, with component PRs stacked into that integration
branch when they depend on shared in-progress interfaces.

All four deliverables are complete on `mesh/v0`. PR #36 remains the integration
gate to `main`; later milestones must not expand its implementation scope.

### M8.1 — arbitrary N-stage execution

Tracking: [#31](https://github.com/fijimunkii/mlx-swarm/issues/31)

Remove the current producer/consumer assumption before adding scheduling policy.

- [x] represent an execution plan as an ordered list of stages rather than one
  producer and one consumer
- [x] support explicit 2-, 3-, 4-, and 5-stage plans without worker-count-specific
  generation code
- [x] validate contiguous layer coverage, ownership, checkpoint identity, and
  terminal output semantics before inference
- [x] run prefill and cached decode through every planned stage in order
- [x] make multi-stage sequence open/close transactional and rollback-safe
- [x] generalize KV, memory, timing, and transfer observations to per-stage data
- [x] preserve full-reference correctness and the existing two-stage proof as a
  regression path

### M8.2 — five real Mac workers with a Linux coordinator

Tracking: [#32](https://github.com/fijimunkii/mlx-swarm/issues/32)

Use Linux for orchestration so every available macOS slot can perform real MLX
inference. Implementation and the canonical evidence contract are documented in
[`docs/five-mac-scale-proof.md`](docs/five-mac-scale-proof.md); the checklist
is complete after the canonical on-demand workflow published a fully passing
artifact. Smaller MLX and pooled-memory workflows remain the pull-request gates.

```text
Linux coordinator
       |
       +-- Mac worker 0: input + early layers
       +-- Mac worker 1: middle layers
       +-- Mac worker 2: middle layers
       +-- Mac worker 3: middle layers
       +-- Mac worker 4: late layers + output head
```

- [x] start five independent Apple-silicon `swarmd` workers plus one Linux
  coordinator in one private-tailnet workflow
- [x] prove five-stage 270M generation with reference-matched logits and tokens
- [x] reuse the same five-worker pool to measure 2-, 3-, 4-, and 5-stage plans
  under one fixed model/prompt/token plan
- [x] record stage count versus per-worker memory, load time, TTFT, inter-token
  latency, throughput, boundary bytes, wire bytes, and failure surface
- [x] run the 12B pooled-memory proof across all five serving Macs using a
  memory-aware split rather than equal layer counts by assumption
- [x] keep the full-model oracle off the serving Macs and preserve physical
  memory evidence for every worker
- [x] guarantee all worker jobs exit after success or bounded failure so scarce
  macOS concurrency is released

The expected benefit of more serial stages is **capacity and lower per-worker
memory**, not automatically lower single-sequence latency. The scaling curve is
an explicit measurement deliverable.

### M8.3 — membership and automatic placement

Tracking: [#35](https://github.com/fijimunkii/mlx-swarm/issues/35)

Once explicit N-stage execution is correct, let the control plane build the plan.

- [x] define a versioned capability and inventory record
- [x] register workers and expire stale membership through heartbeats
- [x] distinguish lease liveness from server-stamped dynamic-status freshness
- [x] measure or ingest compute latency, RTT, bandwidth, memory, health, and
  retained-shard state
- [x] represent plan constraints independently from a fixed worker count or
  backend
- [x] place complementary ranges subject to memory, adapter, fingerprint,
  topology, input/output ownership, and retained-state constraints
- [x] score plans by observed compute, transfer cost, health, memory pressure,
  and shard reuse
- [x] replan between sequences when workers join, leave, slow down, or fail
- [x] emit selected stages plus rejected-node reasons as machine-readable evidence

The first scheduler should prefer a simple valid contiguous-layer plan over a
clever opaque one. A worker may be excluded from a request when using it would
make that execution plan worse.

### M8.4 — control-plane scale and churn

Tracking: [#34](https://github.com/fijimunkii/mlx-swarm/issues/34)

Use Linux concurrency to test a much larger membership set without consuming
additional macOS inference slots.

- [x] create protocol-compatible synthetic workers with configurable memory,
  backend, adapter, fingerprint, latency, topology, retained shards, and health
- [x] handle at least 32 concurrently visible worker records without fixed-size
  assumptions
- [x] exercise join, expiry, rejoin, duplicate identity, slowdown, and
  incompatible capability changes
- [x] bound network probing and scheduler work as membership grows
- [x] record candidate rejection reasons and scheduler decision latency
- [x] combine real Mac serving workers with many synthetic Linux peers and prove
  the chosen real plan still generates reference-correct output

Synthetic membership tests control-plane behavior; they do not claim that every
Linux peer performs model math.

### M8 acceptance boundary

M8 is complete only when all of the following are true:

- arbitrary 2–5 stage execution is correct and regression-tested;
- five real Mac workers generate correct text under a Linux coordinator;
- the 2/3/4/5-stage scaling curve is published as machine-readable evidence;
- the scheduler automatically chooses a valid plan from current capabilities and
  topology rather than consuming a hard-coded split;
- join/leave/replan behavior is bounded between sequences; and
- at least 32 visible worker records are handled with deterministic placement
  rejection and scheduling evidence.

Linux **model execution** is deliberately the next milestone rather than an M8
merge gate.

## M9 — recover an interrupted sequence by replay

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

## M10 — range-aware checkpoint staging

Tracking: [#47](https://github.com/fijimunkii/mlx-swarm/issues/47)

Make checkpoint acquisition and resident memory proportional to an assigned
range. A middle-stage worker should not need to download input/output-only files
when the upstream safetensor layout permits a narrower manifest.

- [ ] derive verified range manifests from the checkpoint index, layer range,
  and input/output ownership
- [ ] download only required files while recording unavoidable upstream overlap
- [ ] bind every file and manifest to immutable revision, hash, tokenizer, and
  adapter identity
- [ ] distinguish disk cache, resident weights, allocator cache, and sequence/KV
  state in capability and evidence records
- [ ] load, evict, and reload a range without restarting the worker
- [ ] preserve intermediate-output tolerance and exact token parity after reload
- [ ] let placement price cache locality, download cost, disk capacity, and
  retained compatible ranges without trusting stale membership
- [ ] fail partial downloads, missing tensors, insufficient storage, and hash
  mismatches before advertising the range as available

File-level range selection may still download overlapping tensors when an
upstream shard mixes logical owners. Evidence must report that physical overlap
rather than claiming ideal disjoint storage.

## M11 — distributed out-of-core inference

Tracking: [#48](https://github.com/fijimunkii/mlx-swarm/issues/48)

Decouple logical model ranges from concurrently resident physical workers. Run
more logical stages than can fit at once by loading verified ranges in bounded
waves and preserving or reconstructing each range's sequence state.

- [ ] represent an immutable ordered wave plan with more logical ranges than
  concurrently resident workers
- [ ] load, restore, execute, checkpoint or reconstruct, and evict every range
  under explicit deadlines and mutation identities
- [ ] prove the checkpoint and measured fully resident footprint exceed the
  aggregate active memory budget
- [ ] keep every worker within physical and configured memory limits without any
  worker retaining the complete checkpoint
- [ ] preserve or deterministically reconstruct per-range KV state across
  eviction and reassignment
- [ ] match independent reference logits and at least 32 exact greedy tokens
- [ ] replace one lost worker after a committed token and continue without
  restarting unaffected workers
- [ ] record weight I/O, state I/O or replay, compute, transfer, eviction, idle,
  TTFT, inter-token latency, throughput, and recovery cost
- [ ] release proof-owned weights, state, reservations, and temporary artifacts
  after success or bounded failure

Resident placement remains preferred whenever the model fits and satisfies the
request's latency constraints. Out-of-core mode is a capacity fallback, not a
claim that repeatedly paging a dense model is fast.

## M12 — resumable binary tensor and sequence-state transport

Tracking: [#26](https://github.com/fijimunkii/mlx-swarm/issues/26)

Replace temporary JSON/base64 activation relaying with a bounded binary path
that also supports the state movement required by recovery and out-of-core
execution. Transport performance is secondary to mutation and resume semantics.

- [ ] implement versioned framed tensor and sequence-state messages with raw
  contiguous bytes
- [ ] reuse persistent multiplexed connections across prefill, decode, replay,
  and state-transfer calls
- [ ] preserve request IDs, deadlines, response modes, mutation identity,
  sequence ownership, committed position, and structured errors
- [ ] support bounded reconnect/resume without applying an acknowledged tensor
  or state chunk twice
- [ ] enforce frame, tensor, metadata, state, and in-flight request limits
- [ ] detect malformed, truncated, corrupt, mismatched, and incompatible frames
- [ ] retain HTTP/JSON temporarily as a diagnostic compatibility path
- [ ] compare legacy and binary paths with identical hardware, prompt, split,
  token plan, correctness, and failure gates
- [ ] demonstrate lower wire bytes and serialization overhead without weakening
  cleanup or recovery bounds

Correctness-first M9 and M11 work may use the existing transport. Final
performance characterization should use M12.

## M13 — heterogeneous replacement capacity

Linux should become a real inference participant in two steps: prove the current
MLX Swift worker on ordinary Linux CPU first, then extend the same protocol path
to NVIDIA/CUDA. The strategic value is elastic replacement capacity, not merely
adding another supported backend.

### M13.1 — Linux MLX CPU worker

Tracking: [#33](https://github.com/fijimunkii/mlx-swarm/issues/33)

- [ ] build the existing Swift `MLXWorker` on Linux in MLX CPU mode
- [ ] run health, capability, checkpoint metadata, retained-shard, cache, and
  memory proofs on a standard Linux runner
- [ ] retain only an assigned checkpoint range rather than a full model
- [ ] compose at least one Linux CPU middle stage with macOS Metal stages
- [ ] match mixed-plan logits at an explicit tolerance and at least 32 exact
  greedy tokens
- [ ] report Linux CPU as a valid but potentially low-priority scheduler resource
  using measured compute and topology rather than OS labels
- [ ] compare mixed Linux/macOS performance with an Apple-only plan under the
  same prompt and token sequence
- [ ] replace one interrupted macOS stage with a compatible Linux range and
  continue through the M9 recovery contract

Linux CPU participation is useful even if it is too slow for latency-critical
decode. The scheduler should include it only when its memory or capacity has
positive marginal value for that plan.

### M13.2 — Linux MLX CUDA / NVIDIA worker

Tracking: [#28](https://github.com/fijimunkii/mlx-swarm/issues/28)

- [ ] extend the Linux worker path to NVIDIA/CUDA after #33 proves portability
- [ ] prefer MLX CUDA when model support, partial loading, quantization, KV-cache,
  deterministic execution, and memory accounting satisfy the proof contract
- [ ] advertise CUDA device and memory capabilities through the same inventory
- [ ] load only the assigned checkpoint range on the NVIDIA worker
- [ ] place complementary stages across Apple Metal and NVIDIA CUDA workers
- [ ] match reference logits at an explicit tolerance and at least 32 exact
  greedy tokens
- [ ] publish mixed-hardware versions, placement, memory, transfer, correctness,
  performance, and failure evidence
- [ ] prove that a CUDA worker can join as later replacement capacity without a
  GPU-specific coordinator or sequence protocol

Standard GitHub-hosted Linux jobs are appropriate for the CPU proof and control
plane but do not provide the NVIDIA device required for this CUDA validation.
Use a documented self-hosted or external GPU workflow for the hardware proof.
The Go control plane must remain backend-neutral if a runtime fallback is ever
required.

## M14 — failure-aware expert paging experiment

Tracking: [#29](https://github.com/fijimunkii/mlx-swarm/issues/29)

Treat sparse experts as evictable, replaceable ranges rather than adding generic
MoE execution for its own sake. Expert routing can reduce active compute and
make storage-backed execution practical, but fan-out, cache misses, and failure
recovery may erase the benefit on ordinary networks.

- [ ] select one real open-weight MoE checkpoint and expose shared modules,
  routers, and individually placeable experts through an adapter
- [ ] record expert activation frequency, co-activation, skew, fan-out, and
  communication volume
- [ ] page cold experts from verified checkpoint storage and keep hot experts
  resident using measured activation evidence
- [ ] place experts across at least three workers using memory, storage locality,
  topology, compute, and observed routing data
- [ ] selectively replicate hot or latency-critical experts as recovery capacity
- [ ] match router aggregation and final outputs against an independent reference
- [ ] compare MoE placement with contiguous pipeline sharding on capacity,
  cache-hit rate, latency, throughput, memory, storage I/O, wire bytes, and failures
- [ ] lose an assigned expert worker and either continue through bounded
  replacement or return a structured failure without corrupting aggregation
- [ ] test at least one LAN and one higher-latency private-network condition
- [ ] publish a go/no-go report defining where expert placement wins and loses

## Dependency order

```text
M7 proof release (#24) may proceed independently

M8 Dynamic Trusted Mesh (#25 / PR #36)
├── M9 same-sequence replay recovery (#27)
├── M10 range-aware checkpoint staging (#47)
└── M12 resumable binary transport (#26)

M9 + M10 -> M11 distributed out-of-core inference (#48)
M9 + M10 -> M13.1 Linux CPU replacement (#33)
M13.1 -> M13.2 CUDA/NVIDIA replacement (#28)
M10 + M11 + M12 + M13 -> M14 failure-aware expert paging (#29)
```

M9 is the next correctness milestone after M8. M10 may proceed in parallel
because it changes storage and residency rather than committed-token semantics.
M12 can also proceed behind the stable M8 peer boundary, but it must not delay a
correct replay proof. M11 combines recovery and staging only after both have
independent evidence.

## CI scale strategy

Use runner classes according to what they prove:

- **macOS jobs:** scarce real MLX/Metal inference workers; keep the existing
  resident proofs and use bounded worker pools for recovery and wave execution.
- **Linux coordinator:** run the Go control plane, discovery, plan submission,
  assertions, artifact aggregation, and teardown without spending a macOS slot.
- **Linux CPU worker:** prove protocol-compatible replacement and low-priority
  capacity through #33 after same-sequence recovery is stable.
- **Linux synthetic workers:** use larger inexpensive concurrency to exercise
  membership churn, capability incompatibility, health changes, and scheduler
  candidate scale without pretending each node performs model math.
- **NVIDIA Linux:** use a self-hosted or external GPU runner for #28; keep the
  same worker and swarm protocol where possible.

Hosted CI is evidence infrastructure, not the production topology. Every result
must record the actual hardware, route, model, plan, and revision that produced
it.

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
8. **Scale claims distinguish control plane from execution.** Synthetic Linux
   membership may prove scheduler scale, but only real model workers count as
   pooled inference capacity.
9. **Logical scale is not physical concurrency.** Out-of-core claims must record
   logical ranges, concurrently resident workers, storage traffic, and every
   load/eviction transition separately.
10. **Resident execution wins by default.** Paging or replay is selected only
    when capacity or recovery requires it; it is never presented as a free
    performance optimization.

## Later

The following remain deliberately deferred while recovery and out-of-core
execution are unproven:

- peer authentication, encryption, authorization, and multi-tenant isolation
- public membership, NAT traversal, and malicious-worker verification
- synchronous KV replication, delayed hedging, and first-valid-result selection
- continuous batching and concurrent-request scheduling
- speculative decoding and draft-model placement
- activation quantization or compression across shard boundaries
- advanced execution islands using stable high-speed collective groups
- general tensor-parallel feature parity and RDMA-specific optimization
- broad model catalogs, dashboard work, and additional API compatibility
- mobile and opportunistic background workers
- compute credits, accounting, and payments
