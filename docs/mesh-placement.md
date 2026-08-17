# Mesh placement eligibility

The first automatic-placement primitive evaluates every current inventory
worker against the hard constraints for one proposed contiguous model stage.
It is deterministic, backend-neutral, and deliberately separate from the
policy that will choose layer boundaries or score complete plans.

Call `placement.EvaluateCandidates` with a membership inventory and a
`placement.StageRequirement`. The requirement pins:

- model ID, checkpoint fingerprint, and total layer count;
- the expected execution shard ID when retained reuse is possible;
- the proposed layer range and input/output ownership;
- the model-family adapter and worker operations implied by that ownership;
- incremental model-load and per-sequence retained-memory estimates; and
- transport protocol, tensor encoding, and optional TLS requirement.

The result records the inventory revision and generation time, the resolved
status-freshness window, normalized requirements, required operations, and one
candidate record per worker. Candidate records are sorted by stable worker ID.
Identical inputs therefore produce identical output.

## Hard constraints

A candidate is eligible only when all of these conditions hold:

- its server-observed dynamic status is current and healthy;
- it advertises the required adapter, checkpoint fingerprint when its
  fingerprint allowlist is non-empty, and every required operation;
- it supports the requested transport, tensor encoding, and TLS mode;
- its current available memory covers the required incremental allocation;
- the new sequence fits its retained-byte budget; and
- a reused retained shard has open-sequence capacity.

Status defaults to the inventory lease TTL as its maximum age. A caller may
set a stricter positive `statusMaxAgeMillis`. Lease liveness never makes stale
capacity, health, or retained state schedulable.

Available memory and retained-byte admission are separate constraints. Loading
a new shard requires the stage's estimated model bytes plus its per-sequence
reserve. Exact retained-shard reuse requires only the sequence reserve. The
retained-byte budget covers sequence KV/replay/output state rather than model
weights, so it is checked against the current retained bytes plus that reserve.

A retained shard is reusable only when its runtime shard ID, model ID,
checkpoint fingerprint, layer range, and input/output ownership all match
exactly. When a proposed stage does not yet have a runtime shard ID, the
evaluator conservatively budgets a new load. Reuse is reported in the candidate
result so the later plan scorer can price it without weakening any hard
constraint.

## Rejection codes

Every failed hard constraint emits a stable code and a human-readable detail:

| Code | Meaning |
|---|---|
| `stale_status` | Dynamic status is missing, future-dated, or older than the allowed window. |
| `unhealthy` | Worker health is not `healthy`. |
| `unsupported_adapter` | Required model adapter is not advertised. |
| `incompatible_checkpoint` | A non-empty checkpoint allowlist excludes the requested fingerprint. |
| `unsupported_operation` | One or more stage operations are not advertised. |
| `unsupported_transport` | Requested transport protocol is not advertised. |
| `unsupported_tensor_encoding` | Transport does not support the requested tensor encoding. |
| `tls_required` | The request requires TLS but the transport does not advertise it. |
| `insufficient_memory` | Current available memory is below the incremental requirement. |
| `retained_budget_exceeded` | The sequence reserve would exceed retained-state admission. |
| `sequence_capacity_exhausted` | A reusable shard has reached its open-sequence limit. |

The evaluator reports every applicable rejection in a stable order rather than
stopping at the first one. This gives operators and future scheduler evidence
enough to explain why a worker was not considered.

## Topology and compute profile

`placement.ProfileStore` retains bounded rolling evidence for later plan
scoring. Its default five-minute window keeps at most 64 observations per
series and 4,096 total series; callers may set smaller or larger explicit
limits. Link and compute updates are concurrency-safe, and a complete N-stage
sample is admitted atomically so a rejected batch cannot leave a partial
profile.

Directional link observations are keyed by source and target process
identities, protocol, and tensor encoding. Every observation carries an RTT.
Payload probes may also carry bytes and elapsed time, producing an effective
bytes-per-second distribution. The same shape represents coordinator-to-worker
links today and direct worker-to-worker links if a future transport makes those
relevant. Process identities prevent measurements from silently crossing a
worker restart or coordinator run.

Compute observations are keyed by worker process, backend, model/checkpoint,
operation, and exact layer range. Their summaries preserve input-token and
compute-time distributions rather than averaging unlike model ranges together.
`ObservePlannedSample` converts a successful arbitrary N-stage generation
sample into these per-worker observations using the inventory's instance and
backend identity. The plan must pin the supplied inventory's exact revision.

Generation stage overhead is not treated as bandwidth: it combines HTTP,
queueing, serialization, and network time. Link throughput therefore requires
an explicit measurement rather than manufacturing a misleading value from a
normal inference call.

Every update supplies a server-controlled acceptance time. Observations that
are already stale or dated after acceptance are rejected before they can
consume or evict rolling evidence. Snapshots cannot rewind before the latest
accepted update. Fresh updates reclaim expired series before enforcing the
global series limit. Snapshots record their revision, generation time,
freshness and storage bounds, remove expired observations, and sort every
profile by its stable identity. The same accepted observation history and
snapshot time therefore produce identical machine-readable evidence.

Current memory pressure, health, failures, restart counts, and retained-shard
state remain in the server-stamped membership inventory rather than being
duplicated into historical profiles. The planner must combine a fresh inventory
snapshot with a fresh topology profile.

## Current boundary

This package evaluates a proposed stage; it does not yet choose stage count,
layer boundaries, workers, or a complete execution plan. The profile accepts
probe and successful-generation evidence but does not schedule active probes or
score compute and transfer cost. Those policy layers consume these contracts
next, while the existing explicit-plan path remains the diagnostic fallback.
