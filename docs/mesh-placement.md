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

## Current boundary

This package evaluates a proposed stage; it does not yet choose stage count,
layer boundaries, workers, or a complete execution plan. It also does not
measure topology or score compute and transfer cost. Those policy layers will
consume this contract next, while the existing explicit-plan path remains the
diagnostic fallback.
