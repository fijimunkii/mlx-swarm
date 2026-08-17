# Mesh placement and scoring

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
profile. Identity labels are bounded and retained once per series; rolling
samples contain only timestamps and numeric measurements.

Directional link observations are keyed by source and target process
identities, protocol, and tensor encoding. Every observation carries an RTT.
Payload probes may also carry bytes and the payload-transfer interval excluding
the separately reported RTT, producing an effective bytes-per-second
distribution. The same shape represents coordinator-to-worker links today and
direct worker-to-worker links if a future transport makes those relevant.
Process identities prevent measurements from silently crossing a worker
restart or coordinator run.

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

## Complete-plan scoring

`placement.ScorePlan` validates and scores a caller-proposed immutable
`generation.ExecutionPlan`. The plan must pin the exact decimal inventory
revision, and the inventory and profile snapshots must share one generation
time. This prevents a score from silently mixing worker state and measurements
from different scheduling instants.

The scoring request supplies the adapter, transport, coordinator process
identity, prefill token count, decode step count, and one resource estimate per
stage. Every stage estimate includes load and sequence memory, prefill and
per-decode wire bytes, and conservative prefill and per-decode compute
fallbacks. RTT and bandwidth fallbacks are also mandatory. A plan can therefore
still be scored when the mesh has incomplete measurements; the result records
the normalized request, which components used exact profile evidence, and the
exact profile summaries selected for those components.

For every stage, the scorer first calls the hard-constraint evaluator against
the complete inventory. Its result retains every accepted and rejected worker,
plus the selected target's candidate evidence. An ineligible selected target
makes the complete plan ineligible and leaves all cost and risk fields at zero.
Eligibility is never traded for a lower estimated cost.

Fresh compute evidence matches the exact worker process, backend, model and
checkpoint, operation, and layer range. The scorer scales the median measured
compute time by the requested token count, using the supplied fallback when no
exact series exists. A worker restart changes its process identity and therefore
cannot inherit an earlier process's compute or link evidence.

The current runtime relays every stage through the coordinator, so transfer
cost uses the exact coordinator-process-to-worker-process link. Prefill cost is
one RTT plus its payload bytes at the median measured bandwidth. Decode cost is
the same calculation for each requested decode step. Missing RTT or bandwidth
components use their explicit fallbacks independently. Direct worker-to-worker
links can replace this topology model when the runtime transport changes.

The plan score keeps unlike signals visible and compares eligible plans in this
stable order:

1. estimated sequential compute plus transfer time;
2. recent worker failures;
3. current memory pressure;
4. model bytes that require a new load;
5. total additional memory required;
6. worker restart count;
7. stage count; and
8. immutable semantic plan identity as the final lexical tie-breaker.

Exact retained runtime shard identity reduces both new-load and additional
memory cost without relaxing sequence-capacity or retained-budget constraints.
The evidence also reports stage and operation counts backed by profiles, making
fallback-heavy scores distinguishable from well-measured ones.

## Automatic plan construction

`placement.ConstructPlan` searches caller-supplied contiguous range estimates
and returns the lowest-scoring valid complete plan. A range estimate supplies
the full `StageCostEstimate` for one allowed `[layerStart, layerEnd)` edge.
Model adapters can publish every supported range or a smaller set of meaningful
split points without embedding a worker count or fixed layout in the scheduler.

The request also supplies the model/checkpoint identity, terminal response
mode, complete scoring context, maximum stage count, and a work budget. The
scoring context has no stage estimates: the constructor aligns range estimates
with each candidate plan before calling `ScorePlan`.

For each range, the constructor evaluates every inventory worker once using the
same hard constraints as explicit plan scoring. The result preserves this full
range/worker eligibility matrix, including stable rejection codes for ranges
that are not selected. Eligible range/worker pairs receive their exact compute,
transfer, health, memory, and retained-state cost before search begins.

The search advances from layer zero through the range graph. Workers are used
at most once, stages remain contiguous, and only complete coverage can produce
a plan. Dynamic programming retains the best partial plan for each layer
boundary and selected-worker set; a discarded partial has the same possible
suffixes and a strictly worse stable score. Complete candidates are rebuilt as
immutable inventory-pinned execution plans and rescored through `ScorePlan` as
an invariant check before selection.

Search work is bounded by `maxSearchOperations`, which counts the initial
range-by-worker matrix and every partial-plan transition. The default is
100,000 operations and the default stage ceiling is five. If the budget is too
small, construction returns `ErrPlanSearchLimit`, marks the result as limited,
and does not expose a partially searched plan as selected. A fully searched
mesh with no valid coverage returns a successful result with `selectedPlan`
absent and all rejection evidence intact.

Runtime shard IDs are derived from model/checkpoint identity, layer range, and
input/output ownership rather than from the inventory or complete plan
revision. Thus an unchanged compatible resident shard can be reused after a
membership/status revision. The complete plan revision still pins the exact
inventory and topology, so stale plans remain invalid for a later scheduling
decision.

## Current boundary

This package now evaluates proposed stages, scores complete plans, and chooses a
bounded deterministic plan from supplied model range estimates. It does not yet
derive those estimates from a real checkpoint adapter, expose construction
through the control-plane API, bind the selected endpoints into a generation
session, or schedule active probes. Live between-sequence replanning and the
five-Mac scheduler-selected correctness run consume these contracts next, while
the existing explicit-plan path remains the diagnostic fallback.
