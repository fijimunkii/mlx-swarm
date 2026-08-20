# Synthetic mesh scale and churn

The Linux-only mesh proof stresses membership and placement independently from
scarce macOS model runners. It creates backend-neutral synthetic worker specs,
sends their versioned registrations, heartbeats, removals, and re-registrations
through the production membership HTTP API, and passes the resulting inventory
and rolling topology/compute profiles to the production plan constructor.

Synthetic peers advertise the same configurable fields as a real worker:
backend/runtime identity, memory and admission limits, adapters, checkpoint
fingerprints, operations, transport, health and failure counters, retained
shards, RTT/bandwidth hints, and per-range prefill/decode latency. Their
endpoints are evidence identities only; this proof does not call them or claim
that they perform model math.

## Run it

```bash
go run ./cmd/swarm-mesh-smoke > mesh-scale-churn.json
```

The defaults require at least 32 concurrently visible records, cap each plan
search at 20,000 counted operations, cap each complete scheduling decision at
five seconds, and finish the whole proof within 30 seconds. These can be made
stricter for local experiments:

```bash
go run ./cmd/swarm-mesh-smoke \
  -workers 64 \
  -max-search-operations 40000 \
  -decision-bound 2s \
  -timeout 30s > mesh-scale-churn.json
```

The wall-time limit is a regression guard on a shared CI runner, not a
production latency claim. `searchOperationCount` is the deterministic policy
work bound.

## Scenario and evidence

The proof records:

- three eligible workers joining concurrently, followed by enough unsuitable
  workers to reach the configured membership size;
- deterministic rejection of low-memory, unhealthy, wrong-adapter,
  wrong-checkpoint, and wrong-transport candidates;
- an unchanged valid plan after the unsuitable workers join;
- duplicate stable-identity rejection, lease expiry, and rejoin under a new
  process instance;
- a fresh incompatible capability update and its checkpoint rejection;
- a materially better eligible join changing the next plan, followed by
  removal preventing stale target reuse; and
- fresh slow compute observations moving the next plan off the affected worker.

Each decision records visible and eligible worker counts, inventory/profile
revisions, selected stages, stable per-worker rejection codes, complete-plan and
retained-state counts, counted search operations, and measured decision time.
Transitions preserve the membership revision, visible identities, and outcome.

The current placement path performs no active network fan-out: it consumes
bounded rolling link observations collected before scheduling. Evidence
therefore records a network-probe count and limit of zero, while separately
recording membership HTTP request count and retained link/compute series. A
future active probe service must change that explicit bound rather than hiding
probe work inside placement.

## Five-Mac hybrid gate

The five-Mac workflow supplies the real-model counterpart. A Linux coordinator
runs `swarm-control`; all five `swarmd` processes register their stable worker
IDs, process instances, serving endpoints, capabilities, and fresh state. After
the explicit 12B pooled-memory run establishes successful measured stage costs,
the coordinator waits for clean current membership and adds 27 checkpoint-
incompatible synthetic Linux peers.

The production sequence scheduler consumes that 32-member snapshot, records
every synthetic checkpoint rejection, binds five selected real process
instances over the identity-aware HTTP transport, and generates the same 32
reference tokens through the scheduler-selected plan. The evidence includes
the full placement decision, selected bindings, synthetic rejection codes,
stage/run metrics, and direct post-run sequence/KV cleanup observations.

This closes M8.4 without claiming that synthetic peers execute model math or
that the current JSON/base64 transport is representative of final network
performance.
