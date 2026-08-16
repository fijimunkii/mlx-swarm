# Trusted mesh membership

The first Dynamic Trusted Mesh control-plane primitive is a leased inventory of
workers on a private LAN or tailnet. `swarm-control` owns the inventory;
participating `swarmd` processes register themselves and refresh their live
state through heartbeats. This is trusted-network discovery, not public peer
discovery or an authentication boundary.

## Run the control plane

On the coordinator:

```bash
SWARM_CONTROL_ADDR=0.0.0.0:9090 \
SWARM_CONTROL_LEASE_TTL=30s \
go run ./cmd/swarm-control
```

On each worker, keep the existing private `swarmd` endpoint and opt into
membership:

```bash
MLX_SWARM_WORKER="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker" \
SWARMD_ADDR=0.0.0.0:8080 \
SWARMD_CONTROL_URL=http://COORDINATOR_TAILNET_IP:9090 \
SWARMD_WORKER_ID=mac-a \
SWARMD_PUBLIC_URL=http://WORKER_TAILNET_IP:8080 \
go run ./cmd/swarmd
```

`SWARMD_WORKER_ID` is stable across restarts. `swarmd` generates a fresh
process `instanceID` unless `SWARMD_INSTANCE_ID` is supplied. The default
heartbeat interval is 10 seconds and can be changed with
`SWARMD_HEARTBEAT_INTERVAL`; it must be shorter than the controller lease TTL.
`SWARMD_BACKEND` defaults to `mlx` but remains a free backend identifier for
future Swift, Linux CPU, CUDA, and synthetic workers.

## Inventory contract

Every worker record uses membership schema version 1 and contains:

- stable worker and process-instance identities plus the serving endpoint;
- backend, runtime, OS, architecture, device, and physical memory;
- supported model adapters, worker operations, checkpoint fingerprints, and
  transports;
- concurrency, open-sequence, and retained-byte admission limits;
- health, schedulable memory available under the backend's configured runtime
  limit, process pressure, restart/failure counters, and
  open sequence state; and
- retained shard ranges, ownership, checkpoint identities, memory, and open
  sequence counts.

The controller supplies `registeredAt`, `lastSeen`, and `expiresAt`. Inventory
snapshots are sorted by worker ID. Their monotonic revision changes for joins,
removals, expiry, capability changes, and live status changes, but not for an
otherwise identical lease renewal. This gives later placement decisions a
stable input revision without turning every heartbeat into a new shard plan.

## HTTP API

| Method | Path | Effect |
|---|---|---|
| `GET` | `/healthz` | Reports controller health, inventory revision, and active worker count. |
| `GET` | `/v1/membership` | Returns the current versioned inventory and expires overdue leases before responding. |
| `POST` | `/v1/membership/workers` | Registers or refreshes one complete worker record. |
| `POST` | `/v1/membership/workers/{id}/heartbeat` | Refreshes dynamic state for the active process instance. |
| `DELETE` | `/v1/membership/workers/{id}?instanceID=...` | Explicitly removes the caller-owned lease. |

An active worker ID cannot be taken over by a different `instanceID`. The
controller returns machine-readable `duplicate_worker`, `duplicate_instance`,
`duplicate_endpoint`, `stale_instance`, `lease_expired`, and `worker_not_found`
errors. A new process may claim the
stable ID after the old lease expires or is explicitly removed. If the
controller restarts, healthy workers see the missing lease and register again
on their next heartbeat.

## Current boundary

The inventory is in memory and intentionally has no public authentication,
durable consensus, or malicious-worker verification. Bind both controller and
workers only to a trusted LAN or private tailnet. Automatic topology profiling
and shard placement consume this record in the next #35 slices; the existing
explicit-plan path remains unchanged.
