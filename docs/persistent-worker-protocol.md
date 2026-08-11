# Persistent worker protocol

`swarmd` owns one long-lived MLX worker process. The Go daemon starts
`MLXWorker serve-stdio`, checks it with `health`, and shuts it down when the
daemon exits. Swift owns checkpoint loading and MLX execution; Go owns process
lifetime, request cancellation, networking, and error translation.

The current v0 transport is newline-delimited JSON over the child's standard
input and output. Each line is one complete UTF-8 JSON object. Tensor data is
the base64 JSON encoding of contiguous bytes. This framing is intentionally
simple while `proto/swarm.proto` records the language-neutral message shape for
a future binary transport.

## Request envelope

Every request contains:

```json
{"requestID":"request-1","command":"health"}
```

`requestID` must be non-empty and is copied to the response. Go generates it
when a caller omits it and uses it to correlate responses. Commands are handled
serially by the worker so MLX state is mutated in a deterministic order.

The supported commands are:

| Command | Payload | Effect |
|---|---|---|
| `health` | none | Proves the child is responsive. |
| `loadShard` | `loadShard` | Resolves `modelID`, selects its registered `model_type` adapter, loads the requested layer range, and retains it under `shardID`. |
| `unloadShard` | `shard` | Releases a shard after all of its sequences are closed and clears the MLX cache. |
| `openSequence` | `sequence` | Registers `sequenceID` under one loaded `shardID`. |
| `closeSequence` | `sequence` | Removes that sequence registration and its future shard-local state. |
| `forward` | `forward` | Runs the retained stage from token IDs or an upstream hidden tensor. |
| `state` | none | Reports loaded ranges, adapter/model type, sequence and reuse counts, and MLX memory. |
| `shutdown` | none | Releases all shards, clears the MLX cache, acknowledges shutdown, and exits cleanly. Only the supervising daemon sends it. |

Layer ranges use `[layerStart, layerEnd)`. `ownsInput` declares that the stage
owns the model input embedding; `ownsOutput` declares that it owns the final
normalization and output head. `inputKind` is `tokens` for an input-owning stage
or `hidden` for an intermediate stage. A forward also carries `shardID`,
`sequenceID`, and `position`; the worker rejects an unknown shard or a sequence
that was not opened on that shard.

## Response and errors

A success has `ok: true` and a command-specific `result`:

```json
{"requestID":"request-1","ok":true,"result":{"status":"ok"}}
```

A validation or execution failure has `ok: false` and `error`. It does not stop
the worker, so later frames can continue. Through the trusted-network HTTP
proxy at `POST /v1/worker/request`, worker failures use HTTP 422 and worker
process/transport failures use HTTP 502. `GET /v1/worker/state` exposes the
same state snapshot, and `/healthz` is healthy only while the child answers.

Unexpected process exit, malformed output, and EOF are reported to pending Go
callers as bounded errors that include captured worker stderr when present. A
zero exit is considered clean only after the worker acknowledges `shutdown`.

Canceling or timing out a Go request stops that caller's wait and discards any
late response while leaving the supervised worker available. MLX kernels are
not forcibly preempted in v0; an invocation that is already executing may
finish before the serial worker handles the next frame.

## Security and limits

The HTTP proxy is currently unauthenticated and unencrypted. Bind it only to a
loopback or trusted LAN/tailnet address. `swarmd` limits request bodies to 64
MiB, and the Go clients limit responses to 128 MiB. Public membership,
authentication, and transport encryption are outside the MVP.
