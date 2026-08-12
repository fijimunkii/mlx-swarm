# Pooled-memory generation proof

This runbook reproduces the Distributed Inference Proof's central
memory-pooling claim with two
independent 7 GiB Apple-silicon workers. Each worker receives a 6 GiB MLX
scheduling threshold and loads one half of
`mlx-community/gemma-3-12b-it-4bit`. Neither serving process loads the full
checkpoint or a correctness oracle.

## Fixed inputs

| Input | Value |
|---|---|
| Checkpoint | `mlx-community/gemma-3-12b-it-4bit` |
| Adapter model type | `gemma3` |
| Layers | 48, split 0–23 / 24–47 |
| Checkpoint bytes | 8,063,329,713 |
| Per-worker physical memory | 7,516,192,768 bytes (7 GiB) |
| Per-worker MLX scheduling threshold | 6,442,450,944 bytes (6 GiB) |
| MLX allocator cache limit | 67,108,864 bytes (64 MiB) |
| Prompt | `Write a short story about two computers working together:` |
| Required output | 32 greedy tokens |

The checked-in reference at
[`testdata/pooled-memory/gemma-3-12b-it-4bit.json`](../testdata/pooled-memory/gemma-3-12b-it-4bit.json)
pins checkpoint fingerprint
`ec3c7b60c388290a6b8adf28d5f2812b00f0f9fbf5d3c404d755fcdd699518a2`.
An upstream full-model run on a 24 GiB Apple M4 measured a maximum MLX
footprint of 7,416,426,311 bytes and a macOS lifetime process peak of
7,647,434,848 bytes. The process peak and checkpoint byte count each exceed
either serving worker's 7 GiB physical memory. The same reference run proved
exact logit and greedy-token parity before the token plan was recorded.

Standard GitHub-hosted Apple-silicon macOS runners expose 7 GB RAM. The 6 GiB
MLX `memoryLimit` value is an allocator scheduling threshold, not a hard
allocation cap, so the proof does not rely on it to establish impossibility.
It leaves the VM and control-plane processes headroom while the `Pooled Memory
Proof` workflow records actual physical memory, the configured threshold, and
both MLX allocator and macOS process peaks during load/prefill/decode from both
fresh runners in its JSON artifact. A missing process measurement fails closed.

## Prepare both Macs

Each Mac needs enough disk space for the 8.06 GB checkpoint. Build and resolve
the checkpoint before changing DNS or starting the serving process:

```bash
./scripts/build-mlx-worker.sh
worker="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker"
"$worker" checkpoint-info mlx-community/gemma-3-12b-it-4bit
```

`checkpoint-info` downloads, fingerprints, and inventories the checkpoint. It
does not create model modules or load weights into MLX.

Set the same limits on both Macs:

```bash
export MLX_SWARM_MEMORY_THRESHOLD_BYTES=6442450944
export MLX_SWARM_CACHE_LIMIT_BYTES=67108864
export MLX_SWARM_WORKER="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker"
```

On the consumer Mac:

```bash
SWARMD_ADDR=0.0.0.0:8080 go run ./cmd/swarmd
```

On the producer Mac:

```bash
SWARMD_ADDR=127.0.0.1:8080 go run ./cmd/swarmd
```

## Run the proof

From the producer Mac, replace `MAC_B_LAN_IP` with the trusted-network address
of the consumer:

```bash
go run ./cmd/swarm-pooled-memory \
  -producer http://127.0.0.1:8080 \
  -peer http://MAC_B_LAN_IP:8080 \
  -reference testdata/pooled-memory/gemma-3-12b-it-4bit.json \
  -memory-threshold-bytes 6442450944 \
  -minimum-tokens 32 \
  -forward-timeout 2m \
  -timeout 20m \
  > pooled-memory.json

jq -e '.checks.allPassed' pooled-memory.json
```

The command starts no local MLX worker. It requires both remote workers to be
clean, loads exactly one complementary shard on each, generates 32 tokens, and
compares the prompt and generated token IDs with the pinned reference. It
fails unless:

- the two resolved checkpoint fingerprints and byte counts match the reference;
- the full checkpoint and measured full inference footprint exceed both workers' physical memory;
- each serving worker's macOS lifetime process peak stays within its physical memory through load, prefill, and decode;
- the serving workers retain only complementary ranges, never a full range;
- all 32 greedy tokens match the upstream reference; and
- both sequence caches and retained mutation outputs return to zero.

The JSON preserves hardware inventory, shard ownership, phase MLX/process memory,
generation timings, token IDs, teardown state, and every acceptance check.
After capturing that evidence, the command unloads both proof-owned shards on
success or failure so the same clean daemons can run the proof again.

## Regenerate the reference

Reference creation intentionally runs separately on a capable Mac. It first
captures a distributed logit trace from two shard workers, releases them, and
then replays that trace against an independent upstream full-model worker, so
use a Mac with at least 16 GiB unified memory:

```bash
go run ./cmd/swarm-pooled-memory \
  -create-reference \
  -worker "$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker" \
  -model mlx-community/gemma-3-12b-it-4bit \
  -max-tokens 32 \
  -forward-timeout 2m \
  -timeout 30m \
  > regenerated-reference.json
```

Review any fingerprint or token change before replacing the checked-in
manifest. Reference mode requires every distributed logit vector and greedy
token to match the upstream full-model path exactly.

## Trust boundary

The serving HTTP endpoint remains unauthenticated and unencrypted. Run this
proof only on a trusted LAN or private tailnet. The memory proof is a validation
experiment, not a hard multi-tenant resource sandbox.
