# Pooled-memory generation proof

This runbook reproduces the MVP's central memory-pooling claim with two
independent Apple-silicon workers. Each worker receives a 6 GiB MLX memory
limit and loads one half of
`mlx-community/gemma-3-text-12b-it-4bit`. Neither serving process loads the
full checkpoint or a correctness oracle.

## Fixed inputs

| Input | Value |
|---|---|
| Checkpoint | `mlx-community/gemma-3-text-12b-it-4bit` |
| Adapter model type | `gemma3` |
| Layers | 48, split 0–23 / 24–47 |
| Checkpoint bytes | 7,220,708,353 |
| Per-worker MLX limit | 6,442,450,944 bytes (6 GiB) |
| MLX allocator cache limit | 67,108,864 bytes (64 MiB) |
| Prompt | `Write a short story about two computers working together:` |
| Required output | 32 greedy tokens |

The checked-in reference at
[`testdata/pooled-memory/gemma-3-text-12b-it-4bit.json`](../testdata/pooled-memory/gemma-3-text-12b-it-4bit.json)
pins checkpoint fingerprint
`768a1be0202b088d531630c6b7ff26fad1988e1fb1e402c625de7b71dd2904b0`.
An upstream full-model run on a 24 GiB Apple M4 measured a maximum MLX
footprint of 7,416,404,295 bytes. That is larger than the serving workers'
6 GiB usable MLX budgets. The same run proved exact logit and greedy-token
parity before the token plan was recorded.

Standard GitHub-hosted Apple-silicon macOS runners expose 7 GB RAM. The 6 GiB
limit leaves the VM and control-plane processes headroom instead of treating
all physical RAM as available to MLX. The `Pooled Memory Proof` workflow
records the actual physical memory, configured MLX limit, and load/prefill/
decode peaks from both fresh runners in its JSON artifact.

## Prepare both Macs

Each Mac needs enough disk space for the 7.22 GB checkpoint. Build and resolve
the checkpoint before changing DNS or starting the serving process:

```bash
./scripts/build-mlx-worker.sh
worker="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker"
"$worker" checkpoint-info mlx-community/gemma-3-text-12b-it-4bit
```

`checkpoint-info` downloads, fingerprints, and inventories the checkpoint. It
does not create model modules or load weights into MLX.

Set the same limits on both Macs:

```bash
export MLX_SWARM_MEMORY_LIMIT_BYTES=6442450944
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
  -reference testdata/pooled-memory/gemma-3-text-12b-it-4bit.json \
  -memory-limit-bytes 6442450944 \
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
- the full checkpoint and measured full inference footprint exceed both 6 GiB limits;
- each serving worker stays below its limit during load, prefill, and decode;
- the serving workers retain only complementary ranges, never a full range;
- all 32 greedy tokens match the upstream reference; and
- both sequence caches and retained mutation outputs return to zero.

The JSON preserves hardware inventory, shard ownership, phase memory,
generation timings, token IDs, teardown state, and every acceptance check.

## Regenerate the reference

Reference creation intentionally runs separately on a capable Mac. It starts
two shard workers plus an independent upstream full-model worker, so use a Mac
with at least 24 GiB unified memory:

```bash
go run ./cmd/swarm-pooled-memory \
  -create-reference \
  -worker "$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker" \
  -model mlx-community/gemma-3-text-12b-it-4bit \
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
proof only on a trusted LAN or private tailnet. The memory proof is an MVP
experiment, not a hard multi-tenant resource sandbox.
