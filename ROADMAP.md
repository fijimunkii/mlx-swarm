# Roadmap

## M0 — bootstrap

- [x] establish Go control-plane / Swift worker boundary
- [x] define initial wire messages
- [x] build `swarmd` and `mlx-worker` in CI
- [x] add local worker health/capability handshake

## M1 — one machine, one worker

- [x] load a small supported model through MLX Swift LM
- [x] expose worker capabilities and memory information
- [x] run local prefill/decode through the worker command API
- [x] establish deterministic reference outputs

## M2 — two-Mac pipeline

- [x] load complementary contiguous layer ranges from a real checkpoint
- [x] retain assigned shards in supervised workers across repeated forwards
- [x] send hidden state from worker A to worker B
- [x] preserve per-shard KV cache
- [x] compare final logits with single-node inference at an explicit tolerance
- [x] tokenize prompts and generate user-visible text with cached greedy decode
- [x] measure p50/p95 latency, TTFT, and tokens/sec with warm repeated samples

Current correctness proof (`mlx-community/gemma-3-270m-it-4bit`):

- layers 0–8 run on the producer; layers 9–17 plus final norm/head run on the consumer
- the `bfloat16` boundary is 7,680 tensor bytes; output logits are `[1, 6, 262144]`
- distributed logits match full-checkpoint inference at `rtol=atol=1e-4`
- under a configured 128 MiB worker budget, measured peaks are about 119 MiB and 121 MiB while full-checkpoint inference peaks near 150 MiB
- paired macOS CI runners exercise the boundary over Tailscale and require both correctness and memory proofs
- checkpoint orchestration and filtered loading are architecture-neutral; `model_type` selects a registered adapter, with Gemma 3 as the first validated family
- `swarmd` supervises a long-lived worker; CI loads one real shard once, reuses it for 100 forwards across two sequence IDs, then proves unload, shutdown, and crash behavior
- each shard owns adapter-defined KV caches keyed by shard and sequence; CI prefills two interleaved prompts once and reuses both shard caches for 32 decode steps per sequence
- every distributed cached step matches the upstream full-checkpoint cached path at `rtol=atol=1e-4`; stale, skipped, conflicting, unknown, and closed sequence positions fail deterministically
- KV memory is reported separately from resident weights and allocator cache; sequence teardown returns KV accounting to zero while the shard remains loaded
- cached mutations are retry-safe by exact replay, and adapter-provided context/cache estimates enforce retained-state and open-sequence admission limits before inference
- the Go generation session discovers the model's layer count, caches complementary shard assignments, tokenizes through the checkpoint tokenizer, and stops greedy decode on EOS or a configured maximum
- generation requires identical resolved-checkpoint fingerprints across producer, consumer, and reference workers; private sequence owners make ambiguous cleanup safe under caller-supplied ID collisions
- local and paired-macOS proofs generate 32 tokens whose complete greedy token sequence matches a cached full-checkpoint reference; a second request reuses the loaded shards and both requests release their sequence caches
- generation and smoke proofs share bounded worker cleanup and rollback-safe multi-shard sequence orchestration; smoke proofs also share request assertions and accumulated final-logit comparison metrics so later experiments add only proof-specific behavior
- the paired-macOS benchmark warms already-loaded shards, records five fresh prefills and 100 cached decode steps, and reports p50/p95 producer, serialization, transport, consumer, end-to-end, TTFT, inter-token, throughput, transfer, and memory metrics alongside the same-token cached full-model baseline
- benchmark JSON includes its model fingerprint, prompt and token plan, split, hardware, Tailscale route, tolerances, aggregate distributions, and raw samples; CI uploads it without enforcing hosted-runner performance thresholds

## M3 — chaos harness

- [x] inject latency and jitter
- [ ] throttle bandwidth
- [x] pause/kill worker processes
- [x] disconnect/reconnect a peer
- [x] record recovery behavior and failed-token rate

Current failure proof:

- every `forward`, `prefill`, and `decode` carries an absolute deadline; the Go coordinator applies a fresh per-call timeout and the Swift worker rejects missing or expired deadlines before and after inference
- a timed-out non-preemptible MLX call kills its worker instead of preserving ambiguous KV state; `swarmd` starts a clean worker without replaying the failed mutation, and a fresh session reloads its shards
- the deterministic, model-free harness injects process pause, process kill, long delay, bounded jitter, and a loopback HTTP disconnect without using the public network
- pause, kill, delay, and disconnect fail the active sequence with its shard, phase, position, and last accepted token; every scenario proves bounded termination, released sequence state, worker reuse for a new sequence, and a machine-readable failed-token rate in CI

## M4 — fault-tolerant scheduler

- [ ] worker health scoring
- [x] request deadlines
- [ ] replicas
- [ ] delayed hedged execution
- [ ] first-valid-result selection
- [ ] KV recovery strategy

## M5 — residential WAN

- [ ] connect the two Macs to remote consumer GPU workers
- [ ] validate MLX CUDA worker path on RTX hardware
- [ ] topology/RTT-aware placement
- [ ] compare LAN vs. WAN decode behavior

## M6 — pooled-memory demo

- [ ] select a model that cannot fit on either Mac individually
- [ ] host it across multiple workers
- [ ] demonstrate generation and controlled worker failure

The M2 checkpoint proof establishes the loader and budget accounting needed for M6, but does not claim that the 270M validation model exceeds either Mac's physical memory.

## Later

Public membership, NAT traversal, trust/verification, MoE expert placement, speculative decoding, compute credits, and mobile opportunistic workers are deliberately deferred until the basic execution model has data behind it.
