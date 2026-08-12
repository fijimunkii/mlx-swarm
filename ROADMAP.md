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
- normal generation samples greedily on the output-owning shard and returns one token instead of full-vocabulary logits; verification mode retains full logits for tolerance checks
- benchmark JSON pairs full-logit and token-only runs, requires exact generated-token parity, and records both terminal response sizes and latency/throughput distributions; CI uploads it without enforcing hosted-runner latency thresholds

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

- [x] select a model whose checkpoint and full inference footprint exceed each serving worker's physical memory
- [x] host complementary ranges without materializing a full model in either serving process
- [x] generate 32 reference-matched tokens and record load/prefill/decode memory evidence

Current pooled-memory proof (`mlx-community/gemma-3-12b-it-4bit`):

- the 48 layers split 0–23 / 24–47 across two fresh 7 GiB Apple-silicon workers, each configured with a 6 GiB MLX scheduling threshold and a 64 MiB allocator-cache limit
- the resolved checkpoint contains 8,063,329,713 bytes; a separate upstream full-model run on a 24 GiB M4 measured a 7,416,426,311-byte MLX peak and a 7,647,434,848-byte macOS process peak, so neither the checkpoint nor full-model working set can fit in one serving runner's physical memory
- the checked-in fingerprint, prompt token plan, and 32 greedy output tokens were produced only after every distributed logit vector and token exactly matched the upstream full-model path
- serving proof mode requires clean remote workers, loads exactly one complementary range on each, and does not create a reference worker or full-range shard
- machine-readable evidence records physical memory, configured MLX thresholds, load/prefill/decode MLX and process peaks, shard ownership, generation timings, exact reference parity, and zero retained sequence state after teardown
- paired GitHub-hosted macOS runners reproduce the proof over Tailscale and upload the result; the full-model oracle never runs on either 7 GB serving VM

The independent failure proof in M3 covers bounded worker loss and
next-sequence recovery. Transparent same-sequence recovery remains post-MVP.

## Later

Public membership, NAT traversal, trust/verification, MoE expert placement, speculative decoding, compute credits, and mobile opportunistic workers are deliberately deferred until the basic execution model has data behind it.
