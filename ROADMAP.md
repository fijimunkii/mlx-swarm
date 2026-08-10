# Roadmap

## M0 — bootstrap

- [x] establish Go control-plane / Swift worker boundary
- [x] define initial wire messages
- [ ] build `swarmd` and `mlx-worker` in CI
- [ ] add local worker health handshake

## M1 — one machine, one worker

- [ ] load a small supported model through MLX Swift LM
- [ ] expose worker capabilities and memory information
- [ ] run local prefill/decode through the worker API
- [ ] establish deterministic reference outputs

## M2 — two-Mac pipeline

- [ ] load complementary contiguous layer ranges
- [ ] send hidden state from worker A to worker B
- [ ] preserve per-shard KV cache
- [ ] compare final logits with single-node inference
- [ ] measure transfer size, p50/p95 latency, TTFT, and tokens/sec

## M3 — chaos harness

- [ ] inject latency and jitter
- [ ] throttle bandwidth
- [ ] pause/kill worker processes
- [ ] disconnect/reconnect a peer
- [ ] record recovery behavior and failed-token rate

## M4 — fault-tolerant scheduler

- [ ] worker health scoring
- [ ] request deadlines
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

## Later

Public membership, NAT traversal, trust/verification, MoE expert placement, speculative decoding, compute credits, and mobile opportunistic workers are deliberately deferred until the basic execution model has data behind it.
