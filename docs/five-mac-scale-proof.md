# Five-Mac scale proof

This proof is the M8.2 bridge from arbitrary local N-stage execution to a real
multi-host mesh. One Linux coordinator drives five independent Apple-silicon
MLX workers over a private tailnet. Linux performs orchestration only; every
model stage runs on macOS.

The proof is implemented by
[`cmd/swarm-scale`](../cmd/swarm-scale/main.go), with reusable orchestration and
evidence types in [`internal/scaleproof`](../internal/scaleproof). The canonical
run is [`.github/workflows/five-mac-scale.yml`](../.github/workflows/five-mac-scale.yml).

## Topology and trust boundary

The workflow starts six separately identified jobs:

```text
Linux coordinator
  -> mac-0: input owner + early layers
  -> mac-1: middle layers
  -> mac-2: middle layers
  -> mac-3: middle layers
  -> mac-4: late layers + output owner/sampler
```

`swarmd` remains a trusted-network API. The workflow binds each daemon to its
Tailscale address, does not expose it publicly, and records the coordinator ID,
five stable node IDs, endpoints, probe time, device/runtime capabilities,
physical memory, and MLX memory limits in the result artifact.

## Three gates in one run

The coordinator requires clean workers before loading any shard and runs these
gates sequentially on the same five persistent worker processes:

1. **Independent correctness.** Gemma 3 270M runs through five tensor-returning
   stages for 32 forced tokens. The coordinator captures each final-logit
   tensor, unloads the serving stages, and temporarily loads a full checkpoint
   on `mac-0` to compare every logit and greedy token at `rtol=atol=1e-4`. The
   oracle is unloaded before the scaling and pooled-memory gates.
2. **Scaling curve.** The verified prompt and token plan runs through explicit
   2-, 3-, 4-, and 5-stage terminal-sampling plans. Every run must reproduce the
   verified token IDs. The artifact records shard load time and memory plus
   prefill, decode, and combined per-stage latency, throughput, tensor bytes,
   wire bytes, serialization, KV, and process-memory evidence. Stage count and
   network-boundary count make the serial failure surface explicit.
3. **Pooled memory.** The checked-in Gemma 3 12B reference runs across all five
   Macs without a serving full-model oracle. The explicit layout reserves
   capacity on the embedding/input and norm/output owners, then balances the 48
   transformer layers across the five nodes. Prompt IDs and all 32 generated
   IDs must match the reference. Each process peak must remain within that
   node's physical memory and each node must report the configured MLX limit.

The 12B split is evidence for one pinned checkpoint and worker inventory. It is
not automatic placement policy; capability-driven placement belongs to M8.3.

## Reading the artifact

The uploaded `mlx-swarm-five-mac-scale-<run>-<attempt>` artifact contains one
JSON document. The top-level `checks.allPassed` gate is true only when:

- all five node identities and endpoints are distinct;
- the workers are clean at the beginning and after final shard teardown;
- every small-model logit and greedy token matches the independent oracle;
- the 2/3/4/5 runs all reproduce the verified tokens;
- the 12B prompt and generated tokens match the checked-in reference;
- all five 12B stages are complementary and no stage is a full-model oracle;
- all five process peaks stay within physical memory; and
- every sequence releases its KV and retained state.

For the scaling curve, `prefill.tokenLatencyMicros` is TTFT,
`decode.tokenLatencyMicros` is inter-token latency, `all.tokensPerSecond` is
single-sequence throughput, and `all.stages` contains the stage-local compute,
transport, wire, KV, and memory distributions. More serial stages are expected
to increase capacity and reduce per-worker model memory; they are not expected
to improve single-sequence latency automatically.

## Bounded completion

The workflow builds the Swift 6.3 MLX worker once in a short macOS 26 job and
passes the runtime artifact to all five stable macOS 15 serving jobs. This keeps
the long-lived inference workers on the reliable runner image and avoids five
duplicate native builds.

Every model request has a deadline, the coordinator proof has a 45-minute
limit, every macOS daemon has a 50-minute watchdog, and all six jobs have a
60-minute job timeout. The coordinator posts the workflow-only completion
signal to every discovered worker on both success and failure. Proof-owned
shards are unloaded between gates and again in a bounded final cleanup.

The existing local N-stage, two-Mac correctness, failure, and pooled-memory
workflows remain independent regression paths. This proof adds real five-host
evidence; it does not replace those smaller failure-isolation gates.
