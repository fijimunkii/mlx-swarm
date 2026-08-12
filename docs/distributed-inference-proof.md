# Distributed Inference Proof validation runbook

This is the authoritative validation procedure for the `mlx-swarm`
Distributed Inference Proof. It
takes two Apple-silicon Macs from clean checkouts to distributed generated text
and produces machine-readable correctness, performance, failure, and
physically pooled-memory evidence.

The procedure has two serving phases:

1. the 270M correctness and benchmark proof; and
2. the 12B pooled-memory proof on fresh daemons.

Do not reuse the first phase's daemons for the pooled-memory proof. The pooled
proof deliberately fails unless both workers start without loaded shards or
open sequences.

## Trust boundary

The HTTP transport is unauthenticated and unencrypted. Use a private LAN or
private tailnet, bind the consumer to its private address, and block port 8080
from the public Internet. Do not run these commands on an untrusted network or
with untrusted clients. The worker API is an experimental proof interface, not
a multi-tenant resource sandbox.

## Supported release configuration

| Item | Validated configuration |
|---|---|
| Serving hardware | Two Apple-silicon Macs running macOS 14 or later |
| Control plane | Go 1.24 or newer |
| MLX worker | Swift 6.3 package built by Xcode |
| Validated model family | Gemma 3 (`model_type`-selected adapter) |
| Correctness model | `mlx-community/gemma-3-270m-it-4bit` |
| Correctness split | 18 layers: 0-8 / 9-17 plus final norm and head |
| Numerical tolerance | `rtol=atol=1e-4` |
| Pooled model | `mlx-community/gemma-3-12b-it-4bit` |
| Pooled split | 48 layers: 0-23 / 24-47 plus final norm and head |
| Pooled checkpoint | 8,063,329,713 bytes; fingerprint `ec3c7b60c388290a6b8adf28d5f2812b00f0f9fbf5d3c404d755fcdd699518a2` |
| Pooled worker limits | 6 GiB MLX scheduling threshold; 64 MiB allocator cache |
| Pooled proof hosts | Two fresh 7 GiB Apple-silicon workers |
| Required pooled output | 32 deterministic greedy tokens |

The 7 GiB physical-memory requirement is what makes the checked-in pooled
claim reproducible: both the 8,063,329,713-byte checkpoint and the separately
measured 7,647,434,848-byte full-model process peak exceed either worker's
7,516,192,768 bytes of RAM. Macs with more memory can run the command, but do
not prove that the full inference footprint cannot fit on either host.

Each Mac needs Xcode with command-line tools, Go 1.24 or newer, `git`, `curl`,
and `jq`, plus enough free disk space for the source dependencies, build
products, and the 8.06 GB checkpoint. Both machines must be able to resolve the
same Hugging Face checkpoint contents before the serving run.

## 1. Check out one exact revision on both Macs

Choose the exact commit under review, not a moving branch name. Run the
following on both Macs, replacing `PROOF_REVISION` with the same full commit SHA:

```bash
export PROOF_REVISION="<40-character candidate commit>"
git clone https://github.com/fijimunkii/mlx-swarm.git
cd mlx-swarm
git checkout --detach "$PROOF_REVISION"
test "$(git rev-parse HEAD)" = "$PROOF_REVISION"
test -z "$(git status --porcelain)"
```

Record the environment on each machine:

```bash
mkdir -p validation-evidence
{
  git rev-parse HEAD
  sw_vers
  uname -m
  sysctl -n hw.memsize
  go version
  xcodebuild -version
  swift --version
  jq --version
} > validation-evidence/environment.txt
```

The commit IDs in the two `environment.txt` files must be identical and both
architectures must be `arm64`.

## 2. Build and prefetch before changing network configuration

Run on both Macs:

```bash
go test ./...
./scripts/build-mlx-worker.sh
export MLX_SWARM_WORKER="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker"
"$MLX_SWARM_WORKER" health
"$MLX_SWARM_WORKER" capabilities | tee validation-evidence/capabilities.json
"$MLX_SWARM_WORKER" checkpoint-info mlx-community/gemma-3-270m-it-4bit \
  | tee validation-evidence/gemma-3-270m-checkpoint.json
"$MLX_SWARM_WORKER" checkpoint-info mlx-community/gemma-3-12b-it-4bit \
  | tee validation-evidence/gemma-3-12b-checkpoint.json
```

Prefetching on both machines ensures later setup cannot silently select
different checkpoint revisions. It also avoids dependency or checkpoint
downloads after a VPN changes the macOS DNS resolver.

## 3. Start the trusted-network workers

Choose the consumer's private LAN or tailnet IPv4 address and make it reachable
from the producer. The examples call it `CONSUMER_IP`.

In a dedicated shell on the consumer Mac:

```bash
cd /path/to/mlx-swarm
export MLX_SWARM_WORKER="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker"
export CONSUMER_IP="<private consumer IPv4 address>"
SWARMD_ADDR="${CONSUMER_IP}:8080" go run ./cmd/swarmd
```

In a dedicated shell on the producer Mac:

```bash
cd /path/to/mlx-swarm
export MLX_SWARM_WORKER="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker"
SWARMD_ADDR=127.0.0.1:8080 go run ./cmd/swarmd
```

From a second producer shell, set the same consumer address and wait for both
daemons:

```bash
cd /path/to/mlx-swarm
export MLX_SWARM_WORKER="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker"
export CONSUMER_IP="<private consumer IPv4 address>"
curl -fsS http://127.0.0.1:8080/healthz
curl -fsS "http://${CONSUMER_IP}:8080/healthz"
```

Both responses must report `"status":"ok"`.

## 4. Generate distributed text

From the producer's second shell:

```bash
go run ./cmd/swarm-generate \
  -worker "$MLX_SWARM_WORKER" \
  -producer http://127.0.0.1:8080 \
  -peer "http://${CONSUMER_IP}:8080" \
  -model mlx-community/gemma-3-270m-it-4bit \
  -prompt "Write a short story about two computers working together:" \
  -max-tokens 32 \
  > validation-evidence/generation.json

jq -e '
  (.model == "mlx-community/gemma-3-270m-it-4bit") and
  (.modelType == "gemma3_text") and
  (.shardPlan.producer.layerStart == 0) and
  (.shardPlan.producer.layerEnd == 9) and
  (.shardPlan.consumer.layerStart == 9) and
  (.shardPlan.consumer.layerEnd == 18) and
  ((.generatedTokenIDs | length) > 0) and
  (.failure == null)
' validation-evidence/generation.json
```

Normal generation asks the terminal shard to sample greedily and returns only
the token ID instead of a full-vocabulary logit tensor.

## 5. Prove correctness and retained-session behavior

The deterministic generation smoke uses the remote consumer plus local
producer/reference workers, generates 32 tokens, checks each greedy token
against full-checkpoint cached inference, repeats a request without reloading
the retained shards, and requires all sequence state to be released:

```bash
go run ./cmd/swarm-generate-smoke \
  -worker "$MLX_SWARM_WORKER" \
  -peer "http://${CONSUMER_IP}:8080" \
  > validation-evidence/generation-proof.json

jq -e '
  .greedyTokenIDsMatch and
  (.generatedTokenCount >= 32) and
  .repeatedRequestValidated and
  .sequenceTeardownValidated
' validation-evidence/generation-proof.json
```

For the stronger full-logit and cache-position proof:

```bash
go run ./cmd/swarm-cache-smoke \
  -worker "$MLX_SWARM_WORKER" \
  -peer "http://${CONSUMER_IP}:8080" \
  > validation-evidence/cache-proof.json

jq -e '
  .allFinalLogitsMatch and
  .positionsValidated and
  .sequenceIsolationValidated and
  (.producerAfterTeardownBytes == 0) and
  (.consumerAfterTeardownBytes == 0) and
  (.referenceAfterTeardownBytes == 0)
' validation-evidence/cache-proof.json
```

The comparison uses `rtol=atol=1e-4`. A successful command with a false
predicate is not a valid proof; the `jq -e` checks are part of the procedure.

## 6. Record warm performance evidence

Describe the actual producer hardware and route rather than leaving them
implicit:

```bash
hardware="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
route="private LAN or tailnet description"
go run ./cmd/swarm-benchmark \
  -worker "$MLX_SWARM_WORKER" \
  -peer "http://${CONSUMER_IP}:8080" \
  -hardware "$hardware" \
  -route "$route" \
  -prefill-samples 5 \
  -decode-samples 100 \
  > validation-evidence/distributed-benchmark.json

jq -e '
  (.schemaVersion == "2") and
  (.configuration.distributed == true) and
  (.prefill.sampleCount >= 5) and
  (.decode.sampleCount >= 100) and
  .verification.allFinalLogitsMatch and
  .verification.greedyTokenIDsMatch and
  (.tokenOnly.decode.sampleCount >= 100) and
  .tokenOnly.generatedTokenIDsMatch and
  (.tokenOnly.decode.consumerResponseTensorBytes.maxBytes == 0) and
  (.tokenOnly.decode.consumerResponseWireBytes.p50Bytes <
    .decode.consumerResponseWireBytes.p50Bytes) and
  (.memory.producer.postRunKVCacheBytes == 0) and
  (.memory.consumer.postRunKVCacheBytes == 0) and
  (.memory.reference.postRunKVCacheBytes == 0)
' validation-evidence/distributed-benchmark.json
```

The JSON includes raw samples and nearest-rank p50/p95 summaries for setup,
prefill TTFT, decode, serialization, transport, end-to-end token latency,
throughput, boundary bytes, terminal-response bytes, and memory. Hosted or
residential latency is observational; correctness and payload reduction
are gates, but a particular latency threshold is not. Terminal sampling is not
needed to establish the central pooled-memory claim; it is nevertheless the
integrated default serving path, so exact token parity and removal of the
full-logit response are release regression checks.

## 7. Record bounded failure behavior

The failure harness is model-free and runs on either Mac. It injects pause,
kill, delay, jitter, and loopback HTTP disconnect behavior:

```bash
go run ./cmd/swarm-failure-smoke > validation-evidence/failure-characterization.json

jq -e '
  (.schemaVersion == "1") and
  .allTerminatedBounded and
  .allCleanupReleased and
  .allRecoveryReady and
  (([.scenarios[].name] | sort) == ["delay", "disconnect", "jitter", "kill", "pause"])
' validation-evidence/failure-characterization.json
```

The expected behavior is bounded failure of the active sequence followed by
worker replacement and readiness for a new sequence. Transparent continuation
of the failed sequence is deliberately not claimed.

## 8. Restart into a clean pooled-memory proof

Stop both `swarmd` processes with Control-C and wait for their Swift children to
exit. Then start fresh daemons with the pooled limits.

On the consumer Mac:

```bash
export MLX_SWARM_MEMORY_THRESHOLD_BYTES=6442450944
export MLX_SWARM_CACHE_LIMIT_BYTES=67108864
MLX_SWARM_WORKER="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker" \
SWARMD_ADDR="${CONSUMER_IP}:8080" \
go run ./cmd/swarmd
```

On the producer Mac:

```bash
export MLX_SWARM_MEMORY_THRESHOLD_BYTES=6442450944
export MLX_SWARM_CACHE_LIMIT_BYTES=67108864
MLX_SWARM_WORKER="$PWD/worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker" \
SWARMD_ADDR=127.0.0.1:8080 \
go run ./cmd/swarmd
```

After both `/healthz` endpoints respond, run from the producer's second shell:

```bash
go run ./cmd/swarm-pooled-memory \
  -producer http://127.0.0.1:8080 \
  -peer "http://${CONSUMER_IP}:8080" \
  -reference testdata/pooled-memory/gemma-3-12b-it-4bit.json \
  -memory-threshold-bytes 6442450944 \
  -minimum-tokens 32 \
  -forward-timeout 2m \
  -timeout 30m \
  > validation-evidence/pooled-memory.json

jq -e '
  (.schemaVersion == 1) and
  .checks.allPassed and
  .checks.cleanWorkersAtStart and
  .checks.checkpointMatchesReference and
  .checks.checkpointExceedsProducerPhysicalMemory and
  .checks.checkpointExceedsConsumerPhysicalMemory and
  .checks.fullInferenceExceedsProducerPhysicalMemory and
  .checks.fullInferenceExceedsConsumerPhysicalMemory and
  .checks.producerProcessWithinPhysicalMemory and
  .checks.consumerProcessWithinPhysicalMemory and
  .checks.complementaryShardsOnly and
  .checks.noServingFullModelOracle and
  .checks.generatedTokensMatchReference and
  .checks.sequenceStateReleased and
  ((.generation.generatedTokenIDs | length) >= 32)
' validation-evidence/pooled-memory.json
```

This check passes only on hosts whose reported physical memory establishes the
claim. The checked-in reference was created separately on a capable Mac and
contains the full-model memory evidence and exact token plan; neither serving
worker loads that oracle.

## Automated clean-machine reproduction

The [`CI`](../.github/workflows/ci.yml),
[`MLX Worker`](../.github/workflows/mlx.yml), and
[`Pooled Memory Proof`](../.github/workflows/pooled-memory.yml) workflows mirror
the gates above on fresh GitHub-hosted macOS workers and upload the benchmark,
failure, and pooled-memory JSON artifacts. Changes to this runbook trigger both
paired-macOS workflows so its exact commit cannot be approved using only older
model evidence. Repository-backed build caches reduce compilation time, but
each job receives a clean checkout and the serving proofs require fresh worker
process state.

## 9. Validation evidence checklist

The proof is validated only when all of the following refer to the same
commit:

- both clean checkouts recorded the candidate SHA and `arm64` environment;
- the 270M generation, generation smoke, and cache proof passed;
- the benchmark contains five prefills, 100 full-logit decode samples, and 100
  token-only decode samples with exact token parity and zero retained KV state;
- every failure scenario terminated within its bound, released state, and
  admitted a new sequence;
- the fresh 12B serving proof reports `checks.allPassed`, 32 matching tokens,
  complementary shards, no serving oracle, and process peaks below both
  workers' physical memory; and
- the `CI`, `MLX Worker`, and `Pooled Memory Proof` workflows are green on the
  fully integrated proof pull request.

Archive the two `environment.txt` files and the producer's
`validation-evidence/*.json` files together. These files are the auditable record of
the run; prose observations alone are not release evidence.

## Deliberate proof limitations

- Only the Gemma 3 adapter is validated; architecture-neutral orchestration is
  not a claim of arbitrary checkpoint compatibility.
- Shards are contiguous and manually split; there is no automatic placement,
  health scoring, replica selection, or hedged execution.
- The transport uses JSON/base64 HTTP bodies, not the intended binary tensor
  framing, and is limited to trusted private networks.
- There is no authentication, encryption, malicious-worker verification,
  public membership, stable public API, or multi-tenant isolation.
- A timed-out MLX mutation kills and replaces the worker. The active sequence
  fails; only a later sequence recovers.
- Latency results characterize the measured hardware and route. The proof makes no
  universal interactive-latency guarantee.
- CUDA/RTX workers, WAN-aware placement, batching, speculative decoding,
  bandwidth throttling, and same-sequence KV recovery are future work.

The lower-level pooled proof and wire contracts are documented in
[`pooled-memory-proof.md`](pooled-memory-proof.md) and
[`persistent-worker-protocol.md`](persistent-worker-protocol.md).
