# Upstream checkpoint-shard loader seam

Issue #2 deliberately keeps model math in MLX Swift LM. The local implementation is split into three seams:

- `CheckpointShardRuntime` owns model resolution, injected-registry dispatch, stage requests, boundary envelopes, correctness, and memory budgets.
- `CheckpointWeightLoader` owns safetensor discovery, pre/post-sanitizer filtering, and partial `Module` updates.
- model-family adapters own parameter paths, quantization replacement, and execution semantics. `Gemma3CheckpointShardAdapter` is the first validated adapter; its embedding, transformer blocks, final RMS norm, and language head remain upstream modules.
- `WorkerCheckpointShards` composes the runtime by registering adapters and providing the current model, token, tolerance, and budget defaults.

The remaining generally useful upstream change is a filtered variant of `loadWeights` with this shape:

```swift
public func loadWeights(
    modelDirectory: URL,
    model: Module,
    perLayerQuantization: QuantizationConfig?,
    selecting: (String) -> Bool
) throws
```

Its implementation should preserve the existing loader sequence:

1. read only selected safetensor keys from each indexed file;
2. run the model sanitizer so prefix normalization, vocabulary trimming, and tied-weight synthesis stay model-owned;
3. filter again after sanitization;
4. quantize and update only selected leaf modules, including non-zero array ranges without constructing sparse child arrays;
5. evaluate only retained modules so unowned random parameters do not materialize.

The validation case exercises the tricky parts: adapter selection from `model_type`, a non-zero Gemma layer range, quantized layers, a tied `lm_head`, final norm/head ownership, two independent processes, final-logit equality, and peak-memory accounting. Once the API is accepted upstream, `CheckpointWeightLoader` can delegate its file loading and filtering while adapters continue to retain and execute architecture-specific modules; no checkpoint format or model fork is required.
