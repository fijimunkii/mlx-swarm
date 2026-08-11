# Upstream checkpoint-shard loader seam

Issue #2 deliberately keeps model math in MLX Swift LM. The local implementation only selects and applies checkpoint parameters; Gemma 3 embedding, transformer blocks, final RMS norm, and language head remain upstream modules.

The generally useful upstream change is a filtered variant of `loadWeights` with this shape:

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

The validation case in this repository exercises the tricky parts: a non-zero Gemma layer range, quantized layers, a tied `lm_head`, final norm/head ownership, two independent processes, final-logit equality, and peak-memory accounting. Once the API is accepted upstream, `Gemma3CheckpointShard.loadStage` can collapse to selection predicates plus retained module references; no checkpoint format or model fork is required.
