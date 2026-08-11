import Foundation
import MLX
@_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon
import MLXNN

private struct Gemma3ResourceConfiguration: Decodable {
    let headDim: Int
    let kvHeads: Int
    let maximumSequenceLength: Int

    private enum CodingKeys: String, CodingKey {
        case headDim = "head_dim"
        case kvHeads = "num_key_value_heads"
        case maximumSequenceLength = "max_position_embeddings"
    }

    private enum RootKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: RootKeys.self)
        let values = if root.contains(.textConfig) {
            try root.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
        } else {
            try decoder.container(keyedBy: CodingKeys.self)
        }
        headDim = try values.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        kvHeads = try values.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        maximumSequenceLength =
            try values.decodeIfPresent(Int.self, forKey: .maximumSequenceLength) ?? 32_768
    }

    func conservativeCacheBytesPerLayerPosition() throws -> Int {
        guard headDim > 0, kvHeads > 0, maximumSequenceLength > 0 else {
            throw CheckpointShardError.invalidBoundary("Gemma cache geometry must be positive")
        }
        // Two tensors (key/value), conservatively budgeted at Float32 width.
        let (headBytes, headOverflow) = headDim.multipliedReportingOverflow(by: 4)
        let (headsBytes, headsOverflow) = headBytes.multipliedReportingOverflow(by: kvHeads)
        let (total, totalOverflow) = headsBytes.multipliedReportingOverflow(by: 2)
        guard !headOverflow, !headsOverflow, !totalOverflow else {
            throw CheckpointShardError.invalidBoundary("Gemma cache geometry overflows Int")
        }
        return total
    }
}

private final class Gemma3CheckpointSequenceCache: CheckpointShardSequenceCache {
    let layers: [KVCache]
    let bytesPerLayerPosition: Int

    init(layers: [KVCache], bytesPerLayerPosition: Int) {
        self.layers = layers
        self.bytesPerLayerPosition = bytesPerLayerPosition
    }

    var position: Int {
        layers.first?.offset ?? 0
    }

    var memoryBytes: Int {
        layers.flatMap(\.state).reduce(0) { $0 + $1.nbytes }
    }

    func estimatedMemoryBytes(at position: Int) throws -> Int {
        guard position >= 0 else {
            throw CheckpointShardError.invalidSequenceCache("negative estimated position")
        }
        var total = 0
        for layer in layers {
            // RotatingKVCache retains an oversized first multi-token write in
            // full and only applies its window when a later update trims it.
            let retainedPositions = layer.state.isEmpty
                ? position
                : min(position, layer.maxSize ?? position)
            let (layerBytes, layerOverflow) = retainedPositions.multipliedReportingOverflow(
                by: bytesPerLayerPosition
            )
            let (nextTotal, totalOverflow) = total.addingReportingOverflow(layerBytes)
            guard !layerOverflow, !totalOverflow else {
                throw CheckpointShardError.invalidSequenceCache(
                    "estimated cache memory overflows Int"
                )
            }
            total = nextTotal
        }
        return max(total, memoryBytes)
    }
}

private final class Gemma3CheckpointStage: CheckpointShardStage {
    let embedding: Embedding?
    let layers: [Gemma3TransformerBlock]
    let cachePrototypes: [KVCache]
    let finalNorm: (any UnaryLayer)?
    let lmHead: (any UnaryLayer)?
    let referenceModel: Gemma3TextModel?
    let hiddenSize: Int
    let vocabularySize: Int
    let maximumSequenceLength: Int
    let cacheBytesPerLayerPosition: Int
    let ownsOutput: Bool
    let weightKeyCount: Int

    init(
        embedding: Embedding?,
        layers: [Gemma3TransformerBlock],
        cachePrototypes: [KVCache],
        finalNorm: (any UnaryLayer)?,
        lmHead: (any UnaryLayer)?,
        referenceModel: Gemma3TextModel? = nil,
        hiddenSize: Int,
        vocabularySize: Int,
        maximumSequenceLength: Int,
        cacheBytesPerLayerPosition: Int,
        ownsOutput: Bool,
        weightKeyCount: Int
    ) {
        self.embedding = embedding
        self.layers = layers
        self.cachePrototypes = cachePrototypes
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.referenceModel = referenceModel
        self.hiddenSize = hiddenSize
        self.vocabularySize = vocabularySize
        self.maximumSequenceLength = maximumSequenceLength
        self.cacheBytesPerLayerPosition = cacheBytesPerLayerPosition
        self.ownsOutput = ownsOutput
        self.weightKeyCount = weightKeyCount
    }

    var inputMetadata: CheckpointStageInputMetadata {
        CheckpointStageInputMetadata(
            tokenDType: .int32,
            tokenRank: 2,
            vocabularySize: vocabularySize,
            hiddenDTypes: [.bfloat16],
            hiddenRank: 3,
            hiddenSize: hiddenSize,
            maximumSequenceLength: maximumSequenceLength
        )
    }

    func makeSequenceCache() -> any CheckpointShardSequenceCache {
        if let referenceModel {
            return Gemma3CheckpointSequenceCache(
                layers: referenceModel.newCache(),
                bytesPerLayerPosition: cacheBytesPerLayerPosition
            )
        }
        let caches = cachePrototypes.map { $0.copy() }
        return Gemma3CheckpointSequenceCache(
            layers: caches,
            bytesPerLayerPosition: cacheBytesPerLayerPosition
        )
    }

    func estimatedOutputBytes(inputLength: Int) throws -> Int {
        guard inputLength > 0 else {
            throw CheckpointShardError.invalidBoundary("output length must be positive")
        }
        let width = ownsOutput ? vocabularySize : hiddenSize
        let (elements, elementOverflow) = inputLength.multipliedReportingOverflow(by: width)
        // Conservatively budget the retained result at Float32 width.
        let (bytes, byteOverflow) = elements.multipliedReportingOverflow(by: 4)
        guard !elementOverflow, !byteOverflow else {
            throw CheckpointShardError.invalidBoundary("estimated output memory overflows Int")
        }
        return bytes
    }

    func forward(tokens: MLXArray) throws -> MLXArray {
        if let referenceModel {
            let output = referenceModel(tokens)
            eval(output)
            return output
        }
        guard let embedding else {
            throw CheckpointShardError.missingInputModule("model.embed_tokens")
        }
        let embedded = embedding(tokens)
        let scale = MLXArray(sqrt(Float(hiddenSize)), dtype: .bfloat16)
        return try forward(hidden: embedded * scale.asType(embedded.dtype))
    }

    func forward(hidden input: MLXArray) throws -> MLXArray {
        var hidden = input
        for layer in layers {
            hidden = layer(hidden, mask: .causal, cache: nil)
        }
        if let finalNorm, let lmHead {
            hidden = lmHead(finalNorm(hidden))
        }
        eval(hidden)
        return hidden
    }

    func prefill(
        tokens: MLXArray,
        cache: any CheckpointShardSequenceCache
    ) throws -> MLXArray {
        try cachedForward(tokens: tokens, cache: cache)
    }

    func prefill(
        hidden: MLXArray,
        cache: any CheckpointShardSequenceCache
    ) throws -> MLXArray {
        try cachedForward(hidden: hidden, cache: cache)
    }

    func decode(
        tokens: MLXArray,
        cache: any CheckpointShardSequenceCache
    ) throws -> MLXArray {
        try cachedForward(tokens: tokens, cache: cache)
    }

    func decode(
        hidden: MLXArray,
        cache: any CheckpointShardSequenceCache
    ) throws -> MLXArray {
        try cachedForward(hidden: hidden, cache: cache)
    }

    private func cachedForward(
        tokens: MLXArray,
        cache: any CheckpointShardSequenceCache
    ) throws -> MLXArray {
        let cache = try gemmaCache(cache)
        if let referenceModel {
            let output = referenceModel(tokens, cache: cache.layers)
            eval(output)
            return output
        }
        guard let embedding else {
            throw CheckpointShardError.missingInputModule("model.embed_tokens")
        }
        let embedded = embedding(tokens)
        let scale = MLXArray(sqrt(Float(hiddenSize)), dtype: .bfloat16)
        return try cachedForward(
            hidden: embedded * scale.asType(embedded.dtype),
            cache: cache
        )
    }

    private func cachedForward(
        hidden input: MLXArray,
        cache sequenceCache: any CheckpointShardSequenceCache
    ) throws -> MLXArray {
        try cachedForward(hidden: input, cache: gemmaCache(sequenceCache))
    }

    private func cachedForward(
        hidden input: MLXArray,
        cache sequenceCache: Gemma3CheckpointSequenceCache
    ) throws -> MLXArray {
        guard referenceModel == nil else {
            throw CheckpointShardError.invalidSequenceCache(
                "a full-checkpoint reference accepts token input only"
            )
        }
        guard sequenceCache.layers.count == layers.count else {
            throw CheckpointShardError.invalidSequenceCache(
                "expected \(layers.count) layer caches, got \(sequenceCache.layers.count)"
            )
        }
        var hidden = input
        for (localIndex, layer) in layers.enumerated() {
            let cache = sequenceCache.layers[localIndex]
            let mask = createAttentionMask(
                h: hidden,
                cache: cache,
                windowSize: cache.maxSize
            )
            hidden = layer(hidden, mask: mask, cache: cache)
        }
        if let finalNorm, let lmHead {
            hidden = lmHead(finalNorm(hidden))
        }
        eval(hidden)
        return hidden
    }

    private func gemmaCache(
        _ cache: any CheckpointShardSequenceCache
    ) throws -> Gemma3CheckpointSequenceCache {
        guard let cache = cache as? Gemma3CheckpointSequenceCache else {
            throw CheckpointShardError.invalidSequenceCache(
                "Gemma stage received \(String(describing: type(of: cache)))"
            )
        }
        return cache
    }
}

/// Gemma-specific ownership and execution behind the generic checkpoint seam.
struct Gemma3CheckpointShardAdapter: CheckpointShardAdapter {
    let supportedModelTypes: Set<String> = ["gemma3", "gemma3_text"]

    func layerCount(checkpoint: ResolvedCheckpoint) throws -> Int {
        try configuration(checkpoint).hiddenLayers
    }

    func fullReference(
        checkpoint: ResolvedCheckpoint,
        tokens: MLXArray
    ) async throws -> CheckpointReferenceOutput {
        let baseConfig = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: checkpoint.configData
        )
        let languageModel = try await LLMModelFactory.shared.typeRegistry.createModel(
            configuration: checkpoint.configData,
            modelType: checkpoint.modelType
        )
        guard let model = languageModel as? Gemma3TextModel else {
            throw CheckpointShardError.unsupportedModel(String(describing: type(of: languageModel)))
        }

        try loadWeights(
            modelDirectory: checkpoint.directory,
            model: model,
            perLayerQuantization: baseConfig.perLayerQuantization
        )
        let output = model(tokens)
        eval(output)
        return CheckpointReferenceOutput(
            tensor: WireTensor(output),
            memory: CheckpointMemory.snapshot()
        )
    }

    func loadStage(
        checkpoint: ResolvedCheckpoint,
        request: CheckpointStageRequest
    ) async throws -> any CheckpointShardStage {
        let baseConfig = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: checkpoint.configData
        )
        let resourceConfig = try JSONDecoder().decode(
            Gemma3ResourceConfiguration.self,
            from: checkpoint.configData
        )
        let cacheBytesPerLayerPosition =
            try resourceConfig.conservativeCacheBytesPerLayerPosition()
        let languageModel = try await LLMModelFactory.shared.typeRegistry.createModel(
            configuration: checkpoint.configData,
            modelType: checkpoint.modelType
        )
        guard let model = languageModel as? Gemma3TextModel else {
            throw CheckpointShardError.unsupportedModel(String(describing: type(of: languageModel)))
        }
        let inner = model.model
        let range = request.layerRange
        guard range.lowerBound >= 0, range.upperBound <= inner.layers.count else {
            throw CheckpointShardError.invalidRange(range, inner.layers.count)
        }
        guard !request.ownsInput || range.lowerBound == 0 else {
            throw CheckpointShardError.invalidBoundary("Gemma input owner must start at layer 0")
        }
        guard !request.ownsOutput || range.upperBound == inner.layers.count else {
            throw CheckpointShardError.invalidBoundary("Gemma output owner must end at the final layer")
        }

        // A full input/output range is used as the correctness oracle. Keep it
        // on the upstream model call path so distributed cached logits are not
        // compared against the same custom stage implementation.
        if request.ownsInput,
           request.ownsOutput,
           range == 0 ..< inner.layers.count
        {
            try loadWeights(
                modelDirectory: checkpoint.directory,
                model: model,
                perLayerQuantization: baseConfig.perLayerQuantization
            )
            eval(model)
            return Gemma3CheckpointStage(
                embedding: inner.embedTokens,
                layers: inner.layers,
                cachePrototypes: [],
                finalNorm: nil,
                lmHead: nil,
                referenceModel: model,
                hiddenSize: inner.config.hiddenSize,
                vocabularySize: inner.embedTokens.weight.shape[0],
                maximumSequenceLength: resourceConfig.maximumSequenceLength,
                cacheBytesPerLayerPosition: cacheBytesPerLayerPosition,
                ownsOutput: true,
                weightKeyCount: model.parameters().flattened().count
            )
        }

        let selection = CheckpointWeightSelection(
            includesRaw: { key in
                ownsRaw(key: key, request: request)
            },
            includesSanitized: { key in
                owns(key: key, request: request)
            }
        )
        let selected = try CheckpointWeightLoader.load(
            modelDirectory: checkpoint.directory,
            selection: selection,
            sanitize: { weights, metadata in
                model.sanitize(weights: weights, metadata: metadata)
            }
        ).arrays

        // Quantize and update owned leaves individually. Updating a sparse
        // non-zero layers array would make MLXNN replace empty child entries.
        if request.ownsInput {
            let prefix = "model.embed_tokens."
            if let quantization = baseConfig.perLayerQuantization,
               selected["\(prefix)scales"] != nil,
               let config = quantization.quantization(layer: "model.embed_tokens")
            {
                let embedding = QuantizedEmbedding(
                    inner.embedTokens,
                    groupSize: config.groupSize,
                    bits: config.bits,
                    mode: config.mode
                )
                try inner.update(
                    modules: ModuleChildren.unflattened([("embed_tokens", embedding)]),
                    verify: [.noUnusedKeys]
                )
            }
            try CheckpointWeightLoader.update(
                module: inner.embedTokens,
                from: selected,
                prefix: prefix
            )
        }

        for index in range {
            let layer = inner.layers[index]
            let prefix = "model.layers.\(index)."
            if let quantization = baseConfig.perLayerQuantization {
                quantize(model: layer) { path, _ in
                    let fullPath = "model.layers.\(index).\(path)"
                    guard selected["\(fullPath).scales"] != nil else {
                        return nil
                    }
                    return quantization.quantization(layer: fullPath)?.asTuple
                }
            }
            try CheckpointWeightLoader.update(module: layer, from: selected, prefix: prefix)
        }

        var finalNorm: (any UnaryLayer)?
        var lmHead: (any UnaryLayer)?
        if request.ownsOutput {
            if let quantization = baseConfig.perLayerQuantization {
                quantize(model: model) { path, _ in
                    guard path == "lm_head", selected["lm_head.scales"] != nil else {
                        return nil
                    }
                    return quantization.quantization(layer: path)?.asTuple
                }
            }
            let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            guard let norm = modules["model.norm"] as? any UnaryLayer else {
                throw CheckpointShardError.missingOutputModule("model.norm")
            }
            guard let head = modules["lm_head"] as? any UnaryLayer else {
                throw CheckpointShardError.missingOutputModule("lm_head")
            }
            try CheckpointWeightLoader.update(
                module: norm,
                from: selected,
                prefix: "model.norm."
            )
            try CheckpointWeightLoader.update(
                module: head,
                from: selected,
                prefix: "lm_head."
            )
            finalNorm = norm
            lmHead = head
        }

        let stage = Gemma3CheckpointStage(
            embedding: request.ownsInput ? inner.embedTokens : nil,
            layers: Array(inner.layers[range]),
            cachePrototypes: Array(model.newCache()[range]),
            finalNorm: finalNorm,
            lmHead: lmHead,
            hiddenSize: inner.config.hiddenSize,
            vocabularySize: inner.embedTokens.weight.shape[0],
            maximumSequenceLength: resourceConfig.maximumSequenceLength,
            cacheBytesPerLayerPosition: cacheBytesPerLayerPosition,
            ownsOutput: request.ownsOutput,
            weightKeyCount: selected.count
        )
        if let embedding = stage.embedding {
            eval(embedding)
        }
        for layer in stage.layers {
            eval(layer)
        }
        if let finalNorm = stage.finalNorm {
            eval(finalNorm)
        }
        if let lmHead = stage.lmHead {
            eval(lmHead)
        }
        return stage
    }

    private func configuration(
        _ checkpoint: ResolvedCheckpoint
    ) throws -> Gemma3TextConfiguration {
        try JSONDecoder().decode(Gemma3TextConfiguration.self, from: checkpoint.configData)
    }

    private func ownsRaw(key: String, request: CheckpointStageRequest) -> Bool {
        if owns(key: key, request: request) {
            return true
        }
        // Gemma commonly ties lm_head to embed_tokens. Admit those source keys
        // for the sanitizer, then discard them from a final-only stage.
        return request.ownsOutput && normalized(key).hasPrefix("model.embed_tokens.")
    }

    private func owns(key originalKey: String, request: CheckpointStageRequest) -> Bool {
        let key = normalized(originalKey)
        if request.ownsInput && key.hasPrefix("model.embed_tokens.") {
            return true
        }
        if request.ownsOutput && (key.hasPrefix("model.norm.") || key.hasPrefix("lm_head.")) {
            return true
        }
        guard key.hasPrefix("model.layers.") else {
            return false
        }
        let remainder = key.dropFirst("model.layers.".count)
        guard let first = remainder.split(separator: ".").first,
              let index = Int(first)
        else {
            return false
        }
        return request.layerRange.contains(index)
    }

    private func normalized(_ key: String) -> String {
        key.hasPrefix("language_model.")
            ? String(key.dropFirst("language_model.".count))
            : key
    }
}
