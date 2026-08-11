import Foundation
import MLX
@_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon
import MLXNN

private final class Gemma3CheckpointStage: CheckpointShardStage {
    let embedding: Embedding?
    let layers: [Gemma3TransformerBlock]
    let finalNorm: (any UnaryLayer)?
    let lmHead: (any UnaryLayer)?
    let hiddenSize: Int
    let weightKeyCount: Int

    init(
        embedding: Embedding?,
        layers: [Gemma3TransformerBlock],
        finalNorm: (any UnaryLayer)?,
        lmHead: (any UnaryLayer)?,
        hiddenSize: Int,
        weightKeyCount: Int
    ) {
        self.embedding = embedding
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hiddenSize = hiddenSize
        self.weightKeyCount = weightKeyCount
    }

    func forward(tokens: MLXArray) throws -> MLXArray {
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
            finalNorm: finalNorm,
            lmHead: lmHead,
            hiddenSize: inner.config.hiddenSize,
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
