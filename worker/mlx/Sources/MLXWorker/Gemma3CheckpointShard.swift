import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
@_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon
import MLXNN

private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

struct StageMemory: Codable {
    let activeBytes: Int
    let cacheBytes: Int
    let peakBytes: Int
}

struct CheckpointShardSmokeResult: Codable {
    let model: String
    let layers: Int
    let splitLayer: Int
    let workerBudgetBytes: Int
    let firstStageWeightKeys: Int
    let secondStageWeightKeys: Int
    let fullCheckpointAfterForward: StageMemory
    let firstStageLoaded: StageMemory
    let firstStageAfterForward: StageMemory
    let secondStageLoaded: StageMemory
    let secondStageAfterForward: StageMemory
    let firstStageWithinBudget: Bool
    let secondStageWithinBudget: Bool
    let fullCheckpointExceedsBudget: Bool
    let passesMemoryProof: Bool
    let boundaryBytes: Int
    let boundaryDType: String
    let outputShape: [Int]
    let rtol: Float
    let atol: Float
    let matchesFullCheckpoint: Bool
}

private struct CheckpointShardBoundary: Codable {
    let version: Int
    let model: String
    let layers: Int
    let splitLayer: Int
    let workerBudgetBytes: Int
    let tokens: [Int32]
    let tensor: WireTensor
    let firstStageWeightKeys: Int
    let firstStageLoaded: StageMemory
    let firstStageAfterForward: StageMemory
}

private struct ReferenceOutput {
    let tensor: WireTensor
    let memory: StageMemory
}

private enum CheckpointShardError: LocalizedError {
    case unsupportedModel(String)
    case invalidBoundary(String)
    case invalidRange(Range<Int>, Int)
    case missingEmbedding
    case missingOutputModule(String)
    case noSafetensors(URL)

    var errorDescription: String? {
        switch self {
        case .unsupportedModel(let type):
            return "checkpoint sharding requires Gemma3TextModel, got \(type)"
        case .invalidBoundary(let reason):
            return "invalid checkpoint shard boundary: \(reason)"
        case .invalidRange(let range, let count):
            return "invalid layer range \(range) for model with \(count) layers"
        case .missingEmbedding:
            return "first checkpoint stage requires token embeddings"
        case .missingOutputModule(let path):
            return "final checkpoint stage requires upstream module \(path)"
        case .noSafetensors(let directory):
            return "no safetensors found in \(directory.path)"
        }
    }
}

private final class Gemma3CheckpointStage {
    let embedding: Embedding?
    let layers: [Gemma3TransformerBlock]
    let finalNorm: (any UnaryLayer)?
    let lmHead: (any UnaryLayer)?
    let hiddenSize: Int
    let layerRange: Range<Int>
    let weightKeyCount: Int

    init(
        embedding: Embedding?,
        layers: [Gemma3TransformerBlock],
        finalNorm: (any UnaryLayer)?,
        lmHead: (any UnaryLayer)?,
        hiddenSize: Int,
        layerRange: Range<Int>,
        weightKeyCount: Int
    ) {
        self.embedding = embedding
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hiddenSize = hiddenSize
        self.layerRange = layerRange
        self.weightKeyCount = weightKeyCount
    }

    func forward(tokens: MLXArray) throws -> MLXArray {
        guard let embedding else {
            throw CheckpointShardError.missingEmbedding
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

enum Gemma3CheckpointShard {
    static let defaultModelID = "mlx-community/gemma-3-270m-it-4bit"
    static let defaultWorkerBudgetBytes = 128 * 1024 * 1024
    static let rtol: Float = 1e-4
    static let atol: Float = 1e-4

    private static let boundaryVersion = 1
    private static let defaultTokens: [Int32] = [1, 2, 3, 4, 5, 6]

    static func run(modelID: String = defaultModelID) async throws -> CheckpointShardSmokeResult {
        let payload = try await produceBoundaryPayload(modelID: modelID)
        return try await finishBoundary(from: payload)
    }

    static func produceBoundaryPayload(modelID: String = defaultModelID) async throws -> Data {
        let (directory, configData, gemmaConfig) = try await resolveModel(modelID: modelID)
        let layerCount = gemmaConfig.hiddenLayers
        let split = layerCount / 2
        let tokens = MLXArray(defaultTokens, [1, defaultTokens.count])

        Memory.clearCache()
        Memory.peakMemory = 0
        var firstStage: Gemma3CheckpointStage? = try await loadStage(
            directory: directory,
            configData: configData,
            range: 0 ..< split,
            includeEmbedding: true,
            includeOutput: false
        )
        Memory.clearCache()
        let firstLoaded = memory()
        let firstWeightKeys = firstStage!.weightKeyCount
        let boundaryTensor = WireTensor(try firstStage!.forward(tokens: tokens))
        let firstAfterForward = memory()
        firstStage = nil
        Memory.clearCache()

        let boundary = CheckpointShardBoundary(
            version: boundaryVersion,
            model: modelID,
            layers: layerCount,
            splitLayer: split,
            workerBudgetBytes: defaultWorkerBudgetBytes,
            tokens: defaultTokens,
            tensor: boundaryTensor,
            firstStageWeightKeys: firstWeightKeys,
            firstStageLoaded: firstLoaded,
            firstStageAfterForward: firstAfterForward
        )
        return try JSONEncoder().encode(boundary)
    }

    static func finishBoundary(from payload: Data) async throws -> CheckpointShardSmokeResult {
        let boundary = try JSONDecoder().decode(CheckpointShardBoundary.self, from: payload)
        guard boundary.version == boundaryVersion else {
            throw CheckpointShardError.invalidBoundary("unsupported version \(boundary.version)")
        }
        guard boundary.workerBudgetBytes > 0 else {
            throw CheckpointShardError.invalidBoundary("worker budget must be positive")
        }

        let (directory, configData, gemmaConfig) = try await resolveModel(modelID: boundary.model)
        guard boundary.layers == gemmaConfig.hiddenLayers else {
            throw CheckpointShardError.invalidBoundary(
                "producer reported \(boundary.layers) layers; checkpoint has \(gemmaConfig.hiddenLayers)"
            )
        }
        guard boundary.splitLayer > 0, boundary.splitLayer < boundary.layers else {
            throw CheckpointShardError.invalidBoundary("split \(boundary.splitLayer) is not internal")
        }
        guard !boundary.tokens.isEmpty else {
            throw CheckpointShardError.invalidBoundary("token sequence is empty")
        }
        let tokens = MLXArray(boundary.tokens, [1, boundary.tokens.count])

        // The complete checkpoint is a correctness and memory oracle only. Its
        // MLX allocations leave memory before the consumer stage is loaded.
        Memory.clearCache()
        Memory.peakMemory = 0
        let expected = try await fullReference(
            directory: directory,
            configData: configData,
            tokens: tokens
        )
        Memory.clearCache()

        Memory.peakMemory = 0
        var secondStage: Gemma3CheckpointStage? = try await loadStage(
            directory: directory,
            configData: configData,
            range: boundary.splitLayer ..< boundary.layers,
            includeEmbedding: false,
            includeOutput: true
        )
        Memory.clearCache()
        let secondLoaded = memory()
        let secondWeightKeys = secondStage!.weightKeyCount
        let output = try secondStage!.forward(hidden: boundary.tensor.materialize())
        let secondAfterForward = memory()
        secondStage = nil

        let matches = allClose(
            expected.tensor.materialize(),
            output,
            rtol: Double(rtol),
            atol: Double(atol)
        ).item(Bool.self)
        let firstWithinBudget = boundary.firstStageAfterForward.peakBytes <= boundary.workerBudgetBytes
        let secondWithinBudget = secondAfterForward.peakBytes <= boundary.workerBudgetBytes
        let fullExceedsBudget = expected.memory.peakBytes > boundary.workerBudgetBytes

        return CheckpointShardSmokeResult(
            model: boundary.model,
            layers: boundary.layers,
            splitLayer: boundary.splitLayer,
            workerBudgetBytes: boundary.workerBudgetBytes,
            firstStageWeightKeys: boundary.firstStageWeightKeys,
            secondStageWeightKeys: secondWeightKeys,
            fullCheckpointAfterForward: expected.memory,
            firstStageLoaded: boundary.firstStageLoaded,
            firstStageAfterForward: boundary.firstStageAfterForward,
            secondStageLoaded: secondLoaded,
            secondStageAfterForward: secondAfterForward,
            firstStageWithinBudget: firstWithinBudget,
            secondStageWithinBudget: secondWithinBudget,
            fullCheckpointExceedsBudget: fullExceedsBudget,
            passesMemoryProof: firstWithinBudget && secondWithinBudget && fullExceedsBudget,
            boundaryBytes: boundary.tensor.data.count,
            boundaryDType: boundary.tensor.dtype.rawValue,
            outputShape: output.shape,
            rtol: rtol,
            atol: atol,
            matchesFullCheckpoint: matches
        )
    }

    private static func resolveModel(
        modelID: String
    ) async throws -> (URL, Data, Gemma3TextConfiguration) {
        let resolved = try await resolve(
            configuration: ModelConfiguration(id: modelID),
            from: #hubDownloader(),
            useLatest: false,
            progressHandler: { _ in }
        )
        let directory = resolved.modelDirectory
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let gemmaConfig = try JSONDecoder().decode(Gemma3TextConfiguration.self, from: configData)
        return (directory, configData, gemmaConfig)
    }

    private static func fullReference(
        directory: URL,
        configData: Data,
        tokens: MLXArray
    ) async throws -> ReferenceOutput {
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        let languageModel = try await LLMModelFactory.shared.typeRegistry.createModel(
            configuration: configData,
            modelType: baseConfig.modelType
        )
        guard let model = languageModel as? Gemma3TextModel else {
            throw CheckpointShardError.unsupportedModel(String(describing: type(of: languageModel)))
        }

        try loadWeights(
            modelDirectory: directory,
            model: model,
            perLayerQuantization: baseConfig.perLayerQuantization
        )
        let output = model(tokens)
        eval(output)
        let tensor = WireTensor(output)
        let snapshot = memory()
        return ReferenceOutput(tensor: tensor, memory: snapshot)
    }

    private static func loadStage(
        directory: URL,
        configData: Data,
        range: Range<Int>,
        includeEmbedding: Bool,
        includeOutput: Bool
    ) async throws -> Gemma3CheckpointStage {
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        let languageModel = try await LLMModelFactory.shared.typeRegistry.createModel(
            configuration: configData,
            modelType: baseConfig.modelType
        )
        guard let model = languageModel as? Gemma3TextModel else {
            throw CheckpointShardError.unsupportedModel(String(describing: type(of: languageModel)))
        }
        let inner = model.model
        guard range.lowerBound >= 0, range.upperBound <= inner.layers.count else {
            throw CheckpointShardError.invalidRange(range, inner.layers.count)
        }

        var selected = [String: MLXArray]()
        var metadata = [String: String]()
        for url in try safetensorURLs(in: directory) {
            autoreleasepool {
                if let loaded = try? loadArraysAndMetadata(url: url) {
                    let (fileWeights, fileMetadata) = loaded
                    for (key, value) in fileWeights where ownsRaw(
                        key: key,
                        range: range,
                        includeEmbedding: includeEmbedding,
                        includeOutput: includeOutput
                    ) {
                        selected[key] = value
                    }
                    if metadata.isEmpty {
                        metadata = fileMetadata
                    }
                }
            }
        }

        // Sanitization can normalize prefixes and synthesize tied lm_head
        // weights from embed_tokens. The final filter drops that temporary
        // embedding input from a consumer stage.
        selected = model.sanitize(weights: selected, metadata: metadata)
        selected = selected.filter {
            owns(
                key: $0.key,
                range: range,
                includeEmbedding: includeEmbedding,
                includeOutput: includeOutput
            )
        }

        // Apply each owned module independently. A sparse second-half `layers`
        // update begins with empty array entries, which MLXNN cannot use when it
        // replaces quantized children on the complete model.
        if includeEmbedding {
            let prefix = "model.embed_tokens."
            if let quantization = baseConfig.perLayerQuantization,
               selected["\(prefix)scales"] != nil,
               let configuration = quantization.quantization(layer: "model.embed_tokens")
            {
                let embedding = QuantizedEmbedding(
                    inner.embedTokens,
                    groupSize: configuration.groupSize,
                    bits: configuration.bits,
                    mode: configuration.mode
                )
                try inner.update(
                    modules: ModuleChildren.unflattened([("embed_tokens", embedding)]),
                    verify: [.noUnusedKeys]
                )
            }
            try update(module: inner.embedTokens, from: selected, prefix: prefix)
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
            try update(module: layer, from: selected, prefix: prefix)
        }

        var finalNorm: (any UnaryLayer)?
        var lmHead: (any UnaryLayer)?
        if includeOutput {
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
            try update(module: norm, from: selected, prefix: "model.norm.")
            try update(module: head, from: selected, prefix: "lm_head.")
            finalNorm = norm
            lmHead = head
        }

        // Keep only owned module references. The parent and all unowned random
        // parameters can deallocate when this function returns.
        let stage = Gemma3CheckpointStage(
            embedding: includeEmbedding ? inner.embedTokens : nil,
            layers: Array(inner.layers[range]),
            finalNorm: finalNorm,
            lmHead: lmHead,
            hiddenSize: inner.config.hiddenSize,
            layerRange: range,
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

    private static func update(
        module: Module,
        from weights: [String: MLXArray],
        prefix: String
    ) throws {
        let local: [String: MLXArray] = Dictionary(
            uniqueKeysWithValues: weights.compactMap { key, value -> (String, MLXArray)? in
                guard key.hasPrefix(prefix) else {
                    return nil
                }
                return (String(key.dropFirst(prefix.count)), value)
            }
        )
        try module.update(
            parameters: ModuleParameters.unflattened(local),
            verify: [.noUnusedKeys, .shapeMismatch]
        )
    }

    private static func ownsRaw(
        key: String,
        range: Range<Int>,
        includeEmbedding: Bool,
        includeOutput: Bool
    ) -> Bool {
        if owns(
            key: key,
            range: range,
            includeEmbedding: includeEmbedding,
            includeOutput: includeOutput
        ) {
            return true
        }
        // Gemma checkpoints commonly tie lm_head to embed_tokens. Feed only
        // those source keys to sanitize, then discard them above.
        return includeOutput && normalized(key).hasPrefix("model.embed_tokens.")
    }

    private static func owns(
        key originalKey: String,
        range: Range<Int>,
        includeEmbedding: Bool,
        includeOutput: Bool
    ) -> Bool {
        let key = normalized(originalKey)
        if includeEmbedding && key.hasPrefix("model.embed_tokens.") {
            return true
        }
        if includeOutput && (key.hasPrefix("model.norm.") || key.hasPrefix("lm_head.")) {
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
        return range.contains(index)
    }

    private static func normalized(_ key: String) -> String {
        key.hasPrefix("language_model.")
            ? String(key.dropFirst("language_model.".count))
            : key
    }

    private static func safetensorURLs(in directory: URL) throws -> [URL] {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let index = try JSONDecoder().decode(
                SafetensorsIndex.self,
                from: Data(contentsOf: indexURL)
            )
            return Set(index.weightMap.values)
                .sorted()
                .map { directory.appendingPathComponent($0) }
        }

        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: nil
        ) else {
            throw CheckpointShardError.noSafetensors(directory)
        }
        let urls = enumerator.compactMap { item -> URL? in
            guard let url = item as? URL, url.pathExtension == "safetensors" else {
                return nil
            }
            return url
        }.sorted { $0.path < $1.path }
        guard !urls.isEmpty else {
            throw CheckpointShardError.noSafetensors(directory)
        }
        return urls
    }

    private static func memory() -> StageMemory {
        let snapshot = Memory.snapshot()
        return StageMemory(
            activeBytes: snapshot.activeMemory,
            cacheBytes: snapshot.cacheMemory,
            peakBytes: snapshot.peakMemory
        )
    }
}
