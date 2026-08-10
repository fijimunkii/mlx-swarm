import Foundation
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

private struct StageMemory: Codable {
    let activeBytes: Int
    let cacheBytes: Int
    let peakBytes: Int
}

struct CheckpointShardSmokeResult: Codable {
    let model: String
    let layers: Int
    let splitLayer: Int
    let firstStageWeightKeys: Int
    let secondStageWeightKeys: Int
    let firstStageLoaded: StageMemory
    let firstStageAfterForward: StageMemory
    let secondStageLoaded: StageMemory
    let secondStageAfterForward: StageMemory
    let boundaryBytes: Int
    let boundaryDType: String
    let outputShape: [Int]
    let matchesFullCheckpoint: Bool
}

private enum CheckpointShardError: LocalizedError {
    case unsupportedModel(String)
    case invalidRange(Range<Int>, Int)
    case missingEmbedding
    case noSafetensors(URL)

    var errorDescription: String? {
        switch self {
        case .unsupportedModel(let type):
            return "checkpoint shard smoke requires Gemma3TextModel, got \(type)"
        case .invalidRange(let range, let count):
            return "invalid layer range \(range) for model with \(count) layers"
        case .missingEmbedding:
            return "first checkpoint stage requires token embeddings"
        case .noSafetensors(let directory):
            return "no safetensors found in \(directory.path)"
        }
    }
}

private final class Gemma3CheckpointStage {
    let embedding: Embedding?
    let layers: [Gemma3TransformerBlock]
    let hiddenSize: Int
    let layerRange: Range<Int>
    let weightKeyCount: Int

    init(
        embedding: Embedding?,
        layers: [Gemma3TransformerBlock],
        hiddenSize: Int,
        layerRange: Range<Int>,
        weightKeyCount: Int
    ) {
        self.embedding = embedding
        self.layers = layers
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
        return forward(hidden: embedded * scale.asType(embedded.dtype))
    }

    func forward(hidden input: MLXArray) -> MLXArray {
        var hidden = input
        for layer in layers {
            hidden = layer(hidden, mask: .causal, cache: nil)
        }
        eval(hidden)
        return hidden
    }
}

enum Gemma3CheckpointShard {
    static let defaultModelID = "mlx-community/gemma-3-270m-it-4bit"

    static func run(modelID: String = defaultModelID) async throws -> CheckpointShardSmokeResult {
        let resolved = try await resolve(
            configuration: ModelConfiguration(id: modelID),
            from: #hubDownloader(),
            useLatest: false,
            progressHandler: { _ in }
        )
        let directory = resolved.modelDirectory
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let gemmaConfig = try JSONDecoder().decode(Gemma3TextConfiguration.self, from: configData)
        let layerCount = gemmaConfig.hiddenLayers
        let split = layerCount / 2
        let tokens = MLXArray([1, 2, 3, 4, 5, 6], [1, 6])

        // Establish the correctness oracle, then retain only CPU bytes so the
        // complete checkpoint can leave MLX memory before either shard is measured.
        let expected = try await fullReference(
            directory: directory,
            configData: configData,
            tokens: tokens
        )
        Memory.clearCache()

        var firstStage: Gemma3CheckpointStage? = try await loadStage(
            directory: directory,
            configData: configData,
            range: 0 ..< split,
            includeEmbedding: true
        )
        Memory.clearCache()
        Memory.peakMemory = 0
        let firstLoaded = memory()
        let firstWeightKeys = firstStage!.weightKeyCount
        let boundary = WireTensor(try firstStage!.forward(tokens: tokens))
        let firstAfterForward = memory()
        firstStage = nil
        Memory.clearCache()

        var secondStage: Gemma3CheckpointStage? = try await loadStage(
            directory: directory,
            configData: configData,
            range: split ..< layerCount,
            includeEmbedding: false
        )
        Memory.clearCache()
        Memory.peakMemory = 0
        let secondLoaded = memory()
        let secondWeightKeys = secondStage!.weightKeyCount
        let output = secondStage!.forward(hidden: boundary.materialize())
        let secondAfterForward = memory()
        secondStage = nil

        let expectedArray = expected.materialize()
        let matches = allClose(expectedArray, output, rtol: 1e-4, atol: 1e-4).item(Bool.self)

        return CheckpointShardSmokeResult(
            model: modelID,
            layers: layerCount,
            splitLayer: split,
            firstStageWeightKeys: firstWeightKeys,
            secondStageWeightKeys: secondWeightKeys,
            firstStageLoaded: firstLoaded,
            firstStageAfterForward: firstAfterForward,
            secondStageLoaded: secondLoaded,
            secondStageAfterForward: secondAfterForward,
            boundaryBytes: boundary.data.count,
            boundaryDType: boundary.dtype.rawValue,
            outputShape: output.shape,
            matchesFullCheckpoint: matches
        )
    }

    private static func fullReference(
        directory: URL,
        configData: Data,
        tokens: MLXArray
    ) async throws -> WireTensor {
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

        var hidden = embeddedInput(model.model, tokens: tokens)
        for layer in model.model.layers {
            hidden = layer(hidden, mask: .causal, cache: nil)
        }
        eval(hidden)
        return WireTensor(hidden)
    }

    private static func loadStage(
        directory: URL,
        configData: Data,
        range: Range<Int>,
        includeEmbedding: Bool
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
                    for (key, value) in fileWeights where owns(
                        key: key,
                        range: range,
                        includeEmbedding: includeEmbedding
                    ) {
                        selected[key] = value
                    }
                    if metadata.isEmpty {
                        metadata = fileMetadata
                    }
                }
            }
        }

        // Sanitizers may normalize prefixes or synthesize tied weights. Filter
        // again afterward so this stage never retains parameters it does not own.
        selected = model.sanitize(weights: selected, metadata: metadata)
        selected = selected.filter {
            owns(key: $0.key, range: range, includeEmbedding: includeEmbedding)
        }

        if let quantization = baseConfig.perLayerQuantization {
            quantize(model: model) { path, _ in
                guard selected["\(path).scales"] != nil else {
                    return nil
                }
                return quantization.quantization(layer: path)?.asTuple
            }
        }

        let parameters = ModuleParameters.unflattened(selected)
        try model.update(
            parameters: parameters,
            verify: [.noUnusedKeys, .shapeMismatch]
        )

        // Keep only the owned module references. When this function returns the
        // parent Gemma3TextModel and every unowned random parameter can deallocate.
        let stage = Gemma3CheckpointStage(
            embedding: includeEmbedding ? inner.embedTokens : nil,
            layers: Array(inner.layers[range]),
            hiddenSize: inner.config.hiddenSize,
            layerRange: range,
            weightKeyCount: selected.count
        )

        // Force selected parameters to materialize before we take steady-memory
        // measurements; do not eval(model), which would materialize unowned layers.
        if let embedding = stage.embedding {
            eval(embedding)
        }
        for layer in stage.layers {
            eval(layer)
        }
        return stage
    }

    private static func embeddedInput(_ inner: Gemma3Model, tokens: MLXArray) -> MLXArray {
        let embedded = inner.embedTokens(tokens)
        let scale = MLXArray(sqrt(Float(inner.config.hiddenSize)), dtype: .bfloat16)
        return embedded * scale.asType(embedded.dtype)
    }

    private static func owns(
        key originalKey: String,
        range: Range<Int>,
        includeEmbedding: Bool
    ) -> Bool {
        let key = originalKey.hasPrefix("language_model.")
            ? String(originalKey.dropFirst("language_model.".count))
            : originalKey

        if includeEmbedding && key.hasPrefix("model.embed_tokens.") {
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
