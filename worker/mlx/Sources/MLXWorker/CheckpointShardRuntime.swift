import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon

struct StageMemory: Codable {
    let activeBytes: Int
    let cacheBytes: Int
    let peakBytes: Int
}

struct CheckpointShardResult: Codable {
    let model: String
    let modelType: String
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

struct ResolvedCheckpoint {
    let modelID: String
    let modelType: String
    let directory: URL
    let configData: Data
}

struct CheckpointStageRequest {
    let layerRange: Range<Int>
    let ownsInput: Bool
    let ownsOutput: Bool
}

struct CheckpointReferenceOutput {
    let tensor: WireTensor
    let memory: StageMemory
}

struct CheckpointStageInputMetadata {
    let tokenDType: WireTensor.ElementType
    let tokenRank: Int
    let vocabularySize: Int
    let hiddenDTypes: Set<WireTensor.ElementType>
    let hiddenRank: Int
    let hiddenSize: Int
}

protocol CheckpointShardStage: AnyObject {
    var weightKeyCount: Int { get }
    var inputMetadata: CheckpointStageInputMetadata { get }
    func forward(tokens: MLXArray) throws -> MLXArray
    func forward(hidden: MLXArray) throws -> MLXArray
}

/// Model-family seam: checkpoint transport, budgeting, and orchestration stay
/// generic while embeddings, masks, cache semantics, and module paths remain
/// owned by the architecture adapter.
protocol CheckpointShardAdapter: Sendable {
    var supportedModelTypes: Set<String> { get }
    func layerCount(checkpoint: ResolvedCheckpoint) throws -> Int
    func loadStage(
        checkpoint: ResolvedCheckpoint,
        request: CheckpointStageRequest
    ) async throws -> any CheckpointShardStage
    func fullReference(
        checkpoint: ResolvedCheckpoint,
        tokens: MLXArray
    ) async throws -> CheckpointReferenceOutput
}

struct CheckpointShardAdapterRegistry: Sendable {
    private let adapters: [any CheckpointShardAdapter]

    init(_ adapters: [any CheckpointShardAdapter]) {
        self.adapters = adapters
    }

    func adapter(for modelType: String) throws -> any CheckpointShardAdapter {
        guard let adapter = adapters.first(where: { $0.supportedModelTypes.contains(modelType) }) else {
            throw CheckpointShardError.unsupportedModelType(modelType)
        }
        return adapter
    }

    var supportedModelTypes: [String] {
        adapters.flatMap(\.supportedModelTypes).sorted()
    }
}

struct CheckpointShardRuntimeConfiguration: Sendable {
    let adapterRegistry: CheckpointShardAdapterRegistry
    let workerBudgetBytes: Int
    let tokens: [Int32]
    let rtol: Float
    let atol: Float
}

enum CheckpointShardError: LocalizedError {
    case unsupportedModelType(String)
    case unsupportedModel(String)
    case invalidBoundary(String)
    case invalidRange(Range<Int>, Int)
    case missingInputModule(String)
    case missingOutputModule(String)
    case noSafetensors(URL)

    var errorDescription: String? {
        switch self {
        case .unsupportedModelType(let type):
            return "no checkpoint shard adapter is registered for model type \(type)"
        case .unsupportedModel(let type):
            return "checkpoint adapter received unsupported model \(type)"
        case .invalidBoundary(let reason):
            return "invalid checkpoint shard boundary: \(reason)"
        case .invalidRange(let range, let count):
            return "invalid layer range \(range) for model with \(count) layers"
        case .missingInputModule(let path):
            return "first checkpoint stage requires input module \(path)"
        case .missingOutputModule(let path):
            return "final checkpoint stage requires output module \(path)"
        case .noSafetensors(let directory):
            return "no safetensors found in \(directory.path)"
        }
    }
}

enum CheckpointMemory {
    static func snapshot() -> StageMemory {
        let snapshot = Memory.snapshot()
        return StageMemory(
            activeBytes: snapshot.activeMemory,
            cacheBytes: snapshot.cacheMemory,
            peakBytes: snapshot.peakMemory
        )
    }
}

private struct CheckpointShardBoundary: Codable {
    let version: Int
    let model: String
    let modelType: String
    let layers: Int
    let splitLayer: Int
    let workerBudgetBytes: Int
    let rtol: Float
    let atol: Float
    let tokens: [Int32]
    let tensor: WireTensor
    let firstStageWeightKeys: Int
    let firstStageLoaded: StageMemory
    let firstStageAfterForward: StageMemory
}

enum CheckpointShardRuntime {
    private static let boundaryVersion = 2

    static func run(
        modelID: String,
        configuration: CheckpointShardRuntimeConfiguration
    ) async throws -> CheckpointShardResult {
        let payload = try await produceBoundaryPayload(
            modelID: modelID,
            configuration: configuration
        )
        return try await finishBoundary(from: payload, configuration: configuration)
    }

    static func produceBoundaryPayload(
        modelID: String,
        configuration: CheckpointShardRuntimeConfiguration
    ) async throws -> Data {
        let checkpoint = try await resolveCheckpoint(modelID: modelID)
        let adapter = try configuration.adapterRegistry.adapter(for: checkpoint.modelType)
        let layerCount = try adapter.layerCount(checkpoint: checkpoint)
        let split = layerCount / 2
        guard split > 0, split < layerCount else {
            throw CheckpointShardError.invalidRange(0 ..< split, layerCount)
        }
        guard configuration.workerBudgetBytes > 0 else {
            throw CheckpointShardError.invalidBoundary("worker budget must be positive")
        }
        guard !configuration.tokens.isEmpty else {
            throw CheckpointShardError.invalidBoundary("token sequence is empty")
        }
        guard configuration.rtol >= 0, configuration.atol >= 0 else {
            throw CheckpointShardError.invalidBoundary("numeric tolerances must be non-negative")
        }
        let tokens = MLXArray(configuration.tokens, [1, configuration.tokens.count])

        Memory.clearCache()
        Memory.peakMemory = 0
        var firstStage: (any CheckpointShardStage)? = try await adapter.loadStage(
            checkpoint: checkpoint,
            request: CheckpointStageRequest(
                layerRange: 0 ..< split,
                ownsInput: true,
                ownsOutput: false
            )
        )
        Memory.clearCache()
        let firstLoaded = CheckpointMemory.snapshot()
        let firstWeightKeys = firstStage!.weightKeyCount
        let boundaryTensor = WireTensor(try firstStage!.forward(tokens: tokens))
        let firstAfterForward = CheckpointMemory.snapshot()
        firstStage = nil
        Memory.clearCache()

        return try JSONEncoder().encode(
            CheckpointShardBoundary(
                version: boundaryVersion,
                model: modelID,
                modelType: checkpoint.modelType,
                layers: layerCount,
                splitLayer: split,
                workerBudgetBytes: configuration.workerBudgetBytes,
                rtol: configuration.rtol,
                atol: configuration.atol,
                tokens: configuration.tokens,
                tensor: boundaryTensor,
                firstStageWeightKeys: firstWeightKeys,
                firstStageLoaded: firstLoaded,
                firstStageAfterForward: firstAfterForward
            )
        )
    }

    static func finishBoundary(
        from payload: Data,
        configuration: CheckpointShardRuntimeConfiguration
    ) async throws -> CheckpointShardResult {
        let boundary = try JSONDecoder().decode(CheckpointShardBoundary.self, from: payload)
        guard boundary.version == boundaryVersion else {
            throw CheckpointShardError.invalidBoundary("unsupported version \(boundary.version)")
        }
        guard boundary.workerBudgetBytes > 0 else {
            throw CheckpointShardError.invalidBoundary("worker budget must be positive")
        }
        guard boundary.rtol >= 0, boundary.atol >= 0 else {
            throw CheckpointShardError.invalidBoundary("numeric tolerances must be non-negative")
        }
        guard boundary.workerBudgetBytes == configuration.workerBudgetBytes else {
            throw CheckpointShardError.invalidBoundary("producer and consumer budgets differ")
        }
        guard boundary.rtol == configuration.rtol, boundary.atol == configuration.atol else {
            throw CheckpointShardError.invalidBoundary("producer and consumer tolerances differ")
        }
        guard boundary.tokens == configuration.tokens else {
            throw CheckpointShardError.invalidBoundary("producer and consumer token plans differ")
        }

        let checkpoint = try await resolveCheckpoint(modelID: boundary.model)
        guard checkpoint.modelType == boundary.modelType else {
            throw CheckpointShardError.invalidBoundary(
                "producer model type \(boundary.modelType) resolved as \(checkpoint.modelType)"
            )
        }
        let adapter = try configuration.adapterRegistry.adapter(for: checkpoint.modelType)
        let layerCount = try adapter.layerCount(checkpoint: checkpoint)
        guard boundary.layers == layerCount else {
            throw CheckpointShardError.invalidBoundary(
                "producer reported \(boundary.layers) layers; checkpoint has \(layerCount)"
            )
        }
        guard boundary.splitLayer > 0, boundary.splitLayer < boundary.layers else {
            throw CheckpointShardError.invalidBoundary("split \(boundary.splitLayer) is not internal")
        }
        guard !boundary.tokens.isEmpty else {
            throw CheckpointShardError.invalidBoundary("token sequence is empty")
        }
        let tokens = MLXArray(boundary.tokens, [1, boundary.tokens.count])

        Memory.clearCache()
        Memory.peakMemory = 0
        let expected = try await adapter.fullReference(checkpoint: checkpoint, tokens: tokens)
        Memory.clearCache()

        Memory.peakMemory = 0
        var secondStage: (any CheckpointShardStage)? = try await adapter.loadStage(
            checkpoint: checkpoint,
            request: CheckpointStageRequest(
                layerRange: boundary.splitLayer ..< boundary.layers,
                ownsInput: false,
                ownsOutput: true
            )
        )
        Memory.clearCache()
        let secondLoaded = CheckpointMemory.snapshot()
        let secondWeightKeys = secondStage!.weightKeyCount
        let output = try secondStage!.forward(hidden: boundary.tensor.materialize())
        let secondAfterForward = CheckpointMemory.snapshot()
        secondStage = nil

        let matches = allClose(
            try expected.tensor.materialize(),
            output,
            rtol: Double(configuration.rtol),
            atol: Double(configuration.atol)
        ).item(Bool.self)
        let firstWithinBudget = boundary.firstStageAfterForward.peakBytes <= boundary.workerBudgetBytes
        let secondWithinBudget = secondAfterForward.peakBytes <= boundary.workerBudgetBytes
        let fullExceedsBudget = expected.memory.peakBytes > boundary.workerBudgetBytes

        return CheckpointShardResult(
            model: boundary.model,
            modelType: boundary.modelType,
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
            rtol: configuration.rtol,
            atol: configuration.atol,
            matchesFullCheckpoint: matches
        )
    }

    static func resolveCheckpoint(modelID: String) async throws -> ResolvedCheckpoint {
        let resolved = try await resolve(
            configuration: ModelConfiguration(id: modelID),
            from: #hubDownloader(),
            useLatest: false,
            progressHandler: { _ in }
        )
        let directory = resolved.modelDirectory
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        return ResolvedCheckpoint(
            modelID: modelID,
            modelType: baseConfig.modelType,
            directory: directory,
            configData: configData
        )
    }
}
