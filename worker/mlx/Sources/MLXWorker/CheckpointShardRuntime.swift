import Foundation
import CryptoKit
import Darwin
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon

struct StageMemory: Codable {
    let activeBytes: Int
    let cacheBytes: Int
    let peakBytes: Int
    let processPhysicalBytes: UInt64
    let processPeakPhysicalBytes: UInt64
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
    let fingerprint: String
    let checkpointBytes: UInt64
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
    let maximumSequenceLength: Int
}

/// Adapter-owned, per-sequence incremental state. The worker can account for
/// and lifecycle-manage it without knowing a model family's concrete cache
/// representation.
protocol CheckpointShardSequenceCache: AnyObject {
    var position: Int { get }
    var memoryBytes: Int { get }
    func estimatedMemoryBytes(at position: Int) throws -> Int
}

protocol CheckpointShardStage: AnyObject {
    var weightKeyCount: Int { get }
    var inputMetadata: CheckpointStageInputMetadata { get }
    func makeSequenceCache() -> any CheckpointShardSequenceCache
    func estimatedOutputBytes(inputLength: Int) throws -> Int
    func forward(tokens: MLXArray) throws -> MLXArray
    func forward(hidden: MLXArray) throws -> MLXArray
    func prefill(
        tokens: MLXArray,
        cache: any CheckpointShardSequenceCache
    ) throws -> MLXArray
    func prefill(
        hidden: MLXArray,
        cache: any CheckpointShardSequenceCache
    ) throws -> MLXArray
    func decode(
        tokens: MLXArray,
        cache: any CheckpointShardSequenceCache
    ) throws -> MLXArray
    func decode(
        hidden: MLXArray,
        cache: any CheckpointShardSequenceCache
    ) throws -> MLXArray
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
    let maxOpenSequencesPerShard: Int
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
    case invalidSequenceCache(String)
    case noSafetensors(URL)
    case incompleteSafetensorIndex(URL, Int, Int)

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
        case .invalidSequenceCache(let reason):
            return "invalid checkpoint shard sequence cache: \(reason)"
        case .noSafetensors(let directory):
            return "no safetensors found in \(directory.path)"
        case .incompleteSafetensorIndex(let directory, let found, let expected):
            return "safetensor index in \(directory.path) resolves \(found) of \(expected) files"
        }
    }
}

enum CheckpointMemory {
    static func snapshot() -> StageMemory {
        let snapshot = Memory.snapshot()
        var usage = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &usage) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        return StageMemory(
            activeBytes: snapshot.activeMemory,
            cacheBytes: snapshot.cacheMemory,
            peakBytes: snapshot.peakMemory,
            processPhysicalBytes: status == 0 ? usage.ri_phys_footprint : 0,
            processPeakPhysicalBytes: status == 0
                ? usage.ri_lifetime_max_phys_footprint
                : 0
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
        let identity = try checkpointIdentity(directory: directory)
        return ResolvedCheckpoint(
            modelID: modelID,
            modelType: baseConfig.modelType,
            fingerprint: identity.fingerprint,
            checkpointBytes: identity.bytes,
            directory: directory,
            configData: configData
        )
    }

    private static func checkpointIdentity(
        directory: URL
    ) throws -> (fingerprint: String, bytes: UInt64) {
        let resourceKeys: [URLResourceKey] = [
            .isRegularFileKey,
            .isSymbolicLinkKey,
        ]
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: resourceKeys,
            options: [.skipsHiddenFiles]
        ) else {
            throw CheckpointShardError.invalidBoundary(
                "cannot enumerate checkpoint directory \(directory.path)"
            )
        }
        let root = directory.standardizedFileURL.path + "/"
        var files = [URL]()
        for case let file as URL in enumerator {
            let values = try file.resourceValues(forKeys: Set(resourceKeys))
            if values.isRegularFile == true || values.isSymbolicLink == true {
                files.append(file.standardizedFileURL)
            }
        }
        files.sort { $0.path < $1.path }
        guard !files.isEmpty else {
            throw CheckpointShardError.invalidBoundary(
                "checkpoint directory \(directory.path) contains no files"
            )
        }

        var hasher = SHA256()
        var totalBytes: UInt64 = 0
        for file in files {
            guard file.path.hasPrefix(root) else {
                throw CheckpointShardError.invalidBoundary(
                    "checkpoint file \(file.path) is outside \(directory.path)"
                )
            }
            let relativePath = String(file.path.dropFirst(root.count))
            hasher.update(data: Data(relativePath.utf8))
            hasher.update(data: Data([0]))
            let handle = try FileHandle(forReadingFrom: file)
            defer {
                try? handle.close()
            }
            // The identity pass is sequential and never rereads file data.
            // Avoid displacing useful model pages when the filesystem supports it.
            _ = fcntl(handle.fileDescriptor, F_NOCACHE, 1)
            while let chunkBytes: Int = try autoreleasepool(invoking: {
                guard let chunk = try handle.read(upToCount: 1024 * 1024), !chunk.isEmpty else {
                    return nil
                }
                hasher.update(data: chunk)
                return chunk.count
            }) {
                let (nextTotal, overflow) = totalBytes.addingReportingOverflow(UInt64(chunkBytes))
                guard !overflow else {
                    throw CheckpointShardError.invalidBoundary(
                        "checkpoint byte count overflows UInt64"
                    )
                }
                totalBytes = nextTotal
            }
            try handle.close()
            hasher.update(data: Data([0xff]))
        }
        return (
            hasher.finalize().map { String(format: "%02x", $0) }.joined(),
            totalBytes
        )
    }
}
