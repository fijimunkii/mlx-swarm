import Foundation
import CryptoKit
import MLX

struct PersistentWorkerRequest: Codable {
    let requestID: String
    let command: String
    let loadShard: PersistentLoadShardRequest?
    let shard: PersistentShardRequest?
    let sequence: PersistentSequenceRequest?
    let forward: PersistentForwardRequest?
}

private struct PersistentWorkerEnvelope: Decodable {
    let requestID: String
}

struct PersistentLoadShardRequest: Codable {
    let modelID: String
    let shardID: String
    let layerStart: Int
    let layerEnd: Int
    let ownsInput: Bool
    let ownsOutput: Bool
}

struct PersistentShardRequest: Codable {
    let shardID: String
}

struct PersistentSequenceRequest: Codable {
    let shardID: String
    let sequenceID: String
}

struct PersistentForwardRequest: Codable {
    let shardID: String
    let sequenceID: String
    let position: UInt64
    let inputKind: String
    let input: WireTensor
}

struct PersistentWorkerResponse: Codable {
    let requestID: String
    let ok: Bool
    let error: String?
    let result: PersistentWorkerResult?

    static func success(
        requestID: String,
        result: PersistentWorkerResult
    ) -> PersistentWorkerResponse {
        PersistentWorkerResponse(requestID: requestID, ok: true, error: nil, result: result)
    }

    static func failure(requestID: String, error: Error) -> PersistentWorkerResponse {
        PersistentWorkerResponse(
            requestID: requestID,
            ok: false,
            error: error.localizedDescription,
            result: nil
        )
    }
}

struct PersistentWorkerResult: Codable {
    var status: String?
    var shard: PersistentShardSnapshot?
    var forward: PersistentForwardResult?
    var state: PersistentWorkerState?
    var shutdown: Bool?
}

struct PersistentForwardResult: Codable {
    let shardID: String
    let sequenceID: String
    let operation: String
    let position: UInt64
    let nextPosition: UInt64
    let output: WireTensor
    let computeMicros: UInt64
    let kvCacheBytes: Int
    let memory: StageMemory
}

struct PersistentShardSnapshot: Codable {
    let shardID: String
    let modelID: String
    let modelType: String
    let layerStart: Int
    let layerEnd: Int
    let ownsInput: Bool
    let ownsOutput: Bool
    let weightKeyCount: Int
    let openSequenceCount: Int
    let forwardCount: Int
    let kvCacheBytes: Int
    let loadedMemory: StageMemory
}

struct PersistentWorkerState: Codable {
    let loadedShards: [PersistentShardSnapshot]
    let loadCount: Int
    let forwardCount: Int
    let kvCacheBytes: Int
    let memory: StageMemory
}

private enum PersistentWorkerError: LocalizedError {
    case invalidRequest(String)
    case shardAlreadyLoaded(String)
    case shardNotFound(String)
    case shardHasOpenSequences(String, Int)
    case sequenceAlreadyOpen(String, String)
    case sequenceNotFound(String, String)
    case sequenceAlreadyPrefilled(String, String)
    case sequenceNotPrefilled(String, String)
    case invalidPosition(String, String, got: UInt64, expected: UInt64)
    case cachePositionMismatch(String, String, got: Int, expected: Int)
    case unsupportedInputKind(String)
    case inputKindMismatch(String, got: String, expected: String)

    var errorDescription: String? {
        switch self {
        case .invalidRequest(let reason):
            return "invalid persistent worker request: \(reason)"
        case .shardAlreadyLoaded(let shardID):
            return "shard \(shardID) is already loaded"
        case .shardNotFound(let shardID):
            return "shard \(shardID) is not loaded"
        case .shardHasOpenSequences(let shardID, let count):
            return "shard \(shardID) has \(count) open sequences"
        case .sequenceAlreadyOpen(let shardID, let sequenceID):
            return "sequence \(sequenceID) is already open on shard \(shardID)"
        case .sequenceNotFound(let shardID, let sequenceID):
            return "sequence \(sequenceID) is not open on shard \(shardID)"
        case .sequenceAlreadyPrefilled(let shardID, let sequenceID):
            return "sequence \(sequenceID) is already prefilled on shard \(shardID)"
        case .sequenceNotPrefilled(let shardID, let sequenceID):
            return "sequence \(sequenceID) is not prefilled on shard \(shardID)"
        case .invalidPosition(let shardID, let sequenceID, let got, let expected):
            return "sequence \(sequenceID) on shard \(shardID) received position \(got); expected \(expected)"
        case .cachePositionMismatch(let shardID, let sequenceID, let got, let expected):
            return "sequence \(sequenceID) on shard \(shardID) cache is at position \(got); expected \(expected)"
        case .unsupportedInputKind(let kind):
            return "unsupported forward input kind \(kind)"
        case .inputKindMismatch(let shardID, let got, let expected):
            return "shard \(shardID) requires input kind \(expected), got \(got)"
        }
    }
}

private final class PersistentSequenceState {
    let cache: any CheckpointShardSequenceCache
    var nextPosition = 0
    var prefilled = false
    var completedMutation: PersistentCompletedMutation?

    init(cache: any CheckpointShardSequenceCache) {
        self.cache = cache
    }
}

private struct PersistentCompletedMutation {
    let operation: String
    let position: UInt64
    let inputKind: String
    let inputShape: [Int]
    let inputDType: WireTensor.ElementType
    let inputDigest: Data
    let result: PersistentForwardResult

    init(
        operation: String,
        request: PersistentForwardRequest,
        result: PersistentForwardResult
    ) {
        self.operation = operation
        self.position = request.position
        self.inputKind = request.inputKind
        self.inputShape = request.input.shape
        self.inputDType = request.input.dtype
        self.inputDigest = Data(SHA256.hash(data: request.input.data))
        self.result = result
    }

    func matches(operation: String, request: PersistentForwardRequest) -> Bool {
        self.operation == operation
            && position == request.position
            && inputKind == request.inputKind
            && inputShape == request.input.shape
            && inputDType == request.input.dtype
            && inputDigest == Data(SHA256.hash(data: request.input.data))
    }
}

private final class PersistentLoadedShard {
    let modelID: String
    let modelType: String
    let layerRange: Range<Int>
    let ownsInput: Bool
    let ownsOutput: Bool
    let stage: any CheckpointShardStage
    let loadedMemory: StageMemory
    var sequences = [String: PersistentSequenceState]()
    var forwardCount = 0

    init(
        modelID: String,
        modelType: String,
        layerRange: Range<Int>,
        ownsInput: Bool,
        ownsOutput: Bool,
        stage: any CheckpointShardStage,
        loadedMemory: StageMemory
    ) {
        self.modelID = modelID
        self.modelType = modelType
        self.layerRange = layerRange
        self.ownsInput = ownsInput
        self.ownsOutput = ownsOutput
        self.stage = stage
        self.loadedMemory = loadedMemory
    }
}

final class PersistentShardService {
    private let configuration: CheckpointShardRuntimeConfiguration
    private var shards = [String: PersistentLoadedShard]()
    private var loadCount = 0
    private var forwardCount = 0

    init(configuration: CheckpointShardRuntimeConfiguration) {
        self.configuration = configuration
    }

    func handle(_ request: PersistentWorkerRequest) async throws -> PersistentWorkerResult {
        guard !request.requestID.isEmpty else {
            throw PersistentWorkerError.invalidRequest("requestID is empty")
        }

        switch request.command {
        case "health":
            return PersistentWorkerResult(status: "ok")

        case "loadShard":
            guard let load = request.loadShard else {
                throw PersistentWorkerError.invalidRequest("loadShard payload is missing")
            }
            return PersistentWorkerResult(shard: try await loadShard(load))

        case "unloadShard":
            guard let shard = request.shard else {
                throw PersistentWorkerError.invalidRequest("shard payload is missing")
            }
            try unloadShard(shard.shardID)
            return PersistentWorkerResult(state: state())

        case "openSequence":
            guard let sequence = request.sequence else {
                throw PersistentWorkerError.invalidRequest("sequence payload is missing")
            }
            try openSequence(sequence)
            return PersistentWorkerResult(state: state())

        case "closeSequence":
            guard let sequence = request.sequence else {
                throw PersistentWorkerError.invalidRequest("sequence payload is missing")
            }
            try closeSequence(sequence)
            return PersistentWorkerResult(state: state())

        case "forward":
            guard let forward = request.forward else {
                throw PersistentWorkerError.invalidRequest("forward payload is missing")
            }
            return PersistentWorkerResult(forward: try forwardRequest(forward, operation: "forward"))

        case "prefill", "decode":
            guard let forward = request.forward else {
                throw PersistentWorkerError.invalidRequest("\(request.command) payload is missing")
            }
            return PersistentWorkerResult(
                forward: try forwardRequest(forward, operation: request.command)
            )

        case "state":
            return PersistentWorkerResult(state: state())

        case "shutdown":
            shards.removeAll()
            Memory.clearCache()
            return PersistentWorkerResult(state: state(), shutdown: true)

        default:
            throw PersistentWorkerError.invalidRequest("unknown command \(request.command)")
        }
    }

    private func loadShard(
        _ request: PersistentLoadShardRequest
    ) async throws -> PersistentShardSnapshot {
        guard !request.shardID.isEmpty else {
            throw PersistentWorkerError.invalidRequest("shardID is empty")
        }
        guard shards[request.shardID] == nil else {
            throw PersistentWorkerError.shardAlreadyLoaded(request.shardID)
        }
        guard request.layerStart >= 0, request.layerEnd > request.layerStart else {
            throw PersistentWorkerError.invalidRequest("layer range is invalid")
        }

        let checkpoint = try await CheckpointShardRuntime.resolveCheckpoint(modelID: request.modelID)
        let adapter = try configuration.adapterRegistry.adapter(for: checkpoint.modelType)
        let layerCount = try adapter.layerCount(checkpoint: checkpoint)
        let range = request.layerStart ..< request.layerEnd
        guard range.upperBound <= layerCount else {
            throw CheckpointShardError.invalidRange(range, layerCount)
        }

        Memory.clearCache()
        Memory.peakMemory = 0
        defer { Memory.clearCache() }
        let stage = try await adapter.loadStage(
            checkpoint: checkpoint,
            request: CheckpointStageRequest(
                layerRange: range,
                ownsInput: request.ownsInput,
                ownsOutput: request.ownsOutput
            )
        )
        Memory.clearCache()
        let loadedMemory = CheckpointMemory.snapshot()
        let loaded = PersistentLoadedShard(
            modelID: request.modelID,
            modelType: checkpoint.modelType,
            layerRange: range,
            ownsInput: request.ownsInput,
            ownsOutput: request.ownsOutput,
            stage: stage,
            loadedMemory: loadedMemory
        )
        shards[request.shardID] = loaded
        loadCount += 1
        return snapshot(shardID: request.shardID, shard: loaded)
    }

    private func unloadShard(_ shardID: String) throws {
        guard let openSequenceCount = shards[shardID]?.sequences.count else {
            throw PersistentWorkerError.shardNotFound(shardID)
        }
        guard openSequenceCount == 0 else {
            throw PersistentWorkerError.shardHasOpenSequences(shardID, openSequenceCount)
        }
        shards.removeValue(forKey: shardID)
        Memory.clearCache()
    }

    private func openSequence(_ request: PersistentSequenceRequest) throws {
        guard !request.sequenceID.isEmpty else {
            throw PersistentWorkerError.invalidRequest("sequenceID is empty")
        }
        guard let shard = shards[request.shardID] else {
            throw PersistentWorkerError.shardNotFound(request.shardID)
        }
        guard shard.sequences[request.sequenceID] == nil else {
            throw PersistentWorkerError.sequenceAlreadyOpen(request.shardID, request.sequenceID)
        }
        shard.sequences[request.sequenceID] = PersistentSequenceState(
            cache: shard.stage.makeSequenceCache()
        )
    }

    private func closeSequence(_ request: PersistentSequenceRequest) throws {
        guard let shard = shards[request.shardID] else {
            throw PersistentWorkerError.shardNotFound(request.shardID)
        }
        guard shard.sequences.removeValue(forKey: request.sequenceID) != nil else {
            throw PersistentWorkerError.sequenceNotFound(request.shardID, request.sequenceID)
        }
        Memory.clearCache()
    }

    private func forwardRequest(
        _ request: PersistentForwardRequest,
        operation: String
    ) throws -> PersistentForwardResult {
        guard let shard = shards[request.shardID] else {
            throw PersistentWorkerError.shardNotFound(request.shardID)
        }
        guard let sequence = shard.sequences[request.sequenceID] else {
            throw PersistentWorkerError.sequenceNotFound(request.shardID, request.sequenceID)
        }
        if operation != "forward",
           let completed = sequence.completedMutation,
           completed.matches(operation: operation, request: request)
        {
            return completed.result
        }

        let input: MLXArray
        let inputLength: Int
        switch request.inputKind {
        case "tokens":
            guard shard.ownsInput else {
                throw PersistentWorkerError.inputKindMismatch(
                    request.shardID,
                    got: request.inputKind,
                    expected: "hidden"
                )
            }
            try validateTokenInput(request.input, metadata: shard.stage.inputMetadata)
            input = try request.input.materialize()
            inputLength = request.input.shape[1]
        case "hidden":
            guard !shard.ownsInput else {
                throw PersistentWorkerError.inputKindMismatch(
                    request.shardID,
                    got: request.inputKind,
                    expected: "tokens"
                )
            }
            try validateHiddenInput(request.input, metadata: shard.stage.inputMetadata)
            input = try request.input.materialize()
            inputLength = request.input.shape[1]
        default:
            throw PersistentWorkerError.unsupportedInputKind(request.inputKind)
        }

        let start = DispatchTime.now().uptimeNanoseconds
        let output: MLXArray
        switch operation {
        case "forward":
            output = try request.inputKind == "tokens"
                ? shard.stage.forward(tokens: input)
                : shard.stage.forward(hidden: input)
        case "prefill":
            guard !sequence.prefilled else {
                throw PersistentWorkerError.sequenceAlreadyPrefilled(
                    request.shardID,
                    request.sequenceID
                )
            }
            try validatePosition(request, sequence: sequence, expected: 0)
            output = try request.inputKind == "tokens"
                ? shard.stage.prefill(tokens: input, cache: sequence.cache)
                : shard.stage.prefill(hidden: input, cache: sequence.cache)
            try advance(
                request,
                sequence: sequence,
                inputLength: inputLength
            )
            sequence.prefilled = true
        case "decode":
            guard sequence.prefilled else {
                throw PersistentWorkerError.sequenceNotPrefilled(
                    request.shardID,
                    request.sequenceID
                )
            }
            guard inputLength == 1 else {
                throw PersistentWorkerError.invalidRequest(
                    "decode input must contain exactly one position, got \(inputLength)"
                )
            }
            try validatePosition(
                request,
                sequence: sequence,
                expected: sequence.nextPosition
            )
            output = try request.inputKind == "tokens"
                ? shard.stage.decode(tokens: input, cache: sequence.cache)
                : shard.stage.decode(hidden: input, cache: sequence.cache)
            try advance(
                request,
                sequence: sequence,
                inputLength: inputLength
            )
        default:
            throw PersistentWorkerError.invalidRequest("unknown inference operation \(operation)")
        }
        let tensor = WireTensor(output)
        let elapsed = DispatchTime.now().uptimeNanoseconds - start
        shard.forwardCount += 1
        forwardCount += 1
        let result = PersistentForwardResult(
            shardID: request.shardID,
            sequenceID: request.sequenceID,
            operation: operation,
            position: request.position,
            nextPosition: UInt64(sequence.nextPosition),
            output: tensor,
            computeMicros: elapsed / 1_000,
            kvCacheBytes: sequence.cache.memoryBytes,
            memory: CheckpointMemory.snapshot()
        )
        if operation == "prefill" || operation == "decode" {
            sequence.completedMutation = PersistentCompletedMutation(
                operation: operation,
                request: request,
                result: result
            )
        }
        return result
    }

    private func validatePosition(
        _ request: PersistentForwardRequest,
        sequence: PersistentSequenceState,
        expected: Int
    ) throws {
        guard request.position == UInt64(expected) else {
            throw PersistentWorkerError.invalidPosition(
                request.shardID,
                request.sequenceID,
                got: request.position,
                expected: UInt64(expected)
            )
        }
        guard sequence.cache.position == expected else {
            throw PersistentWorkerError.cachePositionMismatch(
                request.shardID,
                request.sequenceID,
                got: sequence.cache.position,
                expected: expected
            )
        }
    }

    private func advance(
        _ request: PersistentForwardRequest,
        sequence: PersistentSequenceState,
        inputLength: Int
    ) throws {
        let (nextPosition, overflow) = sequence.nextPosition.addingReportingOverflow(inputLength)
        guard !overflow else {
            throw PersistentWorkerError.invalidRequest("sequence position overflow")
        }
        guard sequence.cache.position == nextPosition else {
            throw PersistentWorkerError.cachePositionMismatch(
                request.shardID,
                request.sequenceID,
                got: sequence.cache.position,
                expected: nextPosition
            )
        }
        sequence.nextPosition = nextPosition
    }

    private func validateTokenInput(
        _ input: WireTensor,
        metadata: CheckpointStageInputMetadata
    ) throws {
        try input.validate()
        guard input.dtype == metadata.tokenDType else {
            throw PersistentWorkerError.invalidRequest(
                "token tensor dtype must be \(metadata.tokenDType.rawValue), got \(input.dtype.rawValue)"
            )
        }
        guard input.shape.count == metadata.tokenRank,
              input.shape.allSatisfy({ $0 > 0 }),
              input.shape.first == 1
        else {
            throw PersistentWorkerError.invalidRequest(
                "token tensor shape must have batch size 1 and \(metadata.tokenRank) positive dimensions"
            )
        }

        let invalidToken = input.data.withUnsafeBytes { bytes -> Int32? in
            for offset in stride(from: 0, to: bytes.count, by: MemoryLayout<Int32>.size) {
                let token = bytes.loadUnaligned(fromByteOffset: offset, as: Int32.self)
                if token < 0 || Int(token) >= metadata.vocabularySize {
                    return token
                }
            }
            return nil
        }
        if let invalidToken {
            throw PersistentWorkerError.invalidRequest(
                "token ID \(invalidToken) is outside vocabulary size \(metadata.vocabularySize)"
            )
        }
    }

    private func validateHiddenInput(
        _ input: WireTensor,
        metadata: CheckpointStageInputMetadata
    ) throws {
        try input.validate()
        guard metadata.hiddenDTypes.contains(input.dtype) else {
            let expected = metadata.hiddenDTypes.map(\.rawValue).sorted().joined(separator: ", ")
            throw PersistentWorkerError.invalidRequest(
                "hidden tensor dtype must be one of [\(expected)], got \(input.dtype.rawValue)"
            )
        }
        guard input.shape.count == metadata.hiddenRank,
              input.shape.allSatisfy({ $0 > 0 }),
              input.shape.first == 1,
              input.shape.last == metadata.hiddenSize
        else {
            throw PersistentWorkerError.invalidRequest(
                "hidden tensor shape must have batch size 1, \(metadata.hiddenRank) positive dimensions, and width \(metadata.hiddenSize)"
            )
        }
    }

    private func state() -> PersistentWorkerState {
        PersistentWorkerState(
            loadedShards: shards.map { snapshot(shardID: $0.key, shard: $0.value) }
                .sorted { $0.shardID < $1.shardID },
            loadCount: loadCount,
            forwardCount: forwardCount,
            kvCacheBytes: shards.values.reduce(0) { $0 + cacheBytes(shard: $1) },
            memory: CheckpointMemory.snapshot()
        )
    }

    private func snapshot(
        shardID: String,
        shard: PersistentLoadedShard
    ) -> PersistentShardSnapshot {
        PersistentShardSnapshot(
            shardID: shardID,
            modelID: shard.modelID,
            modelType: shard.modelType,
            layerStart: shard.layerRange.lowerBound,
            layerEnd: shard.layerRange.upperBound,
            ownsInput: shard.ownsInput,
            ownsOutput: shard.ownsOutput,
            weightKeyCount: shard.stage.weightKeyCount,
            openSequenceCount: shard.sequences.count,
            forwardCount: shard.forwardCount,
            kvCacheBytes: cacheBytes(shard: shard),
            loadedMemory: shard.loadedMemory
        )
    }

    private func cacheBytes(shard: PersistentLoadedShard) -> Int {
        shard.sequences.values.reduce(0) { $0 + $1.cache.memoryBytes }
    }
}

enum PersistentShardWorker {
    static func serveStdio() async {
        let service = PersistentShardService(configuration: WorkerCheckpointShards.configuration)
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        while let line = readLine(strippingNewline: true) {
            var requestID = "unknown"
            var shouldShutdown = false
            let response: PersistentWorkerResponse
            let data = Data(line.utf8)
            if let envelope = try? decoder.decode(PersistentWorkerEnvelope.self, from: data) {
                requestID = envelope.requestID
            }
            do {
                let request = try decoder.decode(
                    PersistentWorkerRequest.self,
                    from: data
                )
                let result = try await service.handle(request)
                shouldShutdown = result.shutdown == true
                response = .success(requestID: requestID, result: result)
            } catch {
                response = .failure(requestID: requestID, error: error)
            }

            do {
                var data = try encoder.encode(response)
                data.append(0x0A)
                FileHandle.standardOutput.write(data)
            } catch {
                FileHandle.standardError.write(Data("encode response: \(error)\n".utf8))
                return
            }
            if shouldShutdown {
                return
            }
        }
    }
}
