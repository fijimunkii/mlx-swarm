import Foundation
import CryptoKit
import MLX
import Tokenizers

struct PersistentWorkerRequest: Codable {
    let requestID: String
    let command: String
    let deadlineUnixMillis: Int64?
    let loadShard: PersistentLoadShardRequest?
    let shard: PersistentShardRequest?
    let sequence: PersistentSequenceRequest?
    let forward: PersistentForwardRequest?
    let model: PersistentModelRequest?
    let text: PersistentTextRequest?
}

private struct PersistentWorkerEnvelope: Decodable {
    let requestID: String
}

struct PersistentLoadShardRequest: Codable {
    let modelID: String
    let shardID: String
    let checkpointFingerprint: String?
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
    let ownerID: String?
}

struct PersistentForwardRequest: Codable {
    let shardID: String
    let sequenceID: String
    let position: UInt64
    let inputKind: String
    let input: WireTensor
    let returnSampledToken: Bool?
}

struct PersistentModelRequest: Codable {
    let modelID: String
}

struct PersistentTextRequest: Codable {
    let modelID: String
    let text: String?
    let tokenIDs: [Int32]?
    let addSpecialTokens: Bool?
    let skipSpecialTokens: Bool?
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
    var model: PersistentModelResult?
    var text: PersistentTextResult?
    var state: PersistentWorkerState?
    var shutdown: Bool?
}

struct PersistentModelResult: Codable {
    let modelID: String
    let modelType: String
    let layerCount: Int
    let checkpointFingerprint: String
    let checkpointBytes: UInt64
}

struct PersistentTextResult: Codable {
    let modelID: String
    let tokenIDs: [Int32]?
    let text: String?
    let eosTokenID: Int32?
}

struct PersistentForwardResult: Codable {
    let shardID: String
    let sequenceID: String
    let operation: String
    let position: UInt64
    let nextPosition: UInt64
    let output: WireTensor
    let sampledTokenID: Int32?
    let computeMicros: UInt64
    let kvCacheBytes: Int
    let memory: StageMemory
}

struct PersistentShardSnapshot: Codable {
    let shardID: String
    let modelID: String
    let modelType: String
    let checkpointFingerprint: String
    let layerStart: Int
    let layerEnd: Int
    let ownsInput: Bool
    let ownsOutput: Bool
    let weightKeyCount: Int
    let openSequenceCount: Int
    let maxOpenSequenceCount: Int
    let forwardCount: Int
    let kvCacheBytes: Int
    let retainedBytes: Int
    let loadedMemory: StageMemory
}

struct PersistentWorkerState: Codable {
    let loadedShards: [PersistentShardSnapshot]
    let loadCount: Int
    let forwardCount: Int
    let kvCacheBytes: Int
    let retainedBytes: Int
    let retainedByteBudget: Int
    let maxOpenSequencesPerShard: Int
    let physicalMemoryBytes: UInt64
    let mlxMemoryLimitBytes: Int
    let mlxCacheLimitBytes: Int
    let memory: StageMemory
}

private enum PersistentWorkerError: LocalizedError {
    case invalidRequest(String)
    case shardAlreadyLoaded(String)
    case shardNotFound(String)
    case shardHasOpenSequences(String, Int)
    case sequenceAlreadyOpen(String, String)
    case openSequenceLimit(String, Int)
    case sequenceNotFound(String, String)
    case sequenceOwnerMismatch(String, String)
    case sequenceAlreadyPrefilled(String, String)
    case sequenceNotPrefilled(String, String)
    case invalidPosition(String, String, got: UInt64, expected: UInt64)
    case cachePositionMismatch(String, String, got: Int, expected: Int)
    case sequenceLengthLimit(String, String, got: Int, maximum: Int)
    case retainedByteBudget(got: Int, maximum: Int)
    case inferenceDeadlineRequired(String)
    case inferenceDeadlineExceeded(String, deadline: Int64, now: Int64)
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
        case .openSequenceLimit(let shardID, let maximum):
            return "shard \(shardID) reached its open sequence limit of \(maximum)"
        case .sequenceNotFound(let shardID, let sequenceID):
            return "sequence \(sequenceID) is not open on shard \(shardID)"
        case .sequenceOwnerMismatch(let shardID, let sequenceID):
            return "sequence \(sequenceID) on shard \(shardID) is owned by another request"
        case .sequenceAlreadyPrefilled(let shardID, let sequenceID):
            return "sequence \(sequenceID) is already prefilled on shard \(shardID)"
        case .sequenceNotPrefilled(let shardID, let sequenceID):
            return "sequence \(sequenceID) is not prefilled on shard \(shardID)"
        case .invalidPosition(let shardID, let sequenceID, let got, let expected):
            return "sequence \(sequenceID) on shard \(shardID) received position \(got); expected \(expected)"
        case .cachePositionMismatch(let shardID, let sequenceID, let got, let expected):
            return "sequence \(sequenceID) on shard \(shardID) cache is at position \(got); expected \(expected)"
        case .sequenceLengthLimit(let shardID, let sequenceID, let got, let maximum):
            return "sequence \(sequenceID) on shard \(shardID) would reach position \(got); maximum is \(maximum)"
        case .retainedByteBudget(let got, let maximum):
            return "retained sequence state would use \(got) bytes; budget is \(maximum)"
        case .inferenceDeadlineRequired(let operation):
            return "\(operation) request requires deadlineUnixMillis"
        case .inferenceDeadlineExceeded(let operation, let deadline, let now):
            return "\(operation) request deadline \(deadline) expired at \(now)"
        case .unsupportedInputKind(let kind):
            return "unsupported forward input kind \(kind)"
        case .inputKindMismatch(let shardID, let got, let expected):
            return "shard \(shardID) requires input kind \(expected), got \(got)"
        }
    }
}

private final class PersistentSequenceState {
    let cache: any CheckpointShardSequenceCache
    let ownerID: String?
    var nextPosition = 0
    var prefilled = false
    var completedMutation: PersistentCompletedMutation?

    init(cache: any CheckpointShardSequenceCache, ownerID: String?) {
        self.cache = cache
        self.ownerID = ownerID
    }
}

private enum PersistentMutationOutcome {
    case success(PersistentForwardResult)
    case failure(String)
}

private struct PersistentCompletedMutation {
    let operation: String
    let position: UInt64
    let inputKind: String
    let inputShape: [Int]
    let inputDType: WireTensor.ElementType
    let inputDigest: Data
    let returnSampledToken: Bool
    let outcome: PersistentMutationOutcome

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
        self.returnSampledToken = request.returnSampledToken == true
        self.outcome = .success(result)
    }

    init(
        operation: String,
        request: PersistentForwardRequest,
        failureReason: String
    ) {
        self.operation = operation
        self.position = request.position
        self.inputKind = request.inputKind
        self.inputShape = request.input.shape
        self.inputDType = request.input.dtype
        self.inputDigest = Data(SHA256.hash(data: request.input.data))
        self.returnSampledToken = request.returnSampledToken == true
        self.outcome = .failure(failureReason)
    }

    var isFailure: Bool {
        if case .failure = outcome {
            return true
        }
        return false
    }

    var result: PersistentForwardResult? {
        if case .success(let result) = outcome {
            return result
        }
        return nil
    }

    func replay() throws -> PersistentForwardResult {
        switch outcome {
        case .success(let result):
            return result
        case .failure(let reason):
            throw PersistentWorkerError.invalidRequest(reason)
        }
    }

    func matches(operation: String, request: PersistentForwardRequest) -> Bool {
        self.operation == operation
            && position == request.position
            && inputKind == request.inputKind
            && inputShape == request.input.shape
            && inputDType == request.input.dtype
            && returnSampledToken == (request.returnSampledToken == true)
            && inputDigest == Data(SHA256.hash(data: request.input.data))
    }
}

private final class PersistentLoadedShard {
    let modelID: String
    let modelType: String
    let checkpointFingerprint: String
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
        checkpointFingerprint: String,
        layerRange: Range<Int>,
        ownsInput: Bool,
        ownsOutput: Bool,
        stage: any CheckpointShardStage,
        loadedMemory: StageMemory
    ) {
        self.modelID = modelID
        self.modelType = modelType
        self.checkpointFingerprint = checkpointFingerprint
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
    private var tokenizers = [String: any Tokenizer]()
    private var checkpoints = [String: ResolvedCheckpoint]()
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

        case "modelInfo":
            guard let model = request.model else {
                throw PersistentWorkerError.invalidRequest("modelInfo payload is missing")
            }
            return PersistentWorkerResult(model: try await modelInfo(model))

        case "tokenize":
            guard let text = request.text else {
                throw PersistentWorkerError.invalidRequest("tokenize payload is missing")
            }
            return PersistentWorkerResult(text: try await tokenize(text))

        case "detokenize":
            guard let text = request.text else {
                throw PersistentWorkerError.invalidRequest("detokenize payload is missing")
            }
            return PersistentWorkerResult(text: try await detokenize(text))

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
            try requireActiveDeadline(request.deadlineUnixMillis, operation: request.command)
            let result = try forwardRequest(forward, operation: "forward")
            try requireActiveDeadline(request.deadlineUnixMillis, operation: request.command)
            return PersistentWorkerResult(forward: result)

        case "prefill", "decode":
            guard let forward = request.forward else {
                throw PersistentWorkerError.invalidRequest("\(request.command) payload is missing")
            }
            try requireActiveDeadline(request.deadlineUnixMillis, operation: request.command)
            let result = try forwardRequest(forward, operation: request.command)
            try requireActiveDeadline(request.deadlineUnixMillis, operation: request.command)
            return PersistentWorkerResult(forward: result)

        case "state":
            return PersistentWorkerResult(state: state())

        case "shutdown":
            shards.removeAll()
            tokenizers.removeAll()
            checkpoints.removeAll()
            Memory.clearCache()
            return PersistentWorkerResult(state: state(), shutdown: true)

        default:
            throw PersistentWorkerError.invalidRequest("unknown command \(request.command)")
        }
    }

    private func requireActiveDeadline(
        _ deadlineUnixMillis: Int64?,
        operation: String
    ) throws {
        guard let deadlineUnixMillis else {
            throw PersistentWorkerError.inferenceDeadlineRequired(operation)
        }
        let now = Int64((Date().timeIntervalSince1970 * 1_000).rounded(.down))
        guard now < deadlineUnixMillis else {
            throw PersistentWorkerError.inferenceDeadlineExceeded(
                operation,
                deadline: deadlineUnixMillis,
                now: now
            )
        }
    }

    private func modelInfo(_ request: PersistentModelRequest) async throws -> PersistentModelResult {
        guard !request.modelID.isEmpty else {
            throw PersistentWorkerError.invalidRequest("modelID is empty")
        }
        let checkpoint = try await checkpoint(modelID: request.modelID)
        let adapter = try configuration.adapterRegistry.adapter(for: checkpoint.modelType)
        return PersistentModelResult(
            modelID: request.modelID,
            modelType: checkpoint.modelType,
            layerCount: try adapter.layerCount(checkpoint: checkpoint),
            checkpointFingerprint: checkpoint.fingerprint,
            checkpointBytes: checkpoint.checkpointBytes
        )
    }

    private func tokenize(_ request: PersistentTextRequest) async throws -> PersistentTextResult {
        guard !request.modelID.isEmpty else {
            throw PersistentWorkerError.invalidRequest("modelID is empty")
        }
        guard let text = request.text else {
            throw PersistentWorkerError.invalidRequest("tokenize text is missing")
        }
        guard request.tokenIDs == nil else {
            throw PersistentWorkerError.invalidRequest("tokenize tokenIDs must be omitted")
        }
        try requireLoadedInputModel(request.modelID)
        let tokenizer = try await tokenizer(modelID: request.modelID)
        let encoded = tokenizer.encode(
            text: text,
            addSpecialTokens: request.addSpecialTokens ?? true
        )
        let tokenIDs = try int32Tokens(encoded)
        return PersistentTextResult(
            modelID: request.modelID,
            tokenIDs: tokenIDs,
            text: nil,
            eosTokenID: try int32Token(tokenizer.eosTokenId)
        )
    }

    private func detokenize(_ request: PersistentTextRequest) async throws -> PersistentTextResult {
        guard !request.modelID.isEmpty else {
            throw PersistentWorkerError.invalidRequest("modelID is empty")
        }
        guard request.text == nil else {
            throw PersistentWorkerError.invalidRequest("detokenize text must be omitted")
        }
        guard let tokenIDs = request.tokenIDs else {
            throw PersistentWorkerError.invalidRequest("detokenize tokenIDs are missing")
        }
        try requireLoadedInputModel(request.modelID)
        let tokenizer = try await tokenizer(modelID: request.modelID)
        return PersistentTextResult(
            modelID: request.modelID,
            tokenIDs: nil,
            text: tokenizer.decode(
                tokens: tokenIDs.map(Int.init),
                skipSpecialTokens: request.skipSpecialTokens ?? true
            ),
            eosTokenID: try int32Token(tokenizer.eosTokenId)
        )
    }

    private func tokenizer(modelID: String) async throws -> any Tokenizer {
        if let tokenizer = tokenizers[modelID] {
            return tokenizer
        }
        let checkpoint = try await checkpoint(modelID: modelID)
        let tokenizer = try await AutoTokenizer.from(modelFolder: checkpoint.directory)
        tokenizers[modelID] = tokenizer
        return tokenizer
    }

    private func checkpoint(modelID: String) async throws -> ResolvedCheckpoint {
        if let checkpoint = checkpoints[modelID] {
            return checkpoint
        }
        let resolved = try await CheckpointShardRuntime.resolveCheckpoint(modelID: modelID)
        checkpoints[modelID] = resolved
        return resolved
    }

    private func requireLoadedInputModel(_ modelID: String) throws {
        guard shards.values.contains(where: { $0.modelID == modelID && $0.ownsInput }) else {
            throw PersistentWorkerError.invalidRequest(
                "model \(modelID) has no loaded input-owning shard"
            )
        }
    }

    private func int32Tokens(_ tokens: [Int]) throws -> [Int32] {
        try tokens.map { token in
            guard let value = Int32(exactly: token), value >= 0 else {
                throw PersistentWorkerError.invalidRequest("token ID \(token) is outside int32")
            }
            return value
        }
    }

    private func int32Token(_ token: Int?) throws -> Int32? {
        guard let token else { return nil }
        guard let value = Int32(exactly: token), value >= 0 else {
            throw PersistentWorkerError.invalidRequest("token ID \(token) is outside int32")
        }
        return value
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

        let checkpoint = try await checkpoint(modelID: request.modelID)
        if let expectedFingerprint = request.checkpointFingerprint,
           expectedFingerprint != checkpoint.fingerprint
        {
            throw PersistentWorkerError.invalidRequest(
                "checkpoint fingerprint \(checkpoint.fingerprint) does not match expected \(expectedFingerprint)"
            )
        }
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
            checkpointFingerprint: checkpoint.fingerprint,
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
        guard let openSequenceCount = shards[shardID]?.sequences.count,
              let modelID = shards[shardID]?.modelID
        else {
            throw PersistentWorkerError.shardNotFound(shardID)
        }
        guard openSequenceCount == 0 else {
            throw PersistentWorkerError.shardHasOpenSequences(shardID, openSequenceCount)
        }
        shards.removeValue(forKey: shardID)
        if !shards.values.contains(where: { $0.modelID == modelID }) {
            tokenizers.removeValue(forKey: modelID)
        }
        Memory.clearCache()
    }

    private func openSequence(_ request: PersistentSequenceRequest) throws {
        guard !request.sequenceID.isEmpty else {
            throw PersistentWorkerError.invalidRequest("sequenceID is empty")
        }
        guard let shard = shards[request.shardID] else {
            throw PersistentWorkerError.shardNotFound(request.shardID)
        }
        if let existing = shard.sequences[request.sequenceID] {
            if let ownerID = request.ownerID, existing.ownerID == ownerID {
                return
            }
            throw PersistentWorkerError.sequenceAlreadyOpen(request.shardID, request.sequenceID)
        }
        guard shard.sequences.count < configuration.maxOpenSequencesPerShard else {
            throw PersistentWorkerError.openSequenceLimit(
                request.shardID,
                configuration.maxOpenSequencesPerShard
            )
        }
        shard.sequences[request.sequenceID] = PersistentSequenceState(
            cache: shard.stage.makeSequenceCache(),
            ownerID: request.ownerID
        )
    }

    private func closeSequence(_ request: PersistentSequenceRequest) throws {
        guard let shard = shards[request.shardID] else {
            throw PersistentWorkerError.shardNotFound(request.shardID)
        }
        guard let sequence = shard.sequences[request.sequenceID] else {
            throw PersistentWorkerError.sequenceNotFound(request.shardID, request.sequenceID)
        }
        if let ownerID = request.ownerID, sequence.ownerID != ownerID {
            throw PersistentWorkerError.sequenceOwnerMismatch(request.shardID, request.sequenceID)
        }
        shard.sequences.removeValue(forKey: request.sequenceID)
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
        if request.returnSampledToken == true && !shard.ownsOutput {
            throw PersistentWorkerError.invalidRequest(
                "returnSampledToken requires an output-owning shard"
            )
        }
        if operation != "forward", let completed = sequence.completedMutation {
            if completed.matches(operation: operation, request: request) {
                return try completed.replay()
            }
            if completed.isFailure {
                throw PersistentWorkerError.invalidRequest(
                    "sequence \(request.sequenceID) cannot continue after failed "
                        + "\(completed.operation); close it before retrying"
                )
            }
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
            try validateMutationBudget(
                request: request,
                shard: shard,
                sequence: sequence,
                inputLength: inputLength
            )
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
            try validateMutationBudget(
                request: request,
                shard: shard,
                sequence: sequence,
                inputLength: inputLength
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
        let tensor: WireTensor
        let sampledTokenID: Int32?
        if request.returnSampledToken == true {
            let finalLogits = output[0..., -1, 0...]
            guard !isNaN(finalLogits).any().item(Bool.self) else {
                throw recordMutationFailure(
                    "cannot sample logits containing NaN",
                    operation: operation,
                    request: request,
                    sequence: sequence
                )
            }
            let token = argMax(finalLogits, axis: -1).item(Int32.self)
            guard token >= 0 else {
                throw recordMutationFailure(
                    "sampled token ID must be non-negative, got \(token)",
                    operation: operation,
                    request: request,
                    sequence: sequence
                )
            }
            tensor = .empty
            sampledTokenID = token
        } else {
            tensor = WireTensor(output)
            sampledTokenID = nil
        }
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
            sampledTokenID: sampledTokenID,
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

    private func recordMutationFailure(
        _ reason: String,
        operation: String,
        request: PersistentForwardRequest,
        sequence: PersistentSequenceState
    ) -> PersistentWorkerError {
        if operation == "prefill" || operation == "decode" {
            sequence.completedMutation = PersistentCompletedMutation(
                operation: operation,
                request: request,
                failureReason: reason
            )
        }
        return .invalidRequest(reason)
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
        let nextPosition = try projectedPosition(sequence: sequence, inputLength: inputLength)
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

    private func validateMutationBudget(
        request: PersistentForwardRequest,
        shard: PersistentLoadedShard,
        sequence: PersistentSequenceState,
        inputLength: Int
    ) throws {
        let nextPosition = try projectedPosition(sequence: sequence, inputLength: inputLength)
        let maximum = shard.stage.inputMetadata.maximumSequenceLength
        guard nextPosition <= maximum else {
            throw PersistentWorkerError.sequenceLengthLimit(
                request.shardID,
                request.sequenceID,
                got: nextPosition,
                maximum: maximum
            )
        }

        let currentSequenceBytes = retainedBytes(sequence: sequence)
        let currentWorkerBytes = retainedBytes()
        let estimatedCacheBytes = try sequence.cache.estimatedMemoryBytes(at: nextPosition)
        let estimatedOutputBytes = try shard.stage.estimatedOutputBytes(inputLength: inputLength)
        let (estimatedSequenceBytes, sequenceOverflow) = estimatedCacheBytes
            .addingReportingOverflow(estimatedOutputBytes)
        let workerWithoutSequence = max(0, currentWorkerBytes - currentSequenceBytes)
        let (estimatedWorkerBytes, workerOverflow) = workerWithoutSequence
            .addingReportingOverflow(estimatedSequenceBytes)
        guard !sequenceOverflow, !workerOverflow,
              estimatedWorkerBytes <= configuration.workerBudgetBytes
        else {
            throw PersistentWorkerError.retainedByteBudget(
                got: sequenceOverflow || workerOverflow ? Int.max : estimatedWorkerBytes,
                maximum: configuration.workerBudgetBytes
            )
        }
    }

    private func projectedPosition(
        sequence: PersistentSequenceState,
        inputLength: Int
    ) throws -> Int {
        let (nextPosition, overflow) = sequence.nextPosition.addingReportingOverflow(inputLength)
        guard !overflow else {
            throw PersistentWorkerError.invalidRequest("sequence position overflow")
        }
        return nextPosition
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
            retainedBytes: retainedBytes(),
            retainedByteBudget: configuration.workerBudgetBytes,
            maxOpenSequencesPerShard: configuration.maxOpenSequencesPerShard,
            physicalMemoryBytes: WorkerRuntimeMemory.physicalMemoryBytes,
            mlxMemoryLimitBytes: Memory.memoryLimit,
            mlxCacheLimitBytes: Memory.cacheLimit,
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
            checkpointFingerprint: shard.checkpointFingerprint,
            layerStart: shard.layerRange.lowerBound,
            layerEnd: shard.layerRange.upperBound,
            ownsInput: shard.ownsInput,
            ownsOutput: shard.ownsOutput,
            weightKeyCount: shard.stage.weightKeyCount,
            openSequenceCount: shard.sequences.count,
            maxOpenSequenceCount: configuration.maxOpenSequencesPerShard,
            forwardCount: shard.forwardCount,
            kvCacheBytes: cacheBytes(shard: shard),
            retainedBytes: retainedBytes(shard: shard),
            loadedMemory: shard.loadedMemory
        )
    }

    private func cacheBytes(shard: PersistentLoadedShard) -> Int {
        shard.sequences.values.reduce(0) { $0 + $1.cache.memoryBytes }
    }

    private func retainedBytes() -> Int {
        shards.values.reduce(0) { total, shard in
            saturatedAdd(total, retainedBytes(shard: shard))
        }
    }

    private func retainedBytes(shard: PersistentLoadedShard) -> Int {
        shard.sequences.values.reduce(0) { total, sequence in
            saturatedAdd(total, retainedBytes(sequence: sequence))
        }
    }

    private func retainedBytes(sequence: PersistentSequenceState) -> Int {
        let replayBytes: Int
        if let result = sequence.completedMutation?.result {
            replayBytes = result.sampledTokenID == nil
                ? result.output.data.count
                : MemoryLayout<Int32>.size
        } else {
            replayBytes = 0
        }
        return saturatedAdd(
            sequence.cache.memoryBytes,
            replayBytes
        )
    }

    private func saturatedAdd(_ left: Int, _ right: Int) -> Int {
        let (result, overflow) = left.addingReportingOverflow(right)
        return overflow ? Int.max : result
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
