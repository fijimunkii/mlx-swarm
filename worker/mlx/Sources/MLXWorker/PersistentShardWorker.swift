import Foundation
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
    let position: UInt64
    let output: WireTensor
    let computeMicros: UInt64
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
    let loadedMemory: StageMemory
}

struct PersistentWorkerState: Codable {
    let loadedShards: [PersistentShardSnapshot]
    let loadCount: Int
    let forwardCount: Int
    let memory: StageMemory
}

private enum PersistentWorkerError: LocalizedError {
    case invalidRequest(String)
    case shardAlreadyLoaded(String)
    case shardNotFound(String)
    case shardHasOpenSequences(String, Int)
    case sequenceAlreadyOpen(String, String)
    case sequenceNotFound(String, String)
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
        case .unsupportedInputKind(let kind):
            return "unsupported forward input kind \(kind)"
        case .inputKindMismatch(let shardID, let got, let expected):
            return "shard \(shardID) requires input kind \(expected), got \(got)"
        }
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
    var sequenceIDs = Set<String>()
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
            return PersistentWorkerResult(forward: try forwardRequest(forward))

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
        guard let openSequenceCount = shards[shardID]?.sequenceIDs.count else {
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
        guard shard.sequenceIDs.insert(request.sequenceID).inserted else {
            throw PersistentWorkerError.sequenceAlreadyOpen(request.shardID, request.sequenceID)
        }
    }

    private func closeSequence(_ request: PersistentSequenceRequest) throws {
        guard let shard = shards[request.shardID] else {
            throw PersistentWorkerError.shardNotFound(request.shardID)
        }
        guard shard.sequenceIDs.remove(request.sequenceID) != nil else {
            throw PersistentWorkerError.sequenceNotFound(request.shardID, request.sequenceID)
        }
    }

    private func forwardRequest(
        _ request: PersistentForwardRequest
    ) throws -> PersistentForwardResult {
        guard let shard = shards[request.shardID] else {
            throw PersistentWorkerError.shardNotFound(request.shardID)
        }
        guard shard.sequenceIDs.contains(request.sequenceID) else {
            throw PersistentWorkerError.sequenceNotFound(request.shardID, request.sequenceID)
        }

        let start = DispatchTime.now().uptimeNanoseconds
        let output: MLXArray
        switch request.inputKind {
        case "tokens":
            guard shard.ownsInput else {
                throw PersistentWorkerError.inputKindMismatch(
                    request.shardID,
                    got: request.inputKind,
                    expected: "hidden"
                )
            }
            output = try shard.stage.forward(tokens: request.input.materialize())
        case "hidden":
            guard !shard.ownsInput else {
                throw PersistentWorkerError.inputKindMismatch(
                    request.shardID,
                    got: request.inputKind,
                    expected: "tokens"
                )
            }
            output = try shard.stage.forward(hidden: request.input.materialize())
        default:
            throw PersistentWorkerError.unsupportedInputKind(request.inputKind)
        }
        let tensor = WireTensor(output)
        let elapsed = DispatchTime.now().uptimeNanoseconds - start
        shard.forwardCount += 1
        forwardCount += 1
        return PersistentForwardResult(
            shardID: request.shardID,
            sequenceID: request.sequenceID,
            position: request.position,
            output: tensor,
            computeMicros: elapsed / 1_000,
            memory: CheckpointMemory.snapshot()
        )
    }

    private func state() -> PersistentWorkerState {
        PersistentWorkerState(
            loadedShards: shards.map { snapshot(shardID: $0.key, shard: $0.value) }
                .sorted { $0.shardID < $1.shardID },
            loadCount: loadCount,
            forwardCount: forwardCount,
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
            openSequenceCount: shard.sequenceIDs.count,
            forwardCount: shard.forwardCount,
            loadedMemory: shard.loadedMemory
        )
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
