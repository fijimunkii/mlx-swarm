import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

private enum WorkerCommand {
    case health
    case capabilities
    case shardSmoke
    case checkpointShardSmoke(modelID: String)
    case shardProduce(path: String)
    case shardFinish(path: String)
    case shardProduceStdio
    case shardFinishStdio
    case checkpointShardProduceStdio
    case checkpointShardFinishStdio
    case generate(prompt: String)

    init(arguments: [String]) throws {
        guard let command = arguments.first else {
            self = .health
            return
        }

        switch command {
        case "health":
            self = .health
        case "capabilities":
            self = .capabilities
        case "shard-smoke":
            self = .shardSmoke
        case "checkpoint-shard-smoke":
            guard arguments.count <= 2 else {
                throw WorkerError.usage("checkpoint-shard-smoke accepts at most one model id")
            }
            self = .checkpointShardSmoke(
                modelID: arguments.count == 2
                    ? arguments[1]
                    : WorkerCheckpointShards.defaultModelID
            )
        case "shard-produce":
            guard arguments.count == 2 else {
                throw WorkerError.usage("shard-produce requires an output path")
            }
            self = .shardProduce(path: arguments[1])
        case "shard-finish":
            guard arguments.count == 2 else {
                throw WorkerError.usage("shard-finish requires an input path")
            }
            self = .shardFinish(path: arguments[1])
        case "shard-produce-stdio":
            self = .shardProduceStdio
        case "shard-finish-stdio":
            self = .shardFinishStdio
        case "checkpoint-shard-produce-stdio":
            self = .checkpointShardProduceStdio
        case "checkpoint-shard-finish-stdio":
            self = .checkpointShardFinishStdio
        case "generate":
            let prompt = arguments.dropFirst().joined(separator: " ")
            guard !prompt.isEmpty else {
                throw WorkerError.usage("generate requires a prompt")
            }
            self = .generate(prompt: prompt)
        default:
            throw WorkerError.usage("unknown command: \(command)")
        }
    }
}

private enum WorkerError: LocalizedError {
    case usage(String)
    case shardMismatch
    case checkpointShardMismatch

    var errorDescription: String? {
        switch self {
        case .usage(let message):
            return "\(message)\nusage: mlx-worker [health | capabilities | shard-smoke | checkpoint-shard-smoke [model-id] | shard-produce <path> | shard-finish <path> | shard-produce-stdio | shard-finish-stdio | checkpoint-shard-produce-stdio | checkpoint-shard-finish-stdio | generate <prompt>]"
        case .shardMismatch:
            return "sharded output did not match the single-range reference"
        case .checkpointShardMismatch:
            return "real checkpoint shard output did not match the full-checkpoint reference"
        }
    }
}

private struct WorkerCapabilities: Codable {
    let runtime: String
    let device: String
    let checkpointShardModelTypes: [String]
    let physicalMemoryBytes: UInt64
    let mlxActiveMemoryBytes: Int
    let mlxCacheMemoryBytes: Int
}

@main
struct MLXWorker {
    static func main() async throws {
        let command = try WorkerCommand(arguments: Array(CommandLine.arguments.dropFirst()))
        let encoder = JSONEncoder()

        switch command {
        case .health:
            print("{\"status\":\"ok\",\"runtime\":\"mlx-swift\"}")

        case .capabilities:
            let capabilities = WorkerCapabilities(
                runtime: "mlx-swift",
                device: Device.defaultDevice().deviceType?.rawValue ?? "unknown",
                checkpointShardModelTypes: WorkerCheckpointShards.configuration
                    .adapterRegistry.supportedModelTypes,
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                mlxActiveMemoryBytes: Memory.activeMemory,
                mlxCacheMemoryBytes: Memory.cacheMemory
            )
            print(String(decoding: try encoder.encode(capabilities), as: UTF8.self))

        case .shardSmoke:
            let result = Gemma3ShardSmoke.run()
            guard result.matchesSingleRange else {
                throw WorkerError.shardMismatch
            }
            print(String(decoding: try encoder.encode(result), as: UTF8.self))

        case .checkpointShardSmoke(let modelID):
            let result = try await CheckpointShardRuntime.run(
                modelID: modelID,
                configuration: WorkerCheckpointShards.configuration
            )
            guard result.matchesFullCheckpoint, result.passesMemoryProof else {
                throw WorkerError.checkpointShardMismatch
            }
            print(String(decoding: try encoder.encode(result), as: UTF8.self))

        case .shardProduce(let path):
            let result = try Gemma3ShardSmoke.produceBoundary(
                to: URL(fileURLWithPath: path)
            )
            print(String(decoding: try encoder.encode(result), as: UTF8.self))

        case .shardFinish(let path):
            let result = try Gemma3ShardSmoke.finishBoundary(
                from: URL(fileURLWithPath: path)
            )
            try printShardResult(result, encoder: encoder)

        case .shardProduceStdio:
            // stdout is intentionally payload-only so the Go daemon can relay it
            // byte-for-byte without parsing MLX-specific data.
            FileHandle.standardOutput.write(
                try Gemma3ShardSmoke.produceBoundaryPayload()
            )

        case .shardFinishStdio:
            let payload = FileHandle.standardInput.readDataToEndOfFile()
            let result = try Gemma3ShardSmoke.finishBoundary(from: payload)
            try printShardResult(result, encoder: encoder)

        case .checkpointShardProduceStdio:
            // The envelope includes the real checkpoint boundary plus producer
            // memory measurements for the remote consumer's proof.
            FileHandle.standardOutput.write(
                try await CheckpointShardRuntime.produceBoundaryPayload(
                    modelID: WorkerCheckpointShards.defaultModelID,
                    configuration: WorkerCheckpointShards.configuration
                )
            )

        case .checkpointShardFinishStdio:
            let payload = FileHandle.standardInput.readDataToEndOfFile()
            let result = try await CheckpointShardRuntime.finishBoundary(
                from: payload,
                configuration: WorkerCheckpointShards.configuration
            )
            guard result.matchesFullCheckpoint, result.passesMemoryProof else {
                throw WorkerError.checkpointShardMismatch
            }
            print(String(decoding: try encoder.encode(result), as: UTF8.self))

        case .generate(let prompt):
            let configuration = LLMRegistry.smolLM_135M_4bit
            let model = try await #huggingFaceLoadModelContainer(
                configuration: configuration
            )

            let session = ChatSession(model)
            let response = try await session.respond(to: prompt)
            print(response)
        }
    }

    private static func printShardResult(
        _ result: ShardSmokeResult,
        encoder: JSONEncoder
    ) throws {
        guard result.matchesSingleRange else {
            throw WorkerError.shardMismatch
        }
        print(String(decoding: try encoder.encode(result), as: UTF8.self))
    }
}
