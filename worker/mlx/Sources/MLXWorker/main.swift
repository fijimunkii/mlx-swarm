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
    case shardSmokeMismatch

    var errorDescription: String? {
        switch self {
        case .usage(let message):
            return "\(message)\nusage: mlx-worker [health | capabilities | shard-smoke | generate <prompt>]"
        case .shardSmokeMismatch:
            return "shard-smoke output did not match the single-range reference"
        }
    }
}

private struct WorkerCapabilities: Codable {
    let runtime: String
    let device: String
    let physicalMemoryBytes: UInt64
    let mlxActiveMemoryBytes: Int
    let mlxCacheMemoryBytes: Int
}

@main
struct MLXWorker {
    static func main() async throws {
        let command = try WorkerCommand(arguments: Array(CommandLine.arguments.dropFirst()))

        switch command {
        case .health:
            print("{\"status\":\"ok\",\"runtime\":\"mlx-swift\"}")

        case .capabilities:
            let capabilities = WorkerCapabilities(
                runtime: "mlx-swift",
                device: Device.defaultDevice().deviceType?.rawValue ?? "unknown",
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                mlxActiveMemoryBytes: Memory.activeMemory,
                mlxCacheMemoryBytes: Memory.cacheMemory
            )
            let data = try JSONEncoder().encode(capabilities)
            print(String(decoding: data, as: UTF8.self))

        case .shardSmoke:
            let result = Gemma3ShardSmoke.run()
            guard result.matchesSingleRange else {
                throw WorkerError.shardSmokeMismatch
            }
            let data = try JSONEncoder().encode(result)
            print(String(decoding: data, as: UTF8.self))

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
}
