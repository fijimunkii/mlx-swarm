import Foundation
import MLX

enum WorkerRuntimeMemoryError: LocalizedError {
    case invalidEnvironmentValue(String, String)
    case limitExceedsPhysicalMemory(String, UInt64)

    var errorDescription: String? {
        switch self {
        case .invalidEnvironmentValue(let name, let value):
            return "\(name) must be a positive byte count, got \(value)"
        case .limitExceedsPhysicalMemory(let name, let physicalBytes):
            return "\(name) exceeds physical memory (\(physicalBytes) bytes)"
        }
    }
}

struct WorkerRuntimeMemory {
    static let physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory

    static func configure(environment: [String: String] = ProcessInfo.processInfo.environment) throws {
        if let value = environment["MLX_SWARM_MEMORY_THRESHOLD_BYTES"] {
            Memory.memoryLimit = try parseLimit(
                name: "MLX_SWARM_MEMORY_THRESHOLD_BYTES",
                value: value
            )
        }
        if let value = environment["MLX_SWARM_CACHE_LIMIT_BYTES"] {
            Memory.cacheLimit = try parseLimit(
                name: "MLX_SWARM_CACHE_LIMIT_BYTES",
                value: value
            )
        }
    }

    private static func parseLimit(name: String, value: String) throws -> Int {
        guard let bytes = UInt64(value), bytes > 0, bytes <= UInt64(Int.max) else {
            throw WorkerRuntimeMemoryError.invalidEnvironmentValue(name, value)
        }
        guard bytes <= physicalMemoryBytes else {
            throw WorkerRuntimeMemoryError.limitExceedsPhysicalMemory(name, physicalMemoryBytes)
        }
        return Int(bytes)
    }
}
