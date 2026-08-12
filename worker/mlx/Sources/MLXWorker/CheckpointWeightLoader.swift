import Foundation
import MLX
import MLXLMCommon
import MLXNN

private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

/// Architecture-neutral rules for retaining checkpoint tensors.
///
/// `includesRaw` may admit sanitizer dependencies such as a tied embedding;
/// `includesSanitized` is the exact set the stage will retain and materialize.
struct CheckpointWeightSelection {
    let includesRaw: (String) -> Bool
    let includesSanitized: (String) -> Bool
}

struct SelectedCheckpointWeights {
    let arrays: [String: MLXArray]
    let metadata: [String: String]
}

/// Shared checkpoint I/O used by model-family adapters.
enum CheckpointWeightLoader {
    static func load(
        modelDirectory: URL,
        selection: CheckpointWeightSelection,
        sanitize: ([String: MLXArray], [String: String]) -> [String: MLXArray]
    ) throws -> SelectedCheckpointWeights {
        var selected = [String: MLXArray]()
        var metadata = [String: String]()

        for url in try safetensorURLs(in: modelDirectory) {
            let (fileWeights, fileMetadata) = try autoreleasepool {
                try loadArraysAndMetadata(url: url)
            }
            for (key, value) in fileWeights where selection.includesRaw(key) {
                selected[key] = value
            }
            if metadata.isEmpty {
                metadata = fileMetadata
            }
        }

        let sanitized = sanitize(selected, metadata)
            .filter { selection.includesSanitized($0.key) }
        // URL safetensors produce lazy Load nodes. Evaluate owned leaves one at
        // a time so each value detaches from its file reader before the stage
        // retains it, while bounding transient sanitizer/view intermediates.
        for key in sanitized.keys.sorted() {
            guard let value = sanitized[key] else {
                continue
            }
            try autoreleasepool {
                try checkedEval(value)
            }
            Memory.clearCache()
        }
        return SelectedCheckpointWeights(arrays: sanitized, metadata: metadata)
    }

    static func update(
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

    private static func safetensorURLs(in directory: URL) throws -> [URL] {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let index = try JSONDecoder().decode(
                SafetensorsIndex.self,
                from: Data(contentsOf: indexURL)
            )
            let indexedURLs = Set(index.weightMap.values)
                .sorted()
                .map { directory.appendingPathComponent($0) }
            guard !indexedURLs.isEmpty else {
                throw CheckpointShardError.noSafetensors(directory)
            }
            let existingCount = indexedURLs.count {
                FileManager.default.fileExists(atPath: $0.path)
            }
            if existingCount == indexedURLs.count {
                return indexedURLs
            }
            if existingCount > 0 {
                throw CheckpointShardError.incompleteSafetensorIndex(
                    directory,
                    existingCount,
                    indexedURLs.count
                )
            }
            // Some converted checkpoints retain an obsolete index beside a
            // newly resharded model. Accept only a canonical, complete
            // replacement set; reject partial or mixed directory contents.
            let discovered = try discoveredSafetensorURLs(in: directory)
            guard isCompleteCanonicalShardSet(discovered) else {
                throw CheckpointShardError.incompleteSafetensorIndex(
                    directory,
                    existingCount,
                    indexedURLs.count
                )
            }
            return discovered
        }

        let urls = try discoveredSafetensorURLs(in: directory)
        guard !urls.isEmpty else {
            throw CheckpointShardError.noSafetensors(directory)
        }
        return urls
    }

    private static func discoveredSafetensorURLs(in directory: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }
            .sorted { $0.path < $1.path }
    }

    private static func isCompleteCanonicalShardSet(_ urls: [URL]) -> Bool {
        guard !urls.isEmpty else {
            return false
        }
        if urls.count == 1 && urls[0].lastPathComponent == "model.safetensors" {
            return true
        }

        var expectedPrefix: String?
        var expectedTotal: Int?
        var indices = Set<Int>()
        for url in urls {
            let name = url.deletingPathExtension().lastPathComponent
            let parts = name.split(separator: "-", omittingEmptySubsequences: false)
            guard parts.count >= 4,
                  parts[parts.count - 2] == "of",
                  let index = Int(parts[parts.count - 3]),
                  let total = Int(parts[parts.count - 1]),
                  index >= 1,
                  index <= total
            else {
                return false
            }
            let prefix = parts.dropLast(3).joined(separator: "-")
            if let expectedPrefix, prefix != expectedPrefix {
                return false
            }
            if let expectedTotal, total != expectedTotal {
                return false
            }
            expectedPrefix = prefix
            expectedTotal = total
            guard indices.insert(index).inserted else {
                return false
            }
        }
        return expectedTotal == urls.count && indices.count == urls.count
    }
}
