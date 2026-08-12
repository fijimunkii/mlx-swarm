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
            let existingCount = indexedURLs.count {
                FileManager.default.fileExists(atPath: $0.path)
            }
            if !indexedURLs.isEmpty && existingCount == indexedURLs.count {
                return indexedURLs
            }
            if existingCount > 0 {
                throw CheckpointShardError.incompleteSafetensorIndex(
                    directory,
                    existingCount,
                    indexedURLs.count
                )
            }
        }

        let urls = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }
            .sorted { $0.path < $1.path }
        guard !urls.isEmpty else {
            throw CheckpointShardError.noSafetensors(directory)
        }
        return urls
    }
}
