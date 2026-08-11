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
            return Set(index.weightMap.values)
                .sorted()
                .map { directory.appendingPathComponent($0) }
        }

        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: nil
        ) else {
            throw CheckpointShardError.noSafetensors(directory)
        }
        let urls = enumerator.compactMap { item -> URL? in
            guard let url = item as? URL, url.pathExtension == "safetensors" else {
                return nil
            }
            return url
        }.sorted { $0.path < $1.path }
        guard !urls.isEmpty else {
            throw CheckpointShardError.noSafetensors(directory)
        }
        return urls
    }
}
