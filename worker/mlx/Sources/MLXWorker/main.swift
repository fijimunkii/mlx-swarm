import Foundation
import MLX
import MLXLLM
import MLXLMCommon

@main
struct MLXWorker {
    static func main() async throws {
        // Keep the initial worker deliberately thin. The first implementation
        // will add a local RPC server and expose MLX Swift LM model/shard
        // operations without reimplementing model math.
        print("mlx-worker: ready (bootstrap)")
    }
}
