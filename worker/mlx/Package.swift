// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "MLXWorker",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            exact: "0.31.6"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            exact: "3.31.4"
        ),
        .package(
            url: "https://github.com/huggingface/swift-huggingface",
            from: "0.9.0"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            from: "1.3.0"
        ),
    ],
    targets: [
        .executableTarget(
            name: "MLXWorker",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        )
    ]
)
