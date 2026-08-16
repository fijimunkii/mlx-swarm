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
        // Pin the merged GemmaEncoder SPI until mlx-swift-lm cuts a release
        // containing https://github.com/ml-explore/mlx-swift-lm/pull/387.
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            revision: "6608a3565178240f1f42a23cb832ee1c59a16208"
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
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        )
    ]
)
