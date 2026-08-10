import Foundation
import MLX
@_spi(GemmaEncoder) import MLXLLM
import MLXNN

struct ShardSmokeResult: Codable {
    let model: String
    let layers: Int
    let splitLayer: Int
    let boundaryBytes: Int
    let boundaryDType: String
    let outputShape: [Int]
    let matchesSingleRange: Bool
}

struct ShardProduceResult: Codable {
    let model: String
    let layers: Int
    let splitLayer: Int
    let boundaryBytes: Int
    let boundaryDType: String
    let boundaryShape: [Int]
}

private enum TinyGemma3Fixture {
    static let seed: UInt64 = 0x4D4C_5853_5741_524D // "MLXSWARM"
    static let tokens = MLXArray([3, 14, 15, 92, 65, 35], [1, 6])
    static let mask = MLXFast.ScaledDotProductAttentionMaskMode.causal

    static func makeModel() -> Gemma3TextModel {
        // Seeding before construction gives independent worker processes the
        // same random weights. This lets the first true process-boundary test
        // stay tiny and zero-download.
        MLXRandom.seed(seed)
        let config = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64,
            hiddenLayers: 8,
            intermediateSize: 64,
            attentionHeads: 4,
            headDim: 64,
            rmsNormEps: 0.00001,
            vocabularySize: 100,
            kvHeads: 4,
            ropeTheta: 1_000_000,
            ropeLocalBaseFreq: 10_000,
            ropeTraditional: false,
            queryPreAttnScalar: 256,
            slidingWindow: 512,
            slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )
        let model = Gemma3TextModel(config)
        eval(model)
        return model
    }

    static func embeddedInput(_ inner: Gemma3Model) -> MLXArray {
        let embedded = inner.embedTokens(tokens)
        let scale = MLXArray(sqrt(Float(inner.config.hiddenSize)), dtype: .bfloat16)
        return embedded * scale.asType(embedded.dtype)
    }

    static func fullReference(_ inner: Gemma3Model) -> MLXArray {
        var hidden = embeddedInput(inner)
        for layer in inner.layers {
            hidden = layer(hidden, mask: mask, cache: nil)
        }
        eval(hidden)
        return hidden
    }
}

enum Gemma3ShardSmoke {
    static func run() -> ShardSmokeResult {
        let model = TinyGemma3Fixture.makeModel()
        let inner = model.model
        let splitLayer = inner.layers.count / 2
        let reference = TinyGemma3Fixture.fullReference(inner)

        var hidden = TinyGemma3Fixture.embeddedInput(inner)
        for layer in inner.layers[..<splitLayer] {
            hidden = layer(hidden, mask: TinyGemma3Fixture.mask, cache: nil)
        }
        eval(hidden)

        let boundary = WireTensor(hidden)
        hidden = boundary.materialize()

        for layer in inner.layers[splitLayer...] {
            hidden = layer(hidden, mask: TinyGemma3Fixture.mask, cache: nil)
        }
        eval(hidden)

        let matches = allClose(reference, hidden, rtol: 1e-5, atol: 1e-5).item(Bool.self)
        return result(
            inner: inner,
            splitLayer: splitLayer,
            boundary: boundary,
            output: hidden,
            matches: matches
        )
    }

    /// First OS process: initialize the deterministic model, execute the first
    /// half, and persist only the language-neutral tensor payload.
    static func produceBoundary(to url: URL) throws -> ShardProduceResult {
        let model = TinyGemma3Fixture.makeModel()
        let inner = model.model
        let splitLayer = inner.layers.count / 2

        var hidden = TinyGemma3Fixture.embeddedInput(inner)
        for layer in inner.layers[..<splitLayer] {
            hidden = layer(hidden, mask: TinyGemma3Fixture.mask, cache: nil)
        }
        eval(hidden)

        let boundary = WireTensor(hidden)
        try JSONEncoder().encode(boundary).write(to: url, options: .atomic)

        return ShardProduceResult(
            model: "gemma3-random-tiny",
            layers: inner.layers.count,
            splitLayer: splitLayer,
            boundaryBytes: boundary.data.count,
            boundaryDType: boundary.dtype.rawValue,
            boundaryShape: boundary.shape
        )
    }

    /// Second OS process: recreate the same deterministic model, consume the
    /// serialized midpoint, execute the remaining layers, and compare against
    /// a full reference computed entirely in this second process. Matching here
    /// proves that the first process's weights and hidden state compose with the
    /// second process's complementary layer range.
    static func finishBoundary(from url: URL) throws -> ShardSmokeResult {
        let boundary = try JSONDecoder().decode(
            WireTensor.self,
            from: Data(contentsOf: url)
        )

        let model = TinyGemma3Fixture.makeModel()
        let inner = model.model
        let splitLayer = inner.layers.count / 2
        let reference = TinyGemma3Fixture.fullReference(inner)

        var hidden = boundary.materialize()
        for layer in inner.layers[splitLayer...] {
            hidden = layer(hidden, mask: TinyGemma3Fixture.mask, cache: nil)
        }
        eval(hidden)

        let matches = allClose(reference, hidden, rtol: 1e-5, atol: 1e-5).item(Bool.self)
        return result(
            inner: inner,
            splitLayer: splitLayer,
            boundary: boundary,
            output: hidden,
            matches: matches
        )
    }

    private static func result(
        inner: Gemma3Model,
        splitLayer: Int,
        boundary: WireTensor,
        output: MLXArray,
        matches: Bool
    ) -> ShardSmokeResult {
        ShardSmokeResult(
            model: "gemma3-random-tiny",
            layers: inner.layers.count,
            splitLayer: splitLayer,
            boundaryBytes: boundary.data.count,
            boundaryDType: boundary.dtype.rawValue,
            outputShape: output.shape,
            matchesSingleRange: matches
        )
    }
}
