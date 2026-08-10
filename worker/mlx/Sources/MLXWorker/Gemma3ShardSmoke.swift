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

enum Gemma3ShardSmoke {
    static func run() -> ShardSmokeResult {
        // Keep this model deliberately tiny and randomly initialized. The goal
        // is to prove that materializing a hidden state at a shard boundary and
        // resuming the same upstream transformer stack produces the same result.
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

        let tokens = MLXArray([3, 14, 15, 92, 65, 35], [1, 6])
        let inner = model.model
        let splitLayer = inner.layers.count / 2

        func embeddedInput() -> MLXArray {
            let embedded = inner.embedTokens(tokens)
            let scale = MLXArray(sqrt(Float(inner.config.hiddenSize)), dtype: .bfloat16)
            return embedded * scale.asType(embedded.dtype)
        }

        // A six-token prompt is shorter than Gemma 3's sliding window, so a
        // causal mask is equivalent for this composition smoke test. Production
        // sharding must preserve Gemma's per-layer global/sliding mask semantics.
        let mask = MLXFast.ScaledDotProductAttentionMaskMode.causal

        var singleRange = embeddedInput()
        for layer in inner.layers {
            singleRange = layer(singleRange, mask: mask, cache: nil)
        }
        eval(singleRange)

        var splitRange = embeddedInput()
        for layer in inner.layers[..<splitLayer] {
            splitRange = layer(splitRange, mask: mask, cache: nil)
        }
        eval(splitRange)

        // This is the real future network boundary: detach the lazy graph,
        // copy the hidden state into a language-neutral byte payload, then
        // construct a fresh MLXArray before resuming the remaining layers.
        let boundary = WireTensor(splitRange)
        splitRange = boundary.materialize()

        for layer in inner.layers[splitLayer...] {
            splitRange = layer(splitRange, mask: mask, cache: nil)
        }
        eval(splitRange)

        let matches = allClose(singleRange, splitRange, rtol: 1e-5, atol: 1e-5).item(Bool.self)

        return ShardSmokeResult(
            model: "gemma3-random-tiny",
            layers: inner.layers.count,
            splitLayer: splitLayer,
            boundaryBytes: boundary.data.count,
            boundaryDType: boundary.dtype.rawValue,
            outputShape: splitRange.shape,
            matchesSingleRange: matches
        )
    }
}
