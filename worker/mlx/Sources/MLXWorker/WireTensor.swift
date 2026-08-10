import Foundation
import MLX

/// Language-neutral tensor payload used at worker boundaries.
///
/// `Data` is contiguous native-endian MLX storage. The protobuf transport will
/// carry the same three fields as raw bytes rather than JSON/base64.
struct WireTensor: Codable {
    enum ElementType: String, Codable {
        case bool
        case uint8
        case uint16
        case uint32
        case uint64
        case int8
        case int16
        case int32
        case int64
        case float16
        case float32
        case bfloat16
        case complex64
        case float64

        init(_ dtype: DType) {
            switch dtype {
            case .bool: self = .bool
            case .uint8: self = .uint8
            case .uint16: self = .uint16
            case .uint32: self = .uint32
            case .uint64: self = .uint64
            case .int8: self = .int8
            case .int16: self = .int16
            case .int32: self = .int32
            case .int64: self = .int64
            case .float16: self = .float16
            case .float32: self = .float32
            case .bfloat16: self = .bfloat16
            case .complex64: self = .complex64
            case .float64: self = .float64
            }
        }

        var mlxDType: DType {
            switch self {
            case .bool: .bool
            case .uint8: .uint8
            case .uint16: .uint16
            case .uint32: .uint32
            case .uint64: .uint64
            case .int8: .int8
            case .int16: .int16
            case .int32: .int32
            case .int64: .int64
            case .float16: .float16
            case .float32: .float32
            case .bfloat16: .bfloat16
            case .complex64: .complex64
            case .float64: .float64
            }
        }
    }

    let shape: [Int]
    let dtype: ElementType
    let data: Data

    init(_ array: MLXArray) {
        let payload = array.asData(access: .copy)
        self.shape = payload.shape
        self.dtype = ElementType(payload.dType)
        self.data = payload.data
    }

    func materialize() -> MLXArray {
        MLXArray(data, shape, dtype: dtype.mlxDType)
    }
}
