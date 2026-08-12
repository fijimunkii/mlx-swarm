import Foundation
import MLX

/// Language-neutral tensor payload used at worker boundaries.
///
/// `Data` is contiguous native-endian MLX storage. The protobuf transport will
/// carry the same three fields as raw bytes rather than JSON/base64.
struct WireTensor: Codable {
    enum ElementType: String, Codable, Hashable {
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

        var byteWidth: Int {
            switch self {
            case .bool, .uint8, .int8:
                1
            case .uint16, .int16, .float16, .bfloat16:
                2
            case .uint32, .int32, .float32:
                4
            case .uint64, .int64, .complex64, .float64:
                8
            }
        }
    }

    let shape: [Int]
    let dtype: ElementType
    let data: Data

    static let empty = WireTensor(shape: [0], dtype: .uint8, data: Data())

    private init(shape: [Int], dtype: ElementType, data: Data) {
        self.shape = shape
        self.dtype = dtype
        self.data = data
    }

    init(_ array: MLXArray) {
        let payload = array.asData(access: .copy)
        self.shape = payload.shape
        self.dtype = ElementType(payload.dType)
        self.data = payload.data
    }

    func validate() throws {
        var elementCount = 1
        for dimension in shape {
            guard dimension >= 0 else {
                throw WireTensorError.invalidShape("dimension \(dimension) is negative")
            }
            let (nextCount, overflow) = elementCount.multipliedReportingOverflow(by: dimension)
            guard !overflow else {
                throw WireTensorError.invalidShape("element count overflows Int")
            }
            elementCount = nextCount
        }
        let (expectedBytes, overflow) = elementCount.multipliedReportingOverflow(
            by: dtype.byteWidth
        )
        guard !overflow else {
            throw WireTensorError.invalidShape("byte count overflows Int")
        }
        guard data.count == expectedBytes else {
            throw WireTensorError.invalidByteCount(
                got: data.count,
                expected: expectedBytes,
                dtype: dtype.rawValue
            )
        }
    }

    func materialize() throws -> MLXArray {
        try validate()
        return MLXArray(data, shape, dtype: dtype.mlxDType)
    }
}

private enum WireTensorError: LocalizedError {
    case invalidShape(String)
    case invalidByteCount(got: Int, expected: Int, dtype: String)

    var errorDescription: String? {
        switch self {
        case .invalidShape(let reason):
            return "invalid wire tensor shape: \(reason)"
        case .invalidByteCount(let got, let expected, let dtype):
            return "invalid wire tensor byte count for \(dtype): got \(got), expected \(expected)"
        }
    }
}
