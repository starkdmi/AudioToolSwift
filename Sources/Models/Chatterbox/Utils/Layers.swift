import MLX
import MLXNN

public final class ManualLayerNorm: Module {
    public let dimensions: Int
    public let eps: Float

    @ParameterInfo public var weight: MLXArray
    @ParameterInfo public var bias: MLXArray

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.dimensions = dimensions
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        self._bias.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let variance = MLX.variance(x, axis: -1, keepDims: true)
        let invStd = MLX.rsqrt(variance + MLXArray(eps))
        return (x - mean) * invStd * weight + bias
    }
}

final class GELU: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXNN.gelu(x)
    }
}

final class Sequential: Module {
    @ModuleInfo var layers: [Module]

    init(_ layers: [Module]) {
        self._layers.wrappedValue = layers
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in layers {
            if let linear = layer as? Linear {
                out = linear(out)
            } else if let gelu = layer as? GELU {
                out = gelu(out)
            } else {
                fatalError("Unsupported layer in Sequential: \(type(of: layer))")
            }
        }
        return out
    }
}
