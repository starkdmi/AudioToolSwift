import Foundation
import MLX
import MLXNN

/// Encoder module for MLX UNet implementation.
public class Encoder: Module {
    @ModuleInfo var conv: ComplexConv2d
    @ModuleInfo var bn: ComplexBatchNorm2d
    
    let negative_slope: Float = 0.01
    
    public init(
        in_channels: Int,
        out_channels: Int,
        kernel_size: (Int, Int),
        stride: (Int, Int),
        padding: (Int, Int)
    ) {
        self._conv.wrappedValue = ComplexConv2d(
            in_channels: in_channels,
            out_channels: out_channels,
            kernel_size: IntOrPair(kernel_size),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding)
        )
        self._bn.wrappedValue = ComplexBatchNorm2d(num_features: out_channels)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv(x)
        out = bn(out)
        out = MLXNN.leakyRelu(out, negativeSlope: negative_slope)
        return out
    }
}
