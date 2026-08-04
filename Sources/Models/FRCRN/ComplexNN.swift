import Foundation
import MLX
import MLXNN

/// Complex-valued 2D convolutional layer for MLX Swift.
/// Processes input tensors with real and imaginary parts.
///
/// Input shape: (batch, height, width, channels, 2) where last dim = [real, imag]
/// Output shape: (batch, height, width, out_channels, 2)
public class ComplexConv2d: Module {
    @ModuleInfo var conv_re: Conv2d
    @ModuleInfo var conv_im: Conv2d
    
    public init(
        in_channels: Int,
        out_channels: Int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
        dilation: IntOrPair = 1,
        bias: Bool = true
    ) {
        self._conv_re.wrappedValue = Conv2d(
            inputChannels: in_channels,
            outputChannels: out_channels,
            kernelSize: kernel_size,
            stride: stride,
            padding: padding,
            dilation: dilation,
            bias: bias
        )
        self._conv_im.wrappedValue = Conv2d(
            inputChannels: in_channels,
            outputChannels: out_channels,
            kernelSize: kernel_size,
            stride: stride,
            padding: padding,
            dilation: dilation,
            bias: bias
        )
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Extract real and imaginary parts
        let x_re = x[.ellipsis, 0]  // (batch, height, width, channels)
        let x_im = x[.ellipsis, 1]  // (batch, height, width, channels)
        
        // Apply convolutions (parallelized by MLX graph)
        let conv_re_re = conv_re(x_re)
        let conv_im_im = conv_im(x_im)
        let conv_im_re = conv_im(x_re)
        let conv_re_im = conv_re(x_im)
        
        // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        let real = conv_re_re - conv_im_im
        let imag = conv_im_re + conv_re_im
        
        return MLX.stacked([real, imag], axis: -1)
    }
}

/// Complex-valued batch normalization for MLX Swift.
/// Normalizes real and imaginary parts independently.
///
/// Input shape: (batch, height, width, channels, 2)
/// Output shape: (batch, height, width, channels, 2)
public class ComplexBatchNorm2d: Module {
    @ModuleInfo var bn_re: BatchNorm
    @ModuleInfo var bn_im: BatchNorm
    
    public init(
        num_features: Int,
        eps: Float = 1e-5,
        momentum: Float = 0.1,
        affine: Bool = true
    ) {
        self._bn_re.wrappedValue = BatchNorm(featureCount: num_features, eps: eps, momentum: momentum, affine: affine)
        self._bn_im.wrappedValue = BatchNorm(featureCount: num_features, eps: eps, momentum: momentum, affine: affine)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Apply BatchNorm to real and imaginary parts independently
        let real = bn_re(x[.ellipsis, 0])
        let imag = bn_im(x[.ellipsis, 1])
        return MLX.stacked([real, imag], axis: -1)
    }
}

/// Complex-valued 2D transposed convolutional layer for MLX Swift.
///
/// Input shape: (batch, height, width, channels, 2)
/// Output shape: (batch, height, width, out_channels, 2)
public class ComplexConvTranspose2d: Module {
    @ModuleInfo var tconv_re: ConvTransposed2d
    @ModuleInfo var tconv_im: ConvTransposed2d
    
    public init(
        in_channels: Int,
        out_channels: Int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
        dilation: IntOrPair = 1,
        bias: Bool = true
    ) {
        self._tconv_re.wrappedValue = ConvTransposed2d(
            inputChannels: in_channels,
            outputChannels: out_channels,
            kernelSize: kernel_size,
            stride: stride,
            padding: padding,
            dilation: dilation,
            bias: bias
        )
        self._tconv_im.wrappedValue = ConvTransposed2d(
            inputChannels: in_channels,
            outputChannels: out_channels,
            kernelSize: kernel_size,
            stride: stride,
            padding: padding,
            dilation: dilation,
            bias: bias
        )
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Extract real and imaginary parts
        let x_re = x[.ellipsis, 0]
        let x_im = x[.ellipsis, 1]
        
        // Apply transposed convolutions
        let tconv_re_re = tconv_re(x_re)
        let tconv_im_im = tconv_im(x_im)
        let tconv_im_re = tconv_im(x_re)
        let tconv_re_im = tconv_re(x_im)
        
        // Complex multiplication
        let real = tconv_re_re - tconv_im_im
        let imag = tconv_re_im + tconv_im_re
        
        return MLX.stacked([real, imag], axis: -1)
    }
}
