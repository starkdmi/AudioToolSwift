// Copyright © 2025
// Matcha - Decoder blocks for flow matching
// Pure MLX port of Python MLX matcha/decoder.py

import Foundation
import MLX
import MLXNN
import MLXFast
import MLXRandom

// MARK: - Supporting Types

/// Layer list container for indexed attribute access
public class LayerList: Module {
    @ModuleInfo(key: "0") public var layer_0: Linear
    @ModuleInfo(key: "1") public var layer_1: Linear

    public init(_ layers: [Linear]) {
        precondition(layers.count == 2, "LayerList expects exactly 2 layers")
        self._layer_0 = ModuleInfo(wrappedValue: layers[0])
        self._layer_1 = ModuleInfo(wrappedValue: layers[1])
        super.init()
    }

    public subscript(index: Int) -> Linear {
        switch index {
        case 0:
            return layer_0
        case 1:
            return layer_1
        default:
            fatalError("LayerList index out of range: \(index)")
        }
    }

    public var count: Int { 2 }
}

// MARK: - Sinusoidal Position Embeddings

/// Sinusoidal position embeddings for timestep encoding
public class SinusoidalPosEmb: Module {
    public let dim: Int
    
    public init(dim: Int) {
        precondition(dim % 2 == 0, "SinusoidalPosEmb requires dim to be even")
        self.dim = dim
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, scale: Float = 1000) -> MLXArray {
        var input = x
        if input.ndim < 1 {
            input = input.expandedDimensions(axis: 0)
        }
        
        let halfDim = dim / 2
        let logBase = log(10000.0) / Float(halfDim - 1)
        var emb = exp(MLXArray(0..<halfDim).asType(.float32) * -logBase)
        emb = scale * input.expandedDimensions(axis: 1) * emb.expandedDimensions(axis: 0)
        emb = concatenated([sin(emb), cos(emb)], axis: -1)
        return emb
    }
}

// MARK: - Timestep Embedding

/// MLP for timestep embedding
public class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") public var linear_1: Linear
    @ModuleInfo(key: "linear_2") public var linear_2: Linear
    let act_fn: String
    
    public init(inChannels: Int, timeEmbedDim: Int, actFn: String = "silu") {
        self._linear_1 = ModuleInfo(wrappedValue: Linear(inChannels, timeEmbedDim))
        self._linear_2 = ModuleInfo(wrappedValue: Linear(timeEmbedDim, timeEmbedDim))
        self.act_fn = actFn
        super.init()
    }
    
    public func callAsFunction(_ sample: MLXArray) -> MLXArray {
        var x = linear_1(sample)
        x = act_fn == "silu" ? silu(x) : gelu(x)
        x = linear_2(x)
        return x
    }
}

// MARK: - Block1D

/// 1D convolutional block with group norm
public class Block1D: Module {
    @ModuleInfo public var conv: Conv1d
    @ModuleInfo public var norm: GroupNorm
    
    public init(dim: Int, dimOut: Int, groups: Int = 8) {
        self._conv = ModuleInfo(wrappedValue: Conv1d(inputChannels: dim, outputChannels: dimOut, kernelSize: 3, padding: 1))
        self._norm = ModuleInfo(wrappedValue: GroupNorm(groupCount: groups, dimensions: dimOut))
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        // x is (B, C, T) but MLX Conv1d expects (B, T, C)
        var xIn = (x * mask).transposed(0, 2, 1)  // (B, C, T) -> (B, T, C)
        var output = conv(xIn)
        output = output.transposed(0, 2, 1)  // (B, T, C) -> (B, C, T)
        output = norm(output)
        output = mish(output)
        return output * mask
    }
}

// MARK: - CausalConv1d

/// Causal 1D convolution with left padding
public class CausalConv1d: Module {
    @ModuleInfo public var conv: Conv1d
    let causalPadding: Int
    
    public init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, dilation: Int = 1, bias: Bool = true) {
        precondition(stride == 1, "CausalConv1d only supports stride=1")
        self._conv = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            bias: bias
        ))
        self.causalPadding = kernelSize - 1
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x.transposed(0, 2, 1)  // (B, C, T) -> (B, T, C)
        // Pad on the left for causal convolution
        input = padded(input, widths: [.init((0, 0)), .init((causalPadding, 0)), .init((0, 0))])
        var output = conv(input)
        output = output.transposed(0, 2, 1)  // (B, T, C) -> (B, C, T)
        return output
    }
}

// MARK: - CausalBlock1D

/// Causal 1D block with LayerNorm
public class CausalBlock1D: Module {
    @ModuleInfo public var conv: CausalConv1d
    @ModuleInfo public var norm: LayerNorm
    
    public init(dim: Int, dimOut: Int) {
        self._conv = ModuleInfo(wrappedValue: CausalConv1d(inChannels: dim, outChannels: dimOut, kernelSize: 3))
        self._norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: dimOut))
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var output = conv(x * mask)
        // Transpose to (B, T, C), apply LayerNorm, transpose back
        output = output.transposed(0, 2, 1)  // (B, C, T) -> (B, T, C)
        output = norm(output)
        output = output.transposed(0, 2, 1)  // (B, T, C) -> (B, C, T)
        output = mish(output)
        return output * mask
    }
}

// MARK: - ResnetBlock1D

/// 1D ResNet block with time embedding
public class ResnetBlock1D: Module {
    @ModuleInfo(key: "mlp_linear") public var mlp_linear: Linear
    @ModuleInfo public var block1: Block1D
    @ModuleInfo public var block2: Block1D
    @ModuleInfo public var res_conv: Conv1d
    
    public init(dim: Int, dimOut: Int, timeEmbDim: Int, groups: Int = 8) {
        self._mlp_linear = ModuleInfo(wrappedValue: Linear(timeEmbDim, dimOut))
        self._block1 = ModuleInfo(wrappedValue: Block1D(dim: dim, dimOut: dimOut, groups: groups))
        self._block2 = ModuleInfo(wrappedValue: Block1D(dim: dimOut, dimOut: dimOut, groups: groups))
        self._res_conv = ModuleInfo(wrappedValue: Conv1d(inputChannels: dim, outputChannels: dimOut, kernelSize: 1))
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> MLXArray {
        var h = block1(x, mask: mask)
        // Original: h += self.mlp(time_emb) where mlp = Sequential(Mish(), Linear())
        h = h + mlp_linear(mish(timeEmb)).expandedDimensions(axis: -1)
        h = block2(h, mask: mask)
        
        // res_conv: (B, C, T) -> transpose -> conv -> transpose back
        let xRes = (x * mask).transposed(0, 2, 1)
        var resOut = res_conv(xRes)
        resOut = resOut.transposed(0, 2, 1)
        
        return h + resOut
    }
}

// MARK: - CausalResnetBlock1D

/// Causal ResNet block
public class CausalResnetBlock1D: Module {
    @ModuleInfo(key: "mlp_linear") public var mlp_linear: Linear
    @ModuleInfo public var block1: CausalBlock1D
    @ModuleInfo public var block2: CausalBlock1D
    @ModuleInfo public var res_conv: Conv1d
    
    public init(dim: Int, dimOut: Int, timeEmbDim: Int, groups: Int = 8) {
        self._mlp_linear = ModuleInfo(wrappedValue: Linear(timeEmbDim, dimOut))
        self._block1 = ModuleInfo(wrappedValue: CausalBlock1D(dim: dim, dimOut: dimOut))
        self._block2 = ModuleInfo(wrappedValue: CausalBlock1D(dim: dimOut, dimOut: dimOut))
        self._res_conv = ModuleInfo(wrappedValue: Conv1d(inputChannels: dim, outputChannels: dimOut, kernelSize: 1))
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> MLXArray {
        var h = block1(x, mask: mask)
        h = h + mlp_linear(mish(timeEmb)).expandedDimensions(axis: -1)
        h = block2(h, mask: mask)
        
        let xRes = (x * mask).transposed(0, 2, 1)
        var resOut = res_conv(xRes)
        resOut = resOut.transposed(0, 2, 1)
        
        return h + resOut
    }
}

// MARK: - Downsample1D

/// 1D downsampling with stride-2 convolution
public class Downsample1D: Module {
    @ModuleInfo public var conv: Conv1d
    
    public init(dim: Int) {
        self._conv = ModuleInfo(wrappedValue: Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 2, padding: 1))
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x.transposed(0, 2, 1)  // (B, C, T) -> (B, T, C)
        var output = conv(input)
        output = output.transposed(0, 2, 1)  // (B, T, C) -> (B, C, T)
        return output
    }
}

// MARK: - Upsample1D

/// 1D upsampling with ConvTranspose1d (or optional nearest neighbor)
public class Upsample1DMatcha: Module {
    @ModuleInfo public var conv: ConvTranspose1d
    let channels: Int
    let useConvTranspose: Bool
    
    public init(channels: Int, useConvTranspose: Bool = true) {
        self.channels = channels
        self.useConvTranspose = useConvTranspose
        self._conv = ModuleInfo(wrappedValue: ConvTranspose1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 4,
            stride: 2,
            padding: 1
        ))
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if useConvTranspose {
            var input = x.transposed(0, 2, 1)  // (B, C, T) -> (B, T, C)
            input = conv(input)
            return input.transposed(0, 2, 1)
        }

        // Nearest neighbor upsampling by 2x along time axis
        let (B, C, T) = (x.shape[0], x.shape[1], x.shape[2])
        let expanded = x.expandedDimensions(axis: 3)
        let repeated = broadcast(expanded, to: [B, C, T, 2])
        return repeated.reshaped([B, C, T * 2])
    }
}
