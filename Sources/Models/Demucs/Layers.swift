import MLX
import MLXNN
import Foundation

// MARK: - LayerScale

/// Learnable scale parameter for layer outputs
public class LayerScale: Module, UnaryLayer {
    public var scale: MLXArray
    
    public init(channels: Int, init_value: Float = 0) {
        self.scale = MLXArray.full([channels], values: MLXArray(init_value))
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (Batch, Time, Channels)
        return scale * x
    }
}

// MARK: - DConv

/// Dilated convolution block with residual connections
public class DConv: Module, UnaryLayer {
    let channels: Int
    let compress: Float
    let depth: Int
    
    // Using individual layers since Sequential requires UnaryLayer
    @ModuleInfo var conv0_0: Conv1d?
    @ModuleInfo var norm0_0: GroupNorm?
    @ModuleInfo var conv0_1: Conv1d?
    @ModuleInfo var norm0_1: GroupNorm?
    @ModuleInfo var scale0: LayerScale?
    
    @ModuleInfo var conv1_0: Conv1d?
    @ModuleInfo var norm1_0: GroupNorm?
    @ModuleInfo var conv1_1: Conv1d?
    @ModuleInfo var norm1_1: GroupNorm?
    @ModuleInfo var scale1: LayerScale?
    
    let gelu: Bool
    let normEnabled: Bool
    
    public init(
        channels: Int,
        compress: Float = 4,
        depth: Int = 2,
        init_value: Float = 1e-4,
        norm: Bool = true,
        attn: Bool = false,
        heads: Int = 4,
        ndecay: Int = 4,
        lstm: Bool = false,
        gelu: Bool = true,
        kernel: Int = 3,
        dilate: Bool = true
    ) {
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        self.gelu = gelu
        self.normEnabled = norm
        
        let actualDilate = depth > 0
        let hidden = Int(Float(channels) / compress)
        
        // Create layer 0
        if self.depth > 0 {
            let dilation0 = actualDilate ? 1 : 1
            let padding0 = dilation0 * (kernel / 2)
            
            self.conv0_0 = Conv1d(inputChannels: channels, outputChannels: hidden, kernelSize: kernel, stride: 1, padding: padding0, dilation: dilation0)
            if norm {
                self.norm0_0 = GroupNorm(groupCount: 1, dimensions: hidden, pytorchCompatible: true)
            }
            self.conv0_1 = Conv1d(inputChannels: hidden, outputChannels: 2 * channels, kernelSize: 1)
            if norm {
                self.norm0_1 = GroupNorm(groupCount: 1, dimensions: 2 * channels, pytorchCompatible: true)
            }
            self.scale0 = LayerScale(channels: channels, init_value: init_value)
        }
        
        // Create layer 1
        if self.depth > 1 {
            let dilation1 = actualDilate ? 2 : 1
            let padding1 = dilation1 * (kernel / 2)
            
            self.conv1_0 = Conv1d(inputChannels: channels, outputChannels: hidden, kernelSize: kernel, stride: 1, padding: padding1, dilation: dilation1)
            if norm {
                self.norm1_0 = GroupNorm(groupCount: 1, dimensions: hidden, pytorchCompatible: true)
            }
            self.conv1_1 = Conv1d(inputChannels: hidden, outputChannels: 2 * channels, kernelSize: 1)
            if norm {
                self.norm1_1 = GroupNorm(groupCount: 1, dimensions: 2 * channels, pytorchCompatible: true)
            }
            self.scale1 = LayerScale(channels: channels, init_value: init_value)
        }
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var output = x
        
        // Layer 0
        if let c0 = conv0_0, let c1 = conv0_1, let sc = scale0 {
            var y = c0(output)
            if let n = norm0_0 { y = n(y) }
            y = gelu ? GELU()(y) : relu(y)
            y = c1(y)
            if let n = norm0_1 { y = n(y) }
            y = GLU(axis: -1)(y)
            y = sc(y)
            output = output + y
        }
        
        // Layer 1
        if let c0 = conv1_0, let c1 = conv1_1, let sc = scale1 {
            var y = c0(output)
            if let n = norm1_0 { y = n(y) }
            y = gelu ? GELU()(y) : relu(y)
            y = c1(y)
            if let n = norm1_1 { y = n(y) }
            y = GLU(axis: -1)(y)
            y = sc(y)
            output = output + y
        }
        
        return output
    }
}

// MARK: - HEncLayer

/// Hybrid Encoder Layer for frequency or time domain
public class HEncLayer: Module {
    let norm_groups: Int
    let empty: Bool
    let context: Int
    let pad_enabled: Bool
    let kernel_size: Int
    let stride: Int
    let freq: Bool
    
    // Using concrete types instead of protocols
    @ModuleInfo var conv2d: Conv2d?
    @ModuleInfo var conv1d: Conv1d?
    @ModuleInfo var norm1: GroupNorm?
    @ModuleInfo var rewrite2d: Conv2d?
    @ModuleInfo var rewrite1d: Conv1d?
    @ModuleInfo var norm2: GroupNorm?
    @ModuleInfo var dconv: DConv?
    
    public init(
        chin: Int,
        chout: Int,
        kernel_size: Int = 8,
        stride: Int = 4,
        norm_groups: Int = 1,
        empty: Bool = false,
        freq: Bool = true,
        dconv: Bool = true,
        norm: Bool = true,
        context: Int = 0,
        dconv_depth: Int = 2,
        dconv_compress: Float = 8,
        dconv_init: Float = 1e-3,
        dconv_gelu: Bool = true,
        pad: Bool = true,
        rewrite: Bool = true
    ) {
        self.norm_groups = norm_groups
        self.empty = empty
        self.context = context
        self.pad_enabled = pad
        self.kernel_size = kernel_size
        self.stride = stride
        self.freq = freq
        
        // Build convolution
        let convPadding = pad ? kernel_size / 4 : 0
        
        if freq {
            // Conv2d for freq branch: kernel (1, K), stride (1, S)
            self.conv2d = Conv2d(
                inputChannels: chin,
                outputChannels: chout,
                kernelSize: IntOrPair([1, kernel_size]),
                stride: IntOrPair([1, stride]),
                padding: IntOrPair([0, convPadding])
            )
        } else {
            // Conv1d for time branch
            self.conv1d = Conv1d(
                inputChannels: chin,
                outputChannels: chout,
                kernelSize: kernel_size,
                stride: stride,
                padding: convPadding
            )
        }
        
        if !empty {
            // Norm
            if norm {
                self.norm1 = GroupNorm(groupCount: norm_groups, dimensions: chout, pytorchCompatible: true)
            }
            
            // Rewrite conv
            if rewrite {
                let k_rw = 1 + 2 * context
                if freq {
                    self.rewrite2d = Conv2d(
                        inputChannels: chout,
                        outputChannels: 2 * chout,
                        kernelSize: IntOrPair(k_rw),
                        stride: IntOrPair(1),
                        padding: IntOrPair(context)
                    )
                } else {
                    self.rewrite1d = Conv1d(
                        inputChannels: chout,
                        outputChannels: 2 * chout,
                        kernelSize: k_rw,
                        stride: 1,
                        padding: context
                    )
                }
                if norm {
                    self.norm2 = GroupNorm(groupCount: norm_groups, dimensions: 2 * chout, pytorchCompatible: true)
                }
            }
            
            // DConv
            if dconv {
                self.dconv = DConv(
                    channels: chout,
                    compress: dconv_compress,
                    depth: dconv_depth,
                    init_value: dconv_init,
                    gelu: dconv_gelu
                )
            }
        }
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, inject: MLXArray? = nil) -> MLXArray {
        var input = x
        
        // Squeeze freq dim if time branch
        if !freq && input.ndim == 4 {
            input = input.squeezed(axis: 2)
        }
        
        // Handle stride alignment for time branch
        if !freq {
            let le = input.shape[1]  // (B, T, C)
            if le % stride != 0 {
                let extraPad = stride - (le % stride)
                input = MLX.padded(input, widths: [IntOrPair(0), IntOrPair([0, extraPad]), IntOrPair(0)])
            }
        }
        
        // Conv
        var y: MLXArray
        if freq, let c = conv2d {
            y = c(input)
        } else if let c = conv1d {
            y = c(input)
        } else {
            y = input
        }
        
        if empty {
            return y
        }
        
        // Inject
        if let inj = inject {
            y = y + inj
        }
        
        // Norm1 + GELU
        if let n = norm1 {
            y = n(y)
        }
        y = GELU()(y)
        
        // DConv (before rewrite)
        if let dc = dconv {
            if freq {
                // y is (B, T, F, C)
                let B = y.shape[0]
                let T = y.shape[1]
                let F = y.shape[2]
                let C = y.shape[3]
                // Reshape to (B*F, T, C) for DConv
                var yReshaped = y.transposed(0, 2, 1, 3).reshaped([B * F, T, C])
                yReshaped = dc(yReshaped)
                // Reshape back
                y = yReshaped.reshaped([B, F, T, C]).transposed(0, 2, 1, 3)
            } else {
                y = dc(y)
            }
        }
        
        // Rewrite
        if freq, let rw = rewrite2d {
            var z = rw(y)
            if let n = norm2 { z = n(z) }
            z = GLU(axis: -1)(z)
            return z
        } else if let rw = rewrite1d {
            var z = rw(y)
            if let n = norm2 { z = n(z) }
            z = GLU(axis: -1)(z)
            return z
        } else {
            return y
        }
    }
}

// MARK: - HDecLayer

/// Hybrid Decoder Layer for frequency or time domain
public class HDecLayer: Module {
    let freq: Bool
    let empty: Bool
    let last: Bool
    let pad_amount: Int
    let kernel_size: Int
    let stride: Int
    
    @ModuleInfo var conv_tr2d: ConvTransposed2d?
    @ModuleInfo var conv_tr1d: ConvTransposed1d?
    @ModuleInfo var norm2: GroupNorm?
    @ModuleInfo var rewrite2d: Conv2d?
    @ModuleInfo var rewrite1d: Conv1d?
    @ModuleInfo var norm1: GroupNorm?
    @ModuleInfo var dconv: DConv?
    
    public init(
        chin: Int,
        chout: Int,
        last: Bool = false,
        kernel_size: Int = 8,
        stride: Int = 4,
        norm_groups: Int = 1,
        empty: Bool = false,
        freq: Bool = true,
        dconv: Bool = true,
        norm: Bool = true,
        context: Int = 1,
        dconv_depth: Int = 2,
        dconv_compress: Float = 8,
        dconv_init: Float = 1e-3,
        dconv_gelu: Bool = true,
        pad: Bool = true,
        context_freq: Bool = true,
        rewrite: Bool = true
    ) {
        self.freq = freq
        self.empty = empty
        self.last = last
        self.pad_amount = pad ? kernel_size / 4 : 0
        self.kernel_size = kernel_size
        self.stride = stride
        
        if freq {
            self.conv_tr2d = ConvTransposed2d(
                inputChannels: chin,
                outputChannels: chout,
                kernelSize: IntOrPair([1, kernel_size]),
                stride: IntOrPair([1, stride])
            )
        } else {
            self.conv_tr1d = ConvTransposed1d(
                inputChannels: chin,
                outputChannels: chout,
                kernelSize: kernel_size,
                stride: stride
            )
        }
        
        if norm {
            self.norm2 = GroupNorm(groupCount: norm_groups, dimensions: chout, pytorchCompatible: true)
        }
        
        if rewrite && !empty {
            let k_rw = 1 + 2 * context
            if freq {
                self.rewrite2d = Conv2d(
                    inputChannels: chin,
                    outputChannels: 2 * chin,
                    kernelSize: IntOrPair(k_rw),
                    stride: IntOrPair(1),
                    padding: IntOrPair(context)
                )
            } else {
                self.rewrite1d = Conv1d(
                    inputChannels: chin,
                    outputChannels: 2 * chin,
                    kernelSize: k_rw,
                    stride: 1,
                    padding: context
                )
            }
            if norm {
                self.norm1 = GroupNorm(groupCount: norm_groups, dimensions: 2 * chin, pytorchCompatible: true)
            }
        }
        
        if dconv && !empty {
            self.dconv = DConv(
                channels: chin,
                compress: dconv_compress,
                depth: dconv_depth,
                init_value: dconv_init,
                gelu: dconv_gelu
            )
        }
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, skip: MLXArray?, length: Int? = nil) -> (MLXArray, MLXArray?) {
        var input = x
        
        if empty { return (input, nil) }
        
        if let sk = skip { input = input + sk }
        
        var y: MLXArray
        if freq, let rw = rewrite2d {
            var z = rw(input)
            if let n = norm1 { z = n(z) }
            z = GLU(axis: -1)(z)
            y = z
        } else if let rw = rewrite1d {
            var z = rw(input)
            if let n = norm1 { z = n(z) }
            z = GLU(axis: -1)(z)
            y = z
        } else {
            y = input
        }
        
        if let dc = dconv {
            if freq {
                let B = y.shape[0], T = y.shape[1], F = y.shape[2], C = y.shape[3]
                var yR = y.transposed(0, 2, 1, 3).reshaped([B * F, T, C])
                yR = dc(yR)
                y = yR.reshaped([B, F, T, C]).transposed(0, 2, 1, 3)
            } else {
                y = dc(y)
            }
        }
        
        var z: MLXArray
        if freq, let ct = conv_tr2d { z = ct(y) }
        else if let ct = conv_tr1d { z = ct(y) }
        else { z = y }
        
        if let n = norm2 { z = n(z) }
        
        if pad_amount > 0 {
            if freq {
                let fDim = z.shape[2]
                z = z[0..., 0..., pad_amount..<(fDim - pad_amount), 0...]
            } else {
                if let len = length {
                    z = z[0..., pad_amount..<(pad_amount + len), 0...]
                } else {
                    let tDim = z.shape[1]
                    z = z[0..., pad_amount..<(tDim - pad_amount), 0...]
                }
            }
        }
        
        if !last { z = GELU()(z) }
        
        return (z, y)
    }
}

// MARK: - ScaledEmbedding

public class ScaledEmbedding: Module {
    @ModuleInfo var embedding: Embedding
    let scale: Float
    let smooth: Bool
    
    public init(numEmbeddings: Int, embeddingDim: Int, scale: Float = 10.0, smooth: Bool = false) {
        self.embedding = Embedding(embeddingCount: numEmbeddings, dimensions: embeddingDim)
        self.scale = scale; self.smooth = smooth
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return embedding(x) * scale
    }
}
