import MLX
import MLXNN

// Helper function for leaky ReLU - using MLX's built-in function
func leakyReLU(_ x: MLXArray, negativeSlope: Float = 0.01) -> MLXArray {
    return leakyRelu(x, negativeSlope: negativeSlope)
}

/// Residual convolutional block with FiLM conditioning
public class ConvBlockRes: Module {
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var bn1: BatchNorm2d
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo var bn2: BatchNorm2d
    @ModuleInfo var shortcut: Conv2d?
    
    let inChannels: Int
    let outChannels: Int
    let stride: Int
    
    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        stride: Int = 1
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.stride = stride
        
        // First conv layer
        self.conv1 = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(kernelSize / 2),
            bias: false
        )
        self.bn1 = BatchNorm2d(numFeatures: inChannels, momentum: 0.01)
        
        // Second conv layer
        self.conv2 = Conv2d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(1),
            padding: IntOrPair(kernelSize / 2),
            bias: false
        )
        self.bn2 = BatchNorm2d(numFeatures: outChannels, momentum: 0.01)
        
        // Shortcut connection if needed
        if stride != 1 || inChannels != outChannels {
            self.shortcut = Conv2d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: 1,
                stride: IntOrPair(stride),
                padding: 0
            )
        }
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, beta1: MLXArray? = nil, beta2: MLXArray? = nil) -> MLXArray {
        // First block: BN -> FiLM -> ReLU -> Conv
        var out = bn1(x)
        
        // Apply FiLM modulation if provided
        if let beta1 = beta1 {
            out = out + beta1
        }
        
        out = leakyReLU(out, negativeSlope: 0.01)
        out = conv1(out)
        
        // Second block: BN -> FiLM -> ReLU -> Conv
        out = bn2(out)
        
        // Apply FiLM modulation if provided
        if let beta2 = beta2 {
            out = out + beta2
        }
        
        out = leakyReLU(out, negativeSlope: 0.01)
        out = conv2(out)
        
        // Shortcut connection
        let identity: MLXArray
        if let shortcut = shortcut {
            identity = shortcut(x)
        } else {
            identity = x
        }
        
        // Residual connection (no activation after)
        out = out + identity
        
        return out
    }
    
    public func callAsFunction(_ x: MLXArray, filmDict: [String: Any]) -> MLXArray {
        // Extract FiLM parameters from nested dict
        let convBlock1Dict = filmDict["conv_block1"] as! (beta1: MLXArray, beta2: MLXArray)
        return callAsFunction(x, beta1: convBlock1Dict.beta1, beta2: convBlock1Dict.beta2)
    }
}

/// Encoder block with residual connection and downsampling
public class EncoderBlockRes1B: Module {
    @ModuleInfo var conv_block1: ConvBlockRes
    
    let poolKernel: (Int, Int)
    let poolStride: (Int, Int)
    
    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        downsample: (Int, Int) = (2, 2)
    ) {
        self.poolKernel = downsample
        self.poolStride = downsample
        
        self.conv_block1 = ConvBlockRes(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: 1
        )
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, beta1: MLXArray? = nil, beta2: MLXArray? = nil) -> (pooled: MLXArray, beforePool: MLXArray) {
        // Apply conv block
        let convOutput = conv_block1(x, beta1: beta1, beta2: beta2)
        
        // Apply pooling for downsampling using built-in AvgPool2d
        let poolOp = AvgPool2d(kernelSize: IntOrPair(poolKernel), stride: IntOrPair(poolStride))
        let pooled = poolOp(convOutput)
        
        return (pooled: pooled, beforePool: convOutput)
    }
    
    public func callAsFunction(_ x: MLXArray, filmDict: [String: Any]) -> (pooled: MLXArray, beforePool: MLXArray) {
        // Extract FiLM parameters from nested dict
        let convBlock1Dict = filmDict["conv_block1"] as! (beta1: MLXArray, beta2: MLXArray)
        return callAsFunction(x, beta1: convBlock1Dict.beta1, beta2: convBlock1Dict.beta2)
    }
}

// MARK: - Decoder Block with FiLM modulation

/// Decoder block with FiLM modulation and transposed convolution
public class DecoderBlockRes1B: Module {
    @ModuleInfo var bn1: BatchNorm2d
    @ModuleInfo var conv1: Conv2d  // This will hold the transposed conv weight - named to match Python
    @ModuleInfo var conv_block2: ConvBlockRes
    
    let inChannels: Int
    let outChannels: Int
    let upsample: (Int, Int)
    
    public init(
        inChannels: Int,
        outChannels: Int,
        upsample: (Int, Int) = (2, 2)
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.upsample = upsample
        
        // Batch norm for input
        self.bn1 = BatchNorm2d(numFeatures: inChannels, momentum: 0.01)
        
        // Conv2d that will hold the transposed convolution weights
        // The actual operation will use convTransposed2d
        self.conv1 = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(upsample),  // kernel size = stride for transposed conv
            stride: IntOrPair(1),  // Not used in transposed conv
            padding: IntOrPair(0),
            bias: false
        )
        
        // Conv block for processing concatenated features
        self.conv_block2 = ConvBlockRes(
            inChannels: outChannels * 2,  // After concatenation with skip
            outChannels: outChannels,
            kernelSize: 3,
            stride: 1
        )
        
        super.init()
    }
    
    public func callAsFunction(
        _ x: MLXArray,
        skipConnection: MLXArray,
        beta1: MLXArray? = nil,
        beta2: MLXArray? = nil
    ) -> MLXArray {
        // Apply batch norm + FiLM
        var h = bn1(x)
        if let beta1 = beta1 {
            // FiLM parameters need to match the channel dimension
            // If shapes don't match, broadcast beta1 properly
            h = h + beta1
        }
        
        // Activation
        h = leakyReLU(h, negativeSlope: 0.01)
        
        // Transposed convolution for upsampling
        // Use the weight from conv1 for the transposed convolution
        h = MLX.convTransposed2d(
            h,
            conv1.weight,
            stride: IntOrPair(upsample),
            padding: IntOrPair(0)
        )
        
        
        // Handle size mismatch by cropping the larger tensor
        var hAligned = h
        var skipAligned = skipConnection
        
        if h.shape[1] != skipConnection.shape[1] || h.shape[2] != skipConnection.shape[2] {
            let targetH = min(h.shape[1], skipConnection.shape[1])
            let targetW = min(h.shape[2], skipConnection.shape[2])
            
            hAligned = h[0..., 0..<targetH, 0..<targetW, 0...]
            skipAligned = skipConnection[0..., 0..<targetH, 0..<targetW, 0...]
            
        }
        
        // Concatenate with skip connection
        let concatenated = MLX.concatenated([hAligned, skipAligned], axis: 3)
        
        // Apply conv block with optional FiLM
        return conv_block2(concatenated, beta1: beta2, beta2: nil)
    }
    
    public func callAsFunction(
        _ x: MLXArray,
        skipConnection: MLXArray,
        filmDict: [String: Any]
    ) -> MLXArray {
        // Extract beta1 and conv_block2 dict from film dict
        let beta1 = filmDict["beta1"] as! MLXArray
        let convBlock2Dict = filmDict["conv_block2"] as! (beta1: MLXArray, beta2: MLXArray)
        
        // Apply batch norm + FiLM
        var h = bn1(x)
        h = h + beta1
        
        // Activation
        h = leakyReLU(h, negativeSlope: 0.01)
        
        // Transposed convolution for upsampling
        h = MLX.convTransposed2d(
            h,
            conv1.weight,
            stride: IntOrPair(upsample),
            padding: IntOrPair(0)
        )
        
        
        // Handle size mismatch by cropping the larger tensor
        var hAligned = h
        var skipAligned = skipConnection
        
        if h.shape[1] != skipConnection.shape[1] || h.shape[2] != skipConnection.shape[2] {
            let targetH = min(h.shape[1], skipConnection.shape[1])
            let targetW = min(h.shape[2], skipConnection.shape[2])
            
            hAligned = h[0..., 0..<targetH, 0..<targetW, 0...]
            skipAligned = skipConnection[0..., 0..<targetH, 0..<targetW, 0...]
            
        }
        
        // Concatenate with skip connection
        let concatenated = MLX.concatenated([hAligned, skipAligned], axis: 3)
        
        // Apply conv block with FiLM
        return conv_block2(concatenated, beta1: convBlock2Dict.beta1, beta2: convBlock2Dict.beta2)
    }
}
