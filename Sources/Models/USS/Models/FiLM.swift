import MLX
import MLXNN

/// FiLM (Feature-wise Linear Modulation) module for conditional generation
public class FiLM: Module {
    @ModuleInfo var beta1: Linear
    @ModuleInfo var beta2: Linear
    
    public init(inputDim: Int, outputDim: Int) {
        // Linear layers to generate modulation parameters
        self.beta1 = Linear(inputDim, outputDim)
        self.beta2 = Linear(inputDim, outputDim)
        
        super.init()
    }
    
    public func callAsFunction(_ conditioning: MLXArray) -> (beta1: MLXArray, beta2: MLXArray) {
        // Generate beta parameters from conditioning vector
        let b1 = beta1(conditioning)
        let b2 = beta2(conditioning)
        
        // Reshape for broadcasting with feature maps
        // From (batch, features) to (batch, 1, 1, features) for 4D tensors
        let beta1Reshaped = b1.expandedDimensions(axes: [1, 2])
        let beta2Reshaped = b2.expandedDimensions(axes: [1, 2])
        
        return (beta1: beta1Reshaped, beta2: beta2Reshaped)
    }
}

/// Hierarchical FiLM module containing all conditioning layers for ResUNet
public class HierarchicalFiLM: Module {
    
    // Encoder FiLMs - these need nested structure for conv_block1
    @ModuleInfo var encoder_block1_conv_block1_beta1: Linear
    @ModuleInfo var encoder_block1_conv_block1_beta2: Linear
    @ModuleInfo var encoder_block2_conv_block1_beta1: Linear
    @ModuleInfo var encoder_block2_conv_block1_beta2: Linear
    @ModuleInfo var encoder_block3_conv_block1_beta1: Linear
    @ModuleInfo var encoder_block3_conv_block1_beta2: Linear
    @ModuleInfo var encoder_block4_conv_block1_beta1: Linear
    @ModuleInfo var encoder_block4_conv_block1_beta2: Linear
    @ModuleInfo var encoder_block5_conv_block1_beta1: Linear
    @ModuleInfo var encoder_block5_conv_block1_beta2: Linear
    @ModuleInfo var encoder_block6_conv_block1_beta1: Linear
    @ModuleInfo var encoder_block6_conv_block1_beta2: Linear
    
    // Center block FiLM - also nested
    @ModuleInfo var conv_block7a_conv_block1_beta1: Linear
    @ModuleInfo var conv_block7a_conv_block1_beta2: Linear
    
    // Decoder FiLMs - need beta1 at block level and nested conv_block2
    @ModuleInfo var decoder_block1_beta1: Linear
    @ModuleInfo var decoder_block1_conv_block2_beta1: Linear
    @ModuleInfo var decoder_block1_conv_block2_beta2: Linear
    
    @ModuleInfo var decoder_block2_beta1: Linear
    @ModuleInfo var decoder_block2_conv_block2_beta1: Linear
    @ModuleInfo var decoder_block2_conv_block2_beta2: Linear
    
    @ModuleInfo var decoder_block3_beta1: Linear
    @ModuleInfo var decoder_block3_conv_block2_beta1: Linear
    @ModuleInfo var decoder_block3_conv_block2_beta2: Linear
    
    @ModuleInfo var decoder_block4_beta1: Linear
    @ModuleInfo var decoder_block4_conv_block2_beta1: Linear
    @ModuleInfo var decoder_block4_conv_block2_beta2: Linear
    
    @ModuleInfo var decoder_block5_beta1: Linear
    @ModuleInfo var decoder_block5_conv_block2_beta1: Linear
    @ModuleInfo var decoder_block5_conv_block2_beta2: Linear
    
    @ModuleInfo var decoder_block6_beta1: Linear
    @ModuleInfo var decoder_block6_conv_block2_beta1: Linear
    @ModuleInfo var decoder_block6_conv_block2_beta2: Linear
    
    
    let conditionSize: Int
    
    public init(conditionSize: Int = 527) {
        self.conditionSize = conditionSize
        
        // Initialize all FiLM layers with appropriate output dimensions
        
        // Encoder blocks - nested structure
        self.encoder_block1_conv_block1_beta1 = Linear(conditionSize, 32)
        self.encoder_block1_conv_block1_beta2 = Linear(conditionSize, 32)
        
        self.encoder_block2_conv_block1_beta1 = Linear(conditionSize, 32)  // Input channels to encoder2
        self.encoder_block2_conv_block1_beta2 = Linear(conditionSize, 64)  // Output channels
        
        self.encoder_block3_conv_block1_beta1 = Linear(conditionSize, 64)   // Input channels to encoder3
        self.encoder_block3_conv_block1_beta2 = Linear(conditionSize, 128)  // Output channels
        
        self.encoder_block4_conv_block1_beta1 = Linear(conditionSize, 128)  // Input channels to encoder4
        self.encoder_block4_conv_block1_beta2 = Linear(conditionSize, 256)  // Output channels
        
        self.encoder_block5_conv_block1_beta1 = Linear(conditionSize, 256)  // Input channels to encoder5
        self.encoder_block5_conv_block1_beta2 = Linear(conditionSize, 384)  // Output channels
        
        self.encoder_block6_conv_block1_beta1 = Linear(conditionSize, 384)
        self.encoder_block6_conv_block1_beta2 = Linear(conditionSize, 384)
        
        // Center block: 384 channels
        self.conv_block7a_conv_block1_beta1 = Linear(conditionSize, 384)
        self.conv_block7a_conv_block1_beta2 = Linear(conditionSize, 384)
        
        // Decoder blocks - beta1 for main block, conv_block2 nested
        self.decoder_block1_beta1 = Linear(conditionSize, 384)  // input: 384
        self.decoder_block1_conv_block2_beta1 = Linear(conditionSize, 768)  // concat: 384*2
        self.decoder_block1_conv_block2_beta2 = Linear(conditionSize, 384)  // output: 384
        
        self.decoder_block2_beta1 = Linear(conditionSize, 384)  // input: 384
        self.decoder_block2_conv_block2_beta1 = Linear(conditionSize, 768)  // concat: 384*2
        self.decoder_block2_conv_block2_beta2 = Linear(conditionSize, 384)  // output: 384
        
        self.decoder_block3_beta1 = Linear(conditionSize, 384)  // input: 384
        self.decoder_block3_conv_block2_beta1 = Linear(conditionSize, 512)  // concat: 256*2
        self.decoder_block3_conv_block2_beta2 = Linear(conditionSize, 256)  // output: 256
        
        self.decoder_block4_beta1 = Linear(conditionSize, 256)  // input: 256
        self.decoder_block4_conv_block2_beta1 = Linear(conditionSize, 256)  // concat: 128*2
        self.decoder_block4_conv_block2_beta2 = Linear(conditionSize, 128)  // output: 128
        
        self.decoder_block5_beta1 = Linear(conditionSize, 128)  // input: 128
        self.decoder_block5_conv_block2_beta1 = Linear(conditionSize, 128)  // concat: 64*2
        self.decoder_block5_conv_block2_beta2 = Linear(conditionSize, 64)   // output: 64
        
        self.decoder_block6_beta1 = Linear(conditionSize, 64)   // input: 64
        self.decoder_block6_conv_block2_beta1 = Linear(conditionSize, 64)   // concat: 32*2
        self.decoder_block6_conv_block2_beta2 = Linear(conditionSize, 32)   // output: 32
        
        super.init()
    }
    
    /// Generate all FiLM parameters from conditioning vector with nested structure
    public func generateAllBetas(_ conditioning: MLXArray) -> [String: Any] {
        var betas: [String: Any] = [:]
        
        // Helper function to reshape betas for broadcasting
        func reshapeBeta(_ beta: MLXArray) -> MLXArray {
            return beta.expandedDimensions(axes: [1, 2])
        }
        
        
        // Encoder blocks - nested structure
        let encoder1_beta1_raw = encoder_block1_conv_block1_beta1(conditioning)
        let encoder1_beta2_raw = encoder_block1_conv_block1_beta2(conditioning)
        
        
        
        betas["encoder_block1"] = [
            "conv_block1": (
                beta1: reshapeBeta(encoder1_beta1_raw),
                beta2: reshapeBeta(encoder1_beta2_raw)
            )
        ]
        
        betas["encoder_block2"] = [
            "conv_block1": (
                beta1: reshapeBeta(encoder_block2_conv_block1_beta1(conditioning)),
                beta2: reshapeBeta(encoder_block2_conv_block1_beta2(conditioning))
            )
        ]
        
        betas["encoder_block3"] = [
            "conv_block1": (
                beta1: reshapeBeta(encoder_block3_conv_block1_beta1(conditioning)),
                beta2: reshapeBeta(encoder_block3_conv_block1_beta2(conditioning))
            )
        ]
        
        betas["encoder_block4"] = [
            "conv_block1": (
                beta1: reshapeBeta(encoder_block4_conv_block1_beta1(conditioning)),
                beta2: reshapeBeta(encoder_block4_conv_block1_beta2(conditioning))
            )
        ]
        
        betas["encoder_block5"] = [
            "conv_block1": (
                beta1: reshapeBeta(encoder_block5_conv_block1_beta1(conditioning)),
                beta2: reshapeBeta(encoder_block5_conv_block1_beta2(conditioning))
            )
        ]
        
        betas["encoder_block6"] = [
            "conv_block1": (
                beta1: reshapeBeta(encoder_block6_conv_block1_beta1(conditioning)),
                beta2: reshapeBeta(encoder_block6_conv_block1_beta2(conditioning))
            )
        ]
        
        // Center block - nested
        betas["conv_block7a"] = [
            "conv_block1": (
                beta1: reshapeBeta(conv_block7a_conv_block1_beta1(conditioning)),
                beta2: reshapeBeta(conv_block7a_conv_block1_beta2(conditioning))
            )
        ]
        
        // Decoder blocks - nested structure with beta1 at top level
        betas["decoder_block1"] = [
            "beta1": reshapeBeta(decoder_block1_beta1(conditioning)),
            "conv_block2": (
                beta1: reshapeBeta(decoder_block1_conv_block2_beta1(conditioning)),
                beta2: reshapeBeta(decoder_block1_conv_block2_beta2(conditioning))
            )
        ]
        
        betas["decoder_block2"] = [
            "beta1": reshapeBeta(decoder_block2_beta1(conditioning)),
            "conv_block2": (
                beta1: reshapeBeta(decoder_block2_conv_block2_beta1(conditioning)),
                beta2: reshapeBeta(decoder_block2_conv_block2_beta2(conditioning))
            )
        ]
        
        betas["decoder_block3"] = [
            "beta1": reshapeBeta(decoder_block3_beta1(conditioning)),
            "conv_block2": (
                beta1: reshapeBeta(decoder_block3_conv_block2_beta1(conditioning)),
                beta2: reshapeBeta(decoder_block3_conv_block2_beta2(conditioning))
            )
        ]
        
        betas["decoder_block4"] = [
            "beta1": reshapeBeta(decoder_block4_beta1(conditioning)),
            "conv_block2": (
                beta1: reshapeBeta(decoder_block4_conv_block2_beta1(conditioning)),
                beta2: reshapeBeta(decoder_block4_conv_block2_beta2(conditioning))
            )
        ]
        
        betas["decoder_block5"] = [
            "beta1": reshapeBeta(decoder_block5_beta1(conditioning)),
            "conv_block2": (
                beta1: reshapeBeta(decoder_block5_conv_block2_beta1(conditioning)),
                beta2: reshapeBeta(decoder_block5_conv_block2_beta2(conditioning))
            )
        ]
        
        betas["decoder_block6"] = [
            "beta1": reshapeBeta(decoder_block6_beta1(conditioning)),
            "conv_block2": (
                beta1: reshapeBeta(decoder_block6_conv_block2_beta1(conditioning)),
                beta2: reshapeBeta(decoder_block6_conv_block2_beta2(conditioning))
            )
        ]
        
        
        return betas
    }
}