import MLX
import MLXNN

/// Base ResUNet30 model containing all the layers
public class ResUNet30_Base: Module {
    // Pre-processing layers
    @ModuleInfo var bn0: FrequencyBatchNorm
    @ModuleInfo var pre_conv: Conv2d
    
    // Encoder blocks
    @ModuleInfo var encoder_block1: EncoderBlockRes1B
    @ModuleInfo var encoder_block2: EncoderBlockRes1B
    @ModuleInfo var encoder_block3: EncoderBlockRes1B
    @ModuleInfo var encoder_block4: EncoderBlockRes1B
    @ModuleInfo var encoder_block5: EncoderBlockRes1B
    @ModuleInfo var encoder_block6: EncoderBlockRes1B
    
    // Center block
    @ModuleInfo var conv_block7a: EncoderBlockRes1B
    
    // Decoder blocks
    @ModuleInfo var decoder_block1: DecoderBlockRes1B
    @ModuleInfo var decoder_block2: DecoderBlockRes1B
    @ModuleInfo var decoder_block3: DecoderBlockRes1B
    @ModuleInfo var decoder_block4: DecoderBlockRes1B
    @ModuleInfo var decoder_block5: DecoderBlockRes1B
    @ModuleInfo var decoder_block6: DecoderBlockRes1B
    
    // Post-processing layers
    @ModuleInfo var after_conv: Conv2d
    
    // STFT/ISTFT for audio processing
    let stft: STFT
    let istft: ISTFT
    
    let inputChannels: Int
    let outputChannels: Int
    
    let K = 3  // Number of masks: magnitude, cos, sin
    
    public init(
        inputChannels: Int = 1,
        outputChannels: Int = 1
    ) {
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        
        // Initialize STFT/ISTFT
        self.stft = STFT(n_fft: 2048, hop_length: 320, win_length: 2048)
        self.istft = ISTFT(n_fft: 2048, hop_length: 320, win_length: 2048)
        
        // Pre-processing
        // Note: STFT with n_fft=2048 produces 1025 frequency bins (2048/2 + 1)
        self.bn0 = FrequencyBatchNorm(numFeatures: 1025, momentum: 0.01)
        self.pre_conv = Conv2d(
            inputChannels: 1,  // Magnitude only after STFT
            outputChannels: 32,
            kernelSize: 1,
            padding: 0,
            bias: true
        )
        
        // Encoder
        self.encoder_block1 = EncoderBlockRes1B(inChannels: 32, outChannels: 32, downsample: (2, 2))
        self.encoder_block2 = EncoderBlockRes1B(inChannels: 32, outChannels: 64, downsample: (2, 2))
        self.encoder_block3 = EncoderBlockRes1B(inChannels: 64, outChannels: 128, downsample: (2, 2))
        self.encoder_block4 = EncoderBlockRes1B(inChannels: 128, outChannels: 256, downsample: (2, 2))
        self.encoder_block5 = EncoderBlockRes1B(inChannels: 256, outChannels: 384, downsample: (2, 2))
        self.encoder_block6 = EncoderBlockRes1B(inChannels: 384, outChannels: 384, downsample: (1, 2))  // Asymmetric
        
        // Center
        self.conv_block7a = EncoderBlockRes1B(inChannels: 384, outChannels: 384, downsample: (1, 1))
        
        // Decoder
        self.decoder_block1 = DecoderBlockRes1B(inChannels: 384, outChannels: 384, upsample: (1, 2))  // Asymmetric
        self.decoder_block2 = DecoderBlockRes1B(inChannels: 384, outChannels: 384, upsample: (2, 2))
        self.decoder_block3 = DecoderBlockRes1B(inChannels: 384, outChannels: 256, upsample: (2, 2))
        self.decoder_block4 = DecoderBlockRes1B(inChannels: 256, outChannels: 128, upsample: (2, 2))
        self.decoder_block5 = DecoderBlockRes1B(inChannels: 128, outChannels: 64, upsample: (2, 2))
        self.decoder_block6 = DecoderBlockRes1B(inChannels: 64, outChannels: 32, upsample: (2, 2))
        
        // Post-processing
        // Output 3 channels per output channel: magnitude mask, cos mask, sin mask
        self.after_conv = Conv2d(
            inputChannels: 32,
            outputChannels: outputChannels * 3,  // K=3 for mag, cos, sin masks
            kernelSize: 1,
            padding: 0
        )
        
        super.init()
        
    }
    
    public func callAsFunction(_ inputs: [String: MLXArray], filmDict: [String: Any]) -> [String: MLXArray] {
        
        // Extract inputs - Python only uses magnitude for the main processing
        guard let specMag = inputs["spectrogram_magnitude"],
              let specCos = inputs["spectrogram_cosine"],
              let specSin = inputs["spectrogram_sine"] else {
            fatalError("Missing required inputs")
        }
        
        
        // Python uses only magnitude for main processing
        var x = specMag
        
        // Store original time length
        let originLen = x.shape[1]
        
        // Input is [B, T, F, C] where F=1025, C=1
        // Python does: (B, C, T, F) -> (B, F, T, C)
        // Since we have [B, T, F, C], we need to transpose to [B, F, T, C]
        x = x.transposed(0, 2, 1, 3)
        
        // Pad time dimension to ensure divisibility by 32
        let timeDim = x.shape[2]
        let padAmount = (32 - timeDim % 32) % 32
        if padAmount > 0 {
            x = MLX.padded(x, widths: [
                IntOrPair(0),  // batch
                IntOrPair(0),  // freq
                IntOrPair([0, padAmount]),  // time
                IntOrPair(0)   // channels
            ])
        }
        
        // FiLM parameters are passed in from the parent ResUNet30
        
        // Apply frequency batch normalization
        
        x = bn0(x)
        
        // Transpose back to MLX format: (B, F, T, C) -> (B, T, F, C)
        x = x.transposed(0, 2, 1, 3)
        
        // Now trim to 512 for even division in the network
        if x.shape[2] % 2 == 1 {
            x = x[0..., 0..., 0..<(x.shape[2]-1), 0...]
        }
        
        // Pre-conv (simple Conv2d, no FiLM)
        x = pre_conv(x)
        
        // Encoder with skip connections
        let encoder1Dict = filmDict["encoder_block1"] as! [String: Any]
        let (x1_pooled, x1_beforePool) = encoder_block1(x, filmDict: encoder1Dict)
        
        let encoder2Dict = filmDict["encoder_block2"] as! [String: Any]
        let (x2_pooled, x2_beforePool) = encoder_block2(x1_pooled, filmDict: encoder2Dict)
        
        let encoder3Dict = filmDict["encoder_block3"] as! [String: Any]
        let (x3_pooled, x3_beforePool) = encoder_block3(x2_pooled, filmDict: encoder3Dict)
        
        let encoder4Dict = filmDict["encoder_block4"] as! [String: Any]
        let (x4_pooled, x4_beforePool) = encoder_block4(x3_pooled, filmDict: encoder4Dict)
        
        let encoder5Dict = filmDict["encoder_block5"] as! [String: Any]
        let (x5_pooled, x5_beforePool) = encoder_block5(x4_pooled, filmDict: encoder5Dict)
        
        let encoder6Dict = filmDict["encoder_block6"] as! [String: Any]
        let (x6_pooled, x6_beforePool) = encoder_block6(x5_pooled, filmDict: encoder6Dict)
        
        
        // Center block
        let centerDict = filmDict["conv_block7a"] as! [String: Any]
        let (x_center, _) = conv_block7a(x6_pooled, filmDict: centerDict)
        x = x_center
        
        
        // Decoder with skip connections
        let decoder1Dict = filmDict["decoder_block1"] as! [String: Any]
        x = decoder_block1(x, skipConnection: x6_beforePool, filmDict: decoder1Dict)
        
        let decoder2Dict = filmDict["decoder_block2"] as! [String: Any]
        x = decoder_block2(x, skipConnection: x5_beforePool, filmDict: decoder2Dict)
        
        let decoder3Dict = filmDict["decoder_block3"] as! [String: Any]
        x = decoder_block3(x, skipConnection: x4_beforePool, filmDict: decoder3Dict)
        
        let decoder4Dict = filmDict["decoder_block4"] as! [String: Any]
        x = decoder_block4(x, skipConnection: x3_beforePool, filmDict: decoder4Dict)
        
        let decoder5Dict = filmDict["decoder_block5"] as! [String: Any]
        x = decoder_block5(x, skipConnection: x2_beforePool, filmDict: decoder5Dict)
        
        let decoder6Dict = filmDict["decoder_block6"] as! [String: Any]
        x = decoder_block6(x, skipConnection: x1_beforePool, filmDict: decoder6Dict)
        
        // After conv (simple Conv2d, no FiLM)
        x = after_conv(x)
        
        // Recover shape - pad frequency back by 1
        x = MLX.padded(x, widths: [
            IntOrPair(0),  // batch
            IntOrPair(0),  // time
            IntOrPair([0, 1]),  // freq - pad by 1 at end
            IntOrPair(0)   // channels
        ])
        
        // Trim to original time length
        x = x[0..., 0..<originLen, 0..., 0...]
        
        // Trim frequency bins to match input magnitude shape
        x = x[0..., 0..., 0..<specMag.shape[2], 0...]
        
        
        // Return the feature maps - the caller will handle mask extraction
        let outputs = [
            "output": x,
            "magnitude": specMag,
            "cosine": specCos,
            "sine": specSin
        ]
        
        
        return outputs
    }
    
    /// Convert waveform to magnitude and phase spectrograms
    private func wavToSpectrogramPhase(_ mixtures: MLXArray) -> (mag: MLXArray, cos: MLXArray, sin: MLXArray) {
        // mixtures: (B, channels, samples)
        
        // let batchSize = mixtures.shape[0]
        // let channels = mixtures.shape[1]
        
        // Use STFT class which handles shapes properly
        let (real, imag) = stft(mixtures)
        
        // STFT returns (batch, channels, time, freq)
        
        
        // Calculate magnitude and phase
        let mag = MLX.sqrt(real * real + imag * imag)
        let eps: Float = 1e-10
        let cos = real / MLX.maximum(mag, eps)
        let sin = imag / MLX.maximum(mag, eps)
        
        // Output format is (B, channels, T, F)
        return (mag: mag, cos: cos, sin: sin)
    }
    
    /// Convert feature maps to waveform
    private func featureMapsToWav(
        inputTensor: MLXArray,
        sp: MLXArray,
        sinIn: MLXArray,
        cosIn: MLXArray,
        audioLength: Int
    ) -> MLXArray {
        // inputTensor: (B, T, F, C*K)
        // sp, sinIn, cosIn: (B, C, T, F)
        
        let batchSize = inputTensor.shape[0]
        let T = inputTensor.shape[1]
        let F = inputTensor.shape[2]
        
        // Reshape input tensor to separate K dimension
        let x = inputTensor.reshaped([batchSize, T, F, outputChannels, K])
        
        // Extract masks
        let maskMag = MLX.sigmoid(x[0..., 0..., 0..., 0..., 0])  // (B, T, F, C)
        let maskReal = MLX.tanh(x[0..., 0..., 0..., 0..., 1])
        let maskImag = MLX.tanh(x[0..., 0..., 0..., 0..., 2])
        
        
        // Convert to (B, C, T, F) format
        let maskMagT = maskMag.transposed(0, 3, 1, 2)
        let maskRealT = maskReal.transposed(0, 3, 1, 2)
        let maskImagT = maskImag.transposed(0, 3, 1, 2)
        
        // Calculate mask phase
        let maskPhaseMag = MLX.sqrt(maskRealT * maskRealT + maskImagT * maskImagT)
        let eps: Float = 1e-10
        let maskCos = maskRealT / MLX.maximum(maskPhaseMag, eps)
        let maskSin = maskImagT / MLX.maximum(maskPhaseMag, eps)
        
        // Apply masks
        let outCos = cosIn * maskCos - sinIn * maskSin
        let outSin = sinIn * maskCos + cosIn * maskSin
        
        
        // Calculate magnitude
        let outMag = MLX.maximum(sp * maskMagT, 0)
        
        // Calculate real and imaginary parts
        let outReal = outMag * outCos
        let outImag = outMag * outSin
        
        // ISTFT expects (batch, channels, time, freq) format
        // outReal, outImag are already in (B, C, T, F) format
        
        // ISTFT
        let waveform = istft(real: outReal, imag: outImag, length: audioLength)
        
        // Ensure output is (B, output_channels, audio_samples)
        if waveform.ndim == 2 {
            // Single channel case: (B, samples) -> (B, 1, samples)
            return waveform.expandedDimensions(axis: 1)
        } else {
            // Multi-channel case: (B, C, samples)
            return waveform
        }
    }
    
    /// Forward pass that processes audio directly
    public func callAsFunction(_ mixtures: MLXArray, filmDict: [String: Any]) -> [String: MLXArray] {
        let audioLength = mixtures.shape[2]
        
        // Convert to spectrogram
        let (mag, cosIn, sinIn) = wavToSpectrogramPhase(mixtures)
        // mag, cosIn, sinIn have shape (B, C, T, F)
        
        // For frequency batch norm, we need to transpose to have frequency dimension
        // From (B, C, T, F) to (B, F, T, C)
        var x = mag
        x = x.transposed(0, 3, 2, 1)
        
        // Batch normalization
        x = bn0(x)
        
        // Transpose back to MLX format: (B, F, T, C) -> (B, T, F, C)
        x = x.transposed(0, 2, 1, 3)
        
        // Store original time length
        let originLen = x.shape[1]
        
        // Pad time dimension to ensure divisibility by 32
        let timeDim = x.shape[1]
        let padAmount = (32 - timeDim % 32) % 32
        if padAmount > 0 {
            x = MLX.padded(x, widths: [
                IntOrPair(0),  // batch
                IntOrPair([0, padAmount]),  // time
                IntOrPair(0),  // freq
                IntOrPair(0)   // channels
            ])
        }
        
        // Let frequency bins be evenly divided by 2
        if x.shape[2] % 2 == 1 {
            x = x[0..., 0..., 0..<(x.shape[2]-1), 0...]  // Trim last frequency bin
        }
        
        // Pre-conv
        x = pre_conv(x)
        
        // Encoder with skip connections
        let encoder1Dict = filmDict["encoder_block1"] as! [String: Any]
        let (x1_pooled, x1_beforePool) = encoder_block1(x, filmDict: encoder1Dict)
        
        
        let encoder2Dict = filmDict["encoder_block2"] as! [String: Any]
        let (x2_pooled, x2_beforePool) = encoder_block2(x1_pooled, filmDict: encoder2Dict)
        
        let encoder3Dict = filmDict["encoder_block3"] as! [String: Any]
        let (x3_pooled, x3_beforePool) = encoder_block3(x2_pooled, filmDict: encoder3Dict)
        
        let encoder4Dict = filmDict["encoder_block4"] as! [String: Any]
        let (x4_pooled, x4_beforePool) = encoder_block4(x3_pooled, filmDict: encoder4Dict)
        
        let encoder5Dict = filmDict["encoder_block5"] as! [String: Any]
        let (x5_pooled, x5_beforePool) = encoder_block5(x4_pooled, filmDict: encoder5Dict)
        
        let encoder6Dict = filmDict["encoder_block6"] as! [String: Any]
        let (x6_pooled, x6_beforePool) = encoder_block6(x5_pooled, filmDict: encoder6Dict)
        
        
        // Center block
        let centerDict = filmDict["conv_block7a"] as! [String: Any]
        let (x_center, _) = conv_block7a(x6_pooled, filmDict: centerDict)
        x = x_center
        
        
        // Decoder with skip connections
        let decoder1Dict = filmDict["decoder_block1"] as! [String: Any]
        x = decoder_block1(x, skipConnection: x6_beforePool, filmDict: decoder1Dict)
        
        let decoder2Dict = filmDict["decoder_block2"] as! [String: Any]
        x = decoder_block2(x, skipConnection: x5_beforePool, filmDict: decoder2Dict)
        
        let decoder3Dict = filmDict["decoder_block3"] as! [String: Any]
        x = decoder_block3(x, skipConnection: x4_beforePool, filmDict: decoder3Dict)
        
        let decoder4Dict = filmDict["decoder_block4"] as! [String: Any]
        x = decoder_block4(x, skipConnection: x3_beforePool, filmDict: decoder4Dict)
        
        let decoder5Dict = filmDict["decoder_block5"] as! [String: Any]
        x = decoder_block5(x, skipConnection: x2_beforePool, filmDict: decoder5Dict)
        
        let decoder6Dict = filmDict["decoder_block6"] as! [String: Any]
        x = decoder_block6(x, skipConnection: x1_beforePool, filmDict: decoder6Dict)
        
        // After conv (simple Conv2d, no FiLM)
        x = after_conv(x)
        
        // Recover shape (PyTorch pads by 1 in freq dimension first)
        x = MLX.padded(x, widths: [
            IntOrPair(0),  // batch
            IntOrPair(0),  // time
            IntOrPair([0, 1]),  // freq - pad by 1 at end
            IntOrPair(0)   // channels
        ])
        
        // Trim to original time length
        x = x[0..., 0..<originLen, 0..., 0...]
        
        // Trim frequency bins to match input magnitude shape
        x = x[0..., 0..., 0..<mag.shape[3], 0...]
        
        // Convert to waveform
        let waveform = featureMapsToWav(
            inputTensor: x,
            sp: mag,
            sinIn: sinIn,
            cosIn: cosIn,
            audioLength: audioLength
        )
        
        return ["waveform": waveform]
    }
}

/// ResUNet30 model with FiLM conditioning - wrapper around base model
public class ResUNet30: Module {
    @ModuleInfo var base: ResUNet30_Base
    @ModuleInfo var film: HierarchicalFiLM
    
    let inputChannels: Int
    let outputChannels: Int
    let conditionSize: Int
    
    // Compiled forward function for performance
    private var forwardCompiled: ((MLXArray, MLXArray) -> MLXArray)?
    
    public init(
        inputChannels: Int = 1,
        outputChannels: Int = 1,
        conditionSize: Int = 527
    ) {
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.conditionSize = conditionSize
        
        // Initialize base model and FiLM
        self.base = ResUNet30_Base(
            inputChannels: inputChannels,
            outputChannels: outputChannels
        )
        self.film = HierarchicalFiLM(conditionSize: conditionSize)
        
        super.init()
        
        // Initialize compiled forward function for better performance
        self.forwardCompiled = MLX.compile(self.forwardCore)
    }
    
    /// Core forward function that can be compiled (takes MLXArrays directly)
    private func forwardCore(_ mixture: MLXArray, _ condition: MLXArray) -> MLXArray {
        // Expand condition dimensions if needed
        var conditions = condition
        if conditions.ndim == 1 {
            conditions = conditions.expandedDimensions(axis: 0)
        }
        
        // Repeat conditions for each sample in batch if needed
        let batchSize = mixture.shape[0]
        if conditions.shape[0] == 1 && batchSize > 1 {
            conditions = repeated(conditions, count: batchSize, axis: 0)
        }
        
        // Generate all FiLM parameters
        let filmDict = film.generateAllBetas(conditions)
        
        // Forward through base model with audio
        let outputs = base(mixture, filmDict: filmDict)
        
        // Return the waveform output
        return outputs["waveform"] ?? MLXArray()
    }
    
    /// Dictionary-based forward function (for compatibility)
    private func forward(_ inputDict: [String: MLXArray]) -> [String: MLXArray] {
        // Extract inputs
        guard let mixture = inputDict["mixture"],
              let condition = inputDict["condition"] else {
            fatalError("Missing mixture or condition input")
        }
        
        // Use compiled or regular forward
        let waveform: MLXArray
        if let compiledFn = forwardCompiled {
            waveform = compiledFn(mixture, condition)
        } else {
            waveform = forwardCore(mixture, condition)
        }
        
        return ["waveform": waveform]
    }
    
    public func callAsFunction(_ inputDict: [String: MLXArray]) -> [String: MLXArray] {
        // Always use the forward function which handles compilation internally
        return forward(inputDict)
    }
    
    /// Enable or disable compilation
    public func setCompile(_ enable: Bool) {
        if enable {
            forwardCompiled = MLX.compile(self.forwardCore)
        } else {
            forwardCompiled = nil
        }
    }

    /// Warm the normalization entry used by this model's owned STFT pair.
    func prewarmNormalization(forSignalLength signalLength: Int) {
        let numFrames = base.stft.frameCount(forSignalLength: signalLength)
        base.istft.prewarmNormalization(numFrames: numFrames)
    }
}
