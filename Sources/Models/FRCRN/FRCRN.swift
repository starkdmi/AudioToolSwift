import Foundation
import MLX
import MLXNN

/// MLX implementation of DCCRN (the core model in FRCRN_SE).
public class DCCRN: Module {
    // STFT parameters
    let win_len: Int
    let win_inc: Int
    let fft_len: Int
    
    // Conv1D-based STFT/iSTFT with learned kernels
    @ModuleInfo public var stft: ConvSTFT
    @ModuleInfo public var istft: ConviSTFT
    
    // UNet models
    @ModuleInfo public var unet: UNet
    @ModuleInfo public var unet2: UNet
    
    public init(
        win_len: Int = 640,
        win_inc: Int = 320,
        fft_len: Int = 640
    ) {
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        
        // Initialize STFT/iSTFT modules (weights loaded separately)
        self._stft.wrappedValue = ConvSTFT(winLen: win_len, winInc: win_inc, fftLen: fft_len)
        self._istft.wrappedValue = ConviSTFT(winLen: win_len, winInc: win_inc, fftLen: fft_len)
        
        // Initialize two UNet models
        self._unet.wrappedValue = UNet()
        self._unet2.wrappedValue = UNet()
        
        super.init()
    }
    
    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        // Input shape: (batch, length) or (batch, length, 1)
        
        // Compute STFT using FFT-based approach
        let (real, imag) = stft(inputs)
        // real, imag shape: (batch, freq, time)
        
        // Preprocessing to match PyTorch DCCRN
        // 1. Concatenate real and imag: [B, D*2, T]
        var cmp_spec = MLX.concatenated([real, imag], axis: 1)
        
        // 2. Add dimension: [B, 1, D*2, T]
        cmp_spec = cmp_spec.expandedDimensions(axis: 1)
        
        // 3. Split and rejoin: [B, 2, D, T]
        let feat_dim = real.shape[1]  // D (frequency bins)
        cmp_spec = MLX.concatenated([
            cmp_spec[0..., 0..., 0..<feat_dim, 0...],  // Real part
            cmp_spec[0..., 0..., feat_dim..., 0...]    // Imaginary part
        ], axis: 1)
        
        // 4. Add dimension: [B, 2, D, T, 1]
        cmp_spec = cmp_spec.expandedDimensions(axis: 4)
        
        // 5. Transpose: [B, 1, D, T, 2]
        cmp_spec = cmp_spec.transposed(0, 4, 2, 3, 1)
        
        // 6. Reshape for MLX UNet: [B, D, T, 1, 2]
        let cmp_spec_for_unet = cmp_spec.transposed(0, 2, 3, 1, 4)
        
        // Apply first UNet
        var unet1_out = unet(cmp_spec_for_unet)  // [B, D, T, 1, 2]
        
        // Reshape back for mask operations: [B, 1, D, T, 2]
        unet1_out = unet1_out.transposed(0, 3, 1, 2, 4)
        
        let cmp_mask1 = MLX.tanh(unet1_out)
        
        // Apply second UNet
        let unet1_out_for_unet2 = unet1_out.transposed(0, 2, 3, 1, 4)
        
        var unet2_out = unet2(unet1_out_for_unet2)
        
        // Reshape back for mask operations: [B, 1, D, T, 2]
        unet2_out = unet2_out.transposed(0, 3, 1, 2, 4)
        
        let cmp_mask2 = MLX.tanh(unet2_out)
        
        // Combine masks
        let mask = cmp_mask2 + cmp_mask1
        
        // Apply mask using complex multiplication formula
        let masked_real = cmp_spec[.ellipsis, 0] * mask[.ellipsis, 0] - cmp_spec[.ellipsis, 1] * mask[.ellipsis, 1]
        let masked_imag = cmp_spec[.ellipsis, 0] * mask[.ellipsis, 1] + cmp_spec[.ellipsis, 1] * mask[.ellipsis, 0]
        
        // Remove singleton dimensions: [B, D, T]
        let real_out = masked_real.squeezed(axis: 1)
        let imag_out = masked_imag.squeezed(axis: 1)
        
        // iSTFT to get time domain signal
        let output = istft(real_out, imag_out)
        
        return output
    }
}


/// MLX implementation of FRCRN_SE for 16kHz audio.
public class FRCRN_SE_16K: Module {
    @ModuleInfo public var model: DCCRN
    
    // Compiled forward function for performance
    private var forwardCompiled: ((MLXArray) -> MLXArray)?
    
    public override init() {
        self._model.wrappedValue = DCCRN(
            win_len: 640,
            win_inc: 320,
            fft_len: 640
        )
        super.init()
        
        // Initialize compiled forward function for better performance
        self.forwardCompiled = MLX.compile(self.forwardCore)
    }
    
    /// Core forward function that can be compiled
    private func forwardCore(_ inputs: MLXArray) -> MLXArray {
        // Ensure correct input shape
        var audio = inputs
        var squeeze_output = false
        
        if audio.ndim == 2 {
            audio = audio.expandedDimensions(axis: -1)
            squeeze_output = true
        }
        
        // Process through model
        var output = model(audio)
        
        // Match input shape
        if squeeze_output && output.ndim == 3 {
            output = output.squeezed(axis: -1)
        }
        
        return output
    }
    
    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        // Use compiled forward if available, otherwise use core
        if let compiledFn = forwardCompiled {
            return compiledFn(inputs)
        }
        return forwardCore(inputs)
    }
    
    /// Enable or disable model compilation
    public func setCompile(_ enable: Bool) {
        if enable {
            forwardCompiled = MLX.compile(self.forwardCore)
        } else {
            forwardCompiled = nil
        }
    }
    
    
    /// Load weights from safetensors file
    public func loadWeights(from weightsPath: String) throws {
        print("Loading weights from \(weightsPath)...")
        
        let weights = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath))
        print("  Found \(weights.count) weights in safetensors")
        
        // Process weights with key mapping
        var processedWeights: [String: MLXArray] = [:]
        var skipped = 0
        
        for (key, value) in weights {
            // Skip non-loadable parameters
            if key.contains("num_batches_tracked") ||
               key.contains("stft.weight") ||
               key.contains("istft.weight") ||
               key.contains("window_sum") ||
               key.contains("enframe") {
                skipped += 1
                continue
            }
            
            // Map key to Swift structure
            var mlxKey = key
            
            // Convert dot notation for array indices to underscore notation
            // model.unet.encoders.0 -> model.unet.encoders_0
            mlxKey = mlxKey.replacingOccurrences(
                of: #"\.(\d+)\."#,
                with: "_$1.",
                options: .regularExpression
            )
            // Handle trailing array index: model.unet.encoders.0 (at end of key)
            mlxKey = mlxKey.replacingOccurrences(
                of: #"\.(\d+)$"#,
                with: "_$1",
                options: .regularExpression
            )
            
            // Map SELayer fc indexed keys to named keys
            // After regex: .fc_r.layers_0. needs to become .fc_r.linear1.
            mlxKey = mlxKey.replacingOccurrences(of: ".fc_r.layers_0.", with: ".fc_r.linear1.")
            mlxKey = mlxKey.replacingOccurrences(of: ".fc_r.layers_2.", with: ".fc_r.linear2.")
            mlxKey = mlxKey.replacingOccurrences(of: ".fc_i.layers_0.", with: ".fc_i.linear1.")
            mlxKey = mlxKey.replacingOccurrences(of: ".fc_i.layers_2.", with: ".fc_i.linear2.")
            
            processedWeights[mlxKey] = value
        }
        
        // Load weights into model
        let nestedParams = NestedDictionary<String, MLXArray>.unflattened(processedWeights)
        try update(parameters: nestedParams, verify: .none)
        
        print("  ✅ Loaded: \(processedWeights.count) weights")
        print("  ⏭️ Skipped: \(skipped) weights")
    }
    
    /// Prepare model for fast inference
    public func prepareForInference() {
        print("Preparing model for fast inference...")
        train(false)  // Set eval mode (like Python model.eval()) - BatchNorm uses running stats
        freeze()
        print("  ✅ Model ready for inference")
    }
}
