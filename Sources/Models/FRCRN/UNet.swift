import Foundation
import MLX
import MLXNN

/// U-Net architecture for MLX Swift handling complex inputs.
/// Uses explicit layer declarations since Swift MLX doesn't support module lists like Python.
public class UNet: Module {
    let model_length: Int = 7  // model_depth / 2 = 14 / 2
    
    // Bottleneck FSMN
    @ModuleInfo var fsmn: ComplexUniDeepFsmn
    
    // Encoders (7 layers)
    @ModuleInfo var encoders_0: Encoder
    @ModuleInfo var encoders_1: Encoder
    @ModuleInfo var encoders_2: Encoder
    @ModuleInfo var encoders_3: Encoder
    @ModuleInfo var encoders_4: Encoder
    @ModuleInfo var encoders_5: Encoder
    @ModuleInfo var encoders_6: Encoder
    
    // SE layers for encoders
    @ModuleInfo var se_layers_enc_0: SELayer
    @ModuleInfo var se_layers_enc_1: SELayer
    @ModuleInfo var se_layers_enc_2: SELayer
    @ModuleInfo var se_layers_enc_3: SELayer
    @ModuleInfo var se_layers_enc_4: SELayer
    @ModuleInfo var se_layers_enc_5: SELayer
    @ModuleInfo var se_layers_enc_6: SELayer
    
    // FSMN for encoders (skip first)
    @ModuleInfo var fsmn_enc_1: ComplexUniDeepFsmn_L1
    @ModuleInfo var fsmn_enc_2: ComplexUniDeepFsmn_L1
    @ModuleInfo var fsmn_enc_3: ComplexUniDeepFsmn_L1
    @ModuleInfo var fsmn_enc_4: ComplexUniDeepFsmn_L1
    @ModuleInfo var fsmn_enc_5: ComplexUniDeepFsmn_L1
    @ModuleInfo var fsmn_enc_6: ComplexUniDeepFsmn_L1
    
    // Decoders (7 layers)
    @ModuleInfo var decoders_0: Decoder
    @ModuleInfo var decoders_1: Decoder
    @ModuleInfo var decoders_2: Decoder
    @ModuleInfo var decoders_3: Decoder
    @ModuleInfo var decoders_4: Decoder
    @ModuleInfo var decoders_5: Decoder
    @ModuleInfo var decoders_6: Decoder
    
    // SE layers for decoders (skip last)
    @ModuleInfo var se_layers_dec_0: SELayer
    @ModuleInfo var se_layers_dec_1: SELayer
    @ModuleInfo var se_layers_dec_2: SELayer
    @ModuleInfo var se_layers_dec_3: SELayer
    @ModuleInfo var se_layers_dec_4: SELayer
    
    // FSMN for decoders (skip last)
    @ModuleInfo var fsmn_dec_0: ComplexUniDeepFsmn_L1
    @ModuleInfo var fsmn_dec_1: ComplexUniDeepFsmn_L1
    @ModuleInfo var fsmn_dec_2: ComplexUniDeepFsmn_L1
    @ModuleInfo var fsmn_dec_3: ComplexUniDeepFsmn_L1
    @ModuleInfo var fsmn_dec_4: ComplexUniDeepFsmn_L1
    @ModuleInfo var fsmn_dec_5: ComplexUniDeepFsmn_L1
    
    // Final linear layer
    @ModuleInfo var linear: ComplexConv2d
    
    // Architecture parameters for model_depth=14
    let enc_channels = [1, 128, 128, 128, 128, 128, 128, 128]
    let enc_kernel_sizes: [(Int, Int)] = [(5, 2), (5, 2), (5, 2), (5, 2), (5, 2), (5, 2), (2, 2)]
    let enc_strides: [(Int, Int)] = [(2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)]
    let enc_paddings: [(Int, Int)] = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    
    let dec_channels = [64, 128, 128, 128, 128, 128, 128, 1]
    let dec_kernel_sizes: [(Int, Int)] = [(2, 2), (5, 2), (5, 2), (5, 2), (6, 2), (5, 2), (5, 2)]
    let dec_strides: [(Int, Int)] = [(2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)]
    let dec_paddings: [(Int, Int)] = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    
    public override init() {
        // Initialize bottleneck FSMN
        self._fsmn.wrappedValue = ComplexUniDeepFsmn(nIn: 128, nHidden: 128, nOut: 128)
        
        // Initialize encoders
        self._encoders_0.wrappedValue = Encoder(in_channels: enc_channels[0], out_channels: enc_channels[1],
                                                 kernel_size: enc_kernel_sizes[0], stride: enc_strides[0], padding: enc_paddings[0])
        self._encoders_1.wrappedValue = Encoder(in_channels: enc_channels[1], out_channels: enc_channels[2],
                                                 kernel_size: enc_kernel_sizes[1], stride: enc_strides[1], padding: enc_paddings[1])
        self._encoders_2.wrappedValue = Encoder(in_channels: enc_channels[2], out_channels: enc_channels[3],
                                                 kernel_size: enc_kernel_sizes[2], stride: enc_strides[2], padding: enc_paddings[2])
        self._encoders_3.wrappedValue = Encoder(in_channels: enc_channels[3], out_channels: enc_channels[4],
                                                 kernel_size: enc_kernel_sizes[3], stride: enc_strides[3], padding: enc_paddings[3])
        self._encoders_4.wrappedValue = Encoder(in_channels: enc_channels[4], out_channels: enc_channels[5],
                                                 kernel_size: enc_kernel_sizes[4], stride: enc_strides[4], padding: enc_paddings[4])
        self._encoders_5.wrappedValue = Encoder(in_channels: enc_channels[5], out_channels: enc_channels[6],
                                                 kernel_size: enc_kernel_sizes[5], stride: enc_strides[5], padding: enc_paddings[5])
        self._encoders_6.wrappedValue = Encoder(in_channels: enc_channels[6], out_channels: enc_channels[7],
                                                 kernel_size: enc_kernel_sizes[6], stride: enc_strides[6], padding: enc_paddings[6])
        
        // Initialize SE layers for encoders
        self._se_layers_enc_0.wrappedValue = SELayer(channel: enc_channels[1], reduction: 8)
        self._se_layers_enc_1.wrappedValue = SELayer(channel: enc_channels[2], reduction: 8)
        self._se_layers_enc_2.wrappedValue = SELayer(channel: enc_channels[3], reduction: 8)
        self._se_layers_enc_3.wrappedValue = SELayer(channel: enc_channels[4], reduction: 8)
        self._se_layers_enc_4.wrappedValue = SELayer(channel: enc_channels[5], reduction: 8)
        self._se_layers_enc_5.wrappedValue = SELayer(channel: enc_channels[6], reduction: 8)
        self._se_layers_enc_6.wrappedValue = SELayer(channel: enc_channels[7], reduction: 8)
        
        // Initialize FSMN for encoders (skip first encoder)
        self._fsmn_enc_1.wrappedValue = ComplexUniDeepFsmn_L1(nIn: 128, nHidden: 128, nOut: 128)
        self._fsmn_enc_2.wrappedValue = ComplexUniDeepFsmn_L1(nIn: 128, nHidden: 128, nOut: 128)
        self._fsmn_enc_3.wrappedValue = ComplexUniDeepFsmn_L1(nIn: 128, nHidden: 128, nOut: 128)
        self._fsmn_enc_4.wrappedValue = ComplexUniDeepFsmn_L1(nIn: 128, nHidden: 128, nOut: 128)
        self._fsmn_enc_5.wrappedValue = ComplexUniDeepFsmn_L1(nIn: 128, nHidden: 128, nOut: 128)
        self._fsmn_enc_6.wrappedValue = ComplexUniDeepFsmn_L1(nIn: 128, nHidden: 128, nOut: 128)
        
        // Initialize decoders
        // First decoder: bottleneck only (128 channels)
        self._decoders_0.wrappedValue = Decoder(in_channels: 128, out_channels: dec_channels[1],
                                                 kernel_size: dec_kernel_sizes[0], stride: dec_strides[0], padding: dec_paddings[0])
        // Other decoders: 256 channels (128 from prev + 128 from skip)
        self._decoders_1.wrappedValue = Decoder(in_channels: 256, out_channels: dec_channels[2],
                                                 kernel_size: dec_kernel_sizes[1], stride: dec_strides[1], padding: dec_paddings[1])
        self._decoders_2.wrappedValue = Decoder(in_channels: 256, out_channels: dec_channels[3],
                                                 kernel_size: dec_kernel_sizes[2], stride: dec_strides[2], padding: dec_paddings[2])
        self._decoders_3.wrappedValue = Decoder(in_channels: 256, out_channels: dec_channels[4],
                                                 kernel_size: dec_kernel_sizes[3], stride: dec_strides[3], padding: dec_paddings[3])
        self._decoders_4.wrappedValue = Decoder(in_channels: 256, out_channels: dec_channels[5],
                                                 kernel_size: dec_kernel_sizes[4], stride: dec_strides[4], padding: dec_paddings[4])
        self._decoders_5.wrappedValue = Decoder(in_channels: 256, out_channels: dec_channels[6],
                                                 kernel_size: dec_kernel_sizes[5], stride: dec_strides[5], padding: dec_paddings[5])
        self._decoders_6.wrappedValue = Decoder(in_channels: 256, out_channels: dec_channels[7],
                                                 kernel_size: dec_kernel_sizes[6], stride: dec_strides[6], padding: dec_paddings[6])
        
        // Initialize SE layers for decoders (not for last decoder)
        self._se_layers_dec_0.wrappedValue = SELayer(channel: dec_channels[1], reduction: 8)
        self._se_layers_dec_1.wrappedValue = SELayer(channel: dec_channels[2], reduction: 8)
        self._se_layers_dec_2.wrappedValue = SELayer(channel: dec_channels[3], reduction: 8)
        self._se_layers_dec_3.wrappedValue = SELayer(channel: dec_channels[4], reduction: 8)
        self._se_layers_dec_4.wrappedValue = SELayer(channel: dec_channels[5], reduction: 8)
        
        // Initialize FSMN for decoders (skip last decoder)
        self._fsmn_dec_0.wrappedValue = ComplexUniDeepFsmn_L1(nIn: 128, nHidden: 128, nOut: 128)
        self._fsmn_dec_1.wrappedValue = ComplexUniDeepFsmn_L1(nIn: 128, nHidden: 128, nOut: 128)
        self._fsmn_dec_2.wrappedValue = ComplexUniDeepFsmn_L1(nIn: 128, nHidden: 128, nOut: 128)
        self._fsmn_dec_3.wrappedValue = ComplexUniDeepFsmn_L1(nIn: 128, nHidden: 128, nOut: 128)
        self._fsmn_dec_4.wrappedValue = ComplexUniDeepFsmn_L1(nIn: 128, nHidden: 128, nOut: 128)
        self._fsmn_dec_5.wrappedValue = ComplexUniDeepFsmn_L1(nIn: 128, nHidden: 128, nOut: 128)
        
        // Final linear layer
        self._linear.wrappedValue = ComplexConv2d(in_channels: dec_channels[7], out_channels: 1, kernel_size: 1)
        
        super.init()
    }
    
    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        var x = inputs
        var xs_se: [MLXArray] = [x]  // SE outputs for skip connections
        
        // Encoder path
        // Encoder 0 (no FSMN)
        x = encoders_0(x)
        xs_se.append(se_layers_enc_0(x))
        
        // Encoder 1 with FSMN
        x = fsmn_enc_1(x)
        x = encoders_1(x)
        xs_se.append(se_layers_enc_1(x))
        
        // Encoder 2 with FSMN
        x = fsmn_enc_2(x)
        x = encoders_2(x)
        xs_se.append(se_layers_enc_2(x))
        
        // Encoder 3 with FSMN
        x = fsmn_enc_3(x)
        x = encoders_3(x)
        xs_se.append(se_layers_enc_3(x))
        
        // Encoder 4 with FSMN
        x = fsmn_enc_4(x)
        x = encoders_4(x)
        xs_se.append(se_layers_enc_4(x))
        
        // Encoder 5 with FSMN
        x = fsmn_enc_5(x)
        x = encoders_5(x)
        xs_se.append(se_layers_enc_5(x))
        
        // Encoder 6 with FSMN
        x = fsmn_enc_6(x)
        x = encoders_6(x)
        xs_se.append(se_layers_enc_6(x))
        
        // Bottleneck FSMN
        x = fsmn(x)
        var p = x
        
        // Decoder path
        // Decoder 0 (no skip concat before first decoder)
        p = decoders_0(p)
        p = fsmn_dec_0(p)
        p = se_layers_dec_0(p)
        // Skip connection from encoder 6
        p = MLX.concatenated([p, xs_se[6]], axis: 3)
        
        // Decoder 1
        p = decoders_1(p)
        p = fsmn_dec_1(p)
        p = se_layers_dec_1(p)
        p = MLX.concatenated([p, xs_se[5]], axis: 3)
        
        // Decoder 2
        p = decoders_2(p)
        p = fsmn_dec_2(p)
        p = se_layers_dec_2(p)
        p = MLX.concatenated([p, xs_se[4]], axis: 3)
        
        // Decoder 3
        p = decoders_3(p)
        p = fsmn_dec_3(p)
        p = se_layers_dec_3(p)
        p = MLX.concatenated([p, xs_se[3]], axis: 3)
        
        // Decoder 4
        p = decoders_4(p)
        p = fsmn_dec_4(p)
        p = se_layers_dec_4(p)
        p = MLX.concatenated([p, xs_se[2]], axis: 3)
        
        // Decoder 5
        p = decoders_5(p)
        p = fsmn_dec_5(p)
        // No SE layer for decoder 5 based on Python logic (i < model_length - 2)
        p = MLX.concatenated([p, xs_se[1]], axis: 3)
        
        // Decoder 6 (last - no FSMN, no SE, no skip after)
        p = decoders_6(p)
        
        // Final output
        let output = linear(p)
        return output
    }
}
