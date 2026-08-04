import Foundation
import MLX
import MLXNN

/// Custom indexed FC module for SE layer weight compatibility.
/// Matches weight key structure: fc.0.weight, fc.2.weight (skipping ReLU at index 1)
class SELayerFC: Module {
    // Named layers matching remapped weight keys: linear1, linear2
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    
    init(in_features: Int, reduced_features: Int) {
        self._linear1.wrappedValue = Linear(in_features, reduced_features)
        self._linear2.wrappedValue = Linear(reduced_features, in_features)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = linear1(x)
        out = MLXNN.relu(out)
        out = linear2(out)
        out = MLX.sigmoid(out)
        return out
    }
}

/// Squeeze-and-Excitation Layer for MLX Swift.
/// Adaptively recalibrates channel-wise feature responses for complex-valued inputs.
///
/// Input shape: (batch, height, width, channels, 2)
/// Output shape: (batch, height, width, channels, 2)
public class SELayer: Module {
    @ModuleInfo var fc_r: SELayerFC
    @ModuleInfo var fc_i: SELayerFC
    
    public init(channel: Int, reduction: Int = 16) {
        let reduced = channel / reduction
        
        self._fc_r.wrappedValue = SELayerFC(in_features: channel, reduced_features: reduced)
        self._fc_i.wrappedValue = SELayerFC(in_features: channel, reduced_features: reduced)
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Get dimensions
        let b = x.shape[0]
        let c = x.shape[3]
        
        // Extract real and imaginary parts
        let x_real = x[.ellipsis, 0]  // (batch, height, width, channels)
        let x_imag = x[.ellipsis, 1]  // (batch, height, width, channels)
        
        // Global average pooling over spatial dimensions (axis 1 and 2)
        let x_r_pooled = MLX.mean(x_real, axes: [1, 2])  // (batch, channels)
        let x_i_pooled = MLX.mean(x_imag, axes: [1, 2])  // (batch, channels)
        
        // Apply FC layers with complex arithmetic (matching PyTorch)
        // y_r = fc_r(x_r) - fc_i(x_i), y_i = fc_r(x_i) + fc_i(x_r)
        let y_r = fc_r(x_r_pooled) - fc_i(x_i_pooled)
        let y_i = fc_r(x_i_pooled) + fc_i(x_r_pooled)
        
        // Reshape to (b, 1, 1, c, 1) for broadcasting
        let y_r_reshaped = y_r.reshaped([b, 1, 1, c, 1])
        let y_i_reshaped = y_i.reshaped([b, 1, 1, c, 1])
        
        // Concatenate like PyTorch: torch.cat([y_r, y_i], 4)
        let y = MLX.concatenated([y_r_reshaped, y_i_reshaped], axis: -1)  // (batch, 1, 1, channels, 2)
        
        // Apply attention by element-wise multiplication
        return x * y
    }
}
