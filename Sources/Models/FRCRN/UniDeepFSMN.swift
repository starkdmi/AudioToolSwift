import Foundation
import MLX
import MLXNN

/// A single layer Deep Feedforward Sequential Memory Network (FSMN) for MLX Swift.
public class UniDeepFsmn: Module {
    let input_dim: Int
    let output_dim: Int
    let lorder: Int
    let hidden_size: Int
    
    @ModuleInfo var linear: Linear
    @ModuleInfo var project: Linear
    @ModuleInfo var conv1: Conv1d
    
    public init(
        input_dim: Int,
        output_dim: Int,
        lorder: Int,
        hidden_size: Int
    ) {
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lorder = lorder
        self.hidden_size = hidden_size
        
        // Linear layers
        self._linear.wrappedValue = Linear(input_dim, hidden_size)
        self._project.wrappedValue = Linear(hidden_size, output_dim, bias: false)
        
        // Depthwise Conv1d using groups
        self._conv1.wrappedValue = Conv1d(
            inputChannels: output_dim,
            outputChannels: output_dim,
            kernelSize: lorder,
            stride: 1,
            padding: 0,
            groups: output_dim,
            bias: false
        )
        
        super.init()
    }
    
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // Apply linear transformation with ReLU
        let f1 = MLXNN.relu(linear(input))
        
        // Project to output dimension
        let p1 = project(f1)
        
        // Causal padding on the left
        let x_padded = MLX.padded(p1, widths: [
            IntOrPair((0, 0)),      // batch
            IntOrPair((lorder - 1, 0)),  // time (causal padding)
            IntOrPair((0, 0))       // channels
        ])
        
        // Apply depthwise convolution
        let out = conv1(x_padded)
        
        // Add residual connection
        let outWithResidual = p1 + out
        
        // Return with input residual
        return input + outWithResidual
    }
}

/// Complex-valued UniDeepFsmn (2-layer) for MLX Swift.
public class ComplexUniDeepFsmn: Module {
    @ModuleInfo var fsmn_re_L1: UniDeepFsmn
    @ModuleInfo var fsmn_im_L1: UniDeepFsmn
    @ModuleInfo var fsmn_re_L2: UniDeepFsmn
    @ModuleInfo var fsmn_im_L2: UniDeepFsmn
    
    public init(nIn: Int, nHidden: Int = 128, nOut: Int = 128) {
        self._fsmn_re_L1.wrappedValue = UniDeepFsmn(input_dim: nIn, output_dim: nHidden, lorder: 20, hidden_size: nHidden)
        self._fsmn_im_L1.wrappedValue = UniDeepFsmn(input_dim: nIn, output_dim: nHidden, lorder: 20, hidden_size: nHidden)
        self._fsmn_re_L2.wrappedValue = UniDeepFsmn(input_dim: nHidden, output_dim: nOut, lorder: 20, hidden_size: nHidden)
        self._fsmn_im_L2.wrappedValue = UniDeepFsmn(input_dim: nHidden, output_dim: nOut, lorder: 20, hidden_size: nHidden)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input: (b, h, w, c, d) where d=2 for complex
        let b = x.shape[0]
        let h = x.shape[1]
        let w = x.shape[2]
        let c = x.shape[3]
        let d = x.shape[4]
        
        // Convert to PyTorch-like layout
        var xT = x.transposed(0, 3, 1, 2, 4)  // (b, c, h, w, d)
        
        // Reshape to (b, c*h, w, d)
        xT = xT.reshaped([b, c * h, w, d])
        
        // Transpose to (b, w, c*h, d) - time first for FSMN
        xT = xT.transposed(0, 2, 1, 3)  // (b, w, c*h, d)
        
        // Process real and imaginary parts through first layer
        let real_L1 = fsmn_re_L1(xT[.ellipsis, 0]) - fsmn_im_L1(xT[.ellipsis, 1])
        let imag_L1 = fsmn_re_L1(xT[.ellipsis, 1]) + fsmn_im_L1(xT[.ellipsis, 0])
        
        // Process second layer
        let real = fsmn_re_L2(real_L1) - fsmn_im_L2(imag_L1)
        let imag = fsmn_re_L2(imag_L1) + fsmn_im_L2(real_L1)
        
        // Combine real and imaginary
        var output = MLX.stacked([real, imag], axis: -1)  // (b, w, c*h, 2)
        
        // Transpose back: (b, w, c*h, 2) -> (b, c*h, w, 2)
        output = output.transposed(0, 2, 1, 3)
        
        // Reshape back to (b, c, h, w, d)
        output = output.reshaped([b, c, h, w, d])
        
        // Convert back to NHWC format
        output = output.transposed(0, 2, 3, 1, 4)  // (b, h, w, c, d)
        
        return output
    }
}

/// Single-layer Complex UniDeepFsmn for MLX Swift.
public class ComplexUniDeepFsmn_L1: Module {
    @ModuleInfo var fsmn_re_L1: UniDeepFsmn
    @ModuleInfo var fsmn_im_L1: UniDeepFsmn
    
    public init(nIn: Int, nHidden: Int = 128, nOut: Int = 128) {
        self._fsmn_re_L1.wrappedValue = UniDeepFsmn(input_dim: nIn, output_dim: nHidden, lorder: 20, hidden_size: nHidden)
        self._fsmn_im_L1.wrappedValue = UniDeepFsmn(input_dim: nIn, output_dim: nHidden, lorder: 20, hidden_size: nHidden)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input: (b, h, w, c, d) where d=2 for complex
        let b = x.shape[0]
        let h = x.shape[1]
        let w = x.shape[2]
        let c = x.shape[3]
        let d = x.shape[4]
        
        // Transpose to (b, w, h, c, d) - time/width first
        var xT = x.transposed(0, 2, 1, 3, 4)
        
        // Reshape to (b*w, h, c, d) for processing
        xT = xT.reshaped([b * w, h, c, d])
        
        // Process real and imaginary parts
        let real = fsmn_re_L1(xT[.ellipsis, 0]) - fsmn_im_L1(xT[.ellipsis, 1])
        let imag = fsmn_re_L1(xT[.ellipsis, 1]) + fsmn_im_L1(xT[.ellipsis, 0])
        
        // Combine results
        var output = MLX.stacked([real, imag], axis: -1)  // (b*w, h, c, 2)
        
        // Reshape back to (b, w, h, c, d)
        output = output.reshaped([b, w, h, c, d])
        
        // Transpose back to (b, h, w, c, d)
        output = output.transposed(0, 2, 1, 3, 4)
        
        return output
    }
}
