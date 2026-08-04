import MLX
import MLXNN

/// BatchNorm2d implementation for MLX Swift
public class BatchNorm2d: Module {
    @ModuleInfo var running_mean: MLXArray
    @ModuleInfo var running_var: MLXArray
    @ModuleInfo var weight: MLXArray
    @ModuleInfo var bias: MLXArray
    
    let numFeatures: Int
    let eps: Float
    let momentum: Float
    let affine: Bool
    let trackRunningStats: Bool
    // Use Module's built-in training property
    
    public init(
        numFeatures: Int,
        eps: Float = 1e-5,
        momentum: Float = 0.1,
        affine: Bool = true,
        trackRunningStats: Bool = true
    ) {
        self.numFeatures = numFeatures
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.trackRunningStats = trackRunningStats
        
        // Initialize parameters
        self.running_mean = MLXArray.zeros([numFeatures])
        self.running_var = MLXArray.ones([numFeatures])
        
        if affine {
            self.weight = MLXArray.ones([numFeatures])
            self.bias = MLXArray.zeros([numFeatures])
        } else {
            self.weight = MLXArray.ones([numFeatures])
            self.bias = MLXArray.zeros([numFeatures])
        }
        
        super.init()
    }
    
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // Input shape: (B, H, W, C) for MLX (channels-last)
        // let shape = input.shape
        // let B = shape[0]
        // let H = shape[1]
        // let W = shape[2]
        // let C = shape[3]
        
        // For inference, always use running statistics
        // MLX Swift doesn't support training mode updates yet
        if false && training && trackRunningStats {
            // Calculate batch statistics
            let batchMean = MLX.mean(input, axes: [0, 1, 2], keepDims: false)
            let batchVar = MLX.variance(input, axes: [0, 1, 2], keepDims: false)
            
            // Update running statistics
            // Note: In MLX Swift, we cannot directly assign to @ModuleInfo properties
            // This should only happen during training, which we don't support yet
            // running_mean = (1 - momentum) * running_mean + momentum * batchMean
            // running_var = (1 - momentum) * running_var + momentum * batchVar
            
            // Normalize using batch statistics
            let normalized = (input - batchMean) / MLX.sqrt(batchVar + eps)
            
            if affine {
                return normalized * weight + bias
            } else {
                return normalized
            }
        } else {
            // Use running statistics for inference
            let normalized = (input - running_mean) / MLX.sqrt(running_var + eps)
            
            if affine {
                return normalized * weight + bias
            } else {
                return normalized
            }
        }
    }
}

/// FrequencyBatchNorm implementation for spectrograms
public class FrequencyBatchNorm: Module {
    @ModuleInfo var running_mean: MLXArray
    @ModuleInfo var running_var: MLXArray
    @ModuleInfo var weight: MLXArray
    @ModuleInfo var bias: MLXArray
    
    let numFeatures: Int
    let eps: Float
    let momentum: Float
    let affine: Bool
    let trackRunningStats: Bool
    // Use Module's built-in training property
    
    public init(
        numFeatures: Int,
        eps: Float = 1e-5,
        momentum: Float = 0.1,
        affine: Bool = true,
        trackRunningStats: Bool = true
    ) {
        self.numFeatures = numFeatures
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.trackRunningStats = trackRunningStats
        
        // Initialize parameters
        self.running_mean = MLXArray.zeros([numFeatures])
        self.running_var = MLXArray.ones([numFeatures])
        
        if affine {
            self.weight = MLXArray.ones([numFeatures])
            self.bias = MLXArray.zeros([numFeatures])
        } else {
            self.weight = MLXArray.ones([numFeatures])
            self.bias = MLXArray.zeros([numFeatures])
        }
        
        super.init()
    }
    
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // Input shape: (B, F, T, C) where F is frequency bins, T is time, C is channels
        // For frequency batch norm, we normalize across (B, T, C) dimensions for each frequency bin
        let shape = input.shape
        let B = shape[0]
        let F = shape[1]  // frequency bins
        let T = shape[2]  // time frames
        let C = shape[3]  // channels
        
        
        // Reshape to (F, B*T*C) to compute stats for each frequency
        // First transpose to (F, B, T, C) then reshape
        let transposed = input.transposed(1, 0, 2, 3)
        let reshaped = transposed.reshaped([F, B * T * C])
        
        // For inference, always use running statistics
        // MLX Swift doesn't support training mode updates yet
        if false && training && trackRunningStats {
            // Calculate statistics per frequency bin
            // For shape (F, B*T*C), compute mean and var over dim 1
            let batchMean = MLX.mean(reshaped, axis: 1, keepDims: true)  // Shape: (F, 1)
            let batchVar = MLX.variance(reshaped, axis: 1, keepDims: true)  // Shape: (F, 1)
            
            // Update running statistics would go here (not supported in Swift)
            
            // Normalize
            let normalized = (reshaped - batchMean) / MLX.sqrt(batchVar + eps)
            
            if affine {
                // Expand weight and bias to (F, 1) for broadcasting
                let weightExpanded = weight.expandedDimensions(axis: 1)  // Shape: (F, 1)
                let biasExpanded = bias.expandedDimensions(axis: 1)      // Shape: (F, 1)
                let scaled = normalized * weightExpanded + biasExpanded
                
                // Reshape back to (F, B, T, C) then transpose to (B, F, T, C)
                let reshaped_back = scaled.reshaped([F, B, T, C])
                return reshaped_back.transposed(1, 0, 2, 3)
            } else {
                let reshaped_back = normalized.reshaped([F, B, T, C])
                return reshaped_back.transposed(1, 0, 2, 3)
            }
        } else {
            // Use running statistics for inference
            // running_mean and running_var have shape (F,)
            
            let meanExpanded = running_mean.expandedDimensions(axis: 1)  // Shape: (F, 1)
            let varExpanded = running_var.expandedDimensions(axis: 1)    // Shape: (F, 1)
            let denominator = MLX.sqrt(varExpanded + eps)
            
            // Debug intermediate values
            let centered = reshaped - meanExpanded
            
            
            let normalized = centered / denominator
            
            if affine {
                let weightExpanded = weight.expandedDimensions(axis: 1)  // Shape: (F, 1)
                let biasExpanded = bias.expandedDimensions(axis: 1)      // Shape: (F, 1)
                let scaled = normalized * weightExpanded + biasExpanded
                
                // Reshape back to (F, B, T, C) then transpose to (B, F, T, C)
                let reshaped_back = scaled.reshaped([F, B, T, C])
                return reshaped_back.transposed(1, 0, 2, 3)
            } else {
                let reshaped_back = normalized.reshaped([F, B, T, C])
                return reshaped_back.transposed(1, 0, 2, 3)
            }
        }
    }
}
