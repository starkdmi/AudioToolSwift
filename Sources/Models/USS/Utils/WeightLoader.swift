import Foundation
import MLX
import MLXNN

/// SafeTensors weight loader for USS model
public class WeightLoader {
    
    /// Load weights from SafeTensors file into model
    public static func loadWeights(model: Module, from path: String) throws {
        // Load safetensors file
        let url = URL(fileURLWithPath: path)
        let weights = try MLX.loadArrays(url: url)
        
        // Process weights for MLX Swift format
        var processedWeights: [String: MLXArray] = [:]
        
        for (name, weight) in weights {
            // MLX Python and MLX Swift use the same weight format
            // No transposition needed - just use weights as-is
            let processedWeight = weight
            
            processedWeights[name] = processedWeight
            
            // Debug FiLM weights
            /*if name.contains("encoder_block1_conv_block1_beta1") {
                print("[WeightLoader] Found FiLM weight: \(name), shape=\(weight.shape), max=\(MLX.max(weight).item(Float.self))")
            }*/
        }
        
        // Debug specific BN weights
        /*if let bnMean = processedWeights["base.bn0.running_mean"] {
            print("[WeightLoader] base.bn0.running_mean shape: \(bnMean.shape)")
            let meanLen = bnMean.shape[0]
            print("[WeightLoader] Last 5 values: \(bnMean[(meanLen-5)..<meanLen].asArray(Float.self))")
        }
        if let bnVar = processedWeights["base.bn0.running_var"] {
            print("[WeightLoader] base.bn0.running_var shape: \(bnVar.shape)")
            let varLen = bnVar.shape[0]
            print("[WeightLoader] Last 5 values: \(bnVar[(varLen-5)..<varLen].asArray(Float.self))")
        }*/
        
        // Create unflattened parameters and update model
        let parameters = ModuleParameters.unflattened(processedWeights)
        
        // Debug: Check parameters before update
        /*print("[WeightLoader] Parameters before update:")
        // Simply check if the weight exists in the processed weights
        if let mean = processedWeights["base.bn0.running_mean"] {
            print("[WeightLoader] Found base.bn0.running_mean in processedWeights")
            print("[WeightLoader] Parameter running_mean shape: \(mean.shape)")
            let meanLen = mean.shape[0]
            print("[WeightLoader] Parameter last 5 values: \(mean[(meanLen-5)..<meanLen].asArray(Float.self))")
        }*/
        
        // Debug: Check FiLM weights before update
        /*if let filmWeight = processedWeights["film.encoder_block1_conv_block1_beta1.weight"] {
            print("[WeightLoader] FiLM weight BEFORE update: shape=\(filmWeight.shape), max=\(MLX.max(filmWeight).item(Float.self)), mean=\(MLX.mean(filmWeight).item(Float.self))")
        }*/
        
        try model.update(parameters: parameters, verify: .none)
        
        // Debug: Check model after update
        /*print("[WeightLoader] Checking model after update...")
        if let resUNet = model as? ResUNet30 {
            print("[WeightLoader] Model is ResUNet30, checking base.bn0...")
            let bn0 = resUNet.base.bn0
            let mean = bn0.running_mean
            print("[WeightLoader] Model running_mean shape after update: \(mean.shape)")
            let meanLen = mean.shape[0]
            print("[WeightLoader] Model last 5 values: \(mean[(meanLen-5)..<meanLen].asArray(Float.self))")
            
            // Check FiLM weights after update
            let filmWeight = resUNet.film.encoder_block1_conv_block1_beta1.weight
            print("[WeightLoader] FiLM weight AFTER update: shape=\(filmWeight.shape), max=\(MLX.max(filmWeight).item(Float.self)), mean=\(MLX.mean(filmWeight).item(Float.self))")
        }*/
        
        print("Successfully loaded \(processedWeights.count) weights")
    }
}
