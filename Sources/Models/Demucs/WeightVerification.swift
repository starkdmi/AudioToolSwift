import MLX
import MLXNN
import Foundation

/// Utility for verifying weight loading between Python and Swift
public struct WeightVerifier {
    
    /// Statistics for a single weight tensor
    public struct WeightStats: Codable {
        public let shape: [Int]
        public let dtype: String
        public let mean: Float
        public let absMax: Float
        
        public init(from array: MLXArray) {
            self.shape = array.shape.map { $0 }
            self.dtype = String(describing: array.dtype)
            
            // Compute statistics
            let arr32 = array.asType(.float32)
            MLX.eval(arr32)
            self.mean = MLX.mean(arr32).item(Float.self)
            self.absMax = MLX.abs(arr32).max().item(Float.self)
        }
    }
    
    /// Extract all weight statistics from a model using flattened parameters
    public static func extractStats(from model: Module) -> [String: WeightStats] {
        var stats: [String: WeightStats] = [:]
        
        // Use flattened() which returns [(String, MLXArray)]
        let flattened = model.parameters().flattened()
        
        for (key, array) in flattened {
            MLX.eval(array)
            stats[key] = WeightStats(from: array)
        }
        
        return stats
    }
    
    /// Load Python weight stats from JSON file
    public struct PythonWeightStats: Codable {
        struct ParamStats: Codable {
            let shape: [Int]
            let dtype: String
            let mean: Double
            let abs_max: Double
            let sum: Double?
        }
        
        let file_weights: [String: ParamStats]
        let config: Config
        
        struct Config: Codable {
            let weights_path: String
            let dtype: String
            let num_weights: Int
        }
    }
    
    public static func loadPythonStats(from path: String) throws -> PythonWeightStats {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PythonWeightStats.self, from: data)
    }
    
    /// Convert Swift key format back to Python format for lookup
    public static func convertSwiftKeyToPython(_ key: String) -> String {
        var result = key
        
        // Convert encoder_0 -> encoder.0
        let patterns = ["encoder", "decoder", "tencoder", "tdecoder"]
        for pattern in patterns {
            let regex = try! NSRegularExpression(pattern: "\(pattern)_(\\d+)")
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "\(pattern).$1")
        }
        
        // Convert DConv layer names back to indices
        if result.contains("dconv.") {
            result = result.replacingOccurrences(of: ".conv0_0.", with: ".layers.0.layers.0.")
            result = result.replacingOccurrences(of: ".norm0_0.", with: ".layers.0.layers.1.")
            result = result.replacingOccurrences(of: ".conv0_1.", with: ".layers.0.layers.3.")
            result = result.replacingOccurrences(of: ".norm0_1.", with: ".layers.0.layers.4.")
            result = result.replacingOccurrences(of: ".scale0.", with: ".layers.0.layers.6.")
            
            result = result.replacingOccurrences(of: ".conv1_0.", with: ".layers.1.layers.0.")
            result = result.replacingOccurrences(of: ".norm1_0.", with: ".layers.1.layers.1.")
            result = result.replacingOccurrences(of: ".conv1_1.", with: ".layers.1.layers.3.")
            result = result.replacingOccurrences(of: ".norm1_1.", with: ".layers.1.layers.4.")
            result = result.replacingOccurrences(of: ".scale1.", with: ".layers.1.layers.6.")
        }
        
        // Convert conv2d -> conv for freq branches
        if result.contains("encoder.") || result.contains("decoder.") {
            if !result.hasPrefix("t") {
                result = result.replacingOccurrences(of: ".conv2d.", with: ".conv.")
                result = result.replacingOccurrences(of: ".conv_tr2d.", with: ".conv_tr.")
                result = result.replacingOccurrences(of: ".rewrite2d.", with: ".rewrite.")
            }
        }
        
        // Convert conv1d -> conv for time branches
        if result.hasPrefix("tencoder.") || result.hasPrefix("tdecoder.") {
            result = result.replacingOccurrences(of: ".conv1d.", with: ".conv.")
            result = result.replacingOccurrences(of: ".conv_tr1d.", with: ".conv_tr.")
            result = result.replacingOccurrences(of: ".rewrite1d.", with: ".rewrite.")
        }
        
        // Crosstransformer layers
        let layersTRegex = try! NSRegularExpression(pattern: "crosstransformer\\.layers_t_(\\d+)")
        let layersTRange = NSRange(result.startIndex..., in: result)
        result = layersTRegex.stringByReplacingMatches(in: result, range: layersTRange, withTemplate: "crosstransformer.layers_t.$1")
        
        let layersRegex = try! NSRegularExpression(pattern: "crosstransformer\\.layers_(\\d+)")
        let layersRange = NSRange(result.startIndex..., in: result)
        result = layersRegex.stringByReplacingMatches(in: result, range: layersRange, withTemplate: "crosstransformer.layers.$1")
        
        return result
    }
    
    /// Print all weight keys and stats for debugging
    public static func printAllWeights(from model: Module) {
        let stats = extractStats(from: model)
        
        print("=== All Model Weights (\(stats.count) total) ===")
        for (key, stat) in stats.sorted(by: { $0.key < $1.key }) {
            let shapeStr = stat.shape.map { String($0) }.joined(separator: "x")
            print(String(format: "  %-60s shape=%@  mean=%.4e  absMax=%.4e", 
                        key, shapeStr, stat.mean, stat.absMax))
        }
    }
    
    /// Check for potential issues in loaded weights
    public static func checkForIssues(in model: Module, verbose: Bool = true) -> [String] {
        let stats = extractStats(from: model)
        var issues: [String] = []
        
        for (key, stat) in stats {
            if stat.absMax < 1e-10 {
                issues.append("WARN: \(key) has absMax=\(stat.absMax) (possibly not loaded)")
            }
            if stat.absMax > 100 {
                issues.append("WARN: \(key) has absMax=\(stat.absMax) (suspiciously large)")
            }
            if stat.mean.isNaN || stat.mean.isInfinite {
                issues.append("ERROR: \(key) has mean=\(stat.mean) (NaN/Inf detected)")
            }
        }
        
        if verbose {
            print("=== Weight Loading Issues ===")
            if issues.isEmpty {
                print("  No issues detected")
            } else {
                for issue in issues {
                    print("  \(issue)")
                }
            }
        }
        
        return issues
    }
    
    /// Compare loaded model weights with Python file weights
    public static func verifyLoadedModelWeights(
        model: Module,
        pythonStatsPath: String,
        verbose: Bool = true
    ) throws -> (matched: Int, mismatched: Int, missing: Int) {
        
        let pythonStats = try loadPythonStats(from: pythonStatsPath)
        let swiftStats = extractStats(from: model)
        
        var matched = 0
        var mismatched = 0
        var missing = 0
        
        if verbose {
            print("=== Loaded Model Weight Verification ===")
            print("Python file weights: \(pythonStats.file_weights.count)")
            print("Swift loaded params: \(swiftStats.count)")
            print()
        }
        
        for (swiftKey, swiftStat) in swiftStats.sorted(by: { $0.key < $1.key }) {
            let pythonKey = convertSwiftKeyToPython(swiftKey)
            
            if let pythonStat = pythonStats.file_weights[pythonKey] {
                let shapesMatch = swiftStat.shape == pythonStat.shape
                let meanDiff = abs(swiftStat.mean - Float(pythonStat.mean))
                let absMaxDiff = abs(swiftStat.absMax - Float(pythonStat.abs_max))
                
                let isMatch = shapesMatch && meanDiff < 1e-4 && absMaxDiff < 1e-4
                
                if isMatch {
                    matched += 1
                } else {
                    mismatched += 1
                    if verbose {
                        print("MISMATCH: \(swiftKey) -> \(pythonKey)")
                        print("  Swift: shape=\(swiftStat.shape) mean=\(swiftStat.mean) absMax=\(swiftStat.absMax)")
                        print("  Python: shape=\(pythonStat.shape) mean=\(pythonStat.mean) absMax=\(pythonStat.abs_max)")
                    }
                }
            } else {
                missing += 1
                if verbose {
                    print("MISSING: \(swiftKey) -> \(pythonKey)")
                }
            }
        }
        
        if verbose {
            print("\n=== Summary ===")
            print("Matched: \(matched)")
            print("Mismatched: \(mismatched)")
            print("Missing: \(missing)")
        }
        
        return (matched, mismatched, missing)
    }
}
