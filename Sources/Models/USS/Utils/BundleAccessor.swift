//
//  BundleAccessor.swift
//  USSMLXSwift
//
//  Provides public access to package bundle resources
//

import Foundation

/// Provides public access to USSMLXSwift package bundle resources
public struct USSBundle {
    
    /// Get the USSMLXSwift module bundle
    public static var module: Bundle {
        return Bundle.module
    }
    
    /// Get URL for model weights
    /// - Parameter fp16: Use FP16 weights (true) or FP32 (false)
    /// - Returns: URL to safetensors file
    public static func weightsURL(fp16: Bool = true) -> URL? {
        let name = fp16 ? "resunet30_fp16" : "resunet30_fp32"
        // SPM copies to bundle root (no subdirectory)
        return Bundle.module.url(forResource: name, withExtension: "safetensors")
    }
    
    /// Get URL for embeddings directory
    public static var embeddingsDirectory: URL? {
        Bundle.module.url(forResource: "Embeddings", withExtension: nil)
    }
}
