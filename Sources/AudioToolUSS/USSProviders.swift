//
//  USSProviders.swift
//  ClearVoiceUSS
//
//  Factory for USS-based providers
//

import Foundation
import ClearVoiceCore
@preconcurrency import USSMLXSwift

// MARK: - USS Providers Factory

/// Factory for creating USS-based providers
public struct USSProviders {
    
    /// Create speech separation provider
    /// - Parameters:
    ///   - embeddingType: Sound type to separate (default: .speech)
    ///   - segmentDuration: Chunk duration in seconds (default: 2.0, no overlap)
    ///   - useFp16: Use FP16 weights for smaller memory (default: true)
    /// - Returns: Speech separation provider ready for loading
    public static func speechSeparation(
        embeddingType: EmbeddingLoader.EmbeddingType = .speech,
        segmentDuration: Float = 2.0,
        useFp16: Bool = true
    ) -> USSMLXProvider {
        USSMLXProvider(
            embeddingType: embeddingType,
            segmentDuration: segmentDuration,
            useFp16: useFp16
        )
    }
    
    /// Create music separation provider
    /// - Parameters:
    ///   - segmentDuration: Chunk duration in seconds (default: 2.0)
    ///   - useFp16: Use FP16 weights (default: true)
    /// - Returns: Music separation provider
    public static func musicSeparation(
        segmentDuration: Float = 2.0,
        useFp16: Bool = true
    ) -> USSMLXProvider {
        USSMLXProvider(
            embeddingType: .music,
            segmentDuration: segmentDuration,
            useFp16: useFp16
        )
    }
    
    /// Create noise separation provider
    /// - Parameters:
    ///   - segmentDuration: Chunk duration in seconds (default: 2.0)
    ///   - useFp16: Use FP16 weights (default: true)
    /// - Returns: Noise separation provider
    public static func noiseSeparation(
        segmentDuration: Float = 2.0,
        useFp16: Bool = true
    ) -> USSMLXProvider {
        USSMLXProvider(
            embeddingType: .noise,
            segmentDuration: segmentDuration,
            useFp16: useFp16
        )
    }
}

// MARK: - ClearVoiceUSS Module Exports

/// Re-export embedding types for convenience
public typealias USSSoundType = EmbeddingLoader.EmbeddingType
