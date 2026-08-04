//
//  USSProviders.swift
//  AudioToolUSS
//
//  Factory for USS-based providers
//

import Foundation
import AudioToolCore
@preconcurrency import USSMLXSwift

// MARK: - USS Providers Factory

/// Factory for creating USS-based providers
///
/// All providers share the same underlying ResUNet30 model (~53MB FP16).
/// The only difference is the initial embedding type loaded.
/// Use `processMultiple()` or `setConditioning()` to separate multiple sound types
/// without reloading the model.
public struct USSProviders {
    
    // MARK: - Generic Factory
    
    /// Create USS separation provider with custom configuration
    /// - Parameters:
    ///   - embeddingType: Initial sound type to separate
    ///   - segmentDuration: Chunk duration in seconds (default: 2.0, no overlap)
    ///   - useFp16: Use FP16 weights for smaller memory (default: true)
    /// - Returns: USS provider ready for loading
    public static func separation(
        type embeddingType: EmbeddingLoader.EmbeddingType,
        segmentDuration: Float = 2.0,
        useFp16: Bool = true
    ) -> USSMLXProvider {
        USSMLXProvider(
            embeddingType: embeddingType,
            segmentDuration: segmentDuration,
            useFp16: useFp16
        )
    }
    
    // MARK: - Convenience Factories
    
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
    
    /// Create animal sound separation provider
    /// - Parameters:
    ///   - segmentDuration: Chunk duration in seconds (default: 2.0)
    ///   - useFp16: Use FP16 weights (default: true)
    /// - Returns: Animal sound separation provider
    public static func animalSeparation(
        segmentDuration: Float = 2.0,
        useFp16: Bool = true
    ) -> USSMLXProvider {
        USSMLXProvider(
            embeddingType: .animal,
            segmentDuration: segmentDuration,
            useFp16: useFp16
        )
    }
    
    /// Create nature sound separation provider (wind, rain, water, etc.)
    /// - Parameters:
    ///   - segmentDuration: Chunk duration in seconds (default: 2.0)
    ///   - useFp16: Use FP16 weights (default: true)
    /// - Returns: Nature sound separation provider
    public static func natureSeparation(
        segmentDuration: Float = 2.0,
        useFp16: Bool = true
    ) -> USSMLXProvider {
        USSMLXProvider(
            embeddingType: .nature,
            segmentDuration: segmentDuration,
            useFp16: useFp16
        )
    }
    
    /// Create human sound separation provider (non-speech: coughing, breathing, footsteps, etc.)
    /// - Parameters:
    ///   - segmentDuration: Chunk duration in seconds (default: 2.0)
    ///   - useFp16: Use FP16 weights (default: true)
    /// - Returns: Human sound separation provider
    public static func humanSeparation(
        segmentDuration: Float = 2.0,
        useFp16: Bool = true
    ) -> USSMLXProvider {
        USSMLXProvider(
            embeddingType: .human,
            segmentDuration: segmentDuration,
            useFp16: useFp16
        )
    }
    
    /// Create things/object sound separation provider (machines, doors, vehicles, etc.)
    /// - Parameters:
    ///   - segmentDuration: Chunk duration in seconds (default: 2.0)
    ///   - useFp16: Use FP16 weights (default: true)
    /// - Returns: Things sound separation provider
    public static func thingsSeparation(
        segmentDuration: Float = 2.0,
        useFp16: Bool = true
    ) -> USSMLXProvider {
        USSMLXProvider(
            embeddingType: .things,
            segmentDuration: segmentDuration,
            useFp16: useFp16
        )
    }
}

// NOTE: USSSoundType is defined in AudioToolCore.Configuration
// Use AudioToolCore.USSSoundType in protocol methods for pipeline integration
