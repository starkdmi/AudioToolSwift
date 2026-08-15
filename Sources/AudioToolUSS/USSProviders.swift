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
/// All providers share the same underlying ResUNet30 model (~106 MB FP32, ~53 MB
/// FP16). The only difference is the initial embedding type loaded.
///
/// FP32 is the default here because it is the default on ``USSMLXProvider`` and
/// because fp16 loses on everything but download size: measured, it is 5% slower,
/// peaks *higher*, and costs 68 dB SI-SDR against the fp32 output. These factories
/// used to default to fp16 and quietly disagree with the provider they construct.
/// Use `processMultiple()` or `setConditioning()` to separate multiple sound types
/// without reloading the model.
public struct USSProviders {
    
    // MARK: - Generic Factory
    
    /// Create USS separation provider with custom configuration
    /// - Parameters:
    ///   - embeddingType: Initial sound type to separate
    ///   - segmentDuration: Chunk duration in seconds (default: 2.0, no overlap)
    ///   - useFp16: Load the FP16 weights file (default: false - see ``USSMLXProvider/init(embeddingType:segmentDuration:useFp16:compile:)``)
    /// - Returns: USS provider ready for loading
    public static func separation(
        type embeddingType: EmbeddingLoader.EmbeddingType,
        segmentDuration: Float = 2.0,
        useFp16: Bool = false
    ) -> USSMLXProvider {
        USSMLXProvider(
            embeddingType: embeddingType,
            segmentDuration: segmentDuration,
            useFp16: useFp16
        )
    }

    /// Create a separation provider backed by a local weights file.
    ///
    /// For development and testing against weights you already have on disk, without
    /// touching HuggingFace. Mirrors the `weightsPath:` initialiser the other MLX
    /// providers offer.
    ///
    /// - Parameters:
    ///   - weightsPath: Path to a resunet30 .safetensors file
    ///   - embeddingType: Initial sound type to separate
    ///   - segmentDuration: Chunk duration in seconds (default: 2.0, no overlap)
    ///   - useFp16: Whether `weightsPath` points at FP16 weights (default: false)
    /// - Returns: USS provider ready for loading
    public static func separation(
        weightsPath: String,
        type embeddingType: EmbeddingLoader.EmbeddingType = .speech,
        segmentDuration: Float = 2.0,
        useFp16: Bool = false
    ) -> USSMLXProvider {
        USSMLXProvider(
            weightsPath: weightsPath,
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
    ///   - useFp16: Load the FP16 weights file (default: false - see ``USSMLXProvider/init(embeddingType:segmentDuration:useFp16:compile:)``)
    /// - Returns: Speech separation provider ready for loading
    public static func speechSeparation(
        embeddingType: EmbeddingLoader.EmbeddingType = .speech,
        segmentDuration: Float = 2.0,
        useFp16: Bool = false
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
    ///   - useFp16: Load the FP16 weights file (default: false - half the download, worse on every other axis)
    /// - Returns: Music separation provider
    public static func musicSeparation(
        segmentDuration: Float = 2.0,
        useFp16: Bool = false
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
    ///   - useFp16: Load the FP16 weights file (default: false - half the download, worse on every other axis)
    /// - Returns: Noise separation provider
    public static func noiseSeparation(
        segmentDuration: Float = 2.0,
        useFp16: Bool = false
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
    ///   - useFp16: Load the FP16 weights file (default: false - half the download, worse on every other axis)
    /// - Returns: Animal sound separation provider
    public static func animalSeparation(
        segmentDuration: Float = 2.0,
        useFp16: Bool = false
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
    ///   - useFp16: Load the FP16 weights file (default: false - half the download, worse on every other axis)
    /// - Returns: Nature sound separation provider
    public static func natureSeparation(
        segmentDuration: Float = 2.0,
        useFp16: Bool = false
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
    ///   - useFp16: Load the FP16 weights file (default: false - half the download, worse on every other axis)
    /// - Returns: Human sound separation provider
    public static func humanSeparation(
        segmentDuration: Float = 2.0,
        useFp16: Bool = false
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
    ///   - useFp16: Load the FP16 weights file (default: false - half the download, worse on every other axis)
    /// - Returns: Things sound separation provider
    public static func thingsSeparation(
        segmentDuration: Float = 2.0,
        useFp16: Bool = false
    ) -> USSMLXProvider {
        USSMLXProvider(
            embeddingType: .things,
            segmentDuration: segmentDuration,
            useFp16: useFp16
        )
    }
}

// NOTE: separation targets are AudioToolCore.SoundEmbedding values. The seven
// presets (.speech, .music, ...) are starting points, not the available set - any
// 527-d AudioSet class vector is a valid target.
