//
//  AudioToolMLXTranslation.swift
//  AudioToolMLXTranslation
//
//  Public exports and convenience factory for MLX-based translation providers
//

import Foundation
import AudioTool
import AudioToolCore

// Re-export core types
@_exported import AudioTool
@_exported import AudioToolCore

/// Factory for creating MLX-based translation providers
public struct MLXTranslationProviders {
    
    // MARK: - TranslateGemma
    
    /// Create TranslateGemma translation provider (55+ languages)
    ///
    /// > Note: Needs a Metal device. The package minimum, iOS 18 / macOS 15, already
/// > covers it - the older "macOS 14+" this said was inherited from mlx-swift and
/// > is below what this package builds for.
    /// > Model (~4GB) is downloaded automatically on first use from HuggingFace.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID (default: mlx-community/translategemma-4b-it-4bit)
    ///   - maxTokens: Maximum tokens to generate (default: 256)
    ///   - temperature: Sampling temperature, 0 = greedy (default: 0.0)
    ///   - progressHandler: Optional callback for model download/load progress
    /// - Returns: Configured TranslateGemmaProvider
    public static func translateGemma(
        modelId: String = "mlx-community/translategemma-4b-it-4bit",
        maxTokens: Int = 256,
        temperature: Float = 0.0,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) -> TranslateGemmaProvider {
        TranslateGemmaProvider(
            modelId: modelId,
            maxTokens: maxTokens,
            temperature: temperature,
            progressHandler: progressHandler
        )
    }
}

/// AudioEngine extension for registering MLX translation providers
extension AudioEngine {
    
    /// Configure with TranslateGemma translation provider
    /// - Parameters:
    ///   - maxTokens: Maximum tokens to generate (default: 256)
    ///   - progressHandler: Optional callback for model download/load progress
    public func configureTranslateGemma(
        maxTokens: Int = 256,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async {
        let provider = MLXTranslationProviders.translateGemma(
            maxTokens: maxTokens,
            progressHandler: progressHandler
        )
        self.register(translator: provider, for: .translateGemma)
    }
}
