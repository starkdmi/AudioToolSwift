//
//  ClearVoiceMLXTranslation.swift
//  ClearVoiceMLXTranslation
//
//  Public exports and convenience factory for MLX-based translation providers
//

import Foundation
import ClearVoice
import ClearVoiceCore

// Re-export core types
@_exported import ClearVoice
@_exported import ClearVoiceCore

/// Factory for creating MLX-based translation providers
public struct MLXTranslationProviders {
    
    // MARK: - TranslateGemma
    
    /// Create TranslateGemma translation provider (55+ languages)
    ///
    /// > Note: Requires macOS 14+ for MLX Metal support.
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

/// ClearVoice extension for registering MLX translation providers
extension ClearVoice {
    
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
