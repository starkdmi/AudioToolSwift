//
//  ClearVoiceTTS.swift
//  ClearVoiceTTS
//
//  Public exports and convenience factory for TTS providers
//

import Foundation
import ClearVoice
import ClearVoiceCore

// Re-export core types
@_exported import ClearVoice
@_exported import ClearVoiceCore

/// Factory for creating TTS providers
public struct TTSProviders {
    
    // MARK: - Kokoro TTS
    
    /// Create Kokoro TTS provider with precision-based repo selection
    /// - Parameters:
    ///   - precision: Model precision (bf16, 4bit, 8bit, etc.)
    ///   - language: Target language (default: American English)
    /// - Returns: Configured KokoroTTSProvider
    public static func kokoro(
        precision: ModelPrecision = .bf16,
        language: KokoroLanguage = .americanEnglish
    ) -> KokoroTTSProvider {
        KokoroTTSProvider(precision: precision, language: language)
    }
    
    /// Create Kokoro TTS provider with explicit repo (for custom repos)
    /// - Parameters:
    ///   - repo: Full HuggingFace repository ID
    ///   - language: Target language (default: American English)
    /// - Returns: Configured KokoroTTSProvider
    public static func kokoro(
        repo: String,
        language: KokoroLanguage = .americanEnglish
    ) -> KokoroTTSProvider {
        KokoroTTSProvider(repo: repo, language: language)
    }
    
    /// Create Kokoro TTS provider with local model path (no download)
    /// - Parameters:
    ///   - modelPath: Path to Kokoro model weights directory
    ///   - language: Target language (default: American English)
    /// - Returns: Configured KokoroTTSProvider
    public static func kokoro(
        modelPath: URL,
        language: KokoroLanguage = .americanEnglish
    ) -> KokoroTTSProvider {
        KokoroTTSProvider(modelPath: modelPath, language: language)
    }
}

/// ClearVoice extension for registering TTS providers
extension ClearVoice {
    
    /// Configure with Kokoro TTS provider
    /// - Parameters:
    ///   - synthesizer: The KokoroTTSProvider instance
    ///   - model: The synthesis model type (default: kokoro)
    public func configure(
        synthesizer: KokoroTTSProvider,
        for model: SynthesisModel = .kokoro(language: .americanEnglish, voice: "af_heart")
    ) async throws {
        try await synthesizer.load()
        self.register(synthesizer: synthesizer, for: model)
    }
}
