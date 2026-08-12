//
//  AudioToolTTS.swift
//  AudioToolTTS
//
//  Public exports and convenience factory for TTS providers
//

import Foundation
import AudioTool
import AudioToolCore

// Re-export core types
@_exported import AudioTool
@_exported import AudioToolCore

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
    
    // MARK: - Apple TTS (AVSpeechSynthesizer)
    
    /// Create Apple TTS provider using AVSpeechSynthesizer
    ///
    /// Supports 60+ languages offline with system voices.
    ///
    /// - Parameter language: BCP-47 language code (e.g., "en-US", "fr-FR", "de-DE", "ru-RU")
    /// - Returns: Configured AppleTTSProvider
    ///
    /// Example:
    /// ```swift
    /// let tts = TTSProviders.apple(language: "fr-FR")
    /// let audio = try await tts.synthesize("Bonjour", voice: "Thomas")
    /// ```
    public static func apple(language: String = "en-US") -> AppleTTSProvider {
        AppleTTSProvider(language: language)
    }
    
    // MARK: - RUAccent (Russian Stress Marking)
    
    /// Create RUAccent text preprocessor for Russian stress marking
    ///
    /// Adds stress marks (+) to Russian text for proper TTS pronunciation.
    /// Example: "привет мир" → "прив+ет м+ир"
    ///
    /// - Parameters:
    ///   - profile: Pipeline profile (lightweight/balanced/max)
    ///   - modelsDir: Directory containing *.mlpackage models
    ///   - assetsDir: Directory containing dictionaries and configs
    /// - Returns: Configured RUAccentProvider
    ///
    /// Example:
    /// ```swift
    /// let ruaccent = try TTSProviders.ruaccent(
    ///     profile: .balanced,
    ///     modelsDir: URL(fileURLWithPath: "path/to/models/balanced"),
    ///     assetsDir: URL(fileURLWithPath: "path/to/assets/balanced")
    /// )
    /// let stressed = try ruaccent.process("привет мир")
    /// ```
    public static func ruaccent(
        profile: RUAccentProfile = .balanced,
        modelsDir: URL,
        assetsDir: URL
    ) throws -> RUAccentProvider {
        try RUAccentProvider(profile: profile, modelsDir: modelsDir, assetsDir: assetsDir)
    }
    
    // MARK: - ChatterBox TTS (Multilingual Voice Cloning)
    
    /// Create ChatterBox TTS provider with precision-based repo selection
    ///
    /// ChatterBox is a multilingual TTS model supporting 25 languages with voice cloning.
    /// Reference audio is required for synthesis.
    ///
    /// - Parameters:
    ///   - precision: Model precision (fp32, fp16, 8bit, 6bit, 4bit)
    ///   - language: Target language (default: English)
    ///   - useRuAccent: Enable automatic RUAccent for Russian (default: true)
    ///   - convertToStressMarks: Convert + to Unicode stress marks for Russian (default: true)
    /// - Returns: Configured ChatterboxTTSProvider
    ///
    /// Example:
    /// ```swift
    /// let tts = TTSProviders.chatterbox(precision: .fp16, language: .english)
    /// try await tts.load()
    /// try await tts.setReferenceAudio(from: referenceURL)
    /// let audio = try await tts.synthesize("Hello world!", voice: "")
    /// ```
    public static func chatterbox(
        precision: ModelPrecision = .fp32,
        language: ChatterboxLanguage = .english,
        useRuAccent: Bool = true,
        convertToStressMarks: Bool = true
    ) -> ChatterboxTTSProvider {
        ChatterboxTTSProvider(
            precision: precision,
            language: language,
            useRuAccent: useRuAccent,
            convertToStressMarks: convertToStressMarks
        )
    }
    
    /// Create ChatterBox TTS provider with explicit repo (for custom repos)
    /// - Parameters:
    ///   - repo: Full HuggingFace repository ID
    ///   - language: Target language (default: English)
    ///   - useRuAccent: Enable automatic RUAccent for Russian (default: true)
    ///   - convertToStressMarks: Convert + to Unicode stress marks (default: true)
    /// - Returns: Configured ChatterboxTTSProvider
    public static func chatterbox(
        repo: String,
        language: ChatterboxLanguage = .english,
        useRuAccent: Bool = true,
        convertToStressMarks: Bool = true
    ) -> ChatterboxTTSProvider {
        ChatterboxTTSProvider(
            repo: repo,
            language: language,
            useRuAccent: useRuAccent,
            convertToStressMarks: convertToStressMarks
        )
    }
    
    /// Create ChatterBox TTS provider with local model path (no download)
    /// - Parameters:
    ///   - modelPath: Path to ChatterBox model weights directory
    ///   - language: Target language (default: English)
    ///   - useRuAccent: Enable automatic RUAccent for Russian (default: true)
    ///   - convertToStressMarks: Convert + to Unicode stress marks (default: true)
    /// - Returns: Configured ChatterboxTTSProvider
    public static func chatterbox(
        modelPath: URL,
        language: ChatterboxLanguage = .english,
        useRuAccent: Bool = true,
        convertToStressMarks: Bool = true
    ) -> ChatterboxTTSProvider {
        ChatterboxTTSProvider(
            modelPath: modelPath,
            language: language,
            useRuAccent: useRuAccent,
            convertToStressMarks: convertToStressMarks
        )
    }
}

/// AudioEngine extension for registering TTS providers
extension AudioEngine {
    
    /// Configure with Kokoro TTS provider
    /// - Parameters:
    ///   - synthesizer: The KokoroTTSProvider instance
    ///   - model: The synthesis model type (default: kokoro)
    public func configure(
        synthesizer: KokoroTTSProvider,
        for model: SynthesisModel = .kokoro(language: .americanEnglish, voice: "af_heart")
    ) async throws {
        // Loaded through the residency manager, so the weights count against the
        // engine's memory budget rather than sitting outside it.
        try await self.preload(synthesizer)
        self.register(synthesizer: synthesizer, for: model)
    }
    
    /// Configure with Apple TTS provider
    /// - Parameters:
    ///   - synthesizer: The AppleTTSProvider instance
    ///   - model: The synthesis model type (default: appleTTS)
    public func configure(
        synthesizer: AppleTTSProvider,
        for model: SynthesisModel = .appleTTS(language: "en-US")
    ) {
        self.register(synthesizer: synthesizer, for: model)
    }
    
    /// Convenience: Configure Apple TTS for a specific language
    /// - Parameter language: BCP-47 language code (e.g., "fr-FR", "de-DE")
    public func configureAppleTTS(language: String = "en-US") {
        let provider = TTSProviders.apple(language: language)
        self.register(synthesizer: provider, for: .appleTTS(language: language))
    }
    
    /// Configure with RUAccent preprocessor
    /// - Parameters:
    ///   - preprocessor: The RUAccentProvider instance
    ///   - model: The preprocessing model type (default: ruaccent balanced)
    public func configure(
        preprocessor: RUAccentProvider,
        for model: TextPreprocessorModel = .ruaccent(profile: .balanced)
    ) {
        self.register(preprocessor: preprocessor, for: model)
    }
    
    /// Configure with ChatterBox TTS provider
    /// - Parameters:
    ///   - synthesizer: The ChatterboxTTSProvider instance
    ///   - model: The synthesis model type (default: chatterbox english)
    public func configure(
        synthesizer: ChatterboxTTSProvider,
        for model: SynthesisModel = .chatterbox(language: .english)
    ) async throws {
        try await self.preload(synthesizer)
        self.register(synthesizer: synthesizer, for: model)
    }
    
    /// Convenience: Configure ChatterBox TTS for a specific language
    /// - Parameters:
    ///   - language: Target language
    ///   - precision: Model precision (default: fp32)
    ///   - useRuAccent: Enable RUAccent for Russian (default: true)
    public func configureChatterbox(
        language: ChatterboxLanguage = .english,
        precision: ModelPrecision = .fp32,
        useRuAccent: Bool = true
    ) async throws {
        let provider = TTSProviders.chatterbox(
            precision: precision,
            language: language,
            useRuAccent: useRuAccent
        )
        try await self.preload(provider)
        self.register(synthesizer: provider, for: .chatterbox(language: language))
    }
}

