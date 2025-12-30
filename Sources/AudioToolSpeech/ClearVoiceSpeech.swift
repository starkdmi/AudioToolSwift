//
//  ClearVoiceSpeech.swift
//  ClearVoiceSpeech
//
//  Public exports and convenience factory for Apple Speech providers
//

import Foundation
import ClearVoice
import ClearVoiceCore

// Re-export core types
@_exported import ClearVoice
@_exported import ClearVoiceCore

// MARK: - Factory for Speech Providers

/// Factory for creating Speech-to-Text providers
public struct SpeechProviders {
    
    // MARK: - Apple Speech (SpeechAnalyzer)
    
    /// Create Apple Speech transcriber using SpeechAnalyzer (iOS 26+)
    ///
    /// On-device speech-to-text with:
    /// - Full offline support
    /// - Word-level timestamps
    /// - Multi-language support
    ///
    /// - Parameter locale: BCP-47 locale identifier (e.g., "en-US", "fr-FR")
    /// - Returns: Configured AppleSpeechTranscriber
    ///
    /// Example:
    /// ```swift
    /// let transcriber = SpeechProviders.appleSpeech(locale: "en-US")
    /// try await transcriber.load()
    /// let result = try await transcriber.transcribe(audio)
    /// ```
    @available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
    public static func appleSpeech(locale: String = "en-US") -> AppleSpeechTranscriber {
        AppleSpeechTranscriber(locale: locale)
    }
    
    /// List all supported locales
    @available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
    public static func supportedLocales() async -> [Locale] {
        await AppleSpeechTranscriber.supportedLocales()
    }
}

// MARK: - ClearVoice Extensions

extension ClearVoice {
    
    /// Configure with Apple Speech transcriber
    /// - Parameters:
    ///   - transcriber: The AppleSpeechTranscriber instance
    ///   - model: The transcription model type (default: appleSpeech)
    @available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
    public func configure(
        transcriber: AppleSpeechTranscriber,
        for model: TranscriptionModel = .appleSpeech
    ) async throws {
        try await transcriber.load()
        self.register(transcriber: transcriber, for: model)
    }
    
    /// Convenience: Configure Apple Speech for a specific locale
    /// - Parameter locale: BCP-47 locale identifier (e.g., "en-US", "de-DE")
    @available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
    public func configureAppleSpeech(locale: String = "en-US") async throws {
        let provider = SpeechProviders.appleSpeech(locale: locale)
        try await provider.load()
        self.register(transcriber: provider, for: .appleSpeech)
    }
}
