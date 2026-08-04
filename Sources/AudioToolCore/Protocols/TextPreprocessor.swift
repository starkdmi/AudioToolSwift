//
//  TextPreprocessor.swift
//  AudioToolCore
//
//  Protocol for text-to-text preprocessing (stress marking, phonetization, etc.)
//

import Foundation

// MARK: - Text Preprocessing Protocol

/// Text preprocessing for TTS synthesis
///
/// Text preprocessors transform input text before synthesis.
/// Examples include stress marking for Russian TTS.
///
/// Usage:
/// ```swift
/// let ruaccent = try RUAccentProvider(profile: .balanced, ...)
/// let stressed = try ruaccent.process("привет мир")
/// // stressed = "прив+ет м+ир"
/// ```
public protocol TextPreprocessor: Sendable {
    /// Process text before TTS synthesis
    /// - Parameter text: Input text
    /// - Returns: Processed text (e.g., with stress marks)
    func process(_ text: String) throws -> String
}
