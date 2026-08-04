//
//  AppleTranslationProvider.swift
//  AudioToolTranslation
//
//  Apple Translation framework provider (iOS 26+, macOS 26+)
//
//  Note: Programmatic TranslationSession requires iOS 26+/macOS 26+.
//  On earlier SDKs, a placeholder implementation is provided.
//

import Foundation
import AudioToolCore

// Check for Translation framework and compiler version
#if canImport(Translation) && compiler(>=6.2)
@preconcurrency import Translation

// MARK: - Apple Translation Provider (Full Implementation)

/// Apple on-device translation provider
///
/// Uses Apple's Translation framework for private, on-device translation.
/// Supports 20+ languages with automatic language detection.
///
/// Usage:
/// ```swift
/// let translator = TranslationProviders.apple()
/// let result = try await translator.translate("Hello", from: "en", to: "es")
/// print(result.translatedText) // "Hola"
/// ```
///
/// > Note: Requires iOS 26+ or macOS 26+ for programmatic (non-SwiftUI) access.
/// > Language models are managed by the system and may be downloaded on first use.
/// > For iOS 18/macOS 15, use SwiftUI's `.translationTask` modifier instead.
@available(iOS 26.0, macOS 26.0, *)
public actor AppleTranslationProvider: TextTranslator {
    
    // MARK: - Properties
    
    /// Current translation session (cached for language pair)
    private var session: TranslationSession?
    
    /// Cached source/target for session reuse check
    private var cachedSource: Locale.Language?
    private var cachedTarget: Locale.Language?
    
    // MARK: - Initialization
    
    public init() {}
    
    // MARK: - TextTranslator Conformance
    
    public func translate(
        _ text: String,
        from source: String?,
        to target: String
    ) async throws -> TranslationResult {
        let session = try await getOrCreateSession(source: source, target: target)
        
        let response = try await session.translate(text)
        
        return TranslationResult(
            sourceText: text,
            translatedText: response.targetText,
            sourceLanguage: response.sourceLanguage.languageCode?.identifier,
            targetLanguage: target
        )
    }
    
    public func translateBatch(
        _ texts: [String],
        from source: String?,
        to target: String
    ) async throws -> BatchTranslationResult {
        let session = try await getOrCreateSession(source: source, target: target)
        
        // Create batch requests with client identifiers
        let requests = texts.enumerated().map { index, text in
            TranslationSession.Request(sourceText: text, clientIdentifier: "\(index)")
        }
        
        // Translate batch
        let responses = try await session.translations(from: requests)
        
        // Map responses back to results
        var translations: [TranslationResult] = []
        for response in responses {
            translations.append(TranslationResult(
                sourceText: response.sourceText,
                translatedText: response.targetText,
                sourceLanguage: response.sourceLanguage.languageCode?.identifier,
                targetLanguage: target
            ))
        }
        
        return BatchTranslationResult(
            translations: translations,
            sourceLanguage: source,
            targetLanguage: target
        )
    }
    
    public func isAvailable(from source: String, to target: String) async -> Bool {
        let sourceLanguage = Locale.Language(identifier: source)
        let targetLanguage = Locale.Language(identifier: target)
        
        let availability = LanguageAvailability()
        let status = await availability.status(from: sourceLanguage, to: targetLanguage)
        
        return status == .installed
    }
    
    public func prepareLanguagePair(from source: String, to target: String) async throws {
        // Clear cached session so it's recreated on next translate call
        session = nil
        cachedSource = Locale.Language(identifier: source)
        cachedTarget = Locale.Language(identifier: target)
    }
    
    // MARK: - Private Methods
    
    private func getOrCreateSession(source: String?, target: String) async throws -> TranslationSession {
        let targetLanguage = Locale.Language(identifier: target)
        let sourceLanguage: Locale.Language? = source.map { Locale.Language(identifier: $0) }
        
        // Check if we can reuse existing session
        if let existingSession = session,
           cachedTarget == targetLanguage,
           cachedSource == sourceLanguage {
            return existingSession
        }
        
        // Create new session using installedSource:target: initializer
        let newSession: TranslationSession
        if let sourceLanguage = sourceLanguage {
            newSession = TranslationSession(installedSource: sourceLanguage, target: targetLanguage)
        } else {
            // When source is nil, use English as default and let translate handle detection
            newSession = TranslationSession(installedSource: .init(identifier: "en"), target: targetLanguage)
        }
        
        self.cachedSource = sourceLanguage
        self.cachedTarget = targetLanguage
        self.session = newSession
        
        return newSession
    }
}

#else

// MARK: - Placeholder Implementation (Pre-Xcode 26 SDK)

/// Placeholder for AppleTranslationProvider when building with Xcode < 26
///
/// This placeholder is compiled when the programmatic TranslationSession API is not available.
/// To use actual Apple Translation, build with Xcode 26+ on macOS 26+.
///
/// Alternative: Use TranslateGemmaProvider from AudioToolMLXTranslation for on-device translation.
@available(iOS 26.0, macOS 26.0, *)
public actor AppleTranslationProvider: TextTranslator {
    
    public init() {}
    
    public func translate(
        _ text: String,
        from source: String?,
        to target: String
    ) async throws -> TranslationResult {
        throw AudioToolError.resourceUnavailable(
            "Apple Translation programmatic API requires macOS 26+ and Xcode 26+ SDK. " +
            "Use TranslateGemmaProvider for on-device translation on macOS 15."
        )
    }
    
    public func translateBatch(
        _ texts: [String],
        from source: String?,
        to target: String
    ) async throws -> BatchTranslationResult {
        throw AudioToolError.resourceUnavailable(
            "Apple Translation programmatic API requires macOS 26+ and Xcode 26+ SDK."
        )
    }
    
    public func isAvailable(from source: String, to target: String) async -> Bool {
        return false
    }
    
    public func prepareLanguagePair(from source: String, to target: String) async throws {
        throw AudioToolError.resourceUnavailable(
            "Apple Translation programmatic API requires macOS 26+ and Xcode 26+ SDK."
        )
    }
}

#endif // canImport(Translation) && compiler(>=6.2)
