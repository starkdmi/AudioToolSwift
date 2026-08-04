//
//  AudioToolTranslation.swift
//  AudioToolTranslation
//
//  Public exports and convenience factory for translation providers
//

import Foundation
import AudioTool
import AudioToolCore

// Re-export core types
@_exported import AudioTool
@_exported import AudioToolCore

/// Factory for creating translation providers
public struct TranslationProviders {
    
    // MARK: - Apple Translation
    
    /// Create Apple Translation provider (iOS 26+, macOS 26+)
    ///
    /// > Note: Requires iOS 26+ or macOS 26+ for programmatic access.
    /// > For iOS 18/macOS 15, use SwiftUI's `.translationTask` modifier instead.
    ///
    /// - Returns: Configured AppleTranslationProvider
    @available(iOS 26.0, macOS 26.0, *)
    public static func apple() -> AppleTranslationProvider {
        AppleTranslationProvider()
    }
}

/// AudioEngine extension for registering translation providers
@available(iOS 26.0, macOS 26.0, *)
extension AudioEngine {
    
    /// Configure with Apple Translation provider
    public func configureAppleTranslation() {
        let provider = TranslationProviders.apple()
        self.register(translator: provider, for: .appleTranslation)
    }
}

