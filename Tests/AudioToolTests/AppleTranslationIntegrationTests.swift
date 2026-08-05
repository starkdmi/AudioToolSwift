//
//  AppleTranslationIntegrationTests.swift
//  AudioToolTranslation
//
//  Integration tests for Apple Translation API
//

import Testing
import Foundation
@testable import AudioTool
@testable import AudioToolCore

#if canImport(Translation)
import Translation
#endif

@Suite("Apple Translation Integration Tests", .enabled(if: TestConfiguration.runIntegrationTests,
        "integration test - set RUN_INTEGRATION_TESTS=1"))
struct AppleTranslationIntegrationTests {
    
    // MARK: - Language Availability Tests (Works on macOS 15+)
    
    #if canImport(Translation)
    @Test("Check language availability - installed languages")
    @available(macOS 15.0, iOS 18.0, *)
    func testLanguageAvailability() async {
        let availability = LanguageAvailability()
        
        // Check English -> French (commonly installed)
        let enToFr = await availability.status(
            from: Locale.Language(identifier: "en"),
            to: Locale.Language(identifier: "fr")
        )
        print("en -> fr: \(enToFr)")
        
        // Check English -> German
        let enToDe = await availability.status(
            from: Locale.Language(identifier: "en"),
            to: Locale.Language(identifier: "de")
        )
        print("en -> de: \(enToDe)")
        
        // Check English -> Italian
        let enToIt = await availability.status(
            from: Locale.Language(identifier: "en"),
            to: Locale.Language(identifier: "it")
        )
        print("en -> it: \(enToIt)")
        
        // Check English -> Russian
        let enToRu = await availability.status(
            from: Locale.Language(identifier: "en"),
            to: Locale.Language(identifier: "ru")
        )
        print("en -> ru: \(enToRu)")
        
        // Check English -> Turkish
        let enToTr = await availability.status(
            from: Locale.Language(identifier: "en"),
            to: Locale.Language(identifier: "tr")
        )
        print("en -> tr: \(enToTr)")
        
        // At least one should be available (or skip if none installed)
        let anyInstalled = [enToFr, enToDe, enToIt, enToRu, enToTr].contains(.installed)
        let anySupported = [enToFr, enToDe, enToIt, enToRu, enToTr].contains(.supported)
        
        if !anyInstalled && !anySupported {
            // No languages installed or supported - this is a machine configuration issue, not a test failure
            print("⚠️ No translation language pairs installed or supported on this machine")
            return
        }
        
        // If at least one is supported but not installed, that's fine - just means user hasn't downloaded
        if !anyInstalled && anySupported {
            print("⚠️ Language pairs are supported but not yet downloaded")
            return
        }
        
        #expect(anyInstalled, "At least one language pair should be installed")
    }
    #endif
    
    // MARK: - TranslationSession Tests (macOS 26+ only)
    // These tests require Xcode 26+ SDK where TranslationSession initializers are available
    
    #if compiler(>=6.2)
    @Test("Direct TranslationSession test - English to French")
    @available(macOS 26.0, iOS 26.0, *)
    func testDirectTranslationSession() async throws {
        // Direct test of Apple's TranslationSession API
        let sourceLanguage = Locale.Language(identifier: "en")
        let targetLanguage = Locale.Language(identifier: "fr")
        
        // Check availability first
        let availability = LanguageAvailability()
        let status = await availability.status(from: sourceLanguage, to: targetLanguage)
        guard status == .installed else {
            print("Skipping: en -> fr not installed (status: \(status))")
            return
        }
        
        let session = TranslationSession(installedSource: sourceLanguage, target: targetLanguage)
        let response = try await session.translate("Hello, world!")
        
        print("Direct translation: 'Hello, world!' -> '\(response.targetText)'")
        
        #expect(!response.targetText.isEmpty)
        #expect(response.targetText.lowercased().contains("bonjour") || 
                response.targetText.lowercased().contains("monde"))
    }
    
    @Test("Direct TranslationSession batch test")
    @available(macOS 26.0, iOS 26.0, *)
    func testDirectBatchTranslation() async throws {
        let sourceLanguage = Locale.Language(identifier: "en")
        let targetLanguage = Locale.Language(identifier: "de")
        
        let availability = LanguageAvailability()
        let status = await availability.status(from: sourceLanguage, to: targetLanguage)
        guard status == .installed else {
            print("Skipping: en -> de not installed (status: \(status))")
            return
        }
        
        let session = TranslationSession(installedSource: sourceLanguage, target: targetLanguage)
        
        let requests = [
            TranslationSession.Request(sourceText: "Hello", clientIdentifier: "0"),
            TranslationSession.Request(sourceText: "Goodbye", clientIdentifier: "1"),
            TranslationSession.Request(sourceText: "Thank you", clientIdentifier: "2"),
        ]
        
        let responses = try await session.translations(from: requests)
        
        print("Batch translations:")
        for response in responses {
            print("  '\(response.sourceText)' -> '\(response.targetText)'")
        }
        
        #expect(responses.count == 3)
    }
    #endif
}
