//
//  TranslateGemmaProviderTests.swift
//  AudioToolMLXTranslationTests
//
//  Integration tests for TranslateGemma translation provider
//

import Testing
import Foundation
@testable import AudioTool
@testable import AudioToolCore
@testable import AudioToolMLXTranslation
import AudioToolTestSupport

/// Whether these tests should run.
///
/// They download a multi-gigabyte Gemma checkpoint from HuggingFace and run it, so
/// they are opt-in like every other suite that leaves the machine. `SKIP_MLX_TESTS`
/// alone was not enough: unset, it meant *run*, so an offline `swift test` spent
/// seven minutes timing out on network calls and then reported five failures.
private var shouldRunMLXTests: Bool {
    TestGate.runIntegrationTests && !TestGate.skipMLXTests
}

/// Trait for enabling tests only when MLX integration tests are opted into
private extension Trait where Self == Testing.ConditionTrait {
    static var enabledForMLX: Self {
        .enabled(if: shouldRunMLXTests,
                 "MLX integration test - set RUN_INTEGRATION_TESTS=1 (downloads a Gemma checkpoint)")
    }
}

@Suite("TranslateGemma Provider Tests")
struct TranslateGemmaProviderTests {
    
    // MARK: - Language Support Tests
    
    @Test("Supported languages list is non-empty")
    func testSupportedLanguagesNotEmpty() {
        let languages = TranslateGemmaProvider.supportedLanguages
        #expect(languages.count > 50)
        #expect(languages.contains("en"))
        #expect(languages.contains("de-DE"))
        #expect(languages.contains("fr-FR"))
        #expect(languages.contains("ja"))
        #expect(languages.contains("zh-CN"))
    }
    
    @Test("isAvailable returns true for supported language pairs")
    func testIsAvailableForSupportedPairs() async {
        let provider = TranslateGemmaProvider()
        
        let available = await provider.isAvailable(from: "en", to: "de-DE")
        #expect(available == true)
        
        let available2 = await provider.isAvailable(from: "en", to: "ja")
        #expect(available2 == true)
    }
    
    @Test("isAvailable returns false for unsupported languages")
    func testIsAvailableForUnsupportedPairs() async {
        let provider = TranslateGemmaProvider()
        
        // "xx" is not a supported language
        let available = await provider.isAvailable(from: "en", to: "xx")
        #expect(available == false)
    }
    
    // MARK: - Integration Tests (Requires Model Download)
    
    @Test("Basic English to German translation", .enabledForMLX)
    func testEnglishToGermanTranslation() async throws {
        let provider = TranslateGemmaProvider(
            maxTokens: 64,
            progressHandler: { progress in
                let pct = Int(progress.fractionCompleted * 100)
                if pct % 20 == 0 {
                    print("Loading model: \(pct)%")
                }
            }
        )
        
        let result = try await provider.translate(
            "Hello, how are you?",
            from: "en",
            to: "de-DE"
        )
        
        print("Translation: '\(result.sourceText)' -> '\(result.translatedText)'")
        
        #expect(!result.translatedText.isEmpty)
        #expect(result.sourceLanguage == "en")
        #expect(result.targetLanguage == "de-DE")
        // Common German greetings
        let lowerResult = result.translatedText.lowercased()
        #expect(lowerResult.contains("hallo") || lowerResult.contains("wie"))
    }
    
    @Test("English to French translation", .enabledForMLX)
    func testEnglishToFrenchTranslation() async throws {
        let provider = TranslateGemmaProvider(maxTokens: 64)
        
        let result = try await provider.translate(
            "Good morning",
            from: "en",
            to: "fr-FR"
        )
        
        print("Translation: '\(result.sourceText)' -> '\(result.translatedText)'")
        
        #expect(!result.translatedText.isEmpty)
        let lowerResult = result.translatedText.lowercased()
        #expect(lowerResult.contains("bonjour") || lowerResult.contains("matin"))
    }
    
    @Test("English to Japanese translation", .enabledForMLX)
    func testEnglishToJapaneseTranslation() async throws {
        let provider = TranslateGemmaProvider(maxTokens: 64)
        
        let result = try await provider.translate(
            "Thank you",
            from: "en",
            to: "ja"
        )
        
        print("Translation: '\(result.sourceText)' -> '\(result.translatedText)'")
        
        #expect(!result.translatedText.isEmpty)
        // Should contain Japanese characters
        let containsJapanese = result.translatedText.unicodeScalars.contains { 
            ($0.value >= 0x3040 && $0.value <= 0x309F) ||  // Hiragana
            ($0.value >= 0x30A0 && $0.value <= 0x30FF) ||  // Katakana
            ($0.value >= 0x4E00 && $0.value <= 0x9FFF)     // Kanji
        }
        #expect(containsJapanese)
    }
    
    @Test("Batch translation", .enabledForMLX)
    func testBatchTranslation() async throws {
        let provider = TranslateGemmaProvider(maxTokens: 64)
        
        let texts = ["Hello", "Goodbye", "Thank you"]
        let result = try await provider.translateBatch(texts, from: "en", to: "es-419")
        
        print("Batch translations:")
        for translation in result.translations {
            print("  '\(translation.sourceText)' -> '\(translation.translatedText)'")
        }
        
        #expect(result.translations.count == 3)
        #expect(result.targetLanguage == "es-419")
        
        for translation in result.translations {
            #expect(!translation.translatedText.isEmpty)
        }
    }
    
    // MARK: - AudioTool Integration
    
    @Test("AudioTool integration with TranslateGemma", .enabledForMLX)
    func testAudioToolIntegration() async throws {
        // Create a fresh AudioTool and explicitly register TranslateGemma
        // Note: AudioTool is a shared actor, so we specify the model explicitly
        let voice = AudioEngine()
        let provider = TranslateGemmaProvider(maxTokens: 64)
        await voice.register(translator: provider, for: .translateGemma)
        
        // Use translateGemma model explicitly to avoid conflicts with other registered translators
        let result = try await voice.translate("Hello", from: "en", to: "de-DE", model: .translateGemma)
        
        print("AudioTool translation: '\(result.sourceText)' -> '\(result.translatedText)'")
        
        #expect(!result.translatedText.isEmpty)
        #expect(result.targetLanguage == "de-DE")
    }
}
