//
//  TranslationTests.swift
//  ClearVoice
//
//  Tests for translation types and protocol
//

import Testing
@testable import ClearVoice
@testable import ClearVoiceCore

@Suite("Translation Tests")
struct TranslationTests {
    
    // MARK: - TranslationResult
    
    @Test("TranslationResult creation")
    func testTranslationResultCreation() {
        let result = TranslationResult(
            sourceText: "Hello, world!",
            translatedText: "¡Hola, mundo!",
            sourceLanguage: "en",
            targetLanguage: "es"
        )
        
        #expect(result.sourceText == "Hello, world!")
        #expect(result.translatedText == "¡Hola, mundo!")
        #expect(result.sourceLanguage == "en")
        #expect(result.targetLanguage == "es")
    }
    
    @Test("TranslationResult with nil source language (auto-detect)")
    func testTranslationResultAutoDetect() {
        let result = TranslationResult(
            sourceText: "Bonjour",
            translatedText: "Hello",
            sourceLanguage: nil,
            targetLanguage: "en"
        )
        
        #expect(result.sourceLanguage == nil)
        #expect(result.targetLanguage == "en")
    }
    
    // MARK: - BatchTranslationResult
    
    @Test("BatchTranslationResult creation")
    func testBatchTranslationResult() {
        let translations = [
            TranslationResult(sourceText: "Hello", translatedText: "Hola", sourceLanguage: "en", targetLanguage: "es"),
            TranslationResult(sourceText: "Goodbye", translatedText: "Adiós", sourceLanguage: "en", targetLanguage: "es"),
        ]
        
        let batch = BatchTranslationResult(
            translations: translations,
            sourceLanguage: "en",
            targetLanguage: "es"
        )
        
        #expect(batch.translations.count == 2)
        #expect(batch.sourceLanguage == "en")
        #expect(batch.targetLanguage == "es")
        #expect(batch.translations[0].translatedText == "Hola")
        #expect(batch.translations[1].translatedText == "Adiós")
    }
    
    // MARK: - TranslationModel
    
    @Test("TranslationModel modelName")
    func testTranslationModelName() {
        let model = TranslationModel.appleTranslation
        #expect(model.modelName == "apple_translation")
    }
    
    // MARK: - Mock TextTranslator
    
    @Test("Mock translator single translation")
    func testMockTranslatorSingle() async throws {
        let translator = MockTextTranslator()
        
        let result = try await translator.translate("Hello", from: "en", to: "es")
        
        #expect(result.sourceText == "Hello")
        #expect(result.translatedText == "[es] Hello")
        #expect(result.targetLanguage == "es")
    }
    
    @Test("Mock translator batch translation")
    func testMockTranslatorBatch() async throws {
        let translator = MockTextTranslator()
        
        let batch = try await translator.translateBatch(["Hello", "World"], from: nil, to: "ja")
        
        #expect(batch.translations.count == 2)
        #expect(batch.translations[0].translatedText == "[ja] Hello")
        #expect(batch.translations[1].translatedText == "[ja] World")
        #expect(batch.targetLanguage == "ja")
    }
    
    @Test("Mock translator availability")
    func testMockTranslatorAvailability() async {
        let translator = MockTextTranslator()
        
        let available = await translator.isAvailable(from: "en", to: "es")
        #expect(available == true)
    }
    
    // MARK: - ClearVoice Integration
    
    @Test("ClearVoice translate throws when no provider registered")
    func testClearVoiceTranslateNoProvider() async {
        let voice = ClearVoice()
        
        do {
            _ = try await voice.translate("Hello", to: "es")
            #expect(Bool(false), "Expected error to be thrown")
        } catch {
            // Expected: modelNotLoaded error
            #expect(error is ClearVoiceError)
        }
    }
    
    @Test("ClearVoice translate with mock provider")
    func testClearVoiceTranslateWithMock() async throws {
        let voice = ClearVoice()
        let mockTranslator = MockTextTranslator()
        
        await voice.register(translator: mockTranslator, for: .appleTranslation)
        
        let result = try await voice.translate("Hello", from: "en", to: "fr")
        
        #expect(result.translatedText == "[fr] Hello")
        #expect(result.targetLanguage == "fr")
    }
    
    @Test("ClearVoice translateBatch with mock provider")
    func testClearVoiceTranslateBatchWithMock() async throws {
        let voice = ClearVoice()
        let mockTranslator = MockTextTranslator()
        
        await voice.register(translator: mockTranslator, for: .appleTranslation)
        
        let batch = try await voice.translateBatch(["One", "Two", "Three"], from: nil, to: "de")
        
        #expect(batch.translations.count == 3)
        #expect(batch.translations[0].translatedText == "[de] One")
        #expect(batch.translations[2].translatedText == "[de] Three")
    }
}

// MARK: - Mock TextTranslator

/// Mock translator for testing - returns "[target] source" pattern
final class MockTextTranslator: TextTranslator, @unchecked Sendable {
    
    func translate(_ text: String, from source: String?, to target: String) async throws -> TranslationResult {
        TranslationResult(
            sourceText: text,
            translatedText: "[\(target)] \(text)",
            sourceLanguage: source,
            targetLanguage: target
        )
    }
    
    func translateBatch(_ texts: [String], from source: String?, to target: String) async throws -> BatchTranslationResult {
        let translations = texts.map { text in
            TranslationResult(
                sourceText: text,
                translatedText: "[\(target)] \(text)",
                sourceLanguage: source,
                targetLanguage: target
            )
        }
        return BatchTranslationResult(translations: translations, sourceLanguage: source, targetLanguage: target)
    }
    
    func isAvailable(from source: String, to target: String) async -> Bool {
        true
    }
    
    func prepareLanguagePair(from source: String, to target: String) async throws {
        // No-op for mock
    }
}
