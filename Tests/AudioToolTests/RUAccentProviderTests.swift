//
//  RUAccentProviderTests.swift
//  AudioToolTests
//
//  Tests for Russian stress/accent preprocessing
//

import Testing
import Foundation
@testable import AudioToolTTS
@testable import AudioToolCore
import AudioToolTestSupport

@Suite("RUAccent Provider Tests", .enabled(if: TestConfiguration.runIntegrationTests,
        "integration test - set RUN_INTEGRATION_TESTS=1"))
struct RUAccentProviderTests {
    
    // RUAccent's dictionaries and models live in the sibling research checkout.
    static let modelsDir = TestGate.reference("Models/ruaccent/models/balanced")
    static let assetsDir = TestGate.reference("Models/ruaccent/assets/balanced")

    /// The RUAccent asset pair, or a `#require` failure naming what is missing.
    static func requireAssets() throws -> (models: URL, assets: URL) {
        let models = try #require(modelsDir,
                                  Comment(rawValue: TestGate.missingReference("Models/ruaccent/models/balanced")))
        let assets = try #require(assetsDir,
                                  Comment(rawValue: TestGate.missingReference("Models/ruaccent/assets/balanced")))
        return (models, assets)
    }
    
    @Test("Basic stress marking")
    func testBasicStressMarking() throws {
        let (modelsDir, assetsDir) = try Self.requireAssets()
        
        let provider = try RUAccentProvider(
            profile: .balanced,
            modelsDir: modelsDir,
            assetsDir: assetsDir
        )
        
        // Test simple word
        let result = try provider.process("привет")
        #expect(result.contains("+"), "Result should contain stress mark: \(result)")
        print("✅ 'привет' → '\(result)'")
    }
    
    @Test("Multiple words")
    func testMultipleWords() throws {
        let (modelsDir, assetsDir) = try Self.requireAssets()
        
        let provider = try RUAccentProvider(
            profile: .balanced,
            modelsDir: modelsDir,
            assetsDir: assetsDir
        )
        
        let result = try provider.process("привет мир")
        #expect(result.contains("+"), "Result should contain stress marks: \(result)")
        print("✅ 'привет мир' → '\(result)'")
    }
    
    @Test("Dictionary lookup")
    func testDictionaryLookup() throws {
        let (modelsDir, assetsDir) = try Self.requireAssets()
        
        let provider = try RUAccentProvider(
            profile: .balanced,
            modelsDir: modelsDir,
            assetsDir: assetsDir
        )
        
        // Test word that should be in dictionary
        let result = try provider.process("закрыли")
        print("✅ 'закрыли' → '\(result)'")
    }
    
    @Test("Punctuation preserved")
    func testPunctuationPreserved() throws {
        let (modelsDir, assetsDir) = try Self.requireAssets()
        
        let provider = try RUAccentProvider(
            profile: .balanced,
            modelsDir: modelsDir,
            assetsDir: assetsDir
        )
        
        // Test with punctuation
        let result = try provider.process("Привет, мир!")
        #expect(result.contains(","), "Punctuation should be preserved: \(result)")
        #expect(result.contains("!"), "Punctuation should be preserved: \(result)")
        print("✅ 'Привет, мир!' → '\(result)'")
    }
    
    @Test("Empty string")
    func testEmptyString() throws {
        let (modelsDir, assetsDir) = try Self.requireAssets()
        
        let provider = try RUAccentProvider(
            profile: .balanced,
            modelsDir: modelsDir,
            assetsDir: assetsDir
        )
        
        let result = try provider.process("")
        #expect(result == "", "Empty input should return empty output")
        print("✅ Empty string handled correctly")
    }
    
    @Test("AudioTool integration")
    func testAudioToolIntegration() async throws {
        let (modelsDir, assetsDir) = try Self.requireAssets()
        
        // Test factory method
        let provider = try TTSProviders.ruaccent(
            modelsDir: modelsDir,
            assetsDir: assetsDir
        )
        
        let result = try provider.process("Москва столица России")
        #expect(result.contains("+"), "Result should contain stress marks: \(result)")
        print("✅ Factory method works: '\(result)'")
    }
    
    @Test("Convert to Unicode stress marks")
    func testConvertToStressMarks() {
        // Test static method
        let input = "м+ы закр+ыли зам+ок"
        let result = RUAccentProvider.convertToStressMarks(input)
        
        // Result should not contain "+"
        #expect(!result.contains("+"), "Result should not contain '+': \(result)")
        
        // Result should be longer than the input without + (due to combining chars)
        let inputWithoutPlus = input.replacingOccurrences(of: "+", with: "")
        #expect(result.unicodeScalars.count > inputWithoutPlus.count, 
                "Result should have combining accents")
        
        // Check the length matches expected (3 vowels get accents = 3 combining chars added)
        // "мы закрыли замок" = 16 chars, add 3 accents = 19 unicode scalars
        #expect(result.unicodeScalars.count == 19, 
                "Expected 19 scalars, got \(result.unicodeScalars.count)")
        
        print("✅ 'м+ы закр+ыли зам+ок' → '\(result)'")
    }
}
