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

@Suite("RUAccent Provider Tests")
struct RUAccentProviderTests {
    
    // Compute project root from source file path
    static let projectRoot: String = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.path
    }()
    
    // Path to ruaccent assets - relative to project root
    static let modelsDir = URL(fileURLWithPath: "\(projectRoot)/Models/ruaccent/models/balanced")
    static let assetsDir = URL(fileURLWithPath: "\(projectRoot)/Models/ruaccent/assets/balanced")
    
    /// Check if models are available (use with try #require)
    static var modelsAvailable: Bool {
        FileManager.default.fileExists(atPath: modelsDir.path)
    }
    
    @Test("Basic stress marking")
    func testBasicStressMarking() throws {
        try #require(Self.modelsAvailable, "RUAccent models not found at \(Self.modelsDir.path)")
        
        let provider = try RUAccentProvider(
            profile: .balanced,
            modelsDir: Self.modelsDir,
            assetsDir: Self.assetsDir
        )
        
        // Test simple word
        let result = try provider.process("привет")
        #expect(result.contains("+"), "Result should contain stress mark: \(result)")
        print("✅ 'привет' → '\(result)'")
    }
    
    @Test("Multiple words")
    func testMultipleWords() throws {
        try #require(Self.modelsAvailable, "RUAccent models not found")
        
        let provider = try RUAccentProvider(
            profile: .balanced,
            modelsDir: Self.modelsDir,
            assetsDir: Self.assetsDir
        )
        
        let result = try provider.process("привет мир")
        #expect(result.contains("+"), "Result should contain stress marks: \(result)")
        print("✅ 'привет мир' → '\(result)'")
    }
    
    @Test("Dictionary lookup")
    func testDictionaryLookup() throws {
        try #require(Self.modelsAvailable, "RUAccent models not found")
        
        let provider = try RUAccentProvider(
            profile: .balanced,
            modelsDir: Self.modelsDir,
            assetsDir: Self.assetsDir
        )
        
        // Test word that should be in dictionary
        let result = try provider.process("закрыли")
        print("✅ 'закрыли' → '\(result)'")
    }
    
    @Test("Punctuation preserved")
    func testPunctuationPreserved() throws {
        try #require(Self.modelsAvailable, "RUAccent models not found")
        
        let provider = try RUAccentProvider(
            profile: .balanced,
            modelsDir: Self.modelsDir,
            assetsDir: Self.assetsDir
        )
        
        // Test with punctuation
        let result = try provider.process("Привет, мир!")
        #expect(result.contains(","), "Punctuation should be preserved: \(result)")
        #expect(result.contains("!"), "Punctuation should be preserved: \(result)")
        print("✅ 'Привет, мир!' → '\(result)'")
    }
    
    @Test("Empty string")
    func testEmptyString() throws {
        try #require(Self.modelsAvailable, "RUAccent models not found")
        
        let provider = try RUAccentProvider(
            profile: .balanced,
            modelsDir: Self.modelsDir,
            assetsDir: Self.assetsDir
        )
        
        let result = try provider.process("")
        #expect(result == "", "Empty input should return empty output")
        print("✅ Empty string handled correctly")
    }
    
    @Test("AudioTool integration")
    func testAudioToolIntegration() async throws {
        try #require(Self.modelsAvailable, "RUAccent models not found")
        
        // Test factory method
        let provider = try TTSProviders.ruaccent(
            modelsDir: Self.modelsDir,
            assetsDir: Self.assetsDir
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
