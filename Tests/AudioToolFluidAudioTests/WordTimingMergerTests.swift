//
//  WordTimingMergerTests.swift
//  ClearVoiceFluidAudioTests
//
//  Unit tests for WordTimingMerger token-to-word merging logic
//

import XCTest
@testable import ClearVoiceFluidAudio
import FluidAudio

final class WordTimingMergerTests: XCTestCase {
    
    // MARK: - Basic Merging Tests
    
    /// Test merging tokens with space-prefixed word boundaries
    func testMergeTokensWithSpacePrefixes() {
        // Simulate BPE tokens: " Hel" + "lo" + " wor" + "ld"
        let tokens = [
            TokenTiming(token: " Hel", tokenId: 1, startTime: 0.08, endTime: 0.16, confidence: 0.9),
            TokenTiming(token: "lo", tokenId: 2, startTime: 0.16, endTime: 0.24, confidence: 0.85),
            TokenTiming(token: " wor", tokenId: 3, startTime: 0.32, endTime: 0.40, confidence: 0.88),
            TokenTiming(token: "ld", tokenId: 4, startTime: 0.40, endTime: 0.48, confidence: 0.92),
        ]
        
        let words = WordTimingMerger.mergeTokensIntoWords(tokens)
        
        XCTAssertEqual(words.count, 2, "Should produce 2 words")
        
        // First word: "Hello"
        XCTAssertEqual(words[0].word, "Hel" + "lo")
        XCTAssertEqual(words[0].startTime, 0.08, accuracy: 0.001)
        XCTAssertEqual(words[0].endTime, 0.24, accuracy: 0.001)
        XCTAssertEqual(words[0].confidence, (0.9 + 0.85) / 2, accuracy: 0.001)
        
        // Second word: "world"
        XCTAssertEqual(words[1].word, "wor" + "ld")
        XCTAssertEqual(words[1].startTime, 0.32, accuracy: 0.001)
        XCTAssertEqual(words[1].endTime, 0.48, accuracy: 0.001)
        XCTAssertEqual(words[1].confidence, (0.88 + 0.92) / 2, accuracy: 0.001)
    }
    
    /// Test merging tokens without leading space (first word scenario)
    func testMergeTokensWithoutLeadingSpace() {
        // First token has no space prefix
        let tokens = [
            TokenTiming(token: "Hel", tokenId: 1, startTime: 0.0, endTime: 0.08, confidence: 0.9),
            TokenTiming(token: "lo", tokenId: 2, startTime: 0.08, endTime: 0.16, confidence: 0.85),
            TokenTiming(token: " world", tokenId: 3, startTime: 0.24, endTime: 0.40, confidence: 0.88),
        ]
        
        let words = WordTimingMerger.mergeTokensIntoWords(tokens)
        
        XCTAssertEqual(words.count, 2, "Should produce 2 words")
        XCTAssertEqual(words[0].word, "Hello")
        XCTAssertEqual(words[1].word, "world")
    }
    
    /// Test single token = single word
    func testSingleToken() {
        let tokens = [
            TokenTiming(token: "Hello", tokenId: 1, startTime: 0.0, endTime: 0.24, confidence: 0.95),
        ]
        
        let words = WordTimingMerger.mergeTokensIntoWords(tokens)
        
        XCTAssertEqual(words.count, 1)
        XCTAssertEqual(words[0].word, "Hello")
        XCTAssertEqual(words[0].startTime, 0.0)
        XCTAssertEqual(words[0].endTime, 0.24)
        XCTAssertEqual(words[0].confidence, 0.95)
    }
    
    /// Test empty input returns empty output
    func testEmptyInput() {
        let words = WordTimingMerger.mergeTokensIntoWords([])
        XCTAssertTrue(words.isEmpty)
    }
    
    // MARK: - Edge Cases
    
    /// Test tokens with newline boundaries
    func testNewlineBoundaries() {
        let tokens = [
            TokenTiming(token: "First", tokenId: 1, startTime: 0.0, endTime: 0.2, confidence: 0.9),
            TokenTiming(token: "\nSecond", tokenId: 2, startTime: 0.3, endTime: 0.5, confidence: 0.85),
        ]
        
        let words = WordTimingMerger.mergeTokensIntoWords(tokens)
        
        XCTAssertEqual(words.count, 2)
        XCTAssertEqual(words[0].word, "First")
        XCTAssertEqual(words[1].word, "Second")
    }
    
    /// Test tokens with tab boundaries
    func testTabBoundaries() {
        let tokens = [
            TokenTiming(token: "Word1", tokenId: 1, startTime: 0.0, endTime: 0.2, confidence: 0.9),
            TokenTiming(token: "\tWord2", tokenId: 2, startTime: 0.3, endTime: 0.5, confidence: 0.85),
        ]
        
        let words = WordTimingMerger.mergeTokensIntoWords(tokens)
        
        XCTAssertEqual(words.count, 2)
        XCTAssertEqual(words[0].word, "Word1")
        XCTAssertEqual(words[1].word, "Word2")
    }
    
    /// Test punctuation attached to words
    func testPunctuationAttached() {
        let tokens = [
            TokenTiming(token: " Hello", tokenId: 1, startTime: 0.0, endTime: 0.16, confidence: 0.9),
            TokenTiming(token: ",", tokenId: 2, startTime: 0.16, endTime: 0.20, confidence: 0.95),
            TokenTiming(token: " world", tokenId: 3, startTime: 0.24, endTime: 0.40, confidence: 0.88),
            TokenTiming(token: "!", tokenId: 4, startTime: 0.40, endTime: 0.44, confidence: 0.92),
        ]
        
        let words = WordTimingMerger.mergeTokensIntoWords(tokens)
        
        XCTAssertEqual(words.count, 2, "Punctuation should attach to preceding word")
        XCTAssertEqual(words[0].word, "Hello,")
        XCTAssertEqual(words[1].word, "world!")
    }
    
    /// Test many subword tokens forming one word
    func testManySubwordsOneWord() {
        // Simulating a long word split into many pieces
        let tokens = [
            TokenTiming(token: " anti", tokenId: 1, startTime: 0.0, endTime: 0.1, confidence: 0.9),
            TokenTiming(token: "dis", tokenId: 2, startTime: 0.1, endTime: 0.2, confidence: 0.85),
            TokenTiming(token: "establish", tokenId: 3, startTime: 0.2, endTime: 0.4, confidence: 0.88),
            TokenTiming(token: "ment", tokenId: 4, startTime: 0.4, endTime: 0.5, confidence: 0.90),
            TokenTiming(token: "arian", tokenId: 5, startTime: 0.5, endTime: 0.6, confidence: 0.87),
            TokenTiming(token: "ism", tokenId: 6, startTime: 0.6, endTime: 0.7, confidence: 0.91),
        ]
        
        let words = WordTimingMerger.mergeTokensIntoWords(tokens)
        
        XCTAssertEqual(words.count, 1)
        XCTAssertEqual(words[0].word, "antidisestablishmentarianism")
        XCTAssertEqual(words[0].startTime, 0.0)
        XCTAssertEqual(words[0].endTime, 0.7)
        
        // Average confidence of all tokens
        let sum: Float = 0.9 + 0.85 + 0.88 + 0.90 + 0.87 + 0.91
        let expectedConfidence = sum / 6.0
        XCTAssertEqual(words[0].confidence, expectedConfidence, accuracy: 0.001)
    }
    
    // MARK: - Confidence Calculation
    
    /// Test confidence averaging across merged tokens
    func testConfidenceAveraging() {
        let tokens = [
            TokenTiming(token: " test", tokenId: 1, startTime: 0.0, endTime: 0.1, confidence: 0.6),
            TokenTiming(token: "ing", tokenId: 2, startTime: 0.1, endTime: 0.2, confidence: 0.8),
            TokenTiming(token: " word", tokenId: 3, startTime: 0.3, endTime: 0.4, confidence: 1.0),
        ]
        
        let words = WordTimingMerger.mergeTokensIntoWords(tokens)
        
        XCTAssertEqual(words.count, 2)
        XCTAssertEqual(words[0].confidence, 0.7, accuracy: 0.001) // (0.6 + 0.8) / 2
        XCTAssertEqual(words[1].confidence, 1.0, accuracy: 0.001) // Single token
    }
}
