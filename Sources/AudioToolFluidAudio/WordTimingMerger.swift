//
//  WordTimingMerger.swift
//  ClearVoiceFluidAudio
//
//  Merges BPE subword tokens into word-level timing information.
//  Adapted from FluidAudio CLI's TranscribeCommand.swift
//

import Foundation
import FluidAudio

// MARK: - Word Timing

/// Word-level timing information derived from token timings
public struct WordTiming: Sendable, Equatable {
    /// The complete word text
    public let word: String
    
    /// Start time in seconds
    public let startTime: TimeInterval
    
    /// End time in seconds
    public let endTime: TimeInterval
    
    /// Average confidence across constituent tokens
    public let confidence: Float
    
    public init(word: String, startTime: TimeInterval, endTime: TimeInterval, confidence: Float) {
        self.word = word
        self.startTime = startTime
        self.endTime = endTime
        self.confidence = confidence
    }
}

// MARK: - Word Timing Merger

/// Helper to merge BPE subword tokens into word-level timings
///
/// This merger assumes that the ASR tokenizer produces subword tokens where:
/// - Tokens starting with whitespace (space, newline, tab) indicate word boundaries
/// - Multiple consecutive tokens without leading whitespace form a single word
/// - This pattern is typical for BPE (Byte Pair Encoding) tokenizers like SentencePiece
///
/// Example: Tokens `[" H", "ello", " wor", "ld"]` -> Words `["Hello", "world"]`
public enum WordTimingMerger {
    
    /// Merge token timings into word-level timings by detecting word boundaries
    ///
    /// - Parameter tokenTimings: Array of token-level timing information from the ASR model
    /// - Returns: Array of word-level timing information with merged tokens
    public static func mergeTokensIntoWords(_ tokenTimings: [TokenTiming]) -> [WordTiming] {
        guard !tokenTimings.isEmpty else { return [] }
        
        var wordTimings: [WordTiming] = []
        var currentWord = ""
        var currentStartTime: TimeInterval?
        var currentEndTime: TimeInterval = 0
        var currentConfidences: [Float] = []
        
        for timing in tokenTimings {
            let token = timing.token
            
            // Check if token starts with whitespace (indicates new word boundary)
            if token.hasPrefix(" ") || token.hasPrefix("\n") || token.hasPrefix("\t") {
                // Finish previous word if exists
                if !currentWord.isEmpty, let startTime = currentStartTime {
                    wordTimings.append(
                        WordTiming(
                            word: currentWord,
                            startTime: startTime,
                            endTime: currentEndTime,
                            confidence: averageConfidence(currentConfidences)
                        ))
                }
                
                // Start new word (trim leading whitespace)
                currentWord = token.trimmingCharacters(in: .whitespacesAndNewlines)
                currentStartTime = timing.startTime
                currentEndTime = timing.endTime
                currentConfidences = [timing.confidence]
            } else {
                // Continue current word or start first word if no whitespace prefix
                if currentStartTime == nil {
                    currentStartTime = timing.startTime
                }
                currentWord += token
                currentEndTime = timing.endTime
                currentConfidences.append(timing.confidence)
            }
        }
        
        // Add final word
        if !currentWord.isEmpty, let startTime = currentStartTime {
            wordTimings.append(
                WordTiming(
                    word: currentWord,
                    startTime: startTime,
                    endTime: currentEndTime,
                    confidence: averageConfidence(currentConfidences)
                ))
        }
        
        return wordTimings
    }
    
    /// Calculate average confidence from an array of confidence scores
    /// - Parameter confidences: Array of confidence values
    /// - Returns: Average confidence, or 0.0 if array is empty
    private static func averageConfidence(_ confidences: [Float]) -> Float {
        confidences.isEmpty ? 0.0 : confidences.reduce(0, +) / Float(confidences.count)
    }
}
