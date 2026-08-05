//
//  WordTimingIntegrationTests.swift
//  AudioToolFluidAudioTests
//
//  Integration tests for word-level timing using harry_potter.wav
//  Compares ASR word timings against manual sentence-level annotations
//

import XCTest
@testable import AudioToolFluidAudio
import AudioToolCore
import AudioUtils
import MLX

final class WordTimingIntegrationTests: XCTestCase {
    
    // MARK: - Reference Data Model
    
    struct ReferenceSegment: Codable {
        let speaker_id: String
        let persona: String
        let start: Double
        let end: Double
        let text: String
    }
    
    // MARK: - Harry Potter Word Timing Test
    
    /// Test word-level timing extraction on harry_potter.wav
    /// Compares against manual sentence-level annotations from harry_potter.json
    func testHarryPotterWordTiming() async throws {
        print("\n=== Harry Potter Word Timing Integration Test ===\n")
        
        // Load test audio from Fixtures
        guard let audioURL = Bundle.module.url(forResource: "speech_long", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("harry_potter.wav not found in Fixtures")
        }
        
        // Load reference annotations from Docs (use hardcoded path since #file resolution varies)
        // Try multiple potential locations
        let possiblePaths = [
            "/path/to/ProjectTwo/Docs/harry_potter.json",
            URL(fileURLWithPath: #file)
                .deletingLastPathComponent() // Tests/AudioToolFluidAudioTests
                .deletingLastPathComponent() // Tests
                .deletingLastPathComponent() // AudioTool
                .deletingLastPathComponent() // ProjectTwo
                .appendingPathComponent("Docs/harry_potter.json").path
        ]
        
        var docsPath: URL?
        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                docsPath = URL(fileURLWithPath: path)
                break
            }
        }
        
        guard let jsonPath = docsPath else {
            throw XCTSkip("harry_potter.json not found at expected locations")
        }
        
        let referenceData = try Data(contentsOf: jsonPath)
        let referenceSegments = try JSONDecoder().decode([ReferenceSegment].self, from: referenceData)
        
        print("Loaded \(referenceSegments.count) reference segments")
        
        // Load audio at 16kHz for ASR
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: audioURL)
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let duration = Double(samples.count) / 16000.0
        print("Audio: \(String(format: "%.1f", duration))s at 16kHz")
        
        // Load transcriber
        print("Loading Parakeet v3 model...")
        let transcriber = FluidAudioProviders.parakeetTranscriber(version: .v3)
        try await transcriber.load()
        
        // Transcribe with word-level timing
        print("Transcribing with word-level timing...")
        let startTime = Date()
        let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        let result = try await transcriber.transcribe(input)
        let transcribeTime = Date().timeIntervalSince(startTime)
        
        print("Transcription completed in \(String(format: "%.2f", transcribeTime))s")
        print("RTF: \(String(format: "%.0f", duration / transcribeTime))x")
        
        // Verify segments were extracted
        XCTAssertFalse(result.segments.isEmpty, "Should have word-level segments")
        print("\nExtracted \(result.segments.count) word segments")
        
        // Print first 20 words with timing
        print("\n--- First 20 Words with Timing ---")
        for (i, segment) in result.segments.prefix(20).enumerated() {
            let startStr = String(format: "%.2f", segment.timeRange.start)
            let endStr = String(format: "%.2f", segment.timeRange.end)
            let confStr = String(format: "%.2f", segment.confidence)
            print("[\(i)] \(startStr)s - \(endStr)s: \"\(segment.text)\" (conf: \(confStr))")
        }
        print("---\n")
        
        // Validate timing alignment with reference segments
        print("=== Timing Alignment Validation ===\n")
        
        var alignedCount = 0
        var totalChecked = 0
        
        for refSegment in referenceSegments {
            // Skip non-speech segments
            if refSegment.text.hasPrefix("[") { continue }
            
            // Find ASR words that overlap with this reference segment
            let overlappingWords = result.segments.filter { word in
                word.timeRange.start < refSegment.end && word.timeRange.end > refSegment.start
            }
            
            if !overlappingWords.isEmpty {
                // Check if first overlapping word is within tolerance of reference start
                let firstWord = overlappingWords[0]
                let startDiff = abs(firstWord.timeRange.start - refSegment.start)
                
                // 0.5s tolerance for segment boundary alignment
                let isAligned = startDiff < 0.5
                if isAligned { alignedCount += 1 }
                totalChecked += 1
                
                let status = isAligned ? "✓" : "⚠"
                print("\(status) Ref[\(String(format: "%.1f", refSegment.start))s]: \"\(refSegment.text.prefix(40))...\"")
                print("  ASR[\(String(format: "%.2f", firstWord.timeRange.start))s]: \"\(overlappingWords.map(\.text).joined(separator: " ").prefix(40))...\"")
                print("  Offset: \(String(format: "%.2f", startDiff))s")
                print("")
            }
        }
        
        // Calculate alignment rate
        let alignmentRate = Double(alignedCount) / Double(max(1, totalChecked))
        print("=== Summary ===")
        print("Aligned segments: \(alignedCount)/\(totalChecked) (\(String(format: "%.0f", alignmentRate * 100))%)")
        print("Total words extracted: \(result.segments.count)")
        print("Full transcription length: \(result.text.count) characters")
        
        // Assertions
        XCTAssertGreaterThan(result.segments.count, 50, "Should have at least 50 word segments for 2min audio")
        XCTAssertGreaterThan(alignmentRate, 0.5, "At least 50% of reference segments should be aligned within 0.5s")
        
        // Verify timing monotonicity (words should be in order)
        var previousEnd: Double = 0
        var outOfOrderCount = 0
        for segment in result.segments {
            if segment.timeRange.start < previousEnd - 0.1 { // 100ms tolerance
                outOfOrderCount += 1
            }
            previousEnd = segment.timeRange.end
        }
        print("Out-of-order words: \(outOfOrderCount)")
        XCTAssertLessThan(outOfOrderCount, 5, "Should have very few out-of-order words")
        
        print("\n✓ Harry Potter word timing test passed")
    }
    
    /// Test that word segments cover the full audio duration
    func testWordSegmentsCoverage() async throws {
        guard let audioURL = Bundle.module.url(forResource: "speech_long", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("harry_potter.wav not found in Fixtures")
        }
        
        let loader = AudioLoader(config: AudioLoader.Configuration(targetSampleRate: 16000))
        let audio = try loader.loadMono(from: audioURL)
        eval(audio)
        let samples = audio.asArray(Float.self)
        let audioDuration = Double(samples.count) / 16000.0
        
        let transcriber = FluidAudioProviders.parakeetTranscriber(version: .v3)
        try await transcriber.load()
        
        let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        let result = try await transcriber.transcribe(input)
        
        guard !result.segments.isEmpty else {
            XCTFail("No segments extracted")
            return
        }
        
        // Check first and last segment timing
        let firstSegment = result.segments.first!
        let lastSegment = result.segments.last!
        
        // First speech should start within first 10 seconds
        XCTAssertLessThan(firstSegment.timeRange.start, 10.0, "First word should appear within 10s")
        
        // Last speech should end within reasonable time of audio end (allowing for trailing silence)
        // The harry_potter.wav has ~18s of trailing content after last speech at 120s
        XCTAssertGreaterThan(lastSegment.timeRange.end, audioDuration - 25.0, "Last word should be within 25s of audio end")
        
        print("Audio duration: \(String(format: "%.1f", audioDuration))s")
        print("First word at: \(String(format: "%.2f", firstSegment.timeRange.start))s")
        print("Last word ends: \(String(format: "%.2f", lastSegment.timeRange.end))s")
        print("Coverage: \(String(format: "%.1f", (lastSegment.timeRange.end - firstSegment.timeRange.start) / audioDuration * 100))%")
    }
    
    /// Test confidence scores are reasonable
    func testWordConfidenceScores() async throws {
        guard let audioURL = Bundle.module.url(forResource: "speech_long", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("harry_potter.wav not found in Fixtures")
        }
        
        let loader = AudioLoader(config: AudioLoader.Configuration(targetSampleRate: 16000))
        let audio = try loader.loadMono(from: audioURL)
        eval(audio)
        
        let transcriber = FluidAudioProviders.parakeetTranscriber(version: .v3)
        try await transcriber.load()
        
        let input = AudioBuffer(samples: audio.asArray(Float.self), sampleRate: 16000, channels: 1)
        let result = try await transcriber.transcribe(input)
        
        let confidences = result.segments.map(\.confidence)
        let avgConfidence = confidences.reduce(0, +) / Float(max(1, confidences.count))
        let minConfidence = confidences.min() ?? 0
        let maxConfidence = confidences.max() ?? 0
        
        print("Confidence stats:")
        print("  Average: \(String(format: "%.3f", avgConfidence))")
        print("  Min: \(String(format: "%.3f", minConfidence))")
        print("  Max: \(String(format: "%.3f", maxConfidence))")
        
        // Confidence should be in valid range
        XCTAssertGreaterThan(avgConfidence, 0.5, "Average confidence should be > 0.5")
        XCTAssertLessThanOrEqual(maxConfidence, 1.0, "Max confidence should be <= 1.0")
        XCTAssertGreaterThanOrEqual(minConfidence, 0.0, "Min confidence should be >= 0.0")
    }
}
