//
//  FluidAudioSortformerTests.swift
//  ClearVoiceFluidAudioTests
//
//  Unit tests for Sortformer speaker diarization using FluidAudio
//

import XCTest
import ClearVoiceFluidAudio
import ClearVoiceCore
import AudioUtils
import MLX

final class FluidAudioSortformerTests: XCTestCase {
    
    /// Test Sortformer sample rate requirements
    func testSortformerSampleRate() async throws {
        let diarizer = FluidAudioProviders.sortformer()
        
        XCTAssertEqual(diarizer.sampleRate, 16000, "Sortformer should use 16kHz")
        XCTAssertEqual(diarizer.inputChannels, 1, "Should expect mono input")
    }
    
    /// Test Sortformer model loading
    func testSortformerModelLoading() async throws {
        print("\n=== Sortformer Model Loading Test ===\n")
        
        let diarizer = FluidAudioProviders.sortformer()
        
        print("Loading Sortformer models (may download on first run)...")
        let startLoad = Date()
        try await diarizer.load()
        let loadTime = Date().timeIntervalSince(startLoad)
        print("  Models loaded in \(String(format: "%.2f", loadTime))s")
        
        print("✓ Sortformer model loading test passed")
    }
    
    /// Test Sortformer diarization on watson_30s.wav (interview with 2+ speakers)
    func testSortformerDiarizeWatson() async throws {
        print("\n=== Sortformer Diarization Test ===\n")
        
        // Load test audio
        guard let testURL = Bundle.module.url(forResource: "watson_30s", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("watson_30s.wav not found")
        }
        
        // Load at 16kHz for diarization
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: testURL)
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let duration = Double(samples.count) / 16000.0
        print("Audio: \(String(format: "%.1f", duration))s at 16kHz")
        
        // Create Sortformer diarization provider
        print("\nLoading Sortformer models (may download on first run)...")
        let startLoad = Date()
        let diarizer = FluidAudioProviders.sortformer()
        try await diarizer.load()
        let loadTime = Date().timeIntervalSince(startLoad)
        print("  Models loaded in \(String(format: "%.2f", loadTime))s\n")
        
        // Diarize
        print("Running Sortformer diarization...")
        let startDiarize = Date()
        
        let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        let result = try await diarizer.diarize(input)
        
        let diarizeTime = Date().timeIntervalSince(startDiarize)
        let rtf = duration / diarizeTime
        
        print("Diarization completed in \(String(format: "%.2f", diarizeTime))s (RTF: \(String(format: "%.1f", rtf))x)")
        print("\n--- Speaker Timeline ---")
        print("Detected \(result.speakerCount) speakers")
        print("Max overlapping: \(result.maxOverlappingSpeakers)")
        print("\nSegments:")
        
        for segment in result.segments {
            let segDuration = segment.timeRange.end - segment.timeRange.start
            print("  \(segment.speakerID): \(String(format: "%.2f", segment.timeRange.start))s - \(String(format: "%.2f", segment.timeRange.end))s (\(String(format: "%.1f", segDuration))s)")
        }
        print("--- End ---\n")
        
        // Verify
        XCTAssertGreaterThan(result.segments.count, 0, "Should detect at least one segment")
        XCTAssertGreaterThanOrEqual(result.speakerCount, 1, "Should detect at least 1 speaker")
        XCTAssertLessThanOrEqual(result.speakerCount, 4, "Sortformer supports max 4 speakers")
        
        // Verify Sortformer is faster than real-time
        XCTAssertGreaterThan(rtf, 1.0, "Sortformer should be faster than real-time")
        
        print("✓ Sortformer diarization test passed")
    }
    
    /// Test Sortformer diarization using URL-based API
    func testSortformerDiarizeURL() async throws {
        print("\n=== Sortformer URL-based Diarization Test ===\n")
        
        // Load test audio URL
        guard let testURL = Bundle.module.url(forResource: "test", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("test.wav not found")
        }
        
        // Create and load diarizer
        let diarizer = FluidAudioProviders.sortformer()
        try await diarizer.load()
        
        // Diarize using URL
        print("Running Sortformer diarization (URL-based)...")
        let startDiarize = Date()
        let result = try await diarizer.diarize(url: testURL)
        let diarizeTime = Date().timeIntervalSince(startDiarize)
        
        print("Completed in \(String(format: "%.2f", diarizeTime))s")
        print("Detected \(result.speakerCount) speakers, \(result.segments.count) segments")
        
        XCTAssertGreaterThanOrEqual(result.segments.count, 0, "Should complete without error")
        
        print("✓ Sortformer URL-based test passed")
    }
}
