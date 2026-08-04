//
//  FluidAudioDiarizationTests.swift
//  AudioToolFluidAudioTests
//
//  Tests for pyannote speaker diarization using FluidAudio
//

import XCTest
import AudioToolFluidAudio
import AudioToolCore
import AudioUtils
import MLX

final class FluidAudioDiarizationTests: XCTestCase {
    
    /// Test diarization on watson_30s.wav (interview with 2+ speakers)
    func testDiarizeWatson() async throws {
        print("\n=== Pyannote Diarization Test ===\n")
        
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
        
        // Create diarization provider
        print("\nLoading pyannote models (may download on first run)...")
        let startLoad = Date()
        let diarizer = FluidAudioProviders.pyannote(threshold: 0.6)
        try await diarizer.load()
        let loadTime = Date().timeIntervalSince(startLoad)
        print("  Models loaded in \(String(format: "%.2f", loadTime))s\n")
        
        // Diarize
        print("Running speaker diarization...")
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
        
        print("✓ Diarization test passed")
    }
    
    /// Test quick diarization sample rate
    func testDiarizationSampleRate() async throws {
        let diarizer = FluidAudioProviders.pyannote()
        
        XCTAssertEqual(diarizer.sampleRate, 16000, "Diarization should use 16kHz")
        XCTAssertEqual(diarizer.inputChannels, 1, "Should expect mono input")
    }
}
