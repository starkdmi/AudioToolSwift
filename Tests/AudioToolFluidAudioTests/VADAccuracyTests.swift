//
//  VADAccuracyTests.swift
//  ClearVoiceFluidAudioTests
//
//  Tests for VAD detection accuracy on various audio files
//

import XCTest
import ClearVoiceFluidAudio
import ClearVoiceCore
import AudioUtils
import MLX

final class VADAccuracyTests: XCTestCase {
    
    var vad: FluidAudioVADProvider!
    
    override func setUp() async throws {
        vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        try await vad.load()
    }
    
    /// Test VAD on billions.wav - speech should be in the last half
    func testVADBillions() async throws {
        print("\n=== VAD Test: billions.wav ===")
        print("Expected: Speech in the last half of audio")
        
        guard let testURL = Bundle.module.url(forResource: "billions", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("billions.wav not found")
        }
        
        try await runVADTest(url: testURL)
    }
    
    /// Test VAD on watson_30s.wav - interview with multiple speech segments
    func testVADWatson() async throws {
        print("\n=== VAD Test: watson_30s.wav ===")
        print("Expected: Multiple speech segments from interview")
        
        guard let testURL = Bundle.module.url(forResource: "watson_30s", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("watson_30s.wav not found")
        }
        
        try await runVADTest(url: testURL)
    }
    
    /// Test VAD on original test.wav
    func testVADTestWav() async throws {
        print("\n=== VAD Test: test.wav ===")
        print("Expected: Speech starting around 1s (not 0s)")
        
        guard let testURL = Bundle.module.url(forResource: "test", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("test.wav not found")
        }
        
        try await runVADTest(url: testURL)
    }
    
    private func runVADTest(url: URL) async throws {
        // Load at 16kHz
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: url)
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let duration = Double(samples.count) / 16000.0
        print("Audio duration: \(String(format: "%.2f", duration))s")
        
        let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        let segments = try await vad.detect(input)
        
        print("\nDetected \(segments.count) speech segments:")
        
        var totalSpeech = 0.0
        for (i, segment) in segments.enumerated() {
            let segDuration = segment.timeRange.end - segment.timeRange.start
            totalSpeech += segDuration
            print("  [\(i+1)] \(String(format: "%6.2f", segment.timeRange.start))s - \(String(format: "%6.2f", segment.timeRange.end))s  (\(String(format: "%.1f", segDuration))s)")
        }
        
        let speechPercent = (totalSpeech / duration) * 100
        print("\nTotal speech: \(String(format: "%.2f", totalSpeech))s / \(String(format: "%.2f", duration))s (\(String(format: "%.0f", speechPercent))%)")
        
        // Check for non-zero start times
        if let first = segments.first {
            if first.timeRange.start < 0.3 {
                print("⚠️ First segment starts very early (\(String(format: "%.2f", first.timeRange.start))s) - verify if correct")
            } else {
                print("✓ First segment starts at \(String(format: "%.2f", first.timeRange.start))s")
            }
        }
    }
}
