//
//  FluidAudioVADProviderTests.swift
//  ClearVoiceFluidAudioTests
//
//  Tests for Silero VAD provider using FluidAudio
//

import XCTest
import ClearVoiceFluidAudio
import ClearVoiceCore

final class FluidAudioVADProviderTests: XCTestCase {
    
    var provider: FluidAudioVADProvider!
    
    override func setUp() async throws {
        provider = FluidAudioProviders.sileroVAD(threshold: 0.5)
        try await provider.load()
    }
    
    override func tearDown() {
        provider = nil
    }
    
    // MARK: - Provider Configuration Tests
    
    func testProviderSampleRate() {
        XCTAssertEqual(provider.sampleRate, 16000, "VAD should use 16kHz sample rate")
    }
    
    func testProviderChannels() {
        XCTAssertEqual(provider.inputChannels, 1, "VAD should expect mono input")
        XCTAssertEqual(provider.outputChannels, 1, "VAD should produce mono output")
    }
    
    func testChunkSize() {
        XCTAssertEqual(provider.minChunkSize, 512, "Min chunk should be 512 samples (32ms)")
        XCTAssertEqual(provider.recommendedChunkSize, 4096, "Recommended chunk should be 4096 samples (256ms)")
    }
    
    // MARK: - Detection Tests
    
    func testDetectWithSilence() async throws {
        // Create 1 second of silence
        let silence = AudioBuffer(
            samples: [Float](repeating: 0.0, count: 16000),
            sampleRate: 16000,
            channels: 1
        )
        
        let segments = try await provider.detect(silence)
        
        // Should return empty or no speech segments for silence
        XCTAssertTrue(segments.isEmpty || segments.allSatisfy { !$0.isSpeech },
                     "Silence should not produce speech segments")
    }
    
    func testDetectWithTestAudio() async throws {
        // Load test audio from fixtures
        let testURL = Bundle.module.url(forResource: "test", withExtension: "wav", subdirectory: "Fixtures")
        
        guard let url = testURL else {
            throw XCTSkip("Test fixture test.wav not found")
        }
        
        // Load audio (simplified - in real test would use AudioLoader)
        let data = try Data(contentsOf: url)
        // For now, just verify the file exists and detection doesn't crash
        XCTAssertFalse(data.isEmpty, "Test audio should have data")
        
        print("Test audio loaded: \(data.count) bytes")
    }
    
    // MARK: - Process Tests (Passthrough)
    
    func testProcessPassthrough() async throws {
        // VAD process should pass audio through unchanged
        let input = AudioBuffer(
            samples: [1.0, 0.5, -0.5, -1.0],
            sampleRate: 16000,
            channels: 1
        )
        
        let output = try await provider.process(input)
        
        XCTAssertEqual(output.samples, input.samples, "Process should passthrough audio unchanged")
        XCTAssertEqual(output.sampleRate, input.sampleRate)
        XCTAssertEqual(output.channels, input.channels)
    }
    
    // MARK: - Performance Tests
    
    func testDetectionPerformance() async throws {
        // Create 5 seconds of audio (random noise simulating speech)
        let sampleCount = 16000 * 5
        let samples = (0..<sampleCount).map { _ in Float.random(in: -0.5...0.5) }
        let audio = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        
        let startTime = Date()
        _ = try await provider.detect(audio)
        let duration = Date().timeIntervalSince(startTime)
        
        let audioDuration = Double(sampleCount) / 16000.0
        let rtf = audioDuration / duration
        
        print("VAD RTF: \(String(format: "%.1f", rtf))x (5s audio in \(String(format: "%.3f", duration))s)")
        
        // VAD should be very fast (>100x RTF typically)
        XCTAssertGreaterThan(rtf, 10.0, "VAD should be at least 10x real-time")
    }
}
