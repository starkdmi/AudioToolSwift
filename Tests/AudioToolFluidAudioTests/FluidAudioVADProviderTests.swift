//
//  FluidAudioVADProviderTests.swift
//  ClearVoiceFluidAudioTests
//
//  Tests for Silero VAD provider using FluidAudio
//

import XCTest
import ClearVoiceFluidAudio
import ClearVoiceCore
import AudioUtils
import MLX

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
        guard let testURL = Bundle.module.url(forResource: "test", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("Test fixture test.wav not found")
        }
        
        // Load and process actual audio
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: testURL)
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let audioBuffer = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        
        // Run VAD detection
        let segments = try await provider.detect(audioBuffer)
        
        // Real assertions for test audio
        XCTAssertFalse(segments.isEmpty, "Test audio should contain speech segments")
        
        // Verify segment properties
        for segment in segments {
            XCTAssertGreaterThanOrEqual(segment.timeRange.start, 0, "Segment start should be >= 0")
            XCTAssertGreaterThan(segment.timeRange.end, segment.timeRange.start, "Segment end should be > start")
            XCTAssertLessThanOrEqual(segment.timeRange.end, audioBuffer.duration, "Segment end should be <= audio duration")
            XCTAssertGreaterThanOrEqual(segment.probability, 0, "Probability should be >= 0")
            XCTAssertLessThanOrEqual(segment.probability, 1, "Probability should be <= 1")
        }
        
        // Calculate total speech
        let totalSpeech = segments.reduce(0.0) { $0 + ($1.timeRange.end - $1.timeRange.start) }
        print("Detected \(segments.count) segments, total speech: \(String(format: "%.2f", totalSpeech))s")
        
        XCTAssertGreaterThan(totalSpeech, 0.5, "Should detect at least 0.5s of speech")
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
        // Create 5 seconds of deterministic pseudo-speech audio
        let sampleCount = 16000 * 5
        var samples = [Float](repeating: 0, count: sampleCount)
        
        // Use deterministic pattern instead of random
        for i in 0..<sampleCount {
            // Deterministic pseudo-speech pattern
            let t = Float(i) / 16000.0
            let base = sin(2.0 * .pi * 200.0 * t)  // 200 Hz fundamental
            let harmonics = sin(2.0 * .pi * 400.0 * t) * 0.5 + sin(2.0 * .pi * 800.0 * t) * 0.25
            let envelope = sin(2.0 * .pi * 4.0 * t) * 0.5 + 0.5  // Slow modulation
            samples[i] = (base + harmonics) * 0.3 * envelope
        }
        
        let audio = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        
        let startTime = Date()
        _ = try await provider.detect(audio)
        let duration = Date().timeIntervalSince(startTime)
        
        let audioDuration = Double(sampleCount) / 16000.0
        let rtf = audioDuration / duration
        
        print("VAD RTF: \(String(format: "%.1f", rtf))x (5s audio in \(String(format: "%.3f", duration))s)")
        
        // Use CI-aware threshold - VAD should be fast but CI may be slower
        let isCI = ProcessInfo.processInfo.environment["CI"] == "1"
        let threshold = isCI ? 3.0 : 10.0
        XCTAssertGreaterThan(rtf, threshold, "VAD should be at least \(threshold)x real-time")
    }
}
