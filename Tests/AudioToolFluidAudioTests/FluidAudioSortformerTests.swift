//
//  FluidAudioSortformerTests.swift
//  AudioToolFluidAudioTests
//
//  Unit tests for Sortformer speaker diarization using FluidAudio
//

import XCTest
import AudioToolTestSupport
import AudioToolFluidAudio
import AudioToolCore
import AudioUtils
import MLX

final class FluidAudioSortformerTests: IntegrationTestCase {
    
    // MARK: - Factory Method Tests
    
    /// Test all Sortformer factory methods create providers with correct configurations
    func testSortformerFactoryMethods() {
        // Default factory
        let defaultProvider = FluidAudioProviders.sortformer()
        XCTAssertEqual(defaultProvider.configuration.chunkLen, 6)
        XCTAssertEqual(defaultProvider.configuration.chunkRightContext, 7)
        
        // Low latency factory (should use .default config)
        let lowLatency = FluidAudioProviders.sortformerLowLatency()
        XCTAssertEqual(lowLatency.configuration.chunkLen, 6)
        XCTAssertEqual(lowLatency.configuration.chunkRightContext, 7)
        
        // High latency factory (should use .nvidiaHighLatency config)
        let highLatency = FluidAudioProviders.sortformerHighLatency()
        XCTAssertEqual(highLatency.configuration.chunkLen, 340)
        XCTAssertEqual(highLatency.configuration.chunkRightContext, 40)
        
        // NVIDIA low latency factory
        let nvidiaLow = FluidAudioProviders.sortformerNVIDIALowLatency()
        XCTAssertEqual(nvidiaLow.configuration.chunkLen, 6)
        XCTAssertEqual(nvidiaLow.configuration.fifoLen, 188, "NVIDIA low latency uses larger FIFO")
    }
    
    /// Test estimatedLatency property calculations
    func testEstimatedLatencyProperty() {
        // Low latency: (6 + 7) * 8 * 160 / 16000 = 1.04s
        let lowLatency = FluidAudioProviders.sortformerLowLatency()
        XCTAssertEqual(lowLatency.estimatedLatency, 1.04, accuracy: 0.01)
        
        // High latency: (340 + 40) * 8 * 160 / 16000 = 30.4s
        let highLatency = FluidAudioProviders.sortformerHighLatency()
        XCTAssertEqual(highLatency.estimatedLatency, 30.4, accuracy: 0.01)
        
        // NVIDIA low latency should also be ~1.04s (same chunk/context as default)
        let nvidiaLow = FluidAudioProviders.sortformerNVIDIALowLatency()
        XCTAssertEqual(nvidiaLow.estimatedLatency, 1.04, accuracy: 0.01)
    }
    
    /// Test configuration property exposes correct values
    func testConfigurationProperty() {
        let provider = FluidAudioProviders.sortformerLowLatency()
        let config = provider.configuration
        
        // Verify key config values are exposed
        XCTAssertEqual(config.numSpeakers, 4)
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.subsamplingFactor, 8)
        XCTAssertEqual(config.melStride, 160)
        XCTAssertEqual(config.melFeatures, 128)
    }
    
    // MARK: - Sample Rate Tests
    
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
    
    /// Test Sortformer diarization on the redistributable two-speaker dialogue.
    func testSortformerDiarizeDialogue() async throws {
        print("\n=== Sortformer Diarization Test ===\n")
        
        // Load test audio
        guard let testURL = Bundle.module.url(forResource: "speech_dialogue", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("speech_dialogue.wav not found")
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
        
        // A count is never negative; `>= 0` asserted that the call returned, which
        // the `try` above already established. The fixture is speech.
        XCTAssertGreaterThan(result.segments.count, 0, "Sortformer found no speech in a speech fixture")
        XCTAssertGreaterThanOrEqual(result.speakerCount, 1, "Sortformer found no speakers")
        
        print("✓ Sortformer URL-based test passed")
    }
}
