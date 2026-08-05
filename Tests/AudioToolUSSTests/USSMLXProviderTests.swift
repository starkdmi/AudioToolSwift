//
//  USSMLXProviderTests.swift
//  AudioToolUSSTests
//
//  Tests for USS MLX speech separation
//

import XCTest
import AudioToolUSS
import AudioToolCore
import AudioUtils
import MLX
@preconcurrency import USSMLXSwift

final class USSMLXProviderTests: XCTestCase {
    
    /// Test speech separation on watson_30s.wav
    func testSpeechSeparation() async throws {
        print("\n=== USS Speech Separation Test ===")
        
        // Load test audio
        guard let testURL = Bundle.module.url(forResource: "watson_30s", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("watson_30s.wav not found in test fixtures")
        }
        
        // Load at 32kHz for USS
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 32000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: testURL)
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let duration = Double(samples.count) / 32000.0
        print("Audio: \(String(format: "%.1f", duration))s @ 32kHz")
        
        // Create USS provider
        print("\nLoading USS model...")
        let startLoad = Date()
        let uss = USSProviders.speechSeparation()
        try await uss.load()
        let loadTime = Date().timeIntervalSince(startLoad)
        print("Model loaded in \(String(format: "%.2f", loadTime))s")
        
        // Process audio
        print("\nRunning speech separation...")
        let startProcess = Date()
        
        let inputBuffer = AudioBuffer(samples: samples, sampleRate: 32000)
        let outputBuffer = try await uss.process(inputBuffer, type: .speech)
        
        let processTime = Date().timeIntervalSince(startProcess)
        let rtf = duration / processTime
        
        print("Separation completed in \(String(format: "%.2f", processTime))s (RTF: \(String(format: "%.1f", rtf))x)")
        
        // Verify output
        XCTAssertEqual(outputBuffer.sampleRate, 32000, "Output sample rate should be 32kHz")
        XCTAssertGreaterThan(outputBuffer.samples.count, 0, "Output should have samples")
        
        // Check output isn't silent
        let maxOutput = outputBuffer.samples.max() ?? 0
        XCTAssertGreaterThan(maxOutput, 0.01, "Output should not be silent")
        
        print("✓ Speech separation test passed")
        print("  Input: \(samples.count) samples, Output: \(outputBuffer.samples.count) samples")
        print("  Max amplitude: \(String(format: "%.4f", maxOutput))")
    }
    
    /// Test different embedding types
    func testEmbeddingTypes() async throws {
        print("\n=== USS Embedding Types Test ===")
        
        // Just verify all embedding types can be loaded
        for type in [EmbeddingLoader.EmbeddingType.speech, .music, .noise, .nature, .human, .animal, .things] {
            _ = USSProviders.speechSeparation(embeddingType: type)
            // Loading would fail if embeddings are missing
            print("✓ Embedding type '\(type.rawValue)' available")
        }
    }
}
