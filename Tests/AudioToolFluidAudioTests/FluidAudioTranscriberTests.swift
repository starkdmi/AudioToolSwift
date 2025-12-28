//
//  FluidAudioTranscriberTests.swift
//  ClearVoiceFluidAudioTests
//
//  Tests for Parakeet transcription using FluidAudio
//

import XCTest
import ClearVoiceFluidAudio
import ClearVoiceCore
import AudioUtils
import MLX

final class FluidAudioTranscriberTests: XCTestCase {
    
    /// Test transcription on watson_30s.wav
    func testTranscribeWatson() async throws {
        print("\n=== Parakeet v3 Transcription Test ===\n")
        
        // Load test audio
        guard let testURL = Bundle.module.url(forResource: "watson_30s", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("watson_30s.wav not found")
        }
        
        // Load at 16kHz for ASR
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: testURL)
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let duration = Double(samples.count) / 16000.0
        print("Audio: \(String(format: "%.1f", duration))s at 16kHz")
        
        // Create transcriber
        print("Loading Parakeet v3 model (may download on first run)...")
        let startLoad = Date()
        let transcriber = FluidAudioProviders.parakeetTranscriber(version: .v3)
        try await transcriber.load()
        let loadTime = Date().timeIntervalSince(startLoad)
        print("  Model loaded in \(String(format: "%.2f", loadTime))s\n")
        
        // Transcribe
        print("Transcribing...")
        let startTranscribe = Date()
        
        let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        let result = try await transcriber.transcribe(input)
        
        let transcribeTime = Date().timeIntervalSince(startTranscribe)
        let rtf = duration / transcribeTime
        
        print("Transcription completed in \(String(format: "%.2f", transcribeTime))s (RTF: \(String(format: "%.0f", rtf))x)")
        print("\n--- Transcription ---")
        print(result.text)
        print("--- End ---\n")
        
        // Verify
        XCTAssertFalse(result.text.isEmpty, "Transcription should not be empty")
        XCTAssertGreaterThan(result.text.count, 50, "Transcription should have substantial text")
        XCTAssertGreaterThan(rtf, 5, "Transcription should be at least 5x real-time")
        
        print("✓ Transcription test passed")
    }
    
    /// Test transcription performance
    func testTranscriptionPerformance() async throws {
        guard let testURL = Bundle.module.url(forResource: "test", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("test.wav not found")
        }
        
        let loader = AudioLoader(config: AudioLoader.Configuration(targetSampleRate: 16000))
        let audio = try loader.loadMono(from: testURL)
        eval(audio)
        
        let transcriber = FluidAudioProviders.parakeetTranscriber()
        try await transcriber.load()
        
        let input = AudioBuffer(samples: audio.asArray(Float.self), sampleRate: 16000, channels: 1)
        
        let startTime = Date()
        let result = try await transcriber.transcribe(input)
        let duration = Date().timeIntervalSince(startTime)
        
        let audioDuration = Double(input.samples.count) / 16000.0
        let rtf = audioDuration / duration
        
        print("RTF: \(String(format: "%.0f", rtf))x (\(String(format: "%.1f", audioDuration))s audio in \(String(format: "%.2f", duration))s)")
        print("Text: \(result.text.prefix(100))...")
        
        XCTAssertGreaterThan(rtf, 5, "Should be at least 5x real-time")
    }
}
