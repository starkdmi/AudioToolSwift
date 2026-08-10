//
//  TranscriptionTruncationTests.swift
//  AudioToolFluidAudioTests
//
//  Regression tests for transcription truncation bug (mel context issue)
//

import XCTest
import AudioToolTestSupport
import AudioToolFluidAudio
import AudioToolCore
import AudioUtils
import MLX

final class TranscriptionTruncationTests: IntegrationTestCase {
    
    /// Regression test for transcription truncation at chunk boundaries
    ///
    /// Bug: Audio at 118-121s in harry_potter.wav was truncated because
    /// chunk 9 started exactly at a problematic sample boundary, causing
    /// the TDT decoder to predict all blanks due to missing mel context.
    ///
    /// Root cause: The FastConformer encoder's depthwise convolutions need
    /// left context from the previous chunk. Without it, the first mel frames
    /// produce unstable features that the decoder interprets as silence.
    ///
    /// Fix: ChunkProcessor now prepends 1280 samples (80ms = 1 encoder frame)
    /// of context from the previous chunk to provide proper mel spectrogram
    /// left context for subsequent chunks.
    func testHarryPotterTranscriptionNotTruncated() async throws {
        // This exact-content regression remains in the optional private pool;
        // standalone/public clones skip it cleanly.
        let testURL = try reference("Docs/harry_potter.wav")
        
        print("\n=== Transcription Truncation Regression Test ===\n")
        
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: testURL)
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let duration = Double(samples.count) / 16000.0
        print("Audio duration: \(String(format: "%.1f", duration))s")
        XCTAssertGreaterThan(duration, 130, "Audio should be ~136 seconds")
        
        // Create and load transcriber
        print("Loading Parakeet v3 model...")
        let transcriber = FluidAudioProviders.parakeetTranscriber(version: .v3)
        try await transcriber.load()
        print("Model loaded")
        
        // Transcribe
        print("Transcribing...")
        let startTime = Date()
        let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        let result = try await transcriber.transcribe(input)
        let transcribeTime = Date().timeIntervalSince(startTime)
        let rtf = duration / transcribeTime
        
        print("Transcription completed in \(String(format: "%.2f", transcribeTime))s (RTF: \(String(format: "%.0f", rtf))x)")
        print("\nFull transcription:")
        print(result.text)
        print()
        
        // CRITICAL ASSERTION: The phrase "expect great things" at ~118-121s must be present
        // This was the content that was truncated before the fix
        let lowercaseText = result.text.lowercased()
        let hasExpectGreatThings = lowercaseText.contains("expect great things") ||
                                   lowercaseText.contains("expect great thing")
        
        if !hasExpectGreatThings {
            print("ERROR: Missing 'expect great things' phrase!")
            print("This indicates the chunk boundary truncation bug has regressed.")
            print("The phrase should appear around 118-121 seconds in the audio.")
        }
        
        XCTAssertTrue(
            hasExpectGreatThings,
            "Transcription must include 'expect great things' (Dumbledore's line at 118-121s). " +
            "If missing, the chunk boundary truncation bug has regressed. " +
            "Check ChunkProcessor.melContextSamples implementation."
        )
        
        // Also verify we have content near the end of the audio
        // Check for segments past the 115 second mark
        let lateSegments = result.segments.filter { segment in
            segment.timeRange.start > 115.0
        }
        
        print("Segments after 115s: \(lateSegments.count)")
        for segment in lateSegments.prefix(10) {
            print("  [\(String(format: "%.1f", segment.timeRange.start))-\(String(format: "%.1f", segment.timeRange.end))] \(segment.text)")
        }
        
        XCTAssertFalse(lateSegments.isEmpty, "Should have transcribed content past 115 seconds")
        
        print("\n✓ Transcription truncation regression test passed")
    }
    
    /// Test that short audio (< 15s, single chunk) still works correctly
    /// This verifies the mel context logic doesn't break the first chunk
    func testShortAudioTranscription() async throws {
        // Use the redistributable dialogue fixture if available, or skip.
        guard let testURL = Bundle.module.url(forResource: "speech_dialogue", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("speech_dialogue.wav fixture not found")
        }
        
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: testURL)
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        // Take just the first 10 seconds (single chunk, no context needed)
        let shortSamples = Array(samples.prefix(160000))  // 10s at 16kHz
        
        let transcriber = FluidAudioProviders.parakeetTranscriber(version: .v3)
        try await transcriber.load()
        
        let input = AudioBuffer(samples: shortSamples, sampleRate: 16000, channels: 1)
        let result = try await transcriber.transcribe(input)
        
        XCTAssertFalse(result.text.isEmpty, "Short audio should produce transcription")
        print("Short audio transcription: \(result.text.prefix(100))...")
    }
}
