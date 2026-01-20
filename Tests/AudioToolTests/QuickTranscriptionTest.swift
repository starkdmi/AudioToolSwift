//
//  QuickTranscriptionTest.swift
//  Simple test for Apple Speech transcription
//

import Foundation
import Testing
@testable import ClearVoice
@testable import ClearVoiceCore
@testable import ClearVoiceSpeech

@Suite("Quick Transcription Test")
struct QuickTranscriptionTest {
    
    @available(iOS 26.0, macOS 26.0, *)
    @Test("Check supported locales")
    func testCheckLocales() async {
        print("\n=== Apple Speech Locale Check ===")
        
        let locales = await AppleSpeechTranscriber.supportedLocales()
        print("Supported locales: \(locales.count)")
        
        for locale in locales.prefix(10) {
            print("  - \(locale.identifier)")
        }
        
        #expect(locales.count > 0, "Should have at least one supported locale")
        print("=== Done ===\n")
    }
    
    @available(iOS 26.0, macOS 26.0, *)
    @Test("Quick transcribe silence", .timeLimit(.minutes(1)))
    @MainActor
    func testQuickTranscribe() async throws {
        print("\n=== Quick Apple Speech Test ===")
        
        // Check available locales first
        let locales = await AppleSpeechTranscriber.supportedLocales()
        print("Found \(locales.count) locales")
        
        // Find first locale (any will do for testing)
        try #require(!locales.isEmpty, "No speech recognition locales available")
        let firstLocale = locales.first!
        
        print("Using locale: \(firstLocale.identifier)")
        
        // Create simple test audio (1 second of silence)
        let samples = [Float](repeating: 0.0, count: 16000)
        let audioBuffer = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        
        print("Audio buffer: \(samples.count) samples")
        
        // Create transcriber
        let transcriber = AppleSpeechTranscriber(locale: firstLocale)
        
        print("Loading model...")
        try await transcriber.load()
        print("✓ Model loaded")
        
        print("Transcribing...")
        let result = try await transcriber.transcribe(audioBuffer)
        
        print("\n--- Result ---")
        print("Text: '\(result.text)'")
        print("Language: \(result.language ?? "unknown")")
        print("Segments: \(result.segments.count)")
        print("--- Done ---\n")
        
        // Silence should produce empty or minimal text - validate the pipeline completed
        #expect(result.text.count < 50, "Silence should produce little to no text")
    }
}
