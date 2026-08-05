//
//  TranscriptionIntegrationTests.swift
//  Tests for actual transcription with Apple Speech on test.wav
//

import Foundation
import Testing
@testable import AudioTool
@testable import AudioToolCore
@testable import AudioToolSpeech
import AudioUtils
import MLX

@Suite("Transcription Integration Tests", .enabled(if: TestConfiguration.runIntegrationTests,
        "integration test - set RUN_INTEGRATION_TESTS=1"))
struct TranscriptionIntegrationTests {
    
    // Compute project root from source file path
    static let projectRoot: String = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.path
    }()
    
    @available(iOS 26.0, macOS 26.0, *)
    @Test("Transcribe test.wav with Apple Speech", .timeLimit(.minutes(2)))
    @MainActor
    func testTranscribeTestWav() async throws {
        print("\n=== Apple Speech Transcription Test ===")
        
        // Load the audio file
        let testURL = URL(fileURLWithPath: "\(Self.projectRoot)/Models/mossformer2_se_mlx_swift/test.wav")
        
        try #require(FileManager.default.fileExists(atPath: testURL.path), 
                     "test.wav not found at: \(testURL.path)")
        
        // Load audio at 16kHz
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let mlxArray = try loader.loadMono(from: testURL)
        MLX.eval(mlxArray)
        let samples = mlxArray.asArray(Float.self)
        
        let audioBuffer = AudioBuffer(
            samples: samples,
            sampleRate: 16000,
            channels: 1
        )
        
        let duration = Double(samples.count) / 16000.0
        print("Loaded audio: \(String(format: "%.2f", duration))s at 16kHz (\(samples.count) samples)")
        
        // Check available locales
        let locales = await AppleSpeechTranscriber.supportedLocales()
        print("Supported locales: \(locales.count)")
        
        // Try to find an English locale first, otherwise use any
        let englishLocales = locales.filter { $0.identifier.starts(with: "en") }
        let localeToUse: Locale
        
        if let enLocale = englishLocales.first {
            localeToUse = enLocale
            print("Using English locale: \(enLocale.identifier)")
        } else if let firstLocale = locales.first {
            localeToUse = firstLocale
            print("No English locale, using: \(firstLocale.identifier)")
        } else {
            try #require(!locales.isEmpty, "No speech recognition locales available")
            return
        }
        
        // Create transcriber
        let transcriber = AppleSpeechTranscriber(locale: localeToUse)
        
        print("Loading model...")
        try await transcriber.load()
        print("✓ Model loaded")
        
        print("Transcribing...")
        let startTime = Date()
        let result = try await transcriber.transcribe(audioBuffer)
        let elapsed = Date().timeIntervalSince(startTime)
        
        print("\n========== TRANSCRIPTION RESULT ==========")
        print("Text: \(result.text)")
        print("Language: \(result.language ?? "unknown")")
        print("Segments: \(result.segments.count)")
        print("Time: \(String(format: "%.2f", elapsed))s")
        print("==========================================\n")
        
        #expect(!result.text.isEmpty || result.segments.count > 0)
    }
}
