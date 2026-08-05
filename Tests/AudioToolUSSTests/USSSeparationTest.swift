//
//  USSSeparationTest.swift
//  AudioToolUSSTests
//
//  Test USS speech and music separation on test.wav
//

import XCTest
import AudioToolUSS
import AudioToolCore
import AudioUtils
import MLX

final class USSSeparationTest: XCTestCase {
    
    // Compute project root from source file path
    static let projectRoot: String = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.path
    }()
    
    /// Test speech and music separation on test.wav
    func testSpeechAndMusicSeparation() async throws {
        print("\n=== USS Speech & Music Separation Test ===")
        
        // Load test audio at 32kHz (USS native rate)
        let testPath = "\(Self.projectRoot)/Models/uss_mlx_swift/test.wav"
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 32000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: URL(fileURLWithPath: testPath))
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let duration = Double(samples.count) / 32000.0
        print("Input: \(String(format: "%.1f", duration))s @ 32kHz (\(samples.count) samples)")
        
        let inputBuffer = AudioBuffer(samples: samples, sampleRate: 32000)
        
        // Test speech separation
        print("\n--- Speech Separation ---")
        let speechProvider = USSProviders.speechSeparation()
        try await speechProvider.load()
        
        let startSpeech = Date()
        let speechOutput = try await speechProvider.process(inputBuffer, type: .speech)
        let speechTime = Date().timeIntervalSince(startSpeech)
        let speechRTF = duration / speechTime
        
        let speechMax = speechOutput.samples.max() ?? 0
        print("Speech: \(String(format: "%.2f", speechTime))s (RTF: \(String(format: "%.1f", speechRTF))x)")
        print("  Output: \(speechOutput.samples.count) samples, max: \(String(format: "%.4f", speechMax))")
        
        // Test music separation
        print("\n--- Music Separation ---")
        let musicProvider = USSProviders.musicSeparation()
        try await musicProvider.load()  // Will reuse cached model but different embedding
        
        let startMusic = Date()
        let musicOutput = try await musicProvider.process(inputBuffer, type: .music)
        let musicTime = Date().timeIntervalSince(startMusic)
        let musicRTF = duration / musicTime
        
        let musicMax = musicOutput.samples.max() ?? 0
        print("Music: \(String(format: "%.2f", musicTime))s (RTF: \(String(format: "%.1f", musicRTF))x)")
        print("  Output: \(musicOutput.samples.count) samples, max: \(String(format: "%.4f", musicMax))")
        
        // Save outputs for listening
        let outputDir = "\(Self.projectRoot)/Models/uss_mlx_swift"
        try AudioSaver.saveWAV(MLXArray(speechOutput.samples), to: "\(outputDir)/test_speech.wav", sampleRate: 32000)
        try AudioSaver.saveWAV(MLXArray(musicOutput.samples), to: "\(outputDir)/test_music.wav", sampleRate: 32000)
        print("\n✓ Saved: test_speech.wav, test_music.wav")
        
        // Verify outputs have content
        XCTAssertGreaterThan(speechMax, 0.01, "Speech output should not be silent")
        XCTAssertGreaterThan(musicMax, 0.01, "Music output should not be silent")
    }
}
