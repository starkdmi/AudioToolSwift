//
//  VADThresholdTests.swift
//  ClearVoiceFluidAudioTests
//
//  Tests for VAD threshold tuning on different media types
//

import XCTest
import ClearVoiceFluidAudio
import ClearVoiceCore
import AudioUtils
import MLX

final class VADThresholdTests: XCTestCase {
    
    /// Test music.wav (35s trim from 25-60s) with different thresholds and settings
    func testMusicWithVADSettings() async throws {
        print("\n=== VAD Test: music_35s.wav ===")
        print("Testing speech detection in music with singing\n")
        
        guard let testURL = Bundle.module.url(forResource: "music_35s", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("music_35s.wav not found - run: ffmpeg -i music.mp3 -ss 25 -to 60 music_35s.wav")
        }
        
        try await runVADSettingsTest(url: testURL)
    }
    
    /// Test billions.wav with different thresholds and duration settings
    func testBillionsWithVADSettings() async throws {
        print("\n=== VAD Test: billions.wav ===")
        print("Testing speech detection in music/singing with different settings\n")
        
        guard let testURL = Bundle.module.url(forResource: "billions", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("billions.wav not found")
        }
        
        try await runVADSettingsTest(url: testURL)
    }
    
    private func runVADSettingsTest(url: URL) async throws {
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
        
        // Check audio levels
        let maxLevel = samples.map { abs($0) }.max() ?? 0
        let rms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))
        print("Audio levels: max=\(String(format: "%.4f", maxLevel)), RMS=\(String(format: "%.4f", rms))\n")
        
        let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        
        // Test configurations: (threshold, minSpeech, minSilence, description)
        let configs: [(Float, Double, Double, String)] = [
            (0.5, 0.25, 0.4, "Default"),
            (0.3, 0.25, 0.4, "Lower threshold"),
            (0.1, 0.25, 0.4, "Very low threshold"),
            (0.1, 0.1, 0.2, "Very low + short durations (music mode)"),
            (0.1, 0.5, 0.8, "Very low + long durations"),
        ]
        
        for (threshold, minSpeech, minSilence, desc) in configs {
            let vad = FluidAudioProviders.sileroVAD(
                threshold: threshold,
                minSpeechDuration: minSpeech,
                minSilenceDuration: minSilence
            )
            try await vad.load()
            
            let segments = try await vad.detect(input)
            let totalSpeech = segments.reduce(0.0) { $0 + ($1.timeRange.end - $1.timeRange.start) }
            
            print("[\(desc)]")
            print("  threshold=\(threshold), minSpeech=\(minSpeech)s, minSilence=\(minSilence)s")
            print("  Detected \(segments.count) segments, total \(String(format: "%.1f", totalSpeech))s (\(String(format: "%.0f", totalSpeech/duration*100))%)")
            
            for (i, seg) in segments.prefix(3).enumerated() {
                print("    [\(i+1)] \(String(format: "%.2f", seg.timeRange.start))s - \(String(format: "%.2f", seg.timeRange.end))s")
            }
            if segments.count > 3 {
                print("    ... and \(segments.count - 3) more")
            }
            print()
        }
    }
}
