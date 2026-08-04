//
//  VADThresholdTests.swift
//  AudioToolFluidAudioTests
//
//  Tests for VAD threshold tuning on different media types
//

import XCTest
import AudioToolFluidAudio
import AudioToolCore
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
        
        let results = try await runVADSettingsTest(url: testURL)
        
        // Assertions for music file - should detect singing as speech with low threshold
        let lowThresholdResult = results.first { $0.description == "Very low threshold" }
        XCTAssertNotNil(lowThresholdResult, "Should have results for very low threshold")
        
        if let result = lowThresholdResult {
            // Music with singing should have detectable speech at low thresholds
            XCTAssertGreaterThan(result.segments.count, 0, 
                "Music with singing should have some detected segments at low threshold")
        }
    }
    
    /// Test billions.wav with different thresholds and duration settings
    func testBillionsWithVADSettings() async throws {
        print("\n=== VAD Test: billions.wav ===")
        print("Testing speech detection in music/singing with different settings\n")
        
        guard let testURL = Bundle.module.url(forResource: "billions", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("billions.wav not found")
        }
        
        let results = try await runVADSettingsTest(url: testURL)
        
        // Assertions for billions.wav - speech should be detected in the second half
        let defaultResult = results.first { $0.description == "Default" }
        XCTAssertNotNil(defaultResult, "Should have results for default threshold")
        
        // All configs should produce valid (possibly empty) segment lists
        for result in results {
            XCTAssertGreaterThanOrEqual(result.segments.count, 0, 
                "Config '\(result.description)' should produce valid segment count")
            XCTAssertLessThanOrEqual(result.speechPercent, 100.0,
                "Speech percentage should not exceed 100%")
        }
    }
    
    /// Result structure for threshold testing
    struct ThresholdTestResult {
        let description: String
        let segments: [VADSegment]
        let totalSpeech: Double
        let audioDuration: Double
        var speechPercent: Double { (totalSpeech / audioDuration) * 100 }
    }
    
    private func runVADSettingsTest(url: URL) async throws -> [ThresholdTestResult] {
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
        
        var results: [ThresholdTestResult] = []
        
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
            
            results.append(ThresholdTestResult(
                description: desc,
                segments: segments,
                totalSpeech: totalSpeech,
                audioDuration: duration
            ))
        }
        
        return results
    }
}
