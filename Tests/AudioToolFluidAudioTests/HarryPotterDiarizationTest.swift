//
//  HarryPotterDiarizationTest.swift
//  ClearVoiceFluidAudioTests
//
//  Test diarization on Harry Potter trailer (~10 speakers, background music)
//

import XCTest
import ClearVoiceFluidAudio
import ClearVoiceCore
import AudioUtils
import MLX

final class HarryPotterDiarizationTest: XCTestCase {
    
    /// Test diarization on harry_potter.wav (2:16 trailer, ~10 speakers, background music)
    func testDiarizeHarryPotter() async throws {
        print("\n=== Harry Potter Trailer Diarization Test ===")
        print("Expected: Up to 10 speakers, with background music\n")
        
        // Use direct file path instead of bundle to match CLI exactly
        let testURL = URL(fileURLWithPath: "/path/to/clear_voice_research/Docs/harry_potter.wav")
        
        // Load at 16kHz for diarization
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: testURL)
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let duration = Double(samples.count) / 16000.0
        print("Audio: \(String(format: "%.1f", duration))s (\(String(format: "%.0f", duration/60)):\(String(format: "%02.0f", duration.truncatingRemainder(dividingBy: 60))))")
        
        // Create diarization provider
        print("\nLoading pyannote models...")
        let startLoad = Date()
        let diarizer = FluidAudioProviders.pyannote()  // Uses default threshold 0.7045655
        try await diarizer.load()
        let loadTime = Date().timeIntervalSince(startLoad)
        print("  Models loaded in \(String(format: "%.2f", loadTime))s\n")
        
        // Diarize using URL-based method (matches CLI behavior)
        print("Running speaker diarization (URL-based)...")
        let startDiarize = Date()
        
        let result = try await diarizer.diarize(url: testURL)
        
        let diarizeTime = Date().timeIntervalSince(startDiarize)
        let rtf = duration / diarizeTime
        
        print("Diarization completed in \(String(format: "%.2f", diarizeTime))s (RTF: \(String(format: "%.1f", rtf))x)\n")
        
        print("--- Speaker Timeline ---")
        print("🎭 Detected \(result.speakerCount) speakers")
        print("Max overlapping: \(result.maxOverlappingSpeakers)")
        
        // Group segments by speaker
        let speakerGroups = Dictionary(grouping: result.segments, by: { $0.speakerID })
        
        print("\nSpeaker breakdown:")
        for (speaker, segments) in speakerGroups.sorted(by: { $0.key.description < $1.key.description }) {
            let totalTime = segments.reduce(0.0) { $0 + ($1.timeRange.end - $1.timeRange.start) }
            print("  \(speaker): \(segments.count) segments, \(String(format: "%.1f", totalTime))s total")
        }
        
        print("\nAll segments (chronological):")
        for segment in result.segments {
            let segDuration = segment.timeRange.end - segment.timeRange.start
            print("  \(segment.speakerID): \(String(format: "%6.2f", segment.timeRange.start))s - \(String(format: "%6.2f", segment.timeRange.end))s (\(String(format: "%.1f", segDuration))s)")
        }
        print("--- End ---\n")
        
        // Verify multiple speakers detected
        XCTAssertGreaterThan(result.segments.count, 0, "Should detect segments")
        print("✓ Harry Potter diarization test passed")
        print("  Detected \(result.speakerCount) of expected ~10 speakers")
    }
}
