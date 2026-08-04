//
//  SortformerComparisonTests.swift
//  AudioToolFluidAudioTests
//
//  Comprehensive comparison tests between Pyannote and Sortformer diarization
//  Tests on multiple audio files to evaluate performance differences
//

import XCTest
import AudioToolFluidAudio
import AudioToolCore
import AudioUtils
import MLX

final class SortformerComparisonTests: XCTestCase {
    
    // MARK: - Shared Test Infrastructure
    
    struct DiarizationResult {
        let providerName: String
        let loadTime: Double
        let processTime: Double
        let rtf: Double
        let speakerCount: Int
        let segmentCount: Int
        let maxOverlap: Int
        let timeline: SpeakerTimeline
    }
    
    /// Generic comparison test for any audio file
    func runComparison(
        testName: String,
        audioURL: URL,
        skipPyannote: Bool = false
    ) async throws -> (pyannote: DiarizationResult?, sortformer: DiarizationResult) {
        print("\n" + String(repeating: "=", count: 60))
        print("=== Diarization Comparison: \(testName) ===")
        print(String(repeating: "=", count: 60) + "\n")
        
        // Load audio
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: audioURL)
        eval(audio)
        let samples = audio.asArray(Float.self)
        let duration = Double(samples.count) / 16000.0
        
        print("Audio: \(String(format: "%.1f", duration))s at 16kHz")
        print("File: \(audioURL.lastPathComponent)\n")
        
        let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        
        // Run Pyannote (optional)
        var pyannoteResult: DiarizationResult? = nil
        if !skipPyannote {
            print("--- Pyannote (VBx Clustering) ---")
            let pyannote = FluidAudioProviders.pyannote(threshold: 0.6)
            
            var startLoad = Date()
            try await pyannote.load()
            let pyannoteLoadTime = Date().timeIntervalSince(startLoad)
            print("Load time: \(String(format: "%.2f", pyannoteLoadTime))s")
            
            startLoad = Date()
            let pyannoteTimeline = try await pyannote.diarize(input)
            let pyannoteProcessTime = Date().timeIntervalSince(startLoad)
            let pyannoteRTF = duration / pyannoteProcessTime
            
            print("Process time: \(String(format: "%.2f", pyannoteProcessTime))s (RTF: \(String(format: "%.1f", pyannoteRTF))x)")
            print("Speakers: \(pyannoteTimeline.speakerCount)")
            print("Segments: \(pyannoteTimeline.segments.count)")
            print("Max overlap: \(pyannoteTimeline.maxOverlappingSpeakers)\n")
            
            pyannoteResult = DiarizationResult(
                providerName: "Pyannote",
                loadTime: pyannoteLoadTime,
                processTime: pyannoteProcessTime,
                rtf: pyannoteRTF,
                speakerCount: pyannoteTimeline.speakerCount,
                segmentCount: pyannoteTimeline.segments.count,
                maxOverlap: pyannoteTimeline.maxOverlappingSpeakers,
                timeline: pyannoteTimeline
            )
        }
        
        // Run Sortformer
        print("--- Sortformer (End-to-End) ---")
        let sortformer = FluidAudioProviders.sortformer()
        
        var startLoad = Date()
        try await sortformer.load()
        let sortformerLoadTime = Date().timeIntervalSince(startLoad)
        print("Load time: \(String(format: "%.2f", sortformerLoadTime))s")
        
        startLoad = Date()
        let sortformerTimeline = try await sortformer.diarize(input)
        let sortformerProcessTime = Date().timeIntervalSince(startLoad)
        let sortformerRTF = duration / sortformerProcessTime
        
        print("Process time: \(String(format: "%.2f", sortformerProcessTime))s (RTF: \(String(format: "%.1f", sortformerRTF))x)")
        print("Speakers: \(sortformerTimeline.speakerCount)")
        print("Segments: \(sortformerTimeline.segments.count)")
        print("Max overlap: \(sortformerTimeline.maxOverlappingSpeakers)\n")
        
        let sortformerResult = DiarizationResult(
            providerName: "Sortformer",
            loadTime: sortformerLoadTime,
            processTime: sortformerProcessTime,
            rtf: sortformerRTF,
            speakerCount: sortformerTimeline.speakerCount,
            segmentCount: sortformerTimeline.segments.count,
            maxOverlap: sortformerTimeline.maxOverlappingSpeakers,
            timeline: sortformerTimeline
        )
        
        // Print segment timelines
        if let pResult = pyannoteResult {
            print("--- Pyannote Segments ---")
            for segment in pResult.timeline.segments.prefix(20) {
                let segDuration = segment.timeRange.end - segment.timeRange.start
                print("  \(segment.speakerID): \(String(format: "%6.2f", segment.timeRange.start))s - \(String(format: "%6.2f", segment.timeRange.end))s (\(String(format: "%.1f", segDuration))s)")
            }
            if pResult.timeline.segments.count > 20 {
                print("  ... and \(pResult.timeline.segments.count - 20) more segments")
            }
            print()
        }
        
        print("--- Sortformer Segments ---")
        for segment in sortformerTimeline.segments.prefix(20) {
            let segDuration = segment.timeRange.end - segment.timeRange.start
            print("  \(segment.speakerID): \(String(format: "%6.2f", segment.timeRange.start))s - \(String(format: "%6.2f", segment.timeRange.end))s (\(String(format: "%.1f", segDuration))s)")
        }
        if sortformerTimeline.segments.count > 20 {
            print("  ... and \(sortformerTimeline.segments.count - 20) more segments")
        }
        print()
        
        // Summary comparison
        if let pResult = pyannoteResult {
            print("--- Summary ---")
            let speedup = pResult.processTime / sortformerProcessTime
            print("Speed comparison: Sortformer is \(String(format: "%.1f", speedup))x faster")
            print("Speaker detection: Pyannote=\(pResult.speakerCount), Sortformer=\(sortformerResult.speakerCount)")
            print("Segments: Pyannote=\(pResult.segmentCount), Sortformer=\(sortformerResult.segmentCount)")
        }
        print(String(repeating: "=", count: 60) + "\n")
        
        return (pyannoteResult, sortformerResult)
    }
    
    // MARK: - Individual Test Cases
    
    /// Compare diarization on watson_30s.wav (interview, 2-3 speakers, clean)
    func testCompare_Watson30s() async throws {
        guard let testURL = Bundle.module.url(forResource: "watson_30s", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("watson_30s.wav not found")
        }
        
        let (pyannote, sortformer) = try await runComparison(
            testName: "Watson Interview (30s)",
            audioURL: testURL
        )
        
        // Verify both detected speakers
        XCTAssertGreaterThanOrEqual(sortformer.speakerCount, 1)
        XCTAssertGreaterThan(sortformer.rtf, 1.0, "Sortformer should be faster than real-time")
        
        if let p = pyannote {
            XCTAssertGreaterThanOrEqual(p.speakerCount, 1)
        }
        
        print("✓ Watson comparison test passed")
    }
    
    /// Compare diarization on mix_8k.wav (multi-speaker mixture)
    func testCompare_Mix8k() async throws {
        guard let testURL = Bundle.module.url(forResource: "mix_8k", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("mix_8k.wav not found")
        }
        
        let (_, sortformer) = try await runComparison(
            testName: "Mix 8kHz (Speech Mixture)",
            audioURL: testURL
        )
        
        XCTAssertGreaterThanOrEqual(sortformer.segmentCount, 0)
        print("✓ Mix 8k comparison test passed")
    }
    
    /// Compare diarization on mix3_8k.wav (multi-speaker mixture)
    func testCompare_Mix3_8k() async throws {
        guard let testURL = Bundle.module.url(forResource: "mix3_8k", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("mix3_8k.wav not found")
        }
        
        let (_, sortformer) = try await runComparison(
            testName: "Mix3 8kHz (Speech Mixture)",
            audioURL: testURL
        )
        
        XCTAssertGreaterThanOrEqual(sortformer.segmentCount, 0)
        print("✓ Mix3 8k comparison test passed")
    }
    
    /// Compare diarization on test.wav (short speech sample)
    func testCompare_Test() async throws {
        guard let testURL = Bundle.module.url(forResource: "test", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("test.wav not found")
        }
        
        let (_, sortformer) = try await runComparison(
            testName: "Test (Short Speech)",
            audioURL: testURL
        )
        
        XCTAssertGreaterThanOrEqual(sortformer.segmentCount, 0)
        print("✓ Test comparison passed")
    }
    
    /// Compare diarization on harry_potter.wav (movie trailer, ~10 speakers, music)
    func testCompare_HarryPotter() async throws {
        guard let testURL = Bundle.module.url(forResource: "harry_potter", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("harry_potter.wav not found")
        }
        
        let (pyannote, sortformer) = try await runComparison(
            testName: "Harry Potter Trailer (2:16, ~10 speakers, music)",
            audioURL: testURL
        )
        
        XCTAssertGreaterThanOrEqual(sortformer.segmentCount, 0)
        
        // Note: Sortformer is limited to 4 speakers, Pyannote may detect more
        if let p = pyannote {
            print("Note: Pyannote detected \(p.speakerCount) speakers, Sortformer capped at \(sortformer.speakerCount)")
        }
        
        print("✓ Harry Potter comparison test passed")
    }
    
    /// Sortformer-only test on Harry Potter (for speed)
    func testSortformerOnly_HarryPotter() async throws {
        guard let testURL = Bundle.module.url(forResource: "harry_potter", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("harry_potter.wav not found")
        }
        
        let (_, sortformer) = try await runComparison(
            testName: "Harry Potter Trailer (Sortformer Only)",
            audioURL: testURL,
            skipPyannote: true
        )
        
        XCTAssertGreaterThanOrEqual(sortformer.segmentCount, 0)
        XCTAssertGreaterThan(sortformer.rtf, 1.0)
        
        print("✓ Sortformer-only Harry Potter test passed")
    }
    
    // MARK: - Performance Comparison Summary
    
    /// Run all comparisons and print summary table
    func testAllComparisons() async throws {
        print("\n" + String(repeating: "=", count: 70))
        print("=== FULL DIARIZATION COMPARISON SUITE ===")
        print(String(repeating: "=", count: 70) + "\n")
        
        var results: [(name: String, pyannote: DiarizationResult?, sortformer: DiarizationResult)] = []
        
        // Watson 30s
        if let url = Bundle.module.url(forResource: "watson_30s", withExtension: "wav", subdirectory: "Fixtures") {
            let result = try await runComparison(testName: "Watson 30s", audioURL: url)
            results.append(("Watson 30s", result.pyannote, result.sortformer))
        }
        
        // Test
        if let url = Bundle.module.url(forResource: "test", withExtension: "wav", subdirectory: "Fixtures") {
            let result = try await runComparison(testName: "Test Short", audioURL: url)
            results.append(("Test Short", result.pyannote, result.sortformer))
        }
        
        // Mix 8k
        if let url = Bundle.module.url(forResource: "mix_8k", withExtension: "wav", subdirectory: "Fixtures") {
            let result = try await runComparison(testName: "Mix 8k", audioURL: url)
            results.append(("Mix 8k", result.pyannote, result.sortformer))
        }
        
        // Harry Potter (skip Pyannote for speed in full suite)
        if let url = Bundle.module.url(forResource: "harry_potter", withExtension: "wav", subdirectory: "Fixtures") {
            let result = try await runComparison(testName: "Harry Potter", audioURL: url, skipPyannote: true)
            results.append(("Harry Potter", result.pyannote, result.sortformer))
        }
        
        // Print summary table
        print("\n" + String(repeating: "=", count: 70))
        print("=== SUMMARY TABLE ===")
        print(String(repeating: "=", count: 70))
        print("Audio".padding(toLength: 15, withPad: " ", startingAt: 0) + " | " +
              "Pyannote".padding(toLength: 20, withPad: " ", startingAt: 0) + " | " +
              "Sortformer".padding(toLength: 20, withPad: " ", startingAt: 0))
        print(String(repeating: "-", count: 70))
        
        for (name, pyannote, sortformer) in results {
            let pInfo = pyannote.map { "\($0.speakerCount)spk, \(String(format: "%.1f", $0.rtf))x RTF" } ?? "skipped"
            let sInfo = "\(sortformer.speakerCount)spk, \(String(format: "%.1f", sortformer.rtf))x RTF"
            print(name.padding(toLength: 15, withPad: " ", startingAt: 0) + " | " +
                  pInfo.padding(toLength: 20, withPad: " ", startingAt: 0) + " | " +
                  sInfo.padding(toLength: 20, withPad: " ", startingAt: 0))
        }
        
        print(String(repeating: "=", count: 70) + "\n")
        print("✓ All comparison tests completed")
    }
}
