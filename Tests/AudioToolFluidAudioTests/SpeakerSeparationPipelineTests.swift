//
//  SpeakerSeparationPipelineTests.swift
//  ClearVoiceFluidAudioTests
//
//  Tests for speaker separation pipeline with re-identification
//  Phase 3: Full pipeline example/tests
//

import XCTest
@testable import ClearVoice
import ClearVoiceCore
import ClearVoiceFluidAudio
import ClearVoiceMLX
import AudioUtils
import MLX

final class SpeakerSeparationPipelineTests: XCTestCase {
    
    static let projectRoot: String = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.path
    }()
    
    // MARK: - Event Tracker
    
    private actor EventTracker {
        private var overlapEvents: [(timeRange: TimeRange, speakerCount: Int)] = []
        private var trackEvents: [SeparatedSpeakerTrack] = []
        private var progressEvents: [(stage: String, percent: Double)] = []
        private var stageCompleteEvents: [String] = []
        
        func record(event: PipelineEvent) {
            switch event {
            case .overlapDetected(let timeRange, let speakerCount):
                overlapEvents.append((timeRange: timeRange, speakerCount: speakerCount))
            case .trackIdentified(let track):
                trackEvents.append(track)
            case .progress(let stage, let percent):
                progressEvents.append((stage: stage, percent: percent))
            case .stageComplete(let stage, _):
                stageCompleteEvents.append(stage)
            default:
                break
            }
        }
        
        func snapshot() -> (
            overlaps: [(timeRange: TimeRange, speakerCount: Int)],
            tracks: [SeparatedSpeakerTrack],
            progress: [(stage: String, percent: Double)],
            stages: [String]
        ) {
            (overlapEvents, trackEvents, progressEvents, stageCompleteEvents)
        }
    }
    
    // MARK: - Test Fixtures
    
    private func harryPotterURL() throws -> URL {
        if let url = Bundle.module.url(forResource: "harry_potter", withExtension: "wav", subdirectory: "Fixtures") {
            return url
        }
        
        let fallback = URL(fileURLWithPath: "\(Self.projectRoot)/Docs/harry_potter.wav")
        guard FileManager.default.fileExists(atPath: fallback.path) else {
            throw XCTSkip("harry_potter.wav not found in Fixtures or Docs")
        }
        return fallback
    }
    
    private func loadAudio(at url: URL, sampleRate: Int) throws -> AudioBuffer {
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: Double(sampleRate),
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: url)
        eval(audio)
        return AudioBuffer(samples: audio.asArray(Float.self), sampleRate: sampleRate, channels: 1)
    }
    
    // MARK: - Unit Tests
    
    func testSpeakerIdentificationType() {
        // Test SpeakerIdentification type
        let identification = SpeakerIdentification(
            speakerSlot: 1,
            confidence: 0.85,
            allProbabilities: [0.05, 0.85, 0.07, 0.03],
            averageProbabilities: [0.05, 0.85, 0.07, 0.03],
            frameCount: 10
        )
        
        XCTAssertEqual(identification.speakerSlot, 1)
        XCTAssertEqual(identification.confidence, 0.85, accuracy: 0.001)
        XCTAssertTrue(identification.isConfident(threshold: 0.5))
        XCTAssertTrue(identification.isConfident(threshold: 0.8))
        XCTAssertFalse(identification.isConfident(threshold: 0.9))
        XCTAssertEqual(identification.frameCount, 10)
    }
    
    func testSeparatedSpeakerTrackType() {
        // Test SeparatedSpeakerTrack type
        let audio = AudioBuffer.silence(duration: 1.0, sampleRate: 16000)
        let track = SeparatedSpeakerTrack(
            audio: audio,
            speakerSlot: 0,
            speakerID: SpeakerID(0),
            confidence: 0.92,
            sourceTimeRange: TimeRange(start: 5.0, end: 10.0),
            trackIndex: 0
        )
        
        XCTAssertEqual(track.speakerSlot, 0)
        XCTAssertEqual(track.speakerID?.id, "speaker_0")
        XCTAssertEqual(track.confidence, 0.92, accuracy: 0.001)
        XCTAssertEqual(track.sourceTimeRange.start, 5.0)
        XCTAssertEqual(track.sourceTimeRange.end, 10.0)
        XCTAssertEqual(track.sourceTimeRange.duration, 5.0)
        XCTAssertTrue(track.isConfidentlyIdentified(threshold: 0.5))
        XCTAssertTrue(track.isConfidentlyIdentified(threshold: 0.9))
        XCTAssertFalse(track.isConfidentlyIdentified(threshold: 0.95))
    }
    
    func testOverlapHandlingEnum() {
        // Test OverlapHandling options
        XCTAssertEqual(OverlapHandling.skip, OverlapHandling.skip)
        XCTAssertEqual(OverlapHandling.separate, OverlapHandling.separate)
        // Note: separateAndIdentify and separateIdentifyAndMerge are deprecated
        XCTAssertNotEqual(OverlapHandling.skip, OverlapHandling.separate)
    }
    
    func testSeparationModelSelection() {
        // Test model auto-selection based on speaker count
        XCTAssertNil(SeparationModel.forOverlappingSpeakers(0), "0 speakers should return nil")
        XCTAssertNil(SeparationModel.forOverlappingSpeakers(1), "1 speaker should return nil")
        XCTAssertEqual(SeparationModel.forOverlappingSpeakers(2), .mossformerWhamr, "2 speakers should use WHAMR")
        XCTAssertEqual(SeparationModel.forOverlappingSpeakers(3), .mossformer3spk, "3 speakers should use 3spk")
        XCTAssertNil(SeparationModel.forOverlappingSpeakers(4), "4+ speakers should return nil")
        XCTAssertNil(SeparationModel.forOverlappingSpeakers(5), "5+ speakers should return nil")
    }
    
    func testSpeakerTimelineOverlapDetection() {
        // Test overlap detection from SpeakerTimeline
        let segments = [
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 5), speakerID: SpeakerID(0), confidence: 1.0),
            DiarizedSegment(timeRange: TimeRange(start: 3, end: 8), speakerID: SpeakerID(1), confidence: 1.0),
            DiarizedSegment(timeRange: TimeRange(start: 10, end: 15), speakerID: SpeakerID(0), confidence: 1.0)
        ]
        
        let timeline = SpeakerTimeline(segments: segments)
        
        XCTAssertEqual(timeline.speakerCount, 2)
        XCTAssertEqual(timeline.maxOverlappingSpeakers, 2)
        
        let overlaps = timeline.overlappingRanges()
        XCTAssertEqual(overlaps.count, 1, "Should detect one overlap")
        
        if let overlap = overlaps.first {
            XCTAssertEqual(overlap.start, 3.0, accuracy: 0.001)
            XCTAssertEqual(overlap.end, 5.0, accuracy: 0.001)
            XCTAssertEqual(overlap.duration, 2.0, accuracy: 0.001)
        }
    }
    
    func testNoOverlapsTimeline() {
        // Test timeline with no overlaps
        let segments = [
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 5), speakerID: SpeakerID(0), confidence: 1.0),
            DiarizedSegment(timeRange: TimeRange(start: 6, end: 10), speakerID: SpeakerID(1), confidence: 1.0),
            DiarizedSegment(timeRange: TimeRange(start: 11, end: 15), speakerID: SpeakerID(0), confidence: 1.0)
        ]
        
        let timeline = SpeakerTimeline(segments: segments)
        
        XCTAssertEqual(timeline.speakerCount, 2)
        XCTAssertEqual(timeline.maxOverlappingSpeakers, 1)
        
        let overlaps = timeline.overlappingRanges()
        XCTAssertTrue(overlaps.isEmpty, "Should have no overlaps")
    }
    
    func testSingleSpeakerTimeline() {
        // Test timeline with single speaker
        let segments = [
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 5), speakerID: SpeakerID(0), confidence: 1.0),
            DiarizedSegment(timeRange: TimeRange(start: 6, end: 10), speakerID: SpeakerID(0), confidence: 1.0)
        ]
        
        let timeline = SpeakerTimeline(segments: segments)
        
        XCTAssertEqual(timeline.speakerCount, 1)
        XCTAssertEqual(timeline.maxOverlappingSpeakers, 1)
        XCTAssertTrue(timeline.overlappingRanges().isEmpty)
    }
    
    func testThreeSpeakerOverlap() {
        // Test timeline with 3 speakers overlapping
        let segments = [
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 8), speakerID: SpeakerID(0), confidence: 1.0),
            DiarizedSegment(timeRange: TimeRange(start: 2, end: 10), speakerID: SpeakerID(1), confidence: 1.0),
            DiarizedSegment(timeRange: TimeRange(start: 4, end: 12), speakerID: SpeakerID(2), confidence: 1.0)
        ]
        
        let timeline = SpeakerTimeline(segments: segments)
        
        XCTAssertEqual(timeline.speakerCount, 3)
        XCTAssertEqual(timeline.maxOverlappingSpeakers, 3)
        
        let overlaps = timeline.overlappingRanges()
        XCTAssertFalse(overlaps.isEmpty, "Should detect overlaps")
    }
    
    // MARK: - Pipeline Builder Tests
    
    func testPipelineBuilderSeparateOverlapStage() {
        // Test that separateOverlap stage is added correctly
        let voice = ClearVoice()
        let pipeline = voice.pipeline()
            .diarize()
            .separateOverlap(.separate)
        
        XCTAssertEqual(pipeline.stages.count, 2)
        XCTAssertEqual(pipeline.stages[0].name, "diarization")
        XCTAssertEqual(pipeline.stages[1].name, "overlapSeparation")
    }
    
    func testPipelineBuilderWithAllOptions() {
        // Test full pipeline with all overlap options
        let voice = ClearVoice()
        
        // Test .skip option
        let skipPipeline = voice.pipeline()
            .diarize()
            .separateOverlap(.skip)
        XCTAssertEqual(skipPipeline.stages.count, 2)
        
        // Test .separate option (no identification)
        let separatePipeline = voice.pipeline()
            .diarize()
            .separateOverlap(.separate)
        XCTAssertEqual(separatePipeline.stages.count, 2)
        
        // Test .separateAndIdentify option - now defaults to .separate
        let identifyPipeline = voice.pipeline()
            .diarize()
            .separateOverlap()  // Default is .separate
        XCTAssertEqual(identifyPipeline.stages.count, 2)
    }
    
    // MARK: - Integration Tests (require models)
    
    func testSeparateIntegration() async throws {
        // Skip if models aren't available
        let url = try harryPotterURL()
        
        // Load audio at 16kHz for diarization
        let audio = try loadAudio(at: url, sampleRate: 16000)
        XCTAssertGreaterThan(audio.duration, 5.0, "Audio should be at least 5 seconds")
        
        // Initialize ClearVoice
        let voice = ClearVoice()
        
        // Initialize and register Sortformer
        let sortformer = FluidAudioSortformerProvider()
        do {
            try await sortformer.load()
        } catch {
            throw XCTSkip("Sortformer model not available: \(error)")
        }
        await voice.register(diarization: sortformer)
        
        // Initialize and register WHAMR separator
        let separator = MossFormer2SSProvider(model: .twoSpeakerWHAMR)
        do {
            try await separator.load()
        } catch {
            throw XCTSkip("MossFormer2 WHAMR model not available: \(error)")
        }
        await voice.register(separator: separator, for: .mossformerWhamr)
        
        // Run diarization first
        let timeline = try await voice.diarize(audio)
        print("Diarization complete: \(timeline.speakerCount) speakers, \(timeline.maxOverlappingSpeakers) max overlapping")
        
        let overlaps = timeline.overlappingRanges()
        print("Found \(overlaps.count) overlap regions")
        
        if overlaps.isEmpty {
            print("No overlaps detected - skipping separation test")
            return
        }
        
        // Process first overlap
        let firstOverlap = overlaps[0]
        print("Processing overlap: \(firstOverlap.start)-\(firstOverlap.end) (\(firstOverlap.duration)s)")
        
        let overlapAudio = audio.slice(firstOverlap.start..<firstOverlap.end)
        
        // Separate using the separator directly (speaker identification is deprecated)
        let separatedTracks = try await separator.separate(overlapAudio, speakers: 2)
        
        XCTAssertGreaterThan(separatedTracks.count, 0, "Should produce at least one track")
        
        for (idx, track) in separatedTracks.enumerated() {
            print("Track \(idx): \(track.samples.count) samples, duration \(track.duration)s")
            XCTAssertGreaterThan(track.duration, 0)
        }
    }
    
    func testFullPipelineWithSeparateOverlap() async throws {
        // Skip if models aren't available
        let url = try harryPotterURL()
        let audio = try loadAudio(at: url, sampleRate: 16000)
        
        // Initialize ClearVoice with all required providers
        let voice = ClearVoice()
        
        // Register Sortformer
        let sortformer = FluidAudioSortformerProvider()
        do {
            try await sortformer.load()
        } catch {
            throw XCTSkip("Sortformer model not available: \(error)")
        }
        await voice.register(diarization: sortformer)
        
        // Register WHAMR separator
        let separator = MossFormer2SSProvider(model: .twoSpeakerWHAMR)
        do {
            try await separator.load()
        } catch {
            throw XCTSkip("MossFormer2 WHAMR model not available: \(error)")
        }
        await voice.register(separator: separator, for: .mossformerWhamr)
        
        // Track events
        let tracker = EventTracker()
        
        // Run full pipeline with .separate mode (separation only, no identification)
        let result = try await voice.pipeline()
            .diarize()
            .separateOverlap(.separate)
            .onEvent { event in
                await tracker.record(event: event)
            }
            .process(audio: audio)
        
        // Check results
        XCTAssertNotNil(result.analysis, "Should have analysis result")
        
        let snapshot = await tracker.snapshot()
        print("Pipeline completed:")
        print("  - Overlaps detected: \(snapshot.overlaps.count)")
        print("  - Separated tracks: \(result.separatedTracks?.count ?? 0)")
        print("  - Stages completed: \(snapshot.stages)")
        
        // If there were overlaps, we should have separated tracks
        if !snapshot.overlaps.isEmpty {
            XCTAssertNotNil(result.separatedTracks, "Should have separated tracks")
            
            for (index, track) in (result.separatedTracks ?? []).enumerated() {
                print("  Track \(index): \(track.duration)s, \(track.samples.count) samples")
            }
        }
    }
    
    func testPipelineSkipHandling() async throws {
        // Test that .skip handling works correctly
        let url = try harryPotterURL()
        let audio = try loadAudio(at: url, sampleRate: 16000)
        
        let voice = ClearVoice()
        
        // Register Sortformer only (no separator needed for skip)
        let sortformer = FluidAudioSortformerProvider()
        do {
            try await sortformer.load()
        } catch {
            throw XCTSkip("Sortformer model not available: \(error)")
        }
        await voice.register(diarization: sortformer)
        
        // Run pipeline with skip
        let result = try await voice.pipeline()
            .diarize()
            .separateOverlap(.skip)
            .process(audio: audio)
        
        // Should complete without error but have no identified tracks
        XCTAssertNotNil(result.analysis)
        XCTAssertNil(result.identifiedTracks, "Skip mode should not produce identified tracks")
    }
}
