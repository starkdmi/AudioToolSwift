//
//  SegmentTests.swift
//  AudioTool
//
//  Tests for segment types
//

import Testing
@testable import AudioTool
@testable import AudioToolCore

@Suite("Segment Tests")
struct SegmentTests {
    
    // MARK: - TimeRange
    
    @Test("TimeRange properties")
    func testTimeRangeProperties() {
        let range = TimeRange(start: 1.5, end: 3.5)
        
        #expect(range.start == 1.5)
        #expect(range.end == 3.5)
        #expect(range.duration == 2.0)
    }
    
    @Test("TimeRange comparison")
    func testTimeRangeComparison() {
        let a = TimeRange(start: 1.0, end: 2.0)
        let b = TimeRange(start: 2.0, end: 3.0)
        
        #expect(a < b)
        #expect(!(b < a))

        let sameStartShorter = TimeRange(start: 1.0, end: 1.5)
        #expect(sameStartShorter < a)
        #expect(!(a < sameStartShorter))
    }
    
    @Test("TimeRange overlap detection")
    func testTimeRangeOverlap() {
        let a = TimeRange(start: 0.0, end: 2.0)
        let b = TimeRange(start: 1.5, end: 3.0)
        let c = TimeRange(start: 3.0, end: 4.0)
        
        #expect(a.overlaps(with: b))
        #expect(b.overlaps(with: a))
        #expect(!a.overlaps(with: c))  // Adjacent but not overlapping
    }
    
    @Test("TimeRange intersection")
    func testTimeRangeIntersection() {
        let a = TimeRange(start: 0.0, end: 2.0)
        let b = TimeRange(start: 1.5, end: 3.0)
        
        let intersection = a.intersection(with: b)
        #expect(intersection != nil)
        #expect(intersection?.start == 1.5)
        #expect(intersection?.end == 2.0)
        
        let c = TimeRange(start: 3.0, end: 4.0)
        #expect(a.intersection(with: c) == nil)
    }
    
    // MARK: - SpeakerID
    
    @Test("SpeakerID from string")
    func testSpeakerIDString() {
        let speaker = SpeakerID("speaker_alice")
        #expect(speaker.id == "speaker_alice")
        #expect(speaker.description == "speaker_alice")
    }
    
    @Test("SpeakerID from index")
    func testSpeakerIDIndex() {
        let speaker = SpeakerID(0)
        #expect(speaker.id == "speaker_0")
    }
    
    // MARK: - VADSegment
    
    @Test("VADSegment creation")
    func testVADSegment() {
        let segment = VADSegment(
            timeRange: TimeRange(start: 1.0, end: 2.5),
            isSpeech: true,
            probability: 0.95
        )
        
        #expect(segment.isSpeech)
        #expect(segment.probability == 0.95)
        #expect(segment.timeRange.duration == 1.5)
    }
    
    // MARK: - DiarizedSegment
    
    @Test("DiarizedSegment creation")
    func testDiarizedSegment() {
        let segment = DiarizedSegment(
            timeRange: TimeRange(start: 0, end: 3),
            speakerID: SpeakerID("alice"),
            confidence: 0.9
        )
        
        #expect(segment.speakerID.id == "alice")
        #expect(segment.confidence == 0.9)
    }
    
    // MARK: - SpeakerTimeline
    
    @Test("SpeakerTimeline speaker count")
    func testSpeakerTimelineCount() {
        let segments = [
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 2), speakerID: SpeakerID(0), confidence: 0.9),
            DiarizedSegment(timeRange: TimeRange(start: 2, end: 4), speakerID: SpeakerID(1), confidence: 0.85),
            DiarizedSegment(timeRange: TimeRange(start: 4, end: 6), speakerID: SpeakerID(0), confidence: 0.88),
        ]
        
        let timeline = SpeakerTimeline(segments: segments)
        
        #expect(timeline.speakerCount == 2)
        #expect(timeline.segments.count == 3)
    }
    
    @Test("SpeakerTimeline overlap detection")
    func testSpeakerTimelineOverlap() {
        let segments = [
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 3), speakerID: SpeakerID(0), confidence: 0.9),
            DiarizedSegment(timeRange: TimeRange(start: 2, end: 5), speakerID: SpeakerID(1), confidence: 0.85),  // Overlaps with above
            DiarizedSegment(timeRange: TimeRange(start: 6, end: 8), speakerID: SpeakerID(0), confidence: 0.88),
        ]
        
        let timeline = SpeakerTimeline(segments: segments)
        
        #expect(timeline.maxOverlappingSpeakers == 2)
        
        let overlaps = timeline.overlappingRanges()
        #expect(overlaps.count == 1)
        #expect(overlaps[0].start == 2.0)
        #expect(overlaps[0].end == 3.0)
    }

    @Test("Overlap counts distinct simultaneously active speakers")
    func testSpeakerTimelineDistinctConcurrentSpeakers() {
        let speaker0 = SpeakerID(0)
        let timeline = SpeakerTimeline(segments: [
            // Duplicate, nested hypotheses for one speaker must count once.
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 10), speakerID: speaker0, confidence: 0.9),
            DiarizedSegment(timeRange: TimeRange(start: 1, end: 9), speakerID: speaker0, confidence: 0.8),
            DiarizedSegment(timeRange: TimeRange(start: 4, end: 5), speakerID: SpeakerID(1), confidence: 0.9),
        ])

        #expect(timeline.maxOverlappingSpeakers == 2)
        #expect(timeline.overlappingRanges() == [TimeRange(start: 4, end: 5)])
    }

    @Test("Pairwise overlaps that never coincide do not inflate the maximum")
    func testSpeakerTimelineNonCliqueOverlap() {
        let timeline = SpeakerTimeline(segments: [
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 10), speakerID: SpeakerID(0), confidence: 1),
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 4), speakerID: SpeakerID(1), confidence: 1),
            DiarizedSegment(timeRange: TimeRange(start: 6, end: 10), speakerID: SpeakerID(2), confidence: 1),
        ])

        #expect(timeline.maxOverlappingSpeakers == 2)
        #expect(timeline.overlappingRanges() == [
            TimeRange(start: 0, end: 4),
            TimeRange(start: 6, end: 10),
        ])
    }

    @Test("Zero-duration hypotheses do not remain active")
    func testZeroDurationSegmentDoesNotAffectOverlap() {
        let timeline = SpeakerTimeline(segments: [
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 2), speakerID: SpeakerID(0), confidence: 1),
            DiarizedSegment(timeRange: TimeRange(start: 1, end: 1), speakerID: SpeakerID(1), confidence: 1),
        ])

        #expect(timeline.maxOverlappingSpeakers == 1)
        #expect(timeline.overlappingRanges().isEmpty)
    }
    
    @Test("SpeakerTimeline filter by speaker")
    func testSpeakerTimelineFilter() {
        let segments = [
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 2), speakerID: SpeakerID(0), confidence: 0.9),
            DiarizedSegment(timeRange: TimeRange(start: 2, end: 4), speakerID: SpeakerID(1), confidence: 0.85),
            DiarizedSegment(timeRange: TimeRange(start: 4, end: 6), speakerID: SpeakerID(0), confidence: 0.88),
        ]
        
        let timeline = SpeakerTimeline(segments: segments)
        
        let speaker0Segments = timeline.segments(for: SpeakerID(0))
        #expect(speaker0Segments.count == 2)
        
        let speaker1Segments = timeline.segments(for: SpeakerID(1))
        #expect(speaker1Segments.count == 1)
    }
    
    // MARK: - Transcription
    
    @Test("Transcription creation")
    func testTranscription() {
        let segments = [
            TranscriptionSegment(text: "Hello", timeRange: TimeRange(start: 0, end: 1), confidence: 0.95),
            TranscriptionSegment(text: "World", timeRange: TimeRange(start: 1, end: 2), confidence: 0.92),
        ]
        
        let transcription = Transcription(text: "Hello World", segments: segments, language: "en")
        
        #expect(transcription.text == "Hello World")
        #expect(transcription.segments.count == 2)
        #expect(transcription.language == "en")
    }
    
    // MARK: - SoundClassification
    
    @Test("SoundClassification creation")
    func testSoundClassification() {
        let classification = SoundClassification(
            label: "music",
            confidence: 0.85,
            timeRange: TimeRange(start: 5.0, end: 10.0)
        )
        
        #expect(classification.label == "music")
        #expect(classification.confidence == 0.85)
        #expect(classification.timeRange.duration == 5.0)
    }
}
