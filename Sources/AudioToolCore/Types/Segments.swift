//
//  Segments.swift
//  ClearVoice
//
//  Segment types for VAD, diarization, and transcription results
//

import Foundation

// MARK: - Time Range

/// Time range in audio
public struct TimeRange: Sendable, Hashable, Comparable {
    public let start: Double
    public let end: Double
    
    public var duration: Double { end - start }
    
    public init(start: Double, end: Double) {
        precondition(end >= start, "End must be >= start")
        self.start = start
        self.end = end
    }
    
    public static func < (lhs: TimeRange, rhs: TimeRange) -> Bool {
        lhs.start < rhs.start
    }
    
    /// Check if this range overlaps with another
    public func overlaps(with other: TimeRange) -> Bool {
        start < other.end && end > other.start
    }
    
    /// Get intersection with another range
    public func intersection(with other: TimeRange) -> TimeRange? {
        let newStart = max(start, other.start)
        let newEnd = min(end, other.end)
        guard newStart < newEnd else { return nil }
        return TimeRange(start: newStart, end: newEnd)
    }
}

// MARK: - Speaker ID

/// Unique speaker identifier
public struct SpeakerID: Sendable, Hashable, Identifiable, CustomStringConvertible {
    public let id: String
    
    public init(_ id: String) {
        self.id = id
    }
    
    public init(_ index: Int) {
        self.id = "speaker_\(index)"
    }
    
    public var description: String { id }
}

// MARK: - VAD Segment

/// Voice Activity Detection result segment
public struct VADSegment: Sendable, Identifiable, Hashable {
    public let id: UUID
    public let timeRange: TimeRange
    public let isSpeech: Bool
    public let probability: Float
    
    public init(
        id: UUID = UUID(),
        timeRange: TimeRange,
        isSpeech: Bool,
        probability: Float
    ) {
        self.id = id
        self.timeRange = timeRange
        self.isSpeech = isSpeech
        self.probability = probability
    }
}

// MARK: - Diarized Segment

/// Speaker-labeled segment
public struct DiarizedSegment: Sendable, Identifiable, Hashable {
    public let id: UUID
    public let timeRange: TimeRange
    public let speakerID: SpeakerID
    public let confidence: Float
    
    public init(
        id: UUID = UUID(),
        timeRange: TimeRange,
        speakerID: SpeakerID,
        confidence: Float
    ) {
        self.id = id
        self.timeRange = timeRange
        self.speakerID = speakerID
        self.confidence = confidence
    }
}

// MARK: - Speaker Timeline

/// Complete speaker timeline with overlap detection
public struct SpeakerTimeline: Sendable {
    public let segments: [DiarizedSegment]
    public let speakerCount: Int
    public let maxOverlappingSpeakers: Int
    
    public init(segments: [DiarizedSegment]) {
        self.segments = segments.sorted { $0.timeRange.start < $1.timeRange.start }
        
        let uniqueSpeakers = Set(segments.map(\.speakerID))
        self.speakerCount = uniqueSpeakers.count
        
        // Calculate max overlapping speakers
        var maxOverlap = 0
        for i in 0..<segments.count {
            var overlapCount = 1
            for j in (i + 1)..<segments.count {
                if segments[i].timeRange.overlaps(with: segments[j].timeRange) {
                    overlapCount += 1
                }
            }
            maxOverlap = max(maxOverlap, overlapCount)
        }
        self.maxOverlappingSpeakers = maxOverlap
    }
    
    /// Get segments for a specific speaker
    public func segments(for speaker: SpeakerID) -> [DiarizedSegment] {
        segments.filter { $0.speakerID == speaker }
    }
    
    /// Get time ranges where speakers overlap
    public func overlappingRanges() -> [TimeRange] {
        var overlaps: [TimeRange] = []
        
        for i in 0..<segments.count {
            for j in (i + 1)..<segments.count {
                if let intersection = segments[i].timeRange.intersection(with: segments[j].timeRange) {
                    overlaps.append(intersection)
                }
            }
        }
        
        return overlaps.sorted()
    }
}

// MARK: - Transcription

/// Complete transcription result
public struct Transcription: Sendable {
    public let text: String
    public let segments: [TranscriptionSegment]
    public let language: String?
    
    public init(text: String, segments: [TranscriptionSegment], language: String? = nil) {
        self.text = text
        self.segments = segments
        self.language = language
    }
}

/// Single transcription segment
public struct TranscriptionSegment: Sendable, Identifiable, Hashable {
    public let id: UUID
    public let text: String
    public let timeRange: TimeRange
    public let speakerID: SpeakerID?
    public let confidence: Float
    
    public init(
        id: UUID = UUID(),
        text: String,
        timeRange: TimeRange,
        speakerID: SpeakerID? = nil,
        confidence: Float
    ) {
        self.id = id
        self.text = text
        self.timeRange = timeRange
        self.speakerID = speakerID
        self.confidence = confidence
    }
}

// MARK: - Sound Classification

/// Sound classification result
public struct SoundClassification: Sendable, Hashable {
    public let label: String
    public let confidence: Float
    public let timeRange: TimeRange
    
    public init(label: String, confidence: Float, timeRange: TimeRange) {
        self.label = label
        self.confidence = confidence
        self.timeRange = timeRange
    }
}

// MARK: - Translation

/// Translation result
public struct TranslationResult: Sendable {
    /// Original source text
    public let sourceText: String
    
    /// Translated text
    public let translatedText: String
    
    /// Detected/specified source language (BCP-47)
    public let sourceLanguage: String?
    
    /// Target language (BCP-47)
    public let targetLanguage: String
    
    public init(
        sourceText: String,
        translatedText: String,
        sourceLanguage: String?,
        targetLanguage: String
    ) {
        self.sourceText = sourceText
        self.translatedText = translatedText
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
    }
}

/// Batch translation result (multiple strings)
public struct BatchTranslationResult: Sendable {
    /// Individual translation results
    public let translations: [TranslationResult]
    
    /// Source language (auto-detected or specified)
    public let sourceLanguage: String?
    
    /// Target language
    public let targetLanguage: String
    
    public init(translations: [TranslationResult], sourceLanguage: String?, targetLanguage: String) {
        self.translations = translations
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
    }
}
