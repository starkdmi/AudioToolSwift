//
//  Segments.swift
//  AudioTool
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
        precondition(start.isFinite && end.isFinite, "Time range bounds must be finite")
        precondition(end >= start, "End must be >= start")
        self.start = start
        self.end = end
    }
    
    public static func < (lhs: TimeRange, rhs: TimeRange) -> Bool {
        if lhs.start != rhs.start {
            return lhs.start < rhs.start
        }
        return lhs.end < rhs.end
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
        self.segments = segments.sorted { $0.timeRange < $1.timeRange }
        
        let uniqueSpeakers = Set(segments.map(\.speakerID))
        self.speakerCount = uniqueSpeakers.count
        
        self.maxOverlappingSpeakers = Self.overlapAnalysis(segments).maximum
    }
    
    /// Get segments for a specific speaker
    public func segments(for speaker: SpeakerID) -> [DiarizedSegment] {
        segments.filter { $0.speakerID == speaker }
    }
    
    /// Get time ranges where speakers overlap
    ///
    /// Returns merged, non-overlapping time ranges where 2+ speakers are active.
    /// Adjacent or overlapping ranges are merged to prevent duplicate processing.
    public func overlappingRanges() -> [TimeRange] {
        Self.overlapAnalysis(segments).ranges
    }

    private struct SpeakerEvent {
        let time: Double
        let speaker: SpeakerID
        let delta: Int
    }

    /// Sweep distinct speaker activity, retaining per-speaker reference counts so
    /// duplicate/overlapping segments for one speaker never masquerade as overlap.
    private static func overlapAnalysis(
        _ segments: [DiarizedSegment]
    ) -> (maximum: Int, ranges: [TimeRange]) {
        let events = segments.flatMap { segment in
            [
                SpeakerEvent(
                    time: segment.timeRange.start,
                    speaker: segment.speakerID,
                    delta: 1
                ),
                SpeakerEvent(
                    time: segment.timeRange.end,
                    speaker: segment.speakerID,
                    delta: -1
                ),
            ]
        }.sorted { lhs, rhs in
            if lhs.time != rhs.time { return lhs.time < rhs.time }
            return lhs.delta < rhs.delta
        }

        guard !events.isEmpty else { return (0, []) }
        var activeCounts: [SpeakerID: Int] = [:]
        var maximum = 0
        var ranges: [TimeRange] = []
        var previousTime = events[0].time
        var index = 0

        while index < events.count {
            let time = events[index].time
            if previousTime < time, activeCounts.count >= 2 {
                let range = TimeRange(start: previousTime, end: time)
                if let last = ranges.last, last.end == range.start {
                    ranges[ranges.count - 1] = TimeRange(
                        start: last.start,
                        end: range.end
                    )
                } else {
                    ranges.append(range)
                }
            }

            var changes: [SpeakerID: Int] = [:]
            while index < events.count, events[index].time == time {
                let event = events[index]
                changes[event.speaker, default: 0] += event.delta
                index += 1
            }
            for (speaker, delta) in changes {
                let newCount = (activeCounts[speaker] ?? 0) + delta
                if newCount > 0 {
                    activeCounts[speaker] = newCount
                } else {
                    activeCounts.removeValue(forKey: speaker)
                }
            }
            maximum = max(maximum, activeCounts.count)
            previousTime = time
        }

        return (maximum, ranges)
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

// MARK: - Speaker Separation

/// Result of separating overlapping speech with speaker identification
///
/// After separating an overlapping region using MossFormer2 or similar,
/// this type represents a single separated track with its identified speaker.
///
/// ## Usage Flow
/// 1. Run diarization to detect overlap regions
/// 2. Extract overlap region audio
/// 3. Run separation model → `[AudioBuffer]`
/// 4. Re-identify each track → `[SeparatedSpeakerTrack]`
///
/// Example:
/// ```swift
/// let overlaps = timeline.overlappingRanges()
/// for range in overlaps {
///     let overlapAudio = audio.slice(range)
///     let tracks = try await separator.separate(overlapAudio, speakers: 2)
///     let identifiedTracks = try await sortformer.identifySpeakers(tracks)
///     
///     for (track, id) in identifiedTracks {
///         let result = SeparatedSpeakerTrack(
///             audio: track,
///             speakerSlot: id.speakerSlot,
///             speakerID: timeline.speakerID(forSlot: id.speakerSlot),
///             confidence: id.confidence,
///             sourceTimeRange: range
///         )
///     }
/// }
/// ```
public struct SeparatedSpeakerTrack: Sendable, Identifiable {
    public let id: UUID
    
    /// The separated audio for this speaker
    public let audio: AudioBuffer
    
    /// The Sortformer speaker slot (0-3) this track was identified as
    /// - Note: May be nil if identification failed or wasn't performed
    public let speakerSlot: Int?
    
    /// The speaker ID from diarization (e.g., "speaker_0")
    /// - Note: May be nil if speaker mapping isn't available
    public let speakerID: SpeakerID?
    
    /// Confidence score for the speaker identification (0.0 - 1.0)
    public let confidence: Float
    
    /// The time range in the original audio where this overlap occurred
    public let sourceTimeRange: TimeRange
    
    /// Track index in the separation output (0, 1, 2...)
    public let trackIndex: Int
    
    public init(
        id: UUID = UUID(),
        audio: AudioBuffer,
        speakerSlot: Int?,
        speakerID: SpeakerID?,
        confidence: Float,
        sourceTimeRange: TimeRange,
        trackIndex: Int = 0
    ) {
        self.id = id
        self.audio = audio
        self.speakerSlot = speakerSlot
        self.speakerID = speakerID
        self.confidence = confidence
        self.sourceTimeRange = sourceTimeRange
        self.trackIndex = trackIndex
    }
    
    /// Check if identification is confident enough to use
    /// - Parameter threshold: Minimum confidence threshold (default 0.5)
    /// - Returns: True if speaker identification confidence exceeds threshold
    public func isConfidentlyIdentified(threshold: Float = 0.5) -> Bool {
        speakerSlot != nil && confidence >= threshold
    }
}

/// Options for handling overlapping speech in pipelines
///
/// - Note: Speaker re-identification after separation is currently not reliable.
///   The diarizer's speaker cache is contaminated with mixture audio during overlap
///   periods, so matching separated clean audio back to original speakers fails.
///   All modes except `.skip` will separate audio but return tracks without speaker IDs.
public enum OverlapHandling: Sendable, Equatable {
    /// Skip overlap regions (default behavior)
    case skip
    
    /// Separate overlapping speakers into individual tracks (without speaker identification)
    case separate
    
    /// Separate speakers (re-identification not yet implemented - tracks returned without speaker IDs)
    @available(*, deprecated, message: "Re-identification not yet implemented. Use .separate instead.")
    case separateAndIdentify
    
    /// Separate speakers (re-identification not yet implemented - tracks returned without speaker IDs)
    @available(*, deprecated, message: "Re-identification not yet implemented. Use .separate instead.")
    case separateIdentifyAndMerge
}

// MARK: - Speaker Identification

/// Result of speaker identification for a separated audio track
///
/// Used after source separation to determine which speaker slot (from
/// the original diarization) a separated track belongs to.
///
/// Example:
/// ```swift
/// // After separating overlapping audio into tracks
/// let tracks = try await separator.separate(overlapRegion, speakers: 2)
///
/// // Re-identify each track using same Sortformer instance
/// let id1 = try await sortformer.identifySpeaker(tracks[0])
/// // id1.speakerSlot == 0, id1.confidence == 0.92
/// ```
public struct SpeakerIdentification: Sendable, Equatable {
    /// The speaker slot this audio belongs to (0-3 for Sortformer)
    public let speakerSlot: Int
    
    /// Confidence score for the identified slot (probability)
    public let confidence: Float
    
    /// Raw probabilities for all speaker slots [numSpeakers]
    /// For Sortformer: [4] with probabilities for slots 0-3
    public let allProbabilities: [Float]
    
    /// Average probability across all frames (for short segments)
    /// This is used when the audio produces multiple frames
    public let averageProbabilities: [Float]?
    
    /// Number of frames analyzed
    public let frameCount: Int
    
    public init(
        speakerSlot: Int,
        confidence: Float,
        allProbabilities: [Float],
        averageProbabilities: [Float]? = nil,
        frameCount: Int = 1
    ) {
        self.speakerSlot = speakerSlot
        self.confidence = confidence
        self.allProbabilities = allProbabilities
        self.averageProbabilities = averageProbabilities
        self.frameCount = frameCount
    }
    
    /// Check if identification is confident enough to trust
    /// - Parameter threshold: Minimum confidence threshold (default 0.5)
    /// - Returns: True if confidence exceeds threshold
    public func isConfident(threshold: Float = 0.5) -> Bool {
        confidence >= threshold
    }
}

// MARK: - Diarized Transcription

/// Complete transcription result with speaker attribution
///
/// Created by merging a `Transcription` with a `SpeakerTimeline` using timestamp alignment.
/// Each segment is assigned to the speaker with the most overlap at that time range.
///
/// Example:
/// ```swift
/// let result = try await voice.pipeline()
///     .parallel {[
///         PipelineBuilder().transcribe(.parakeet),
///         PipelineBuilder().diarize()
///     ]}
///     .mergeTranscriptionWithDiarization()
///     .process(audio: audio)
///
/// for segment in result.diarizedTranscription?.segments ?? [] {
///     print("\(segment.speakerID?.id ?? "?"): \(segment.text)")
/// }
/// ```
public struct DiarizedTranscription: Sendable {
    /// Full transcription text
    public let text: String
    
    /// Segments with speaker attribution
    public let segments: [DiarizedTranscriptSegment]
    
    /// Detected language (if available)
    public let language: String?
    
    /// Translation (if translation was requested)
    public let translation: String?
    
    /// Number of unique speakers in the transcription
    public var speakerCount: Int {
        Set(segments.compactMap(\.speakerID)).count
    }
    
    /// Get segments for a specific speaker
    public func segments(for speaker: SpeakerID) -> [DiarizedTranscriptSegment] {
        segments.filter { $0.speakerID == speaker }
    }
    
    /// Get segments where speaker attribution is uncertain (overlap regions)
    public func uncertainSegments() -> [DiarizedTranscriptSegment] {
        segments.filter { $0.isOverlapRegion }
    }
    
    public init(
        text: String,
        segments: [DiarizedTranscriptSegment],
        language: String? = nil,
        translation: String? = nil
    ) {
        self.text = text
        self.segments = segments
        self.language = language
        self.translation = translation
    }
}

/// Single transcription segment with speaker attribution
///
/// Contains the original transcription data plus speaker assignment from diarization.
public struct DiarizedTranscriptSegment: Sendable, Identifiable, Hashable {
    public let id: UUID
    
    /// Transcribed text
    public let text: String
    
    /// Time range in the original audio
    public let timeRange: TimeRange
    
    /// Assigned speaker (nil if unattributed or in overlap region)
    public let speakerID: SpeakerID?
    
    /// All speakers active during this segment (for overlap detection)
    /// If count > 1, this segment spans an overlap region
    public let activeSpeakers: [SpeakerID]
    
    /// Transcription confidence (from ASR)
    public let transcriptionConfidence: Float
    
    /// Speaker attribution confidence (from diarization overlap calculation)
    /// Lower confidence indicates the segment spans multiple speakers' regions
    public let attributionConfidence: Float
    
    /// Whether this segment is in an overlap region (multiple speakers active)
    public var isOverlapRegion: Bool {
        activeSpeakers.count > 1
    }
    
    /// Translated text (if translation was requested)
    public let translation: String?
    
    public init(
        id: UUID = UUID(),
        text: String,
        timeRange: TimeRange,
        speakerID: SpeakerID?,
        activeSpeakers: [SpeakerID] = [],
        transcriptionConfidence: Float,
        attributionConfidence: Float,
        translation: String? = nil
    ) {
        self.id = id
        self.text = text
        self.timeRange = timeRange
        self.speakerID = speakerID
        self.activeSpeakers = activeSpeakers.isEmpty && speakerID != nil ? [speakerID!] : activeSpeakers
        self.transcriptionConfidence = transcriptionConfidence
        self.attributionConfidence = attributionConfidence
        self.translation = translation
    }
    
    // Hashable conformance (exclude UUID for value-based comparison)
    public func hash(into hasher: inout Hasher) {
        hasher.combine(text)
        hasher.combine(timeRange)
        hasher.combine(speakerID)
    }
    
    public static func == (lhs: DiarizedTranscriptSegment, rhs: DiarizedTranscriptSegment) -> Bool {
        lhs.text == rhs.text &&
        lhs.timeRange == rhs.timeRange &&
        lhs.speakerID == rhs.speakerID
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
