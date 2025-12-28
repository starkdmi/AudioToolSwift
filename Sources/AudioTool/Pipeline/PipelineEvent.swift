//
//  PipelineEvent.swift
//  ClearVoice
//
//  Pipeline streaming events and results
//

import Foundation
import ClearVoiceCore

// MARK: - Pipeline Events

/// Streaming pipeline events
public enum PipelineEvent: Sendable {
    /// VAD segment detected
    case segmentDetected(VADSegment)
    
    /// Analysis complete (VAD + Diarization)
    case analysisComplete(AnalysisResult)
    
    /// Audio segment enhanced
    case segmentEnhanced(AudioBuffer, timeRange: TimeRange)
    
    /// Speaker separated
    case speakerSeparated(speakerIndex: Int, audio: AudioBuffer)
    
    /// Transcription segment recognized
    case transcriptionSegment(TranscriptionSegment)
    
    /// Sound classified
    case classificationResult(SoundClassification)
    
    /// Pipeline stage completed
    case stageComplete(stage: String, duration: Duration)
    
    /// Progress update
    case progress(stage: String, percent: Double)
}

// MARK: - Analysis Result

/// Combined VAD + Diarization result
public struct AnalysisResult: Sendable {
    public let segments: [VADSegment]
    public let speakers: SpeakerTimeline
    
    public init(segments: [VADSegment], speakers: SpeakerTimeline) {
        self.segments = segments
        self.speakers = speakers
    }
    
    /// Filter to speech segments only
    public var speechSegments: [VADSegment] {
        segments.filter(\.isSpeech)
    }
    
    /// Filter to non-speech segments
    public var nonSpeechSegments: [VADSegment] {
        segments.filter { !$0.isSpeech }
    }
}

// MARK: - Pipeline Result

/// Complete pipeline execution result
public struct PipelineResult: Sendable {
    /// Processed audio (if pipeline includes audio output)
    public let audio: AudioBuffer?
    
    /// Separated speaker tracks (if separation was used)
    public let separatedTracks: [AudioBuffer]?
    
    /// Transcription result (if transcription was used)
    public let transcription: Transcription?
    
    /// Sound classifications (if classification was used)
    public let classifications: [SoundClassification]?
    
    /// Analysis result (VAD + Diarization)
    public let analysis: AnalysisResult?
    
    /// Execution metrics
    public let metrics: PipelineMetrics
    
    public init(
        audio: AudioBuffer? = nil,
        separatedTracks: [AudioBuffer]? = nil,
        transcription: Transcription? = nil,
        classifications: [SoundClassification]? = nil,
        analysis: AnalysisResult? = nil,
        metrics: PipelineMetrics = PipelineMetrics()
    ) {
        self.audio = audio
        self.separatedTracks = separatedTracks
        self.transcription = transcription
        self.classifications = classifications
        self.analysis = analysis
        self.metrics = metrics
    }
}

// MARK: - Pipeline Metrics

/// Execution timing and resource metrics
public struct PipelineMetrics: Sendable {
    /// Total execution time
    public var totalDuration: Duration
    
    /// Per-stage durations
    public var stageDurations: [String: Duration]
    
    /// Peak memory usage (bytes)
    public var peakMemoryUsage: Int
    
    public init(
        totalDuration: Duration = .zero,
        stageDurations: [String: Duration] = [:],
        peakMemoryUsage: Int = 0
    ) {
        self.totalDuration = totalDuration
        self.stageDurations = stageDurations
        self.peakMemoryUsage = peakMemoryUsage
    }
}

// MARK: - Pipeline Context

/// Context available during pipeline execution
public struct PipelineContext: Sendable {
    /// Analysis result (if analyze stage was run)
    public let analysis: AnalysisResult?
    
    /// Current processed audio
    public let currentAudio: AudioBuffer
    
    /// Original input audio (unchanged)
    public let originalAudio: AudioBuffer
    
    public init(
        analysis: AnalysisResult? = nil,
        currentAudio: AudioBuffer,
        originalAudio: AudioBuffer
    ) {
        self.analysis = analysis
        self.currentAudio = currentAudio
        self.originalAudio = originalAudio
    }
}
