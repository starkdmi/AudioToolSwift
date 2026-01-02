//
//  FluidAudioDiarizationProvider.swift
//  ClearVoiceFluidAudio
//
//  Pyannote speaker diarization provider using FluidAudio
//

import Foundation
@preconcurrency import FluidAudio
import ClearVoiceCore

// MARK: - FluidAudio Diarization Provider

/// Pyannote speaker diarization provider using FluidAudio's OfflineDiarizerManager
/// Uses pyannote community-1 pipeline with VBx clustering for offline diarization
public actor FluidAudioDiarizationProvider: DiarizationProvider {
    
    // MARK: - AudioProcessor Conformance
    
    public nonisolated let sampleRate: Int = 16000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1
    
    // MARK: - Private Properties
    
    private var manager: OfflineDiarizerManager?
    private let threshold: Float
    
    // MARK: - Initialization
    
    /// Initialize pyannote diarization provider
    /// - Parameter threshold: Speaker clustering threshold (default 0.7045655, pyannote community-1)
    public init(threshold: Float = 0.7045655) {
        self.threshold = threshold
    }
    
    /// Load the diarization models (downloads if needed)
    public func load() async throws {
        // Use pyannote community-1 default threshold matching CLI
        let config = OfflineDiarizerConfig(
            clusteringThreshold: Double(threshold)
        )
        manager = OfflineDiarizerManager(config: config)
        
        // Use prepareModels() which includes prewarming (same as CLI path)
        try await manager?.prepareModels()
    }
    
    // MARK: - DiarizationProvider Conformance
    
    /// Identify speakers in audio (from URL - recommended)
    /// This uses the memory-mapped streaming API for efficiency
    public func diarize(url: URL) async throws -> SpeakerTimeline {
        guard let manager = manager else {
            throw ClearVoiceError.modelNotLoaded("FluidAudio Diarization")
        }
        
        let result = try await manager.process(url)
        
        // Convert FluidAudio segments to ClearVoice DiarizedSegment
        let segments = result.segments.map { segment in
            DiarizedSegment(
                timeRange: TimeRange(start: Double(segment.startTimeSeconds), end: Double(segment.endTimeSeconds)),
                speakerID: SpeakerID(segment.speakerId),
                confidence: 1.0
            )
        }
        return SpeakerTimeline(segments: segments)
    }
    
    /// Identify speakers in audio (from samples)
    /// - Parameter audio: Input audio buffer (16kHz mono expected)
    /// - Returns: Speaker timeline with labeled segments
    public func diarize(_ audio: AudioBuffer) async throws -> SpeakerTimeline {
        guard let manager = manager else {
            throw ClearVoiceError.modelNotLoaded("FluidAudio Diarization")
        }
        
        let result = try await manager.process(audio: audio.samples)
        
        // Convert FluidAudio segments to ClearVoice DiarizedSegment
        let segments = result.segments.map { segment in
            DiarizedSegment(
                timeRange: TimeRange(start: Double(segment.startTimeSeconds), end: Double(segment.endTimeSeconds)),
                speakerID: SpeakerID(segment.speakerId),
                confidence: 1.0
            )
        }
        return SpeakerTimeline(segments: segments)
    }
    
    /// Diarize with VAD hint for efficiency
    /// Note: FluidAudio performs its own VAD internally, so hint is informational only
    public func diarize(_ audio: AudioBuffer, vadHint: [VADSegment]) async throws -> SpeakerTimeline {
        // FluidAudio has built-in VAD, so we just call regular diarize
        return try await diarize(audio)
    }
    
    // MARK: - AudioProcessor Conformance
    
    /// Process audio (passthrough - diarization doesn't modify audio)
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        return input
    }
}
