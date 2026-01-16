//
//  FluidAudioProviders.swift
//  ClearVoiceFluidAudio
//
//  Factory for FluidAudio-based providers
//

import Foundation
@preconcurrency import FluidAudio
import ClearVoiceCore

// MARK: - FluidAudio Providers Factory

/// Factory for creating FluidAudio-based providers
public struct FluidAudioProviders {
    
    /// Create Silero VAD provider
    /// - Parameters:
    ///   - threshold: Speech probability threshold (0.0-1.0, default 0.5, lower = more sensitive)
    ///   - minSpeechDuration: Minimum speech segment duration in seconds (default 0.25)
    ///   - minSilenceDuration: Minimum silence to split segments in seconds (default 0.4)
    /// - Returns: VAD provider ready for loading
    public static func sileroVAD(
        threshold: Float = 0.5,
        minSpeechDuration: Double = 0.25,
        minSilenceDuration: Double = 0.4
    ) -> FluidAudioVADProvider {
        FluidAudioVADProvider(
            threshold: threshold,
            minSpeechDuration: minSpeechDuration,
            minSilenceDuration: minSilenceDuration
        )
    }
    
    /// Create Parakeet transcription provider
    /// - Parameter version: Model version (.v3 for multilingual, .v2 for English-only faster)
    /// - Returns: Transcriber ready for loading
    public static func parakeetTranscriber(version: ParakeetVersion = .v3) -> FluidAudioTranscriber {
        FluidAudioTranscriber(version: version)
    }
    
    // MARK: - Diarization Providers
    
    /// Create Pyannote diarization provider (multi-stage pipeline)
    /// - Parameter threshold: Speaker clustering threshold (default 0.7045655, pyannote community-1)
    ///   Higher values = more speaker separation, lower values = more merging
    ///
    /// Uses Pyannote community-1 pipeline: powerset segmentation → WeSpeaker embeddings → VBx clustering
    ///
    /// **Best for:**
    /// - Scenarios with >4 speakers
    /// - Non-English audio
    /// - When you need configurable clustering threshold
    ///
    /// **Consider Sortformer for:**
    /// - Real-time/streaming use cases
    /// - ≤4 speakers scenarios
    /// - Better overlapping speech handling
    public static func pyannote(threshold: Float = 0.7045655) -> FluidAudioDiarizationProvider {
        FluidAudioDiarizationProvider(threshold: threshold)
    }
    
    /// Create Sortformer diarization provider (end-to-end neural model)
    /// - Parameter config: Sortformer configuration (default for low latency)
    ///
    /// Uses NVIDIA's Sortformer: single end-to-end neural network for speaker diarization
    ///
    /// **Best for:**
    /// - Real-time/streaming applications
    /// - ≤4 speakers scenarios
    /// - Overlapping speech handling
    /// - Maximum speed (~120x RTF)
    ///
    /// **Consider Pyannote for:**
    /// - >4 speakers scenarios
    /// - Non-English audio
    /// - Fine-tuned clustering control
    public static func sortformer(config: SortformerConfig = .default) -> FluidAudioSortformerProvider {
        FluidAudioSortformerProvider(config: config)
    }
}

// MARK: - ClearVoiceFluidAudio Module Exports

/// Re-export key types for convenience
public typealias SileroVADProvider = FluidAudioVADProvider
public typealias ParakeetTranscriberProvider = FluidAudioTranscriber
public typealias PyannoteDiarizationProvider = FluidAudioDiarizationProvider
public typealias SortformerDiarizationProvider = FluidAudioSortformerProvider
