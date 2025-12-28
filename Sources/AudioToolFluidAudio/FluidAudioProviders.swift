//
//  FluidAudioProviders.swift
//  ClearVoiceFluidAudio
//
//  Factory for FluidAudio-based providers
//

import Foundation
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
    
    /// Create pyannote diarization provider
    /// - Parameter threshold: Speaker clustering threshold (default 0.7045655, pyannote community-1)
    /// Higher values = more speaker separation, lower values = more merging
    public static func pyannote(threshold: Float = 0.7045655) -> FluidAudioDiarizationProvider {
        FluidAudioDiarizationProvider(threshold: threshold)
    }
}

// MARK: - ClearVoiceFluidAudio Module Exports

/// Re-export key types for convenience
public typealias SileroVADProvider = FluidAudioVADProvider
public typealias ParakeetTranscriberProvider = FluidAudioTranscriber
public typealias PyannoteDiarizationProvider = FluidAudioDiarizationProvider
