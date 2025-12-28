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
    
    // Future providers:
    // public static func parakeetTranscriber() -> FluidAudioTranscriber
    // public static func diarization() -> FluidAudioDiarizationProvider
}

// MARK: - ClearVoiceFluidAudio Module Exports

/// Re-export key types for convenience
public typealias SileroVADProvider = FluidAudioVADProvider
