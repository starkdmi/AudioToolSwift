//
//  ResamplingQuality.swift
//  AudioToolCore
//
//  How audio is converted between sample rates
//

import Foundation

/// Resampling quality.
///
/// - Important: These are not simply "worse" and "better". Each model here was
///   converted from a Python implementation, and several deliberately reproduce that
///   pipeline's resampling rather than the highest-fidelity option available - a
///   lower-quality resampler can produce *smaller* error against the reference when
///   it is what the model saw during training. `Models/Chatterbox` replicates scipy's
///   polyphase edge behaviour and librosa's silence trimming; `Models/USS` loads
///   through AVAudioConverter's Mastering algorithm at maximum quality;
///   `Models/MossFormer2SE` carries its own quality setting.
///
///   So do not "upgrade" the default here on signal-processing grounds. Anything that
///   changes which resampler feeds a model changes that model's output, and needs
///   measuring per model against the reference implementation - not reasoning from
///   first principles about anti-aliasing.
///
///   Providers declare what they want via ``AudioProcessor/preferredResamplingQuality``
///   and the facade's edge conversion honours it, so the choice travels with the model
///   rather than being decided by whoever happens to call.
public enum ResamplingQuality: Sendable {
    /// Linear interpolation (fastest, lowest quality)
    case fast
    /// Cubic interpolation - Catmull-Rom (excellent quality, ~84 dB SNR)
    case balanced
    /// AVAudioConverter (professional quality, anti-aliasing)
    case high
}

// MARK: - Resampling Errors

public enum ResamplingError: Error, Sendable {
    case invalidFormat
    case invalidParameters(String)
    case conversionFailed
}

// MARK: - AudioBuffer Resampling Extension
