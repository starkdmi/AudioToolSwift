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
///   converted from a Python implementation, and the resampler that pipeline used is
///   part of what the model was validated against - so this is a correctness setting,
///   not a speed/quality dial. Do not "upgrade" a provider's choice on
///   signal-processing grounds; changing which resampler feeds a model changes that
///   model's output, and needs measuring against the reference.
///
///   That said, the references have now been read, and they agree. Every pipeline
///   under `Models/` resamples with a band-limited method - `scipy.signal.resample`
///   for MossFormer2 SE, `librosa.resample` for SR, `torchaudio.transforms.Resample`
///   for Demucs, `librosa.load` for USS, and AVAudioConverter's Mastering algorithm
///   at maximum quality in every standalone Swift generator. None of them is cubic
///   interpolation.
///
///   Measured on a 48 -> 16 kHz downsample of a signal carrying content to 22 kHz:
///
///   | Compared with Mastering/`.max` | Relative RMS difference |
///   | ------------------------------ | ----------------------- |
///   | ``balanced`` (Catmull-Rom)     | 131%                    |
///   | AVAudioConverter defaults      | 25%                     |
///   | ``balanced``, band-limited input | 0.56%                 |
///
///   The 131% figure is the aliased content exceeding the signal itself. The last row
///   is the same comparison on audio already under the target Nyquist, where there is
///   nothing to fold: aliasing is the entire difference, so this only matters when the
///   caller's audio is wider than the model's band.
///
///   Providers declare what they want via ``AudioProcessor/preferredResamplingQuality``
///   and the facade's edge conversion honours it, so the choice travels with the model
///   rather than being decided by whoever happens to call.
public enum ResamplingQuality: Sendable {
    /// Linear interpolation (fastest, lowest quality)
    case fast
    /// Cubic interpolation - Catmull-Rom.
    ///
    /// Excellent for upsampling and for rate changes on band-limited material. Pure
    /// interpolation with no anti-aliasing stage, so on a downsample it folds content
    /// above the new Nyquist back into the band. The "~84 dB SNR" figure carried over
    /// from AudioUtils describes the interpolator, not downsampling.
    case balanced
    /// AVAudioConverter with the Mastering algorithm at maximum quality.
    ///
    /// Band-limited, and the same request every standalone generator under `Models/`
    /// makes. Not bit-identical to soxr, scipy or torchaudio - the nearest analogue
    /// available on the platform.
    case high
}

// MARK: - Resampling Errors

public enum ResamplingError: Error, Sendable {
    case invalidFormat
    case invalidParameters(String)
    case conversionFailed
}

// MARK: - AudioBuffer Resampling Extension
