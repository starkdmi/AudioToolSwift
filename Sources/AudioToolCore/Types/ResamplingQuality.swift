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
///   The references have been read, and they do not agree with each other:
///
///   | Model            | Reference resampler                                    |
///   | ---------------- | ------------------------------------------------------ |
///   | MossFormer2 SE   | `scipy.signal.resample` (FFT)                           |
///   | MossFormer2 SR   | `librosa.resample`; Swift generator: Mastering/`.max`   |
///   | MossFormer2 SS   | Swift generator: Mastering/`.max`                       |
///   | Demucs           | `torchaudio.transforms.Resample` (windowed sinc)        |
///   | USS              | `librosa.load`; Swift generator: Mastering/`.max`       |
///   | Chatterbox       | none - loads without resampling                         |
///
///   All of the resampling ones are band-limited, and none of them is cubic. It does
///   not follow that they are interchangeable. An FFT method, a windowed sinc and
///   AVAudioConverter's Mastering algorithm differ in transition band, stopband depth
///   and phase, and a separator's output is sensitive enough to input perturbation
///   that those differences survive to the result. "Also anti-aliased" is not "the
///   same filter", and picking the platform's best-sounding option because it is the
///   best-sounding option is the substitution this note exists to prevent.
///
///   So ``high`` is declared only where a *Swift* reference asked for exactly it -
///   MossFormer2 SR, MossFormer2 SS and USS, whose standalone generators request
///   Mastering at maximum quality. The models whose only reference is Python are left
///   on the default until someone measures; matching scipy or torchaudio properly may
///   mean implementing those kernels rather than approximating them with a fourth.
///
///   For scale, on a 48 -> 16 kHz downsample of a signal carrying content to 22 kHz:
///
///   | Compared with Mastering/`.max`   | Relative RMS difference |
///   | -------------------------------- | ----------------------- |
///   | ``balanced`` (Catmull-Rom)       | 131%                    |
///   | AVAudioConverter defaults        | 25%                     |
///   | ``balanced``, band-limited input | 0.56%                   |
///
///   The 131% is aliased content exceeding the signal itself. The last row is the same
///   comparison on audio already under the target Nyquist, where there is nothing to
///   fold: aliasing is the whole difference, so the choice only bites when the
///   caller's audio is wider than the model's band. Within the band the resamplers
///   agree closely and introduce no relative group delay.
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
