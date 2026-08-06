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
///   The references have been read. They do not agree with each other, and the
///   deciding evidence is the *Swift* port in each case - that is the thing that was
///   validated against Python, kaldi filters, FFT conventions and the rest:
///
///   | Model            | Python reference          | Swift generator asks for      |
///   | ---------------- | ------------------------- | ----------------------------- |
///   | MossFormer2 SE   | `scipy.signal.resample`   | `AudioLoader` default: `.auto` |
///   | MossFormer2 SR   | `librosa.resample`        | Mastering / `.max`            |
///   | MossFormer2 SS   | -                         | Mastering / `.max`            |
///   | Demucs           | `torchaudio` Resample     | `AudioLoader` default: `.auto` |
///   | FRCRN            | none - 16 kHz in          | `AudioLoader` default: `.auto` |
///   | USS              | `librosa.load`            | Mastering / `.max`            |
///   | Chatterbox       | -                         | no resampling at all          |
///
///   Two settings, not one, and the split does not follow "which is better". A model
///   whose generator names Mastering gets ``high``; a model whose generator leaves
///   `AudioLoader` alone gets ``auto``, which is cubic upward and AVAudioConverter
///   `Normal` at `.medium` downward. SwiftAudio chose those downsampling settings
///   "for ML model compatibility (matches FluidAudio/pyannote training data
///   resampling)" - the training audio went through an ordinary resampler, so an
///   ordinary resampler is what reproduces it. Mastering would sound better and match
///   less.
///
///   The trap this table exists to close: every Python reference that resamples is
///   band-limited, so it is tempting to conclude they all want the best band-limited
///   option the platform offers. They do not. `scipy`'s FFT method, `torchaudio`'s
///   windowed sinc, librosa's soxr and AVAudioConverter's two algorithms are five
///   different filters differing in transition band, stopband and phase, and a
///   separator's output is sensitive enough to input perturbation that the difference
///   survives to the result. Reproducing the port beats approximating the paper.
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
    /// Band-limited. Requested explicitly by the MossFormer2 SR, MossFormer2 SS and
    /// USS generators. Not bit-identical to soxr, scipy or torchaudio - the nearest
    /// analogue available on the platform, and not automatically the right one.
    case high
    /// What `AudioLoader` does when nothing asks for anything else: cubic when
    /// upsampling, AVAudioConverter `Normal` when downsampling.
    ///
    /// Deliberately *not* the best-sounding option on the downsample side: `Normal`
    /// rather than `Mastering`, because the models were trained on audio that had
    /// been through an ordinary resampler, so reproducing an ordinary resampler is
    /// what matches them. Upsampling has nothing to alias, so cubic is used there.
    ///
    /// - Important: This must track the pinned SwiftAudio, not an idea of what it
    ///   should be. `AudioLoader` at 1.0.0 uses `Normal` at `.high`; an unreleased
    ///   change in the local checkout moves it to `.medium` "for ML model
    ///   compatibility (matches FluidAudio/pyannote training data resampling)". If
    ///   the dependency moves to 1.0.1, this moves with it -
    ///   `AudioLoaderParityTests` compares the two implementations directly and
    ///   fails if they part company.
    ///
    /// This is what any generator that constructs `AudioLoader` without naming a
    /// method gets, which makes it the validated behaviour for every model ported
    /// that way rather than a fallback.
    case auto
}

// MARK: - Resampling Errors

public enum ResamplingError: Error, Sendable {
    case invalidFormat
    case invalidParameters(String)
    case conversionFailed
}

// MARK: - AudioBuffer Resampling Extension
