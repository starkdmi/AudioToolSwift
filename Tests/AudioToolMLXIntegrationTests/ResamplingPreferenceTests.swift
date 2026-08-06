//
//  ResamplingPreferenceTests.swift
//  AudioToolMLXIntegrationTests
//
//  Each provider must declare the resampler its reference uses. No weights needed.
//

import XCTest
import AudioToolCore
@testable import AudioToolMLX

/// Which resampler each model wants at the facade's edge.
///
/// `preferredResamplingQuality` defaults to `.balanced` - cubic interpolation, no
/// anti-aliasing - which is what the facade always did before the seam existed. That
/// default is deliberately unchanged, so a provider nobody has checked against its
/// reference keeps its current output rather than silently acquiring new behaviour.
///
/// The consequence is that declaring nothing is indistinguishable from having audited
/// and chosen cubic. For the models whose references *have* been read, this pins the
/// answer, so dropping a declaration shows up here instead of as a quiet change in
/// output the next time someone runs the benchmarks.
///
/// Not an `IntegrationTestCase`: constructing a provider does not load weights.
final class ResamplingPreferenceTests: XCTestCase {

    /// Every reference under `Models/` resamples with a band-limited method:
    ///
    /// | Provider          | Reference                                          |
    /// | ----------------- | -------------------------------------------------- |
    /// | MossFormer2 SE    | `scipy.signal.resample` (FFT)                       |
    /// | MossFormer2 SR    | `librosa.resample`, and AVAudioConverter Mastering  |
    /// | MossFormer2 SS    | AVAudioConverter Mastering, quality `.max`          |
    /// | Demucs            | `torchaudio.transforms.Resample` (windowed sinc)    |
    /// | USS               | `librosa.load`, and AVAudioConverter Mastering      |
    ///
    /// None of them is cubic interpolation. `.high` is AVAudioConverter Mastering at
    /// maximum quality, which matches the Swift references exactly and is the nearest
    /// available analogue to the Python ones - it is *not* bit-identical to soxr,
    /// scipy's FFT method or torchaudio's sinc. Establishing that requires a parity
    /// run against Python and is not what this test claims.
    func testAuditedProvidersRequestAntiAliasedResampling() {
        let audited: [(name: String, quality: ResamplingQuality)] = [
            ("MossFormer2SE48K", MossFormer2SE48KProvider(precision: .fp32).preferredResamplingQuality),
            ("FRCRNSE16K", FRCRNSE16KProvider(weightsPath: "/nonexistent").preferredResamplingQuality),
            ("MossFormer2SR48K", MossFormer2SR48KProvider(precision: .fp32).preferredResamplingQuality),
            ("MossFormer2SS", MossFormer2SSProvider(model: .twoSpeaker).preferredResamplingQuality),
            ("Demucs", DemucsProvider(weightsDirectory: "/nonexistent").preferredResamplingQuality),
        ]

        for (name, quality) in audited {
            XCTAssertEqual(quality, .high,
                           "\(name) must ask for anti-aliased resampling - its reference does, and the default does not")
        }
    }
}
