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

    /// Only the models with a *Swift* reference declare a preference. Those three
    /// have an unambiguous answer: their standalone generators ask AVAudioConverter
    /// for the Mastering algorithm at maximum quality, which is exactly what `.high`
    /// does, so the declaration reproduces behaviour rather than choosing it.
    ///
    /// | Provider        | Swift reference                                            |
    /// | --------------- | ---------------------------------------------------------- |
    /// | MossFormer2 SR  | `mossformer2_sr_mlx_swift/Sources/Generate/main.swift:29`   |
    /// | MossFormer2 SS  | `mosforrmer2_ss_mlx_swift/Sources/Generate/main.swift:155`  |
    /// | USS             | AVAudioConverter Mastering, quality `.max`                  |
    func testProvidersWithASwiftReferenceDeclareIt() {
        let declared: [(name: String, quality: ResamplingQuality)] = [
            ("MossFormer2SR48K", MossFormer2SR48KProvider(precision: .fp32).preferredResamplingQuality),
            ("MossFormer2SS", MossFormer2SSProvider(model: .twoSpeaker).preferredResamplingQuality),
        ]

        for (name, quality) in declared {
            XCTAssertEqual(quality, .high,
                           "\(name)'s standalone generator asks for AVAudioConverter Mastering at maximum quality; `.high` is how that is spelled here")
        }
    }

    /// The models whose only reference is Python must *not* declare one yet.
    ///
    /// MossFormer2 SE resamples with `scipy.signal.resample` and Demucs with
    /// `torchaudio.transforms.Resample`; FRCRN's reference never resamples at all.
    /// None of those is AVAudioConverter Mastering. They are all anti-aliased, but
    /// "also anti-aliased" is not "the same filter" - they differ in transition band,
    /// stopband depth and phase, and a separator's output is sensitive enough to
    /// input perturbation to make that matter.
    ///
    /// Declaring `.high` for them would substitute Apple's algorithm for the one the
    /// model was validated against, on the reasoning that both suppress aliasing -
    /// which is precisely what `ResamplingQuality`'s documentation says not to do.
    /// The right move is to measure against Python first, and possibly to implement
    /// the actual kernels rather than approximate them.
    ///
    /// This test exists so that "no declaration" reads as a deliberate open question
    /// rather than an oversight, and so that adding one is a decision someone has to
    /// make on purpose.
    func testProvidersWithOnlyAPythonReferenceStayUndeclared() {
        let undeclared: [(name: String, quality: ResamplingQuality)] = [
            ("MossFormer2SE48K", MossFormer2SE48KProvider(precision: .fp32).preferredResamplingQuality),
            ("FRCRNSE16K", FRCRNSE16KProvider(weightsPath: "/nonexistent").preferredResamplingQuality),
            ("Demucs", DemucsProvider(weightsDirectory: "/nonexistent").preferredResamplingQuality),
        ]

        for (name, quality) in undeclared {
            XCTAssertEqual(quality, .balanced,
                           "\(name) has only a Python reference (scipy/torchaudio, or none at all). Declaring a resampler here means claiming parity that has not been measured - do the measurement, then update this test with what it showed")
        }
    }
}
