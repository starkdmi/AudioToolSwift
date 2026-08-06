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

    /// The rest match `AudioLoader`'s `.auto`, because their generators construct the
    /// loader without naming a resampling method.
    ///
    /// | Provider        | Swift reference                                             |
    /// | --------------- | ----------------------------------------------------------- |
    /// | MossFormer2 SE  | `starkdmi/mossformer_se_mlx` swift/Sources/Generate           |
    /// | Demucs          | Demucs MLX generator, `targetSampleRate: 44100`              |
    /// | FRCRN           | FRCRN generator, `normalizationMode: .none`                  |
    ///
    /// `.auto` is cubic upward and AVAudioConverter `Normal` at `.medium` downward -
    /// deliberately not Mastering. SwiftAudio chose those settings "for ML model
    /// compatibility (matches FluidAudio/pyannote training data resampling)": the
    /// models were trained on audio that had been through an ordinary resampler, so
    /// an ordinary resampler is what reproduces them.
    ///
    /// This is the correction that matters most here. An earlier pass set these three
    /// to `.high` on the reasoning that their Python references (`scipy.signal
    /// .resample`, `torchaudio.transforms.Resample`) are anti-aliased and therefore
    /// Mastering is the closest match. Both halves were wrong: the Swift port is the
    /// thing that was validated, and it used the loader's ordinary path, so the
    /// "closest analogue" argument was answering a question that had already been
    /// settled by measurement.
    func testProvidersMatchingTheLoaderDefaultDeclareAuto() {
        let auto: [(name: String, quality: ResamplingQuality)] = [
            ("MossFormer2SE48K", MossFormer2SE48KProvider(precision: .fp32).preferredResamplingQuality),
            ("FRCRNSE16K", FRCRNSE16KProvider(weightsPath: "/nonexistent").preferredResamplingQuality),
            ("Demucs", DemucsProvider(weightsDirectory: "/nonexistent").preferredResamplingQuality),
        ]

        for (name, quality) in auto {
            XCTAssertEqual(quality, .auto,
                           "\(name)'s generator leaves AudioLoader at its default, so `.auto` is what it was validated with - `.high` would be a better-sounding guess, not a match")
        }
    }

    /// `.auto` is not a synonym for either of the others, in either direction.
    /// If it collapsed onto one of them the declarations above would be decorative.
    func testAutoDiffersFromBothFixedChoices() throws {
        let rate = 48000
        let samples = (0..<rate).map { i -> Float in
            let t = Float(i) / Float(rate)
            var sum: Float = 0
            for frequency in stride(from: Float(200), to: Float(22000), by: 700) {
                sum += sin(2 * Float.pi * frequency * t)
            }
            return sum / 32
        }
        let input = AudioBuffer(samples: samples, sampleRate: rate, channels: 1)

        func rms(_ values: [Float]) -> Float {
            sqrt(values.reduce(0) { $0 + $1 * $1 } / Float(max(values.count, 1)))
        }
        func difference(_ a: AudioBuffer, _ b: AudioBuffer) -> Float {
            let n = min(a.samples.count, b.samples.count)
            return rms((0..<n).map { a.samples[$0] - b.samples[$0] }) / max(rms(b.samples), 1e-9)
        }

        // Downsampling: `.auto` is Normal/.medium, distinct from both cubic and Mastering.
        let autoDown = try input.resampled(to: 16000, quality: .auto)
        let cubicDown = try input.resampled(to: 16000, quality: .balanced)
        let masteringDown = try input.resampled(to: 16000, quality: .high)

        XCTAssertGreaterThan(difference(autoDown, cubicDown), 0.1,
                             "`.auto` should not be cubic on the way down - that is the aliasing it exists to avoid")
        XCTAssertGreaterThan(difference(autoDown, masteringDown), 0.01,
                             "`.auto` should not be Mastering either; it is Normal at .medium")

        // Upsampling: `.auto` is cubic, so it should agree with `.balanced` exactly.
        let autoUp = try input.resampled(to: 96000, quality: .auto)
        let cubicUp = try input.resampled(to: 96000, quality: .balanced)
        XCTAssertEqual(autoUp.samples.count, cubicUp.samples.count)
        XCTAssertLessThan(difference(autoUp, cubicUp), 1e-6,
                          "`.auto` upsamples with cubic, so it should be identical to `.balanced` going up")
    }
}
