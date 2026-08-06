//
//  DemucsStereoTests.swift
//  AudioToolMLXIntegrationTests
//
//  Demucs stereo handling against real weights.
//

import XCTest
import MLX
import AudioToolCore
import AudioToolTestSupport
@testable import AudioToolMLX

/// Stereo separation with the real vocals weights.
///
/// `DemucsChannelLayoutTests` covers the deinterleaving arithmetic without a model.
/// This covers what that arithmetic is *for*: that the model actually receives a
/// left/right pair, and that the answer does not depend on how long the audio is.
///
/// The invariant both tests turn on is dual mono. A stereo buffer whose channels are
/// identical carries exactly the information of the mono buffer it was built from,
/// and `stereoPair` maps mono to two identical channels - so separating either must
/// give the same samples. Under the old `reshaped([2, -1])` a dual-mono buffer became
/// the first half of the recording against the second half, which is not dual mono
/// and does not separate to the same thing.
final class DemucsStereoTests: IntegrationTestCase {

    private static let weightsPath = "Models/demucs_mlx_swift/Weights"

    /// 44.1 kHz, the model's rate. Deterministic, and structured enough that the
    /// separation has something to do rather than returning near-silence.
    private func signal(seconds: Double) -> [Float] {
        let count = Int(44100 * seconds)
        return (0..<count).map { i in
            let t = Float(i) / 44100
            let voiceLike = sin(2 * .pi * 220 * t) * 0.4 + sin(2 * .pi * 440 * t) * 0.2
            let bassLike = sin(2 * .pi * 80 * t) * 0.3
            let envelope = 0.5 + 0.5 * sin(2 * .pi * 2 * t)
            return (voiceLike * envelope + bassLike) * 0.5
        }
    }

    private func interleaveDualMono(_ mono: [Float]) -> [Float] {
        var interleaved: [Float] = []
        interleaved.reserveCapacity(mono.count * 2)
        for sample in mono {
            interleaved.append(sample)
            interleaved.append(sample)
        }
        return interleaved
    }

    private func maxAbsoluteDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
    }

    private func loadedProvider() async throws -> DemucsProvider {
        let weights = try reference(Self.weightsPath)
        let provider = DemucsProvider(weightsDirectory: weights.path)
        try await provider.load(stem: .vocals)
        return provider
    }

    /// Short enough for the direct path.
    func testDualMonoStereoMatchesMonoDirect() async throws {
        let provider = try await loadedProvider()
        let mono = signal(seconds: 2.0)

        let fromMono = try await provider.separate(
            AudioBuffer(samples: mono, sampleRate: 44100, channels: 1), stem: .vocals)
        let fromStereo = try await provider.separate(
            AudioBuffer(samples: interleaveDualMono(mono), sampleRate: 44100, channels: 2), stem: .vocals)

        XCTAssertEqual(fromStereo.samples.count, fromMono.samples.count,
                       "stereo input produced a different number of frames than the mono it was built from")
        XCTAssertLessThan(maxAbsoluteDifference(fromStereo.samples, fromMono.samples), 1e-4,
                          "dual-mono stereo must separate to the same samples as mono")
        XCTAssertGreaterThan(fromMono.samples.map { abs($0) }.max() ?? 0, 0.001,
                             "separation returned near-silence; the comparison would be vacuous")
    }

    /// Longer than `maxDirectDuration` (7.8 s), so this goes through the chunked
    /// path - which used to flatten stereo to mono and duplicate it, making long
    /// audio behave differently from short audio.
    func testDualMonoStereoMatchesMonoChunked() async throws {
        let provider = try await loadedProvider()
        let mono = signal(seconds: 9.0)

        let fromMono = try await provider.separate(
            AudioBuffer(samples: mono, sampleRate: 44100, channels: 1), stem: .vocals)
        let fromStereo = try await provider.separate(
            AudioBuffer(samples: interleaveDualMono(mono), sampleRate: 44100, channels: 2), stem: .vocals)

        XCTAssertEqual(fromStereo.samples.count, mono.count,
                       "chunked stereo should produce one output frame per input frame")
        XCTAssertEqual(fromMono.samples.count, mono.count)
        XCTAssertLessThan(maxAbsoluteDifference(fromStereo.samples, fromMono.samples), 1e-4,
                          "dual-mono stereo must separate to the same samples as mono on the chunked path too")
    }

    /// Duration is measured in frames, not samples. Using the flat sample count made
    /// a stereo buffer look twice as long as it is, so a 5-second stereo clip took
    /// the chunked path while the identical mono clip took the direct one.
    func testStereoDurationIsMeasuredInFrames() async throws {
        let provider = try await loadedProvider()

        // 5 s: direct for mono, and must also be direct for stereo. If frames and
        // samples were confused this would be treated as 10 s and chunked, which
        // shows up as a different result for the same content.
        let mono = signal(seconds: 5.0)

        let fromMono = try await provider.separate(
            AudioBuffer(samples: mono, sampleRate: 44100, channels: 1), stem: .vocals)
        let fromStereo = try await provider.separate(
            AudioBuffer(samples: interleaveDualMono(mono), sampleRate: 44100, channels: 2), stem: .vocals)

        XCTAssertLessThan(maxAbsoluteDifference(fromStereo.samples, fromMono.samples), 1e-4,
                          "a 5s stereo clip took a different path than the same 5s of mono")
    }

    /// The point of all of it: genuinely different channels must reach the model as
    /// different channels. If left and right carry different content, the result
    /// cannot equal what either channel alone produces.
    ///
    /// Note that this one passes under the old `reshaped([2, -1])` too, and should:
    /// a misread of the buffer still varies with the right channel's contents. It
    /// establishes that stereo reaches the model at all; the dual-mono tests above
    /// are what establish that it arrives correctly.
    func testDifferingChannelsAffectTheResult() async throws {
        let provider = try await loadedProvider()

        let left = signal(seconds: 2.0)
        let right = left.enumerated().map { index, sample in
            // Phase-inverted and delayed: same energy, different stereo image.
            index >= 441 ? -left[index - 441] : -sample
        }

        var interleaved: [Float] = []
        interleaved.reserveCapacity(left.count * 2)
        for i in 0..<left.count {
            interleaved.append(left[i])
            interleaved.append(right[i])
        }

        let fromTrueStereo = try await provider.separate(
            AudioBuffer(samples: interleaved, sampleRate: 44100, channels: 2), stem: .vocals)
        let fromLeftOnly = try await provider.separate(
            AudioBuffer(samples: left, sampleRate: 44100, channels: 1), stem: .vocals)

        XCTAssertEqual(fromTrueStereo.samples.count, fromLeftOnly.samples.count)

        // Relative to the output's own scale. An absolute threshold would be
        // meaningless here: the vocals stem of a synthetic tone mix is quiet, so a
        // small absolute difference can still be most of the signal.
        let difference = maxAbsoluteDifference(fromTrueStereo.samples, fromLeftOnly.samples)
        let scale = max(fromLeftOnly.samples.map { abs($0) }.max() ?? 0, 1e-9)
        XCTAssertGreaterThan(difference / scale, 0.1,
                             "the right channel changed the result by less than 10% of its own peak - stereo information is not reaching the model")
    }
}
