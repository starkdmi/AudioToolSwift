//
//  ResamplerEffectTests.swift
//  AudioToolMLXIntegrationTests
//
//  How much the resampler choice changes a real model's output.
//

import XCTest
import AudioToolCore
import AudioToolTestSupport
@testable import AudioToolMLX

/// The resampler preference is not a cosmetic setting.
///
/// `ResamplingPreferenceTests` checks that each audited provider declares `.high`;
/// this checks that the declaration is worth making, by running both choices through
/// real weights and comparing what comes out.
///
/// The measurement matters for a practical reason: it means any benchmark numbers
/// gathered before the seam was populated describe a different pipeline than the one
/// that ships. Parity against the Python reference still has to be established
/// separately - matching *an* anti-aliased resampler is not the same as matching
/// torchaudio's.
final class ResamplerEffectTests: IntegrationTestCase {

    private func rms(_ samples: [Float]) -> Float {
        sqrt(samples.reduce(0) { $0 + $1 * $1 } / Float(max(samples.count, 1)))
    }

    /// 48 kHz source carrying content above Demucs' 22.05 kHz Nyquist, so the
    /// downsample to 44.1 kHz has something to alias.
    private func wideband(seconds: Int) -> AudioBuffer {
        let count = 48000 * seconds
        let samples = (0..<count).map { i -> Float in
            let t = Float(i) / 48000
            var sum: Float = 0
            for frequency in stride(from: Float(150), to: Float(23500), by: 450) {
                sum += sin(2 * .pi * frequency * t)
            }
            return sum / 52
        }
        return AudioBuffer(samples: samples, sampleRate: 48000, channels: 1)
    }

    func testResamplerChoiceChangesDemucsOutput() async throws {
        let weights = try reference("Models/demucs_mlx_swift/Weights")
        let provider = DemucsProvider(weightsDirectory: weights.path)
        try await provider.load(stem: .vocals)

        let input = wideband(seconds: 3)
        let viaCubic = try input.resampled(to: 44100, quality: .balanced)
        let viaHigh = try input.resampled(to: 44100, quality: .high)

        // Demucs declares `inputChannels = 2` and `validateInputChannels` enforces it,
        // so a mono buffer is rejected before the model sees it. Callers do not hit
        // that because `AudioEngine.process` adapts the layout first
        // (`AudioEngine.swift:124`); this test drives the provider directly and so has
        // to do the same thing the engine would.
        //
        // Resample first, then upmix - that is the engine's order, and it keeps this
        // measuring the resampler rather than the resampler plus a channel copy.
        let fromCubic = try await provider.separate(
            try viaCubic.converted(toChannels: provider.inputChannels), stem: .vocals
        )
        let fromHigh = try await provider.separate(
            try viaHigh.converted(toChannels: provider.inputChannels), stem: .vocals
        )

        let count = min(fromCubic.samples.count, fromHigh.samples.count)
        let difference = rms((0..<count).map { fromCubic.samples[$0] - fromHigh.samples[$0] })
        let scale = max(rms(fromHigh.samples), 1e-9)

        print(String(format: "resampler choice moves Demucs vocals by %.1f%% relative RMS",
                     100 * difference / scale))

        // The threshold is far below the measured figure: the claim is "this is a
        // first-order effect", not a pinned number, since the exact value depends on
        // the signal. It is printed rather than asserted for the same reason.
        XCTAssertGreaterThan(difference / scale, 0.2,
                             "the resampler choice barely moved the model's output (\(difference / scale)) - either the preference is not reaching the edge conversion, or `.high` has stopped anti-aliasing")
    }

    /// There is deliberately no "band-limited input barely matters" counterpart here.
    /// It is true of the audio - `AntiAliasingTests` measures the two resamplers
    /// agreeing to 0.56% RMS in band, with zero group delay between them - but it is
    /// not true of what comes out of the model. Separating a synthetic tone mix for
    /// vocals yields a residual around 4% of the input, and a 0.56% perturbation of
    /// that input moves the small residual by most of its own size. The model
    /// amplifies; the control belongs at the signal level, where it is a statement
    /// about resampling rather than about conditioning.
}
