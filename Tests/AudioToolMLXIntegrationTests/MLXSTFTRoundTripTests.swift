//
//  MLXSTFTRoundTripTests.swift
//  AudioToolMLXIntegrationTests
//
//  STFT/ISTFT reconstruction, including the winLength < nFFT case
//

import XCTest
import AudioToolTestSupport
import MLX
@testable import AudioToolCoreML

/// Hermetic: synthetic signal, no weights, no CoreML model.
final class MLXSTFTRoundTripTests: MLXTestCase {

    /// Reconstruction error of `signal` after a forward and inverse transform.
    private func roundTripError(
        nFFT: Int,
        hopLength: Int,
        winLength: Int
    ) -> Float {
        let sampleCount = 4096
        let signal = (0..<sampleCount).map { index -> Float in
            0.5 * sin(2 * .pi * 440 * Float(index) / 16000)
                + 0.2 * sin(2 * .pi * 1200 * Float(index) / 16000)
        }
        let input = MLXArray(signal).reshaped([1, sampleCount])
        let window = createPeriodicHannWindow(length: winLength)

        let (real, imaginary) = mlxSTFT(
            input,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: window
        )

        let reconstructed = mlxISTFT(
            realPart: real,
            imagPart: imaginary,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: window,
            audioLength: sampleCount
        )
        eval(reconstructed)

        let output = reconstructed[0].asArray(Float.self)
        XCTAssertEqual(output.count, sampleCount)

        // Ignore the first and last frame, where overlap-add normalization is
        // incomplete for any window.
        var worst: Float = 0
        for index in winLength..<(sampleCount - winLength) {
            worst = max(worst, abs(output[index] - signal[index]))
        }
        return worst
    }

    /// The configuration the GAN provider uses.
    func testRoundTripWithFullLengthWindow() {
        XCTAssertLessThan(roundTripError(nFFT: 400, hopLength: 100, winLength: 400), 1e-3)
    }

    /// The case the public API allows and the inverse transform could not perform:
    /// the forward transform zero-pads each `winLength` frame out to `nFFT`, and the
    /// inverse multiplied an `nFFT`-wide frame by a `winLength`-wide window, which
    /// cannot broadcast at all.
    func testRoundTripWithShortWindow() {
        XCTAssertLessThan(roundTripError(nFFT: 512, hopLength: 100, winLength: 400), 1e-3)
    }
}
