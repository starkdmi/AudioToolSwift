//
//  LSTMKernelTests.swift
//  AudioToolMLXIntegrationTests
//
//  Kokoro's fused LSTM Metal kernel against the same step in plain MLX ops
//

import XCTest
import AudioToolTestSupport
import MLX
import MLXRandom
import KokoroSwift

/// Hermetic: synthetic inputs, no weights, no network.
///
/// This lived in the shipped library as `KokoroSwift.LSTMKernelTest`, a public
/// enum that printed its findings to stdout and returned a `Bool` nobody checked.
/// The kernel it verifies is real and still in use; the verification just had no
/// way to fail a build from there.
final class LSTMKernelTests: MLXTestCase {

    /// One LSTM step in plain MLX ops - the ground truth the kernel must match.
    private func standardLSTMStep(ifgo: MLXArray, cell: MLXArray) -> (MLXArray, MLXArray) {
        let gates = MLX.split(ifgo, parts: 4, axis: -1)
        let inputGate = MLX.sigmoid(gates[0])
        let forgetGate = MLX.sigmoid(gates[1])
        let candidate = MLX.tanh(gates[2])
        let outputGate = MLX.sigmoid(gates[3])

        let newCell = forgetGate * cell + inputGate * candidate
        let newHidden = outputGate * MLX.tanh(newCell)

        return (newCell, newHidden)
    }

    func testFusedKernelMatchesStandardOps() throws {
        // Shapes Kokoro actually runs through the kernel: single-utterance
        // inference at both hidden widths, plus small batches.
        let shapes: [(batchSize: Int, hiddenSize: Int)] = [
            (1, 128),
            (1, 256),
            (2, 128),
            (4, 64),
        ]

        for (batchSize, hiddenSize) in shapes {
            MLXRandom.seed(42)
            let ifgo = MLXRandom.uniform(
                low: -2.0,
                high: 2.0,
                [batchSize, 4 * hiddenSize],
                dtype: .float32
            )
            let cell = MLXRandom.uniform(
                low: -1.0,
                high: 1.0,
                [batchSize, hiddenSize],
                dtype: .float32
            )
            ifgo.eval()
            cell.eval()

            let (expectedCell, expectedHidden) = standardLSTMStep(ifgo: ifgo, cell: cell)
            expectedCell.eval()
            expectedHidden.eval()

            // nil means the kernel declined and the caller falls back to the ops
            // above, so there is nothing to compare - but the fallback is a
            // performance path, not a correctness one, and it silently taking over
            // everywhere is worth knowing about.
            guard let (kernelCell, kernelHidden) = LSTMKernels.fusedLSTMStep(ifgo: ifgo, cell: cell) else {
                throw XCTSkip("fused LSTM kernel unavailable at batch=\(batchSize), hidden=\(hiddenSize)")
            }
            kernelCell.eval()
            kernelHidden.eval()

            let cellMaxDiff: Float = abs(expectedCell - kernelCell).max().item()
            let hiddenMaxDiff: Float = abs(expectedHidden - kernelHidden).max().item()

            // float32 throughout, so agreement should be near-exact.
            let tolerance: Float = 1e-5
            XCTAssertLessThan(
                cellMaxDiff,
                tolerance,
                "cell state diverges at batch=\(batchSize), hidden=\(hiddenSize)"
            )
            XCTAssertLessThan(
                hiddenMaxDiff,
                tolerance,
                "hidden state diverges at batch=\(batchSize), hidden=\(hiddenSize)"
            )
        }
    }
}
