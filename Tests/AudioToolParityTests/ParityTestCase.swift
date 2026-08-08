//
//  ParityTestCase.swift
//  AudioToolParityTests
//
//  Base class and the compare-or-record step every parity suite uses.
//

import AudioToolCore
import AudioToolTestSupport
import AudioUtils
import MLX
import XCTest

/// Skips the whole suite unless parity tests are opted into.
///
/// Two gates, both of which must pass, matching `TestGate`'s existing shape:
/// `RUN_PARITY_TESTS=1`, and the specific artifact being present on disk.
///
/// ```bash
/// swift test                                    # hermetic, no weights, no network
/// RUN_PARITY_TESTS=1 PARITY_RECORD=1 swift test # measure, do not assert
/// RUN_PARITY_TESTS=1 swift test                 # hold the ports to the thresholds
/// ```
class ParityTestCase: XCTestCase {

    /// Whether this suite needs MLX and a Metal device. The CoreML GAN case still
    /// does - its STFT runs through MLX on both sides.
    class var requiresMLX: Bool { true }

    override func setUpWithError() throws {
        try super.setUpWithError()
        try XCTSkipUnless(TestGate.runParityTests, TestGate.parityDisabled)
        if Self.requiresMLX {
            try XCTSkipUnless(!TestGate.skipMLXTests, TestGate.mlxDisabled)
        }
    }

    /// A generated parity case, or a skip.
    func artifact(_ name: String) throws -> ParityArtifact {
        guard let artifact = try TestGate.parityArtifact(name) else {
            throw XCTSkip(TestGate.missingParityArtifact(name))
        }
        return artifact
    }

    /// A file in the sibling research checkout, or a skip.
    func reference(_ relativePath: String) throws -> URL {
        guard let url = TestGate.reference(relativePath) else {
            throw XCTSkip(TestGate.missingReference(relativePath))
        }
        return url
    }

    /// A staged weights file, or a skip.
    ///
    /// Always an explicit path, never the HuggingFace cache: the cache lives under
    /// `~/Documents`, which macOS protects, and the test runner cannot read it.
    func stagedWeights(_ repo: String, _ file: String) throws -> URL {
        guard let url = TestGate.parityWeights(repo, file) else {
            throw XCTSkip(TestGate.missingParityWeights(repo, file))
        }
        return url
    }

    /// Write both sides to wav so a disagreement can be listened to, not just read.
    ///
    /// Off unless `PARITY_DUMP_DIR` is set. A number tells you two signals differ;
    /// it does not tell you whether one of them is broken or whether they are two
    /// defensible answers - and for audio, that is usually a question for the ears.
    /// 41 dB and -1 dB are both "fails the assertion", and only one of them sounds
    /// like anything.
    private func dumpForListening(
        _ candidate: [Float], reference: [Float], label: String, artifact: ParityArtifact
    ) {
        guard let directory = ProcessInfo.processInfo.environment["PARITY_DUMP_DIR"] else { return }
        let base = URL(fileURLWithPath: directory)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)

        let saver = AudioSaver(config: .init(sampleRate: Double(artifact.sampleRate)))
        for (suffix, samples) in [("swift", candidate), ("reference", reference)] {
            let url = base.appendingPathComponent("\(label).\(suffix).wav")
            do {
                try saver.save(MLXArray(samples), to: url)
            } catch {
                print("PARITY dump failed for \(url.lastPathComponent): \(error)")
            }
        }
        if let input = artifact.tensor("input"), artifact.shape("input")?.count == 1 {
            try? saver.save(
                MLXArray(input), to: base.appendingPathComponent("\(artifact.name).input.wav")
            )
        }
    }

    /// Compare one Swift output against one recorded reference tensor.
    ///
    /// Length is checked first and never papered over: several models
    /// legitimately return fewer samples than they were given, but a wrapper that
    /// pads back to the input length is a real difference, and trimming to the
    /// shorter array before measuring would hide it.
    ///
    /// Under `PARITY_RECORD=1` this reports instead of asserting, because the
    /// first run against a new case has nothing to assert against and a threshold
    /// invented before measuring is not a threshold.
    func expectParity(
        _ candidate: [Float],
        matches tensorName: String,
        in artifact: ParityArtifact,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        guard let reference = artifact.tensor(tensorName) else {
            XCTFail(
                "artifact '\(artifact.name)' has no tensor '\(tensorName)'; it has \(artifact.tensorNames)",
                file: file, line: line
            )
            return
        }

        let label = "\(artifact.name).\(tensorName)"
        dumpForListening(candidate, reference: reference, label: label, artifact: artifact)

        guard reference.count == candidate.count else {
            // Report the overlap too. A length difference and a content difference
            // are separate questions, and stopping at the first hides the second:
            // a wrapper that pads back to the input length is a different animal
            // from one that computes different samples, and only the prefix SNR
            // tells them apart.
            let overlap = min(reference.count, candidate.count)
            let prefixSNR = ParityMetrics.snrDB(
                reference: Array(reference.prefix(overlap)),
                candidate: Array(candidate.prefix(overlap))
            )
            let rendered = ParityMetrics.renderSNR(prefixSNR)
            let message = """
                \(label): length mismatch - \
                \(ParityMetrics.lengthReport(reference: reference, candidate: candidate)); \
                first \(overlap) samples agree at \(rendered)
                """
            if TestGate.recordParityBaselines {
                print("PARITY \(message)")
                return
            }
            XCTFail(message, file: file, line: line)
            return
        }

        let snr = ParityMetrics.snrDB(reference: reference, candidate: candidate)
        let worst = ParityMetrics.maxAbsDiff(reference: reference, candidate: candidate)

        if TestGate.recordParityBaselines {
            let rendered = ParityMetrics.renderSNR(snr)
            print(String(
                format: "PARITY %-52@ SNR %@  maxabs %.3e  n=%d",
                label as NSString, rendered as NSString, worst, reference.count
            ))
            return
        }

        guard let threshold = ParityThresholds.minimumSNR[label] else {
            XCTFail(
                """
                \(label): no recorded threshold. Measured \(String(format: "%.1f", snr)) dB. \
                Run with PARITY_RECORD=1, then add it to ParityThresholds.
                """,
                file: file, line: line
            )
            return
        }

        XCTAssertGreaterThan(
            snr, threshold,
            String(
                format: "%@: %.1f dB against the MLX Python reference, below the recorded %.1f dB (max abs diff %.3e)",
                label, snr, threshold, worst
            ),
            file: file, line: line
        )
    }
}
