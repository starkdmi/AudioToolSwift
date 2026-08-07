//
//  ParityInputExportTests.swift
//  AudioToolParityTests
//
//  Exporting Swift-side intermediates for the Python reference to consume.
//

import AudioToolCore
@testable import AudioToolMLX
import AudioToolTestSupport
import AudioUtils
import MLX
import XCTest

/// Writes intermediates that the reference generator reads back, so a stage the
/// two languages cannot agree on stops being part of the comparison.
///
/// Only SR needs this today. Its first step is a 16 -> 48 kHz upsample, and Swift
/// does it with `AVAudioConverter` (Mastering, `.max`) while Python does it with
/// `librosa.resample` (`soxr_hq`). Those two agree at about 45 dB, and SR
/// amplifies the difference - reconstructing the band the upsample has to
/// preserve is the entire job. Comparing them end to end therefore measures the
/// resamplers, not the port, and no amount of fixing Swift closes that gap.
///
/// So the input is handed over instead of re-derived: Swift resamples, writes
/// float32 wav (lossless for the samples it holds), and
/// `Parity/adapters/mossformer2_sr.py` loads that file rather than calling
/// librosa. What remains in the comparison is the model.
///
/// Run when the fixture or the resampler changes:
///
/// ```bash
/// TEST_RUNNER_RUN_PARITY_TESTS=1 TEST_RUNNER_PARITY_EXPORT=1 xcodebuild test \
///   -scheme AudioToolSwift-Package -destination 'platform=macOS' \
///   -only-testing:AudioToolParityTests/ParityInputExportTests
/// ```
///
/// then regenerate the SR artifacts.
final class ParityInputExportTests: ParityTestCase {

    /// Skipped unless explicitly asked for - it writes into the source tree, which
    /// no ordinary test run should do.
    private var exportDirectory: URL {
        get throws {
            try XCTSkipUnless(
                ProcessInfo.processInfo.environment["PARITY_EXPORT"] == "1",
                "export step - set PARITY_EXPORT=1 to regenerate Parity/inputs"
            )
            var url = URL(fileURLWithPath: #filePath)
            for _ in 0..<3 { url.deleteLastPathComponent() }
            let directory = url.appendingPathComponent("Parity/inputs")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            return directory
        }
    }

    func testExportSuperResolutionUpsampledInput() throws {
        let directory = try exportDirectory
        let saver = AudioSaver(config: .init(sampleRate: 48_000))

        for (caseName, filename) in [
            ("mossformer2_sr_48k", "sr_upsampled_48k.wav"),
            ("mossformer2_sr_48k_direct", "sr_upsampled_48k_direct.wav"),
        ] {
            let artifact = try artifact(caseName)
            let input = try XCTUnwrap(artifact.tensor("input"))
            let upsampled = try SuperResolutionResampler.upsample(input, from: 16_000, to: 48_000)

            XCTAssertEqual(
                upsampled.count, input.count * 3,
                "\(caseName): integer-ratio upsample must be exact"
            )

            let url = directory.appendingPathComponent(filename)
            try saver.save(MLXArray(upsampled), to: url)
            print("EXPORT \(filename): \(upsampled.count) samples from \(input.count)")
        }
    }
}
