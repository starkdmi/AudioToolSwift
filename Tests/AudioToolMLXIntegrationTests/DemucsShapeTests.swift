//
//  DemucsShapeTests.swift
//  AudioToolMLXIntegrationTests
//
//  Determines the true rank of the Demucs model output
//

import XCTest
import AudioToolTestSupport
import AudioToolCore
import AudioToolMLX

/// The chunked and non-chunked Demucs paths disagree about the model's output shape:
/// `separateWithChunking` documents `(B, S=4, C=2, T)` and indexes a source, while
/// `separateChunk` averages over axis 0 with no indexing. Only one can be right, and
/// `maxDirectDuration` routes short and long audio to different ones.
///
/// Set AUDIOTOOL_DEMUCS_WEIGHTS to the directory holding <stem>.safetensors.
final class DemucsShapeTests: IntegrationTestCase {

    private var weightsDirectory: String? {
        guard let v = ProcessInfo.processInfo.environment["AUDIOTOOL_DEMUCS_WEIGHTS"],
              !v.isEmpty, FileManager.default.fileExists(atPath: v) else { return nil }
        return v
    }

    private func tone(seconds: Double, rate: Int = 44100) -> AudioBuffer {
        let n = Int(Double(rate) * seconds)
        let s = (0..<n).map { i in
            sin(2 * Float.pi * 220 * Float(i) / Float(rate)) * 0.3
                + sin(2 * Float.pi * 3000 * Float(i) / Float(rate)) * 0.15
        }
        return AudioBuffer(samples: s, sampleRate: rate, channels: 1)
    }

    /// Both routes must produce the same stem for the same input. If the non-chunked
    /// path is averaging four sources together, its output will differ wildly in
    /// energy from the chunked path's properly indexed stem.
    func testShortAndLongPathsAgree() async throws {
        guard let dir = weightsDirectory else {
            throw XCTSkip("set AUDIOTOOL_DEMUCS_WEIGHTS to the directory of <stem>.safetensors files")
        }
        let provider = DemucsProvider(weightsDirectory: dir)
        try await provider.load(stem: .vocals)

        // 4s -> non-chunked path (threshold is 7.8s)
        let short = try await provider.separate(tone(seconds: 4.0), stem: .vocals)
        // 9s -> chunked path
        let long = try await provider.separate(tone(seconds: 9.0), stem: .vocals)

        func rms(_ b: AudioBuffer) -> Float {
            guard !b.samples.isEmpty else { return 0 }
            return (b.samples.reduce(0) { $0 + $1 * $1 } / Float(b.samples.count)).squareRoot()
        }
        let shortRMS = rms(short), longRMS = rms(long)
        print("DEMUCS short(non-chunked) rms=\(shortRMS) count=\(short.samples.count)")
        print("DEMUCS long(chunked)      rms=\(longRMS) count=\(long.samples.count)")

        // Same synthetic content, so the same stem should have comparable energy.
        // A 4-source average would be markedly different.
        let ratio = max(shortRMS, longRMS) / max(min(shortRMS, longRMS), 1e-9)
        print("DEMUCS rms ratio=\(ratio)")
        XCTAssertLessThan(ratio, 3.0,
            "chunked and non-chunked paths produce very different output - they disagree about the model's output rank")

        // The same defect also doubled the length: reducing over the source axis left
        // the result stereo, so a mono buffer came back with 2x the samples and
        // therefore twice the duration at the declared rate.
        XCTAssertEqual(short.samples.count, 4 * 44100,
                       "non-chunked output length should match the input duration")
        XCTAssertEqual(short.duration, 4.0, accuracy: 0.01)
    }
}
