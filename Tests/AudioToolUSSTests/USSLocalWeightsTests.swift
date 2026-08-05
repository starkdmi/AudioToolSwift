//
//  USSLocalWeightsTests.swift
//  AudioToolUSSTests
//
//  USS runs from a local weights file, without a published repo
//

import XCTest
import AudioToolCore
import AudioToolUSS

final class USSLocalWeightsTests: XCTestCase {

    /// Proves the explicit-path escape hatch works end to end: load real weights from
    /// disk and separate, with no HuggingFace access. Skips when the env var is unset.
    func testSeparatesUsingLocalWeights() async throws {
        guard let provider = USSTestWeights.provider() else {
            throw XCTSkip(USSTestWeights.skipReason)
        }

        try await provider.load()

        let rate = 32000
        let samples = (0..<(rate * 2)).map { i in
            sin(2 * Float.pi * 440 * Float(i) / Float(rate)) * 0.3
        }
        let input = AudioBuffer(samples: samples, sampleRate: rate, channels: 1)

        let output = try await provider.separateSound(input, target: .speech)

        XCTAssertEqual(output.sampleRate, rate)
        XCTAssertGreaterThan(output.samples.count, 0)
        XCTAssertFalse(output.samples.allSatisfy { $0 == 0 }, "separation returned silence")
    }
}
