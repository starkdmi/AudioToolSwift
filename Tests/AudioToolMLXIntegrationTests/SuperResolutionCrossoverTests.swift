//
//  SuperResolutionCrossoverTests.swift
//  AudioToolMLXIntegrationTests
//
//  The crossover the SR provider substitutes at, detected without holding the
//  signal it is detected from.
//

import AudioToolCore
@testable import AudioToolMLX
import AudioUtils
import MLX
import XCTest

/// `detectCrossover` streams `detectBandwidth`'s reduction instead of calling it.
///
/// That is only legitimate if it lands on the same answer, and the answer is a
/// single frequency that decides where the upsampled original hands over to the
/// model's reconstruction - so being one 187.5 Hz bin out is not a rounding
/// difference, it moves real audio between two sources. These tests hold the
/// streamed accumulation against whole-signal `detectBandwidth` directly rather
/// than inferring it from an end-to-end parity number.
///
/// No model weights: `detectCrossover` never touches the model, which is why
/// these run unconditionally instead of behind the parity gate.
final class SuperResolutionCrossoverTests: XCTestCase {

    private let inputRate = 16_000
    private let outputRate = 48_000

    /// What `detectBandwidth` returns for the same signal the provider will see.
    private func wholeSignalFHigh(_ samples: [Float]) throws -> Float {
        let upsampled = try SuperResolutionResampler.upsample(
            samples, from: inputRate, to: outputRate
        )
        let array = MLXArray(upsampled)
        eval(array)
        return detectBandwidth(array, fs: outputRate).1
    }

    private func assertAgrees(
        _ samples: [Float],
        _ label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let provider = MossFormer2SR48KProvider(weightsPath: "/dev/null", configPath: "/dev/null")
        let crossover = try await provider.detectCrossover(
            AudioBuffer(samples: samples, sampleRate: inputRate, channels: 1)
        )
        let expected = try wholeSignalFHigh(samples)
        XCTAssertEqual(
            crossover.fHigh, expected,
            "\(label): streamed detection says \(crossover.fHigh) Hz, whole-signal says \(expected) Hz",
            file: file, line: line
        )
    }

    private func tone(_ hz: Float, seconds: Double, amplitude: Float = 0.9) -> [Float] {
        let count = Int(Double(inputRate) * seconds)
        return (0..<count).map { amplitude * sin(2 * .pi * hz * Float($0) / Float(inputRate)) }
    }

    /// Speech-like content, where the crossover lands in the kilohertz range.
    func testAgreesOnSpeechLikeContent() async throws {
        let count = inputRate * 9
        var samples = [Float](repeating: 0, count: count)
        var seed: UInt64 = 0x9E3779B97F4A7C15
        for i in 0..<count {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            let noise = Float(Int64(bitPattern: seed &>> 11)) / Float(1 << 53)
            let t = Float(i) / Float(inputRate)
            let env = 0.5 + 0.5 * sin(2 * .pi * 3 * t)
            samples[i] = env * (0.3 * sin(2 * .pi * 300 * t)
                + 0.2 * sin(2 * .pi * 1800 * t) + 0.1 * noise)
        }
        try await assertAgrees(samples, "speech-like")
    }

    /// Bass-heavy content, which drives the crossover to the bottom bins. Not a
    /// contrived case: a loud 50 Hz hum reaches 187.5 Hz at full scale.
    func testAgreesOnBassHeavyContent() async throws {
        try await assertAgrees(tone(50, seconds: 7), "50 Hz hum")
        try await assertAgrees(tone(100, seconds: 7), "100 Hz tone")
    }

    /// Silence, which `detectBandwidth` answers with Nyquist rather than a
    /// detected edge, and which must therefore come back as passthrough.
    func testSilenceIsPassthrough() async throws {
        let samples = [Float](repeating: 0, count: inputRate * 6)
        let provider = MossFormer2SR48KProvider(weightsPath: "/dev/null", configPath: "/dev/null")
        let crossover = try await provider.detectCrossover(
            AudioBuffer(samples: samples, sampleRate: inputRate, channels: 1)
        )
        XCTAssertTrue(crossover.passthrough, "silence must not be filtered")
        try await assertAgrees(samples, "silence")
    }

    /// Lengths that do not divide the hop, the block, or the chunk. The framing
    /// is what streaming gets wrong, and it gets it wrong at the edges.
    func testAgreesOnAwkwardLengths() async throws {
        for count in [inputRate * 4 + 1, inputRate * 5 + 127, 70_001, 131_073] {
            var samples = [Float](repeating: 0, count: count)
            var seed: UInt64 = 0xDEADBEEFCAFEF00D
            for i in 0..<count {
                seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
                let noise = Float(Int64(bitPattern: seed &>> 11)) / Float(1 << 53)
                let t = Float(i) / Float(inputRate)
                samples[i] = 0.4 * sin(2 * .pi * 700 * t) + 0.15 * noise
            }
            try await assertAgrees(samples, "\(count) samples")
        }
    }

    /// Long enough to cross several accumulator blocks, which is where a
    /// sequential float32 sum would start to drift from MLX's tree reduction.
    func testAgreesAcrossManyBlocks() async throws {
        let count = inputRate * 90
        var samples = [Float](repeating: 0, count: count)
        var seed: UInt64 = 0x1234_5678_9ABC_DEF0
        for i in 0..<count {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            let noise = Float(Int64(bitPattern: seed &>> 11)) / Float(1 << 53)
            let t = Float(i) / Float(inputRate)
            samples[i] = 0.25 * sin(2 * .pi * 220 * t) + 0.2 * sin(2 * .pi * 2400 * t) + 0.2 * noise
        }
        try await assertAgrees(samples, "90 s")
    }
}
