//
//  SuperResolutionResamplerTests.swift
//  AudioToolMLXIntegrationTests
//
//  Regression tests for bounded, fully drained SR sample-rate conversion
//

import XCTest
@testable import AudioToolMLX

final class SuperResolutionResamplerTests: XCTestCase {

    func testIncrementalConversionIsChunkSizeInvariantAndExact() throws {
        let input = (0..<48_017).map { index -> Float in
            let time = Float(index) / 16_000
            return 0.6 * sin(2 * .pi * 317 * time)
                + 0.2 * sin(2 * .pi * 3_701 * time)
        }

        let expected = try SuperResolutionResampler.upsample(
            input,
            from: 16_000,
            to: 48_000
        )
        let stream = try SuperResolutionResampler.Stream(
            input,
            from: 16_000,
            to: 48_000
        )
        var incremental: [Float] = []
        while let chunk = try stream.next(maxFrames: 997) {
            XCTAssertLessThanOrEqual(chunk.count, 997)
            XCTAssertFalse(chunk.isEmpty)
            incremental.append(contentsOf: chunk)
        }

        XCTAssertEqual(expected.count, input.count * 3)
        XCTAssertEqual(incremental.count, expected.count)
        for index in expected.indices {
            XCTAssertEqual(incremental[index], expected[index], accuracy: 1e-6,
                           "converter output changed with sink chunking at frame \(index)")
        }
    }

    /// The 997-frame case above is a smaller pull than the converter's own packet
    /// size. The chunked SR path pulls a whole 4 s chunk first (192000 frames at
    /// 48 kHz) and then one 2 s stride at a time, so the invariance has to hold at
    /// those sizes too - that is the only thing standing between chunked SR and the
    /// contiguous 48 kHz array the reference slices.
    func testConversionIsInvariantAtSuperResolutionChunkSizes() throws {
        // 12 s at 16 kHz: four 4 s chunks at 50% overlap once upsampled.
        let input = (0..<192_000).map { index -> Float in
            let time = Float(index) / 16_000
            return 0.5 * sin(2 * .pi * 220 * time) + 0.25 * sin(2 * .pi * 5_100 * time)
        }

        let expected = try SuperResolutionResampler.upsample(input, from: 16_000, to: 48_000)
        let stream = try SuperResolutionResampler.Stream(input, from: 16_000, to: 48_000)

        var incremental: [Float] = []
        var pull = 192_000
        while let chunk = try stream.next(maxFrames: pull) {
            incremental.append(contentsOf: chunk)
            pull = 96_000
        }

        XCTAssertEqual(incremental.count, expected.count)
        var worst: (index: Int, delta: Float) = (0, 0)
        for index in expected.indices where abs(incremental[index] - expected[index]) > worst.delta {
            worst = (index, abs(incremental[index] - expected[index]))
        }
        XCTAssertLessThan(
            worst.delta, 1e-6,
            "converter output changed with large sink chunking; worst at frame \(worst.index)")
    }

    func testSameRateStreamIsBoundedAndLossless() throws {
        let input = (0..<2_050).map(Float.init)
        let stream = try SuperResolutionResampler.Stream(
            input,
            from: 48_000,
            to: 48_000
        )
        var output: [Float] = []
        while let chunk = try stream.next(maxFrames: 128) {
            XCTAssertLessThanOrEqual(chunk.count, 128)
            output.append(contentsOf: chunk)
        }
        XCTAssertEqual(output, input)
    }
}
