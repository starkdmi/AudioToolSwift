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
