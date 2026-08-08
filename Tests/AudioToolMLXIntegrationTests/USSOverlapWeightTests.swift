//
//  USSOverlapWeightTests.swift
//  AudioToolMLXIntegrationTests
//
//  Regression tests for segmented USS overlap weighting
//

import XCTest
@testable import USSMLXSwift

final class USSOverlapWeightTests: XCTestCase {
    func testPartialFinalSegmentPreservesLeadingCrossfade() {
        let leadingOverlap = USSInference.overlapAddWeight(
            windowWeight: 0.5,
            sampleIndex: 1,
            segmentSamples: 8,
            hopSamples: 6,
            isFirstSegment: false,
            isFinalSegment: true
        )
        let unsharedTail = USSInference.overlapAddWeight(
            windowWeight: 0.75,
            sampleIndex: 2,
            segmentSamples: 8,
            hopSamples: 6,
            isFirstSegment: false,
            isFinalSegment: true
        )

        XCTAssertEqual(leadingOverlap, 0.5)
        XCTAssertEqual(unsharedTail, 1)
    }
}
