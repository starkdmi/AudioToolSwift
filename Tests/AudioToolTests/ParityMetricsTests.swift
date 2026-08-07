//
//  ParityMetricsTests.swift
//  AudioToolTests
//
//  Hermetic cover for the measurement the parity suite trusts. No weights, no
//  artifacts, no Metal - if the metric is wrong, every threshold above it is
//  meaningless, so it is checked where `swift test` will always run it.
//

import XCTest
@testable import AudioToolTestSupport

final class ParityMetricsTests: XCTestCase {

    // MARK: - The non-finite hole

    /// A NaN anywhere used to read as `.infinity`, which clears every threshold.
    ///
    /// `noise` accumulates NaN, and `NaN > 0` is false, so the guard meant for
    /// bit-identical signals fell through and reported perfect agreement for
    /// output that was not merely wrong but unusable.
    func testNaNInCandidateIsNotInfiniteSNR() {
        let reference: [Float] = [0.1, 0.2, 0.3, 0.4]
        let candidate: [Float] = [0.1, .nan, 0.3, 0.4]

        let snr = ParityMetrics.snrDB(reference: reference, candidate: candidate)

        XCTAssertFalse(snr.isNaN, "a NaN sample must not produce a NaN verdict")
        XCTAssertEqual(snr, -.infinity, "a NaN sample must fail, not pass")
        XCTAssertFalse(snr > 60, "must not clear any recorded threshold")
    }

    func testInfiniteSampleIsNotInfiniteSNR() {
        let reference: [Float] = [0.1, 0.2, 0.3]
        let candidate: [Float] = [0.1, .infinity, 0.3]

        XCTAssertEqual(ParityMetrics.snrDB(reference: reference, candidate: candidate), -.infinity)
    }

    func testNaNInReferenceIsNotInfiniteSNR() {
        let reference: [Float] = [0.1, .nan, 0.3]
        let candidate: [Float] = [0.1, 0.2, 0.3]

        XCTAssertEqual(ParityMetrics.snrDB(reference: reference, candidate: candidate), -.infinity)
    }

    // MARK: - Infinity still means what it says

    func testBitIdenticalIsInfinite() {
        let signal: [Float] = [0.1, -0.2, 0.3, -0.4]

        XCTAssertEqual(ParityMetrics.snrDB(reference: signal, candidate: signal), .infinity)
    }

    func testSilentReferenceWithErrorIsNegativeInfinity() {
        XCTAssertEqual(
            ParityMetrics.snrDB(reference: [0, 0, 0], candidate: [0.1, 0.2, 0.3]),
            -.infinity
        )
    }

    // MARK: - Artifact bounds
    //
    // Offsets and shapes come out of a file. Unchecked, a hostile or truncated
    // one traps in `subdata`/`loadUnaligned` instead of throwing, which in a test
    // target reads as a crashed run rather than a failed artifact.

    /// Build a minimal safetensors buffer with a caller-supplied header.
    private func artifactData(header: [String: Any], payloadBytes: Int) throws -> Data {
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var out = Data()
        withUnsafeBytes(of: UInt64(headerData.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(headerData)
        out.append(Data(repeating: 0, count: payloadBytes))
        return out
    }

    private func loadFails(header: [String: Any], payloadBytes: Int) throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try artifactData(header: header, payloadBytes: payloadBytes)
            .write(to: directory.appendingPathComponent("case.safetensors"))
        try Data(#"{"sample_rate": 16000}"#.utf8)
            .write(to: directory.appendingPathComponent("case.json"))

        XCTAssertThrowsError(try ParityArtifact(name: "case", in: directory))
    }

    func testNegativeOffsetsThrowRatherThanTrap() throws {
        try loadFails(
            header: ["t": ["dtype": "F32", "shape": [4], "data_offsets": [-8, 8]]],
            payloadBytes: 16
        )
    }

    func testOverflowingOffsetsThrowRatherThanTrap() throws {
        try loadFails(
            header: ["t": ["dtype": "F32", "shape": [4], "data_offsets": [0, Int.max]]],
            payloadBytes: 16
        )
    }

    func testUnalignedByteRangeThrowsRatherThanTrap() throws {
        // 6 bytes is not a whole number of float32; the decoder would read past
        // the end on its second step.
        try loadFails(
            header: ["t": ["dtype": "F32", "shape": [1], "data_offsets": [0, 6]]],
            payloadBytes: 6
        )
    }

    func testShapeDisagreeingWithByteCountThrows() throws {
        try loadFails(
            header: ["t": ["dtype": "F32", "shape": [99], "data_offsets": [0, 16]]],
            payloadBytes: 16
        )
    }

    // MARK: - Scale

    /// Halving the error should buy about 6 dB.
    func testErrorScalesAtSixDBPerHalving() {
        let reference = (0..<512).map { Float(sin(Double($0) * 0.05)) }
        let coarse = reference.map { $0 + 1e-3 }
        let fine = reference.map { $0 + 5e-4 }

        let gain = ParityMetrics.snrDB(reference: reference, candidate: fine)
            - ParityMetrics.snrDB(reference: reference, candidate: coarse)

        XCTAssertEqual(gain, 6.02, accuracy: 0.1)
    }
}
