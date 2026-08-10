//
//  IncrementalDiscardEdgesTests.swift
//  AudioToolMLXIntegrationTests
//
//  The tiling property that makes discard-edges assembly correct.
//

import Testing
@testable import AudioToolMLX

/// `IncrementalDiscardEdges` against the arithmetic in the MossFormer2
/// references' `generate.py`.
///
/// The property worth pinning is not "it produces some output" but that the kept
/// regions **tile**: contiguous, disjoint, covering everything. Overlap-add gets
/// its correctness from weights summing to one; this gets its correctness from the
/// centres abutting exactly. If `giveUp` is off by one the output silently gains a
/// seam or drops a sample, and nothing downstream would notice.
@Suite("Discard-edges assembly")
struct IncrementalDiscardEdgesTests {

    /// The reference's own numbers: 4 s window, 3 s stride, 0.5 s given up.
    private let chunkSamples = 192_000   // 4 s at 48 kHz
    private let stride = 144_000         // int(window * 0.75)
    private var giveUp: Int { (chunkSamples - stride) / 2 }  // 24_000

    /// A chunk whose every sample equals its absolute output index, so the
    /// assembled result is only correct if each sample came from the right place
    /// in the right chunk.
    private func rampChunk(startIdx: Int) -> [Float] {
        (0..<chunkSamples).map { Float(startIdx + $0) }
    }

    @Test("Assembled output is the identity ramp, so every sample came from the right chunk")
    func testTilingReconstructsInput() {
        let totalLength = stride * 6 + chunkSamples
        var assembler = IncrementalDiscardEdges(
            chunkSamples: chunkSamples, stride: stride, totalLength: totalLength
        )

        var output: [Float] = []
        var startIdx = 0
        while startIdx < totalLength {
            output.append(contentsOf: assembler.add(rampChunk(startIdx: startIdx), startIdx: startIdx))
            startIdx += stride
        }
        output.append(contentsOf: assembler.finish())

        #expect(output.count == totalLength)
        // A gap or a double-write shows up as a discontinuity in the ramp.
        for index in 0..<min(output.count, totalLength) where output[index] != Float(index) {
            Issue.record("sample \(index) is \(output[index]), expected \(index)")
            break
        }
    }

    @Test("First chunk keeps its leading edge, later chunks do not")
    func testFirstChunkKeepsLeadingEdge() {
        var assembler = IncrementalDiscardEdges(
            chunkSamples: chunkSamples, stride: stride, totalLength: stride * 4
        )

        // The reference writes `outputs[0 : window - give_up_length]` for the first
        // chunk and `outputs[idx + give_up : idx + window - give_up]` after it.
        let first = assembler.add(rampChunk(startIdx: 0), startIdx: 0)
        #expect(first.count == chunkSamples - giveUp)
        #expect(first.first == 0)

        let second = assembler.add(rampChunk(startIdx: stride), startIdx: stride)
        #expect(second.count == stride)
        #expect(second.first == Float(stride + giveUp))
    }

    @Test("Each chunk after the first contributes exactly one stride")
    func testSteadyStateContributionIsStride() {
        var assembler = IncrementalDiscardEdges(
            chunkSamples: chunkSamples, stride: stride, totalLength: stride * 10
        )
        _ = assembler.add(rampChunk(startIdx: 0), startIdx: 0)
        for step in 1...4 {
            let idx = stride * step
            #expect(assembler.add(rampChunk(startIdx: idx), startIdx: idx).count == stride)
        }
    }

    @Test("Output never exceeds the declared length")
    func testTrimsToTotalLength() {
        let totalLength = chunkSamples + stride / 2
        var assembler = IncrementalDiscardEdges(
            chunkSamples: chunkSamples, stride: stride, totalLength: totalLength
        )
        var emitted = 0
        var startIdx = 0
        while startIdx < totalLength + chunkSamples {
            emitted += assembler.add(rampChunk(startIdx: startIdx), startIdx: startIdx).count
            startIdx += stride
        }
        emitted += assembler.finish().count
        #expect(emitted == totalLength)
    }

    /// No overlap means nothing to give up, and the assembly degenerates to
    /// concatenation. Worth pinning because `giveUp` is a division.
    @Test("Zero overlap degenerates to concatenation")
    func testZeroOverlap() {
        var assembler = IncrementalDiscardEdges(
            chunkSamples: 100, stride: 100, totalLength: 300
        )
        var output: [Float] = []
        for step in 0..<3 {
            let start = step * 100
            output.append(contentsOf: assembler.add(
                (0..<100).map { Float(start + $0) }, startIdx: start
            ))
        }
        output.append(contentsOf: assembler.finish())
        #expect(output.count == 300)
        #expect(output[0] == 0)
        #expect(output[299] == 299)
    }
}
