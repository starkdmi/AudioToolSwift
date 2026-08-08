//
//  IncrementalOverlapAddTests.swift
//  AudioToolMLXIntegrationTests
//
//  Streaming overlap-add must equal batch overlap-add. No weights, no network.
//

import XCTest
import Testing
import AudioToolTestSupport
import MLX
@testable import AudioToolMLX

/// Streaming reassembly against the batch reassembly it is supposed to match.
///
/// This is the guarantee the super-resolution streaming path was missing. It fed
/// its chunks through a hand-rolled blend that multiplied by the window without
/// dividing by the accumulated weight, so a streamed result was not the batch
/// result: the opening `stride` samples faded in from silence. Nothing caught it,
/// because nothing compared the two.
///
/// Not an `IntegrationTestCase` - this is array arithmetic on synthetic input.
final class IncrementalOverlapAddTests: MLXTestCase {

    /// Deterministic pseudo-random samples, so a failure is reproducible.
    private func samples(count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return Float(state % 2_000_000) / 1_000_000.0 - 1.0
        }
    }

    /// Feed the same chunks to both paths and require identical output.
    private func assertParity(
        totalLength: Int,
        chunkSamples: Int,
        stride: Int,
        window: [Float],
        seed: UInt64,
        line: UInt = #line
    ) {
        // Synthesize a processed chunk per position, exactly as a model would emit.
        var chunks: [(chunk: [Float], startIdx: Int)] = []
        var startIdx = 0
        var chunkSeed = seed
        while startIdx < totalLength {
            chunkSeed &+= 1
            chunks.append((samples(count: chunkSamples, seed: chunkSeed), startIdx))
            startIdx += stride
        }

        // Batch.
        let batch = MLXOverlap.reassembleOverlapAdd(
            processedChunks: chunks.map { (MLXArray($0.chunk), $0.startIdx) },
            chunkSamples: chunkSamples,
            stride: stride,
            window: MLXArray(window),
            originalLength: totalLength
        )
        eval(batch)
        let batchSamples = batch.asArray(Float.self)

        // Streaming.
        var assembler = IncrementalOverlapAdd(
            chunkSamples: chunkSamples,
            stride: stride,
            window: window,
            totalLength: totalLength
        )
        var streamed: [Float] = []
        for (chunk, startIdx) in chunks {
            streamed.append(contentsOf: assembler.add(chunk, startIdx: startIdx))
        }
        streamed.append(contentsOf: assembler.finish())

        XCTAssertEqual(streamed.count, totalLength,
                       "streaming emitted \(streamed.count) samples, expected \(totalLength)",
                       line: line)
        XCTAssertEqual(batchSamples.count, totalLength, line: line)

        for i in 0..<min(streamed.count, batchSamples.count) {
            XCTAssertEqual(streamed[i], batchSamples[i], accuracy: 1e-6,
                           "sample \(i) differs between streaming and batch",
                           line: line)
        }
    }

    func testHannFiftyPercentOverlapMatchesBatch() {
        let chunkSamples = 1024
        let window = MLXOverlap.hannWindow(length: chunkSamples).asArray(Float.self)
        assertParity(totalLength: 5000, chunkSamples: chunkSamples,
                     stride: chunkSamples / 2, window: window, seed: 42)
    }

    func testTriangularTwentyFivePercentOverlapMatchesBatch() {
        let chunkSamples: Int = 800
        let window = MLXOverlap.triangularWindow(length: chunkSamples).asArray(Float.self)
        assertParity(totalLength: 3333, chunkSamples: chunkSamples,
                     stride: chunkSamples - chunkSamples / 4, window: window, seed: 7)
    }

    /// Audio shorter than one chunk still has to come out normalized.
    func testShorterThanOneChunkMatchesBatch() {
        let chunkSamples = 1024
        let window = MLXOverlap.hannWindow(length: chunkSamples).asArray(Float.self)
        assertParity(totalLength: 300, chunkSamples: chunkSamples,
                     stride: chunkSamples / 2, window: window, seed: 11)
    }

    func testExactChunkMultipleMatchesBatch() {
        let chunkSamples = 512
        let window = MLXOverlap.hannWindow(length: chunkSamples).asArray(Float.self)
        assertParity(totalLength: chunkSamples * 4, chunkSamples: chunkSamples,
                     stride: chunkSamples / 2, window: window, seed: 99)
    }

    /// The specific symptom, stated directly: a Hann window rises from zero, so an
    /// unnormalized first chunk starts at silence. Normalized, it does not - the
    /// first sample carries its full amplitude because it has only one contribution
    /// and dividing by that contribution's weight cancels the window.
    func testFirstSamplesAreNotFadedIn() {
        let chunkSamples = 1024
        let stride = chunkSamples / 2
        let window = MLXOverlap.hannWindow(length: chunkSamples).asArray(Float.self)
        let constant = [Float](repeating: 1.0, count: chunkSamples)

        var assembler = IncrementalOverlapAdd(
            chunkSamples: chunkSamples, stride: stride,
            window: window, totalLength: chunkSamples * 3
        )

        let first = assembler.add(constant, startIdx: 0)

        XCTAssertEqual(first.count, stride)
        // Every sample in the first emitted block is a single contribution
        // normalized by its own weight, so a constant input stays constant.
        for (i, value) in first.enumerated() where window[i] > 0 {
            XCTAssertEqual(value, 1.0, accuracy: 1e-5,
                           "sample \(i) of the first block should be 1.0, not window[\(i)] = \(window[i])")
        }
    }
}

@Suite("Incremental overlap-add tail handling")
struct IncrementalOverlapAddTailTests {

    @Test("A partial final chunk preserves its leading overlap weights")
    func partialFinalChunkOverlap() {
        let chunkSamples = 8
        let stride = 6
        let totalLength = 17
        let triangularWindow: [Float] = [0.25, 0.5, 0.75, 1, 1, 0.75, 0.5, 0.25]
        var assembler = IncrementalOverlapAdd(
            chunkSamples: chunkSamples,
            stride: stride,
            window: triangularWindow,
            totalLength: totalLength
        )
        var output: [Float] = []

        for start in [0, 6] {
            output.append(contentsOf: assembler.add(
                [Float](repeating: 0, count: chunkSamples),
                startIdx: start
            ))
        }
        output.append(contentsOf: assembler.add(
            [Float](repeating: 1, count: chunkSamples),
            startIdx: 12
        ))
        output.append(contentsOf: assembler.finish())

        #expect(output.count == totalLength)
        // The final chunk has only five real samples. Its first two are still
        // covered by the preceding chunk, so output 13 must blend weights 0.25
        // and 0.5. Treating the whole short chunk as a tail gives 0.8 instead.
        #expect(abs(output[13] - (0.5 / 0.75)) < 1e-6)
    }
}

@Suite("MLX cache trimming policy")
struct MLXCachePolicyTests {

    @Test("Cache trims are periodic and pressure driven")
    func periodicPressureThreshold() {
        let threshold = MLXCachePolicy.defaultThresholdBytes

        #expect(!MLXCachePolicy.shouldTrim(afterChunk: 0, cacheMemory: threshold))
        #expect(!MLXCachePolicy.shouldTrim(afterChunk: 7, cacheMemory: threshold))
        #expect(!MLXCachePolicy.shouldTrim(afterChunk: 8, cacheMemory: threshold - 1))
        #expect(MLXCachePolicy.shouldTrim(afterChunk: 8, cacheMemory: threshold))
        #expect(!MLXCachePolicy.shouldTrim(afterChunk: 9, cacheMemory: threshold * 2))
        #expect(MLXCachePolicy.shouldTrim(afterChunk: 16, cacheMemory: threshold * 2))
    }

    @Test("Invalid intervals never request a trim")
    func invalidInterval() {
        #expect(!MLXCachePolicy.shouldTrim(
            afterChunk: 1,
            cacheMemory: .max,
            chunkInterval: 0
        ))
    }
}
