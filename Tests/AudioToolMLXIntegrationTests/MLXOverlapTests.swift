//
//  MLXOverlapTests.swift
//  AudioToolMLXIntegrationTests
//
//  Tests for MLXOverlap using xcodebuild for Metal support
//

import XCTest
import AudioToolTestSupport
import MLX
@testable import AudioToolMLX

final class MLXOverlapTests: MLXTestCase {
    
    // MARK: - Window Tests
    
    func testTriangularWindow() {
        // Test triangular window shape
        let window = MLXOverlap.triangularWindow(length: 100)
        eval(window)
        
        let values = window.asArray(Float.self)
        XCTAssertEqual(values.count, 100)
        
        // Window should start low, peak in middle, end low
        XCTAssertLessThan(values[0], values[49], "Window should increase to middle")
        XCTAssertLessThan(values[99], values[50], "Window should decrease from middle")
        
        // Max should be at center
        let maxIdx = values.enumerated().max(by: { $0.element < $1.element })!.offset
        XCTAssertTrue(maxIdx >= 45 && maxIdx <= 55, "Max should be near center, got \(maxIdx)")
        
        // Window should be normalized to [0, 1]
        XCTAssertEqual(values.max()!, 1.0, accuracy: 0.01)
    }
    
    func testHannWindow() {
        // Test Hann window shape
        let window = MLXOverlap.hannWindow(length: 100)
        eval(window)
        
        let values = window.asArray(Float.self)
        XCTAssertEqual(values.count, 100)
        
        // Hann window starts at 0, peaks at center, ends at 0
        XCTAssertEqual(values[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(values[99], 0.0, accuracy: 0.01)
        XCTAssertGreaterThan(values[50], 0.9)
    }
    
    // MARK: - Split Tests
    
    func testSplitChunks() {
        let audio = MLXArray(Array(0..<1000).map { Float($0) })
        let chunkSamples = 300
        let stride = 200  // 33% overlap
        
        let chunks = MLXOverlap.split(audio: audio, chunkSamples: chunkSamples, stride: stride)
        
        // Should have 5 chunks: 0-300, 200-500, 400-700, 600-900, 800-1000(padded)
        XCTAssertEqual(chunks.count, 5)
        
        // Check first chunk
        let first = chunks[0].chunk.asArray(Float.self)
        XCTAssertEqual(first.count, 300)
        XCTAssertEqual(first[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(first[299], 299.0, accuracy: 0.001)
        
        // Check start indices
        XCTAssertEqual(chunks[0].startIdx, 0)
        XCTAssertEqual(chunks[1].startIdx, 200)
        XCTAssertEqual(chunks[2].startIdx, 400)
    }
    
    // MARK: - No Overlap Reassembly
    
    func testNoOverlapReassembly() {
        let audio = MLXArray(Array(0..<1000).map { Float($0) })
        let chunkSamples = 300
        let stride = 300  // No overlap
        
        let chunks = MLXOverlap.split(audio: audio, chunkSamples: chunkSamples, stride: stride)
        let result = MLXOverlap.reassembleNoOverlap(processedChunks: chunks, originalLength: 1000)
        eval(result)
        
        let resultArray = result.asArray(Float.self)
        XCTAssertEqual(resultArray.count, 1000)
        
        // Should match original when no processing
        for i in 0..<1000 {
            XCTAssertEqual(resultArray[i], Float(i), accuracy: 0.001, "Mismatch at \(i)")
        }
    }
    
    // MARK: - Triangular Overlap-Add
    
    func testTriangularOverlapAdd() {
        // Create test signal
        let audio = MLXArray([Float](repeating: 1.0, count: 1000))
        let chunkSamples = 400
        let overlapRatio: Float = 0.25
        let stride = Int(Float(chunkSamples) * (1 - overlapRatio))
        
        let chunks = MLXOverlap.split(audio: audio, chunkSamples: chunkSamples, stride: stride)
        let window = MLXOverlap.triangularWindow(length: chunkSamples)
        
        let result = MLXOverlap.reassembleOverlapAdd(
            processedChunks: chunks,
            chunkSamples: chunkSamples,
            stride: stride,
            window: window,
            originalLength: 1000
        )
        eval(result)
        
        let resultArray = result.asArray(Float.self)
        XCTAssertEqual(resultArray.count, 1000)
        
        // All values should be close to 1.0 (the input constant)
        // After normalization by weights, constant signal should remain constant
        for i in 100..<900 {  // Check middle (edges might differ)
            XCTAssertEqual(resultArray[i], 1.0, accuracy: 0.1, "Value at \(i) = \(resultArray[i])")
        }
    }
    
    // MARK: - Discard Edges
    
    func testDiscardEdgesIdentity() {
        // Test that discard-edges correctly tiles the output
        let audio = MLXArray(Array(0..<1000).map { Float($0) })
        let chunkSamples = 400
        let overlapRatio: Float = 0.25
        let overlapSamples = Int(Float(chunkSamples) * overlapRatio)
        let stride = chunkSamples - overlapSamples
        let giveUp = overlapSamples / 2
        
        let chunks = MLXOverlap.split(audio: audio, chunkSamples: chunkSamples, stride: stride)
        let result = MLXOverlap.reassembleDiscardEdges(
            processedChunks: chunks,
            chunkSamples: chunkSamples,
            stride: stride,
            giveUp: giveUp,
            originalLength: 1000
        )
        eval(result)
        
        let resultArray = result.asArray(Float.self)
        XCTAssertEqual(resultArray.count, 1000)
        
        // First chunk output range: [0, chunkSamples - giveUp) = [0, 350)
        // Should match input
        for i in 0..<350 {
            XCTAssertEqual(resultArray[i], Float(i), accuracy: 0.001, "First chunk mismatch at \(i)")
        }
        
        // Second chunk starts at stride=300, keeps [giveUp, chunkSamples-giveUp) = [50, 350)
        // Output range: [300+50, 300+350) = [350, 650)
        for i in 350..<650 {
            XCTAssertEqual(resultArray[i], Float(i), accuracy: 0.001, "Second chunk mismatch at \(i)")
        }
    }
    
    // MARK: - High-Level API
    
    func testProcessWithChunkingIdentity() async throws {
        // Test the high-level API with identity processing
        let audio = MLXArray(Array(0..<1000).map { Float($0) })
        
        let result = try await MLXOverlap.processWithChunking(
            audio: audio,
            chunkSamples: 400,
            overlapRatio: 0.25,
            strategy: .discardEdges
        ) { chunk in
            // Identity processing - return chunk unchanged
            return chunk
        }
        eval(result)
        
        let resultArray = result.asArray(Float.self)
        XCTAssertEqual(resultArray.count, 1000)
        
        // Check that most of the signal is preserved (mid-section should match exactly)
        for i in 100..<900 {
            XCTAssertEqual(resultArray[i], Float(i), accuracy: 0.001, "Mismatch at \(i)")
        }
    }
    
    func testProcessWithChunkingAllStrategies() async throws {
        // Identity processing must preserve every sample, including window edges
        // and the padded tail.
        let audio = MLXArray([Float](repeating: 1.0, count: 1000))
        
        for strategy in [OverlapStrategy.noOverlap, .overlapAdd, .triangular, .hann, .discardEdges] {
            let result = try await MLXOverlap.processWithChunking(
                audio: audio,
                chunkSamples: 300,
                overlapRatio: strategy == .noOverlap ? 0.0 : 0.25,
                strategy: strategy
            ) { chunk in
                return chunk
            }
            eval(result)
            
            let resultArray = result.asArray(Float.self)
            XCTAssertEqual(resultArray.count, 1000, "Strategy \(strategy) produced wrong length")
            for (index, value) in resultArray.enumerated() {
                XCTAssertEqual(value, 1, accuracy: 1e-6,
                               "Strategy \(strategy) changed identity sample \(index)")
            }
        }
    }

    func testDiscardEdgesProcessesAudioShorterThanItsDiscardMargin() async throws {
        let audio = MLXArray(Array(0..<10).map(Float.init))
        let result = try await MLXOverlap.processWithChunking(
            audio: audio,
            chunkSamples: 300,
            overlapRatio: 0.25,
            strategy: .discardEdges
        ) { $0 }
        eval(result)
        XCTAssertEqual(result.asArray(Float.self), audio.asArray(Float.self))
    }

    func testDualOutputHonorsEveryStrategy() async throws {
        let audio = MLXArray([Float](repeating: 1, count: 1000))
        for strategy in [OverlapStrategy.noOverlap, .overlapAdd, .triangular, .hann, .discardEdges] {
            let (first, second) = try await MLXOverlap.processWithChunkingDual(
                audio: audio,
                chunkSamples: 300,
                overlapRatio: strategy == .noOverlap ? 0 : 0.25,
                strategy: strategy
            ) { chunk in
                (chunk, chunk * 2)
            }
            eval(first, second)
            let firstSamples = first.asArray(Float.self)
            let secondSamples = second.asArray(Float.self)
            XCTAssertEqual(firstSamples.count, 1000)
            XCTAssertEqual(secondSamples.count, 1000)
            for index in 0..<1000 {
                XCTAssertEqual(firstSamples[index], 1, accuracy: 1e-6,
                               "First output failed for \(strategy) at \(index)")
                XCTAssertEqual(secondSamples[index], 2, accuracy: 1e-6,
                               "Second output failed for \(strategy) at \(index)")
            }
        }
    }

    func testMixtureConsistencyPreservesRelativeSourceGain() {
        let mixture: [Float] = [0.6, -0.3, 0.9]
        let sources: [[Float]] = [
            [0.1, -0.05, 0.15],
            [0.4, -0.2, 0.6],
        ]
        let corrected = MossFormer2SSProvider.enforceMixtureConsistency(
            mixture: mixture,
            sources: sources
        )

        XCTAssertEqual(corrected.count, 2)
        for index in mixture.indices {
            XCTAssertEqual(corrected[0][index] + corrected[1][index], mixture[index], accuracy: 1e-6)
        }
        // A quiet estimate stays quieter; it is not independently raised to the
        // mixture RMS.
        XCTAssertLessThan(abs(corrected[0][0]), abs(corrected[1][0]))
    }
    
    // MARK: - Edge Cases
    
    func testShortAudioNoChunking() {
        // Audio shorter than chunk size
        let audio = MLXArray([Float](repeating: 1.0, count: 100))
        let chunks = MLXOverlap.split(audio: audio, chunkSamples: 400, stride: 300)
        
        // Should have 1 chunk, padded
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].chunk.shape[0], 400)
    }
}
