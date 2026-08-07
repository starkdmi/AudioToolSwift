//
//  MLXOverlap.swift
//  AudioToolMLX
//
//  MLX-based overlap-add utilities for audio chunking.
//  Ported from Python benchmark_chunking.py using vectorized MLX operations.
//

import Foundation
import MLX

// MARK: - Overlap Strategy

public enum OverlapStrategy: Sendable {
    case noOverlap      // Sequential chunks, no overlap
    case overlapAdd     // Simple overlap-add (use with window)
    case triangular     // Triangular window blending
    case hann           // Hann window blending
    case discardEdges   // Keep center, discard edges
}

// MARK: - MLX Overlap Utilities

public struct MLXOverlap {
    
    // MARK: - Window Functions
    
    /// Create triangular window (linear fade in/out)
    /// Python: up = arange(1, half+1); down = arange(length-half, 0, -1); window = concat([up, down]) / max
    public static func triangularWindow(length: Int) -> MLXArray {
        let half = length / 2
        
        // Up ramp: 1, 2, ..., half
        let up = MLXArray(Array(1...half).map { Float($0) })
        
        // Down ramp: length - half, length - half - 1, ..., 1
        let downCount = length - half
        let down = MLXArray(Array((1...downCount).reversed()).map { Float($0) })
        
        // Concatenate
        let window = concatenated([up, down], axis: 0)
        
        // Normalize to [0, 1]
        let maxVal = window.max()
        eval(maxVal)
        return window / maxVal
    }
    
    /// Create Hann window
    /// Python: np.hanning(length)
    ///
    /// Uses `sin²(πi/(N-1))`, which is algebraically the same as
    /// `0.5(1 - cos(2πi/(N-1)))` but survives Float32. In the direct form `cos(x)`
    /// rounds to exactly 1 for small `x`, so `1 - cos(x)` cancels to zero: at the
    /// 192000-sample chunk the SR provider uses, that zeroed `i` = 1...7 and the
    /// overlap-add divide-by-weight then left those output samples at silence.
    /// NumPy does not show it because `np.hanning` evaluates in Float64.
    public static func hannWindow(length: Int) -> MLXArray {
        var window = [Float](repeating: 0, count: length)
        for i in 0..<length {
            let s = sin(Float.pi * Float(i) / Float(length - 1))
            window[i] = s * s
        }
        return MLXArray(window)
    }
    
    // MARK: - Chunk Splitting
    
    /// Split audio into overlapping chunks
    /// Returns array of chunks, each of size chunkSamples (zero-padded if needed)
    /// Note: For discard-edges, Python uses: while current_idx + chunk_samples <= total_length + stride
    public static func split(
        audio: MLXArray,
        chunkSamples: Int,
        stride: Int,
        forDiscardEdges: Bool = false
    ) -> [(chunk: MLXArray, startIdx: Int)] {
        eval(audio)
        let totalLength = audio.shape[0]
        var chunks: [(MLXArray, Int)] = []
        
        var currentIdx = 0
        
        // Different loop conditions for different strategies
        while forDiscardEdges ? (currentIdx + chunkSamples <= totalLength + stride) : (currentIdx < totalLength) {
            let endIdx = min(currentIdx + chunkSamples, totalLength)
            
            // Extract chunk using MLX slicing
            var chunk = audio[currentIdx..<endIdx]
            eval(chunk)
            
            // Pad if needed
            let chunkLength = endIdx - currentIdx
            if chunkLength < chunkSamples {
                let padding = MLXArray.zeros([chunkSamples - chunkLength])
                chunk = concatenated([chunk, padding], axis: 0)
                eval(chunk)
            }
            
            chunks.append((chunk, currentIdx))
            currentIdx += stride
        }
        
        return chunks
    }
    
    // MARK: - Reassembly Strategies
    
    /// No overlap: simple concatenation
    public static func reassembleNoOverlap(
        processedChunks: [(chunk: MLXArray, startIdx: Int)],
        originalLength: Int
    ) -> MLXArray {
        guard !processedChunks.isEmpty else {
            return MLXArray.zeros([originalLength])
        }
        
        // Concatenate all chunks and trim
        let allChunks = processedChunks.map { $0.chunk }
        var result = concatenated(allChunks, axis: 0)
        eval(result)
        
        // Trim to original length
        if result.shape[0] > originalLength {
            result = result[0..<originalLength]
            eval(result)
        }
        
        return result
    }
    
    /// Overlap-add with window weighting
    /// Python: result[start:end] += w * output[:chunk_length]; sum_weight[start:end] += w
    public static func reassembleOverlapAdd(
        processedChunks: [(chunk: MLXArray, startIdx: Int)],
        chunkSamples: Int,
        stride: Int,
        window: MLXArray,
        originalLength: Int
    ) -> MLXArray {
        eval(window)
        
        // Pre-allocate output and weight arrays
        var resultArray = [Float](repeating: 0, count: originalLength)
        var weightArray = [Float](repeating: 0, count: originalLength)
        
        // Get window as Float array
        let windowArray = window.asArray(Float.self)
        
        for (chunk, startIdx) in processedChunks {
            eval(chunk)
            let chunkArray = chunk.asArray(Float.self)
            let endIdx = min(startIdx + chunkSamples, originalLength)
            let chunkLength = endIdx - startIdx
            
            // Weighted overlap-add
            for i in 0..<chunkLength {
                let w = i < windowArray.count ? windowArray[i] : 1.0
                let sample = i < chunkArray.count ? chunkArray[i] : 0.0
                resultArray[startIdx + i] += w * sample
                weightArray[startIdx + i] += w
            }
        }
        
        // Normalize by weights
        for i in 0..<originalLength {
            if weightArray[i] > 0 {
                resultArray[i] /= weightArray[i]
            }
        }
        
        return MLXArray(resultArray)
    }
    
    /// Discard edges reassembly
    /// Python: first chunk keeps [0, chunkSamples-giveUp), others keep [giveUp, chunkSamples-giveUp)
    public static func reassembleDiscardEdges(
        processedChunks: [(chunk: MLXArray, startIdx: Int)],
        chunkSamples: Int,
        stride: Int,
        giveUp: Int,  // overlap_samples / 2
        originalLength: Int
    ) -> MLXArray {
        // Pre-allocate output
        var resultArray = [Float](repeating: 0, count: originalLength)
        
        for (idx, chunkData) in processedChunks.enumerated() {
            let (chunk, startIdx) = chunkData
            eval(chunk)
            let chunkArray = chunk.asArray(Float.self)
            
            let isFirst = idx == 0
            
            // Determine valid range in the chunk
            let keepStart = isFirst ? 0 : giveUp
            // keepEnd would be: chunkSamples - giveUp (used implicitly in range calculation below)
            
            // Determine output range
            let outputRangeStart = isFirst ? 0 : startIdx + giveUp
            let outputRangeEnd = isFirst ? chunkSamples - giveUp : startIdx + chunkSamples - giveUp
            let actualEnd = min(outputRangeEnd, originalLength)
            
            // Copy samples
            for outPos in outputRangeStart..<actualEnd {
                let srcIdx = keepStart + (outPos - outputRangeStart)
                if srcIdx >= 0 && srcIdx < chunkArray.count {
                    resultArray[outPos] = chunkArray[srcIdx]
                }
            }
        }
        
        return MLXArray(resultArray)
    }
    
    // MARK: - High-Level API
    
    /// Process audio with chunking using the specified strategy
    /// - Parameters:
    ///   - audio: Input audio as MLXArray [samples]
    ///   - chunkSamples: Number of samples per chunk
    ///   - overlapRatio: Overlap ratio (0.0 to 0.5)
    ///   - strategy: Blending strategy
    ///   - processChunk: Async function to process each chunk
    /// - Returns: Reassembled processed audio
    public static func processWithChunking(
        audio: MLXArray,
        chunkSamples: Int,
        overlapRatio: Float,
        strategy: OverlapStrategy,
        processChunk: (MLXArray) async throws -> MLXArray
    ) async throws -> MLXArray {
        eval(audio)
        let totalLength = audio.shape[0]
        let overlapSamples = Int(Float(chunkSamples) * overlapRatio)
        let stride = chunkSamples - overlapSamples
        
        // Split into chunks - use different loop condition for discard-edges
        let inputChunks = split(
            audio: audio,
            chunkSamples: chunkSamples,
            stride: stride,
            forDiscardEdges: strategy == .discardEdges
        )
        
        // Process each chunk with memory management
        var processedChunks: [(chunk: MLXArray, startIdx: Int)] = []
        for (chunk, startIdx) in inputChunks {
            let processed = try await processChunk(chunk)
            eval(processed)
            processedChunks.append((processed, startIdx))
            
            // Clear GPU cache between chunks to reduce peak memory
            GPU.clearCache()
        }
        
        // Reassemble based on strategy
        switch strategy {
        case .noOverlap:
            return reassembleNoOverlap(processedChunks: processedChunks, originalLength: totalLength)
            
        case .overlapAdd:
            let window = MLXArray([Float](repeating: 1.0, count: chunkSamples))
            return reassembleOverlapAdd(
                processedChunks: processedChunks,
                chunkSamples: chunkSamples,
                stride: stride,
                window: window,
                originalLength: totalLength
            )
            
        case .triangular:
            let window = triangularWindow(length: chunkSamples)
            return reassembleOverlapAdd(
                processedChunks: processedChunks,
                chunkSamples: chunkSamples,
                stride: stride,
                window: window,
                originalLength: totalLength
            )
            
        case .hann:
            let window = hannWindow(length: chunkSamples)
            return reassembleOverlapAdd(
                processedChunks: processedChunks,
                chunkSamples: chunkSamples,
                stride: stride,
                window: window,
                originalLength: totalLength
            )
            
        case .discardEdges:
            let giveUp = overlapSamples / 2
            return reassembleDiscardEdges(
                processedChunks: processedChunks,
                chunkSamples: chunkSamples,
                stride: stride,
                giveUp: giveUp,
                originalLength: totalLength
            )
        }
    }
    
    // MARK: - Dual Output Processing (for background extraction)
    
    /// Process audio in chunks with dual output (enhanced + background)
    /// - Parameters:
    ///   - audio: Input audio as MLXArray [samples]
    ///   - chunkSamples: Number of samples per chunk
    ///   - overlapRatio: Overlap ratio (0.0 to 0.5)
    ///   - strategy: Blending strategy
    ///   - processChunk: Async function that returns (enhanced, background) for each chunk
    /// - Returns: Tuple of (enhanced, background) reassembled audio
    public static func processWithChunkingDual(
        audio: MLXArray,
        chunkSamples: Int,
        overlapRatio: Float,
        strategy: OverlapStrategy,
        processChunk: (MLXArray) async throws -> (MLXArray, MLXArray)
    ) async throws -> (MLXArray, MLXArray) {
        eval(audio)
        let totalLength = audio.shape[0]
        let overlapSamples = Int(Float(chunkSamples) * overlapRatio)
        let stride = chunkSamples - overlapSamples
        
        // Split into chunks
        let inputChunks = split(
            audio: audio,
            chunkSamples: chunkSamples,
            stride: stride,
            forDiscardEdges: strategy == .discardEdges
        )
        
        // Process each chunk - collect both outputs
        var enhancedChunks: [(chunk: MLXArray, startIdx: Int)] = []
        var backgroundChunks: [(chunk: MLXArray, startIdx: Int)] = []
        
        for (chunk, startIdx) in inputChunks {
            let (enhanced, background) = try await processChunk(chunk)
            eval(enhanced, background)
            enhancedChunks.append((enhanced, startIdx))
            backgroundChunks.append((background, startIdx))
            
            // Clear GPU cache between chunks to reduce peak memory
            GPU.clearCache()
        }
        
        // Reassemble both using the same strategy
        let reassembledEnhanced: MLXArray
        let reassembledBackground: MLXArray
        
        switch strategy {
        case .discardEdges:
            let giveUp = overlapSamples / 2
            reassembledEnhanced = reassembleDiscardEdges(
                processedChunks: enhancedChunks,
                chunkSamples: chunkSamples,
                stride: stride,
                giveUp: giveUp,
                originalLength: totalLength
            )
            reassembledBackground = reassembleDiscardEdges(
                processedChunks: backgroundChunks,
                chunkSamples: chunkSamples,
                stride: stride,
                giveUp: giveUp,
                originalLength: totalLength
            )
            
        case .triangular:
            let window = triangularWindow(length: chunkSamples)
            reassembledEnhanced = reassembleOverlapAdd(
                processedChunks: enhancedChunks,
                chunkSamples: chunkSamples,
                stride: stride,
                window: window,
                originalLength: totalLength
            )
            reassembledBackground = reassembleOverlapAdd(
                processedChunks: backgroundChunks,
                chunkSamples: chunkSamples,
                stride: stride,
                window: window,
                originalLength: totalLength
            )
            
        default:  // noOverlap, overlapAdd, hann
            reassembledEnhanced = reassembleNoOverlap(processedChunks: enhancedChunks, originalLength: totalLength)
            reassembledBackground = reassembleNoOverlap(processedChunks: backgroundChunks, originalLength: totalLength)
        }
        
        return (reassembledEnhanced, reassembledBackground)
    }
}

// MARK: - Incremental Overlap-Add

/// Weighted overlap-add performed as chunks arrive, rather than over all of them.
///
/// The same arithmetic as ``MLXOverlap/reassembleOverlapAdd(processedChunks:chunkSamples:stride:window:originalLength:)``,
/// which is the point: a streaming path built on this produces exactly the samples
/// the batch path would, and `IncrementalOverlapAddTests` asserts that rather than
/// leaving it to inspection.
///
/// Providers had been hand-rolling their own streaming blend, and getting it wrong
/// in ways that only showed up as audio. The super-resolution provider emitted its
/// first chunk multiplied by the rising half of a Hann window and never divided by
/// the accumulated weight, so every stream opened with a `stride`-long fade-in from
/// silence - a second and a half at 48 kHz - that batch processing did not have.
///
/// A sample position receives contributions from every chunk that starts at or
/// before it and within `chunkSamples` of it. Once the chunk starting at `startIdx`
/// has been added, nothing below `startIdx + stride` can change, so that region is
/// final and ``add(_:startIdx:)`` returns it. The accumulator holds only the live
/// window, so memory is bounded by the chunk size rather than the length of the
/// audio.
///
/// ```swift
/// var assembler = IncrementalOverlapAdd(chunkSamples: n, stride: s,
///                                       window: w, totalLength: total)
/// for (chunk, startIdx) in chunks {
///     let ready = assembler.add(try process(chunk), startIdx: startIdx)
///     if !ready.isEmpty { yield(ready) }
/// }
/// yield(assembler.finish())
/// ```
public struct IncrementalOverlapAdd: Sendable {

    private let chunkSamples: Int
    private let stride: Int
    private let window: [Float]
    private let totalLength: Int

    /// Weighted sums for the live window, aligned so index 0 is `emittedCount`.
    private var accumulator: [Float]
    /// Accumulated window weight for the same positions, the divisor on the way out.
    private var weights: [Float]
    private var emittedCount = 0

    /// Samples emitted so far. Equals `totalLength` once ``finish()`` has run.
    public var emittedSamples: Int { emittedCount }

    /// - Parameters:
    ///   - chunkSamples: Length of each processed chunk.
    ///   - stride: Hop between consecutive chunk starts.
    ///   - window: Per-sample weight, `chunkSamples` long.
    ///   - totalLength: Length of the finished output, used to trim padding.
    public init(chunkSamples: Int, stride: Int, window: [Float], totalLength: Int) {
        self.chunkSamples = chunkSamples
        self.stride = max(1, stride)
        self.window = window
        self.totalLength = totalLength
        self.accumulator = [Float](repeating: 0, count: chunkSamples)
        self.weights = [Float](repeating: 0, count: chunkSamples)
    }

    /// Fold one processed chunk in, and return whatever is now final.
    ///
    /// - Parameters:
    ///   - chunk: Processed samples for the chunk beginning at `startIdx`.
    ///   - startIdx: Index of the chunk's first sample in the output.
    /// - Returns: Newly completed samples, normalized. Empty when nothing completed.
    public mutating func add(_ chunk: [Float], startIdx: Int) -> [Float] {
        let contributionCount = min(min(chunk.count, chunkSamples),
                                    max(0, totalLength - startIdx))
        let offset = startIdx - emittedCount
        for i in 0..<contributionCount {
            let position = offset + i
            guard position >= 0, position < accumulator.count else { continue }
            let weight = i < window.count ? window[i] : 1.0
            accumulator[position] += weight * chunk[i]
            weights[position] += weight
        }

        // Everything below the next chunk's start is complete.
        return drain(upTo: min(startIdx + stride, totalLength))
    }

    /// Emit the tail - the part of the last chunk beyond the final stride boundary.
    public mutating func finish() -> [Float] {
        drain(upTo: totalLength)
    }

    private mutating func drain(upTo endIndex: Int) -> [Float] {
        let readyCount = endIndex - emittedCount
        guard readyCount > 0 else { return [] }

        var ready = [Float](repeating: 0, count: readyCount)
        for i in 0..<min(readyCount, accumulator.count) where weights[i] > 0 {
            ready[i] = accumulator[i] / weights[i]
        }

        emittedCount = endIndex
        let shift = min(readyCount, accumulator.count)
        accumulator.removeFirst(shift)
        weights.removeFirst(shift)
        accumulator.append(contentsOf: [Float](repeating: 0, count: shift))
        weights.append(contentsOf: [Float](repeating: 0, count: shift))
        return ready
    }
}
