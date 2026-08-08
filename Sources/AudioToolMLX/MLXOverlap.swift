//
//  MLXOverlap.swift
//  AudioToolMLX
//
//  MLX-based overlap-add utilities for audio chunking.
//  Ported from Python benchmark_chunking.py using vectorized MLX operations.
//

import Foundation
import AudioToolCore
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
        guard length > 1 else {
            return MLXArray([Float](repeating: 1, count: max(0, length)))
        }
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
        guard length > 1 else {
            return MLXArray([Float](repeating: 1, count: max(0, length)))
        }
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

            // Weighted overlap-add. The window applies unmodified at the edges -
            // see the note in `IncrementalOverlapAdd.add` for why forcing them to
            // 1 costs 25 dB of parity.
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
        guard chunkSamples > 0, overlapRatio >= 0, overlapRatio < 1 else {
            throw AudioToolError.pipelineConfigurationInvalid(
                "Chunk size must be positive and overlap must be in [0, 1)"
            )
        }
        eval(audio)
        let totalLength = audio.shape[0]
        guard totalLength > 0 else { return MLXArray.zeros([0]) }
        let overlapSamples = strategy == .noOverlap ? 0 : Int(Float(chunkSamples) * overlapRatio)
        let stride = chunkSamples - overlapSamples
        let window = windowArray(for: strategy, length: chunkSamples)
        var incremental = window.map {
            IncrementalOverlapAdd(
                chunkSamples: chunkSamples,
                stride: stride,
                window: $0,
                totalLength: totalLength
            )
        }
        var output = window == nil ? [Float](repeating: 0, count: totalLength) : []
        var streamedOutput: [Float] = []
        if window != nil { streamedOutput.reserveCapacity(totalLength) }

        var startIdx = 0
        var chunkIndex = 0
        while shouldProcessChunk(
            startIdx: startIdx,
            totalLength: totalLength,
            chunkSamples: chunkSamples,
            stride: stride,
            strategy: strategy
        ) {
            try Task.checkCancellation()
            let chunk = paddedChunk(audio, startIdx: startIdx, length: chunkSamples)
            let processed = try await processChunk(chunk)
            eval(processed)
            let samples = processed.asArray(Float.self)
            consume(
                samples,
                startIdx: startIdx,
                chunkIndex: chunkIndex,
                strategy: strategy,
                chunkSamples: chunkSamples,
                stride: stride,
                totalLength: totalLength,
                incremental: &incremental,
                output: &output,
                streamedOutput: &streamedOutput
            )
            MLXCachePolicy.trimIfNeeded(afterChunk: chunkIndex + 1)
            startIdx += stride
            chunkIndex += 1
        }

        if var incremental {
            streamedOutput.append(contentsOf: incremental.finish())
            if streamedOutput.count < totalLength {
                streamedOutput.append(contentsOf: repeatElement(0, count: totalLength - streamedOutput.count))
            }
            return MLXArray(Array(streamedOutput.prefix(totalLength)))
        }
        return MLXArray(output)
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
        guard chunkSamples > 0, overlapRatio >= 0, overlapRatio < 1 else {
            throw AudioToolError.pipelineConfigurationInvalid(
                "Chunk size must be positive and overlap must be in [0, 1)"
            )
        }
        eval(audio)
        let totalLength = audio.shape[0]
        guard totalLength > 0 else { return (MLXArray.zeros([0]), MLXArray.zeros([0])) }
        let overlapSamples = strategy == .noOverlap ? 0 : Int(Float(chunkSamples) * overlapRatio)
        let stride = chunkSamples - overlapSamples
        let window = windowArray(for: strategy, length: chunkSamples)
        var enhancedIncremental = window.map {
            IncrementalOverlapAdd(chunkSamples: chunkSamples, stride: stride, window: $0, totalLength: totalLength)
        }
        var backgroundIncremental = window.map {
            IncrementalOverlapAdd(chunkSamples: chunkSamples, stride: stride, window: $0, totalLength: totalLength)
        }
        var enhancedOutput = window == nil ? [Float](repeating: 0, count: totalLength) : []
        var backgroundOutput = window == nil ? [Float](repeating: 0, count: totalLength) : []
        var enhancedStreamed: [Float] = []
        var backgroundStreamed: [Float] = []
        if window != nil {
            enhancedStreamed.reserveCapacity(totalLength)
            backgroundStreamed.reserveCapacity(totalLength)
        }

        var startIdx = 0
        var chunkIndex = 0
        while shouldProcessChunk(
            startIdx: startIdx,
            totalLength: totalLength,
            chunkSamples: chunkSamples,
            stride: stride,
            strategy: strategy
        ) {
            try Task.checkCancellation()
            let chunk = paddedChunk(audio, startIdx: startIdx, length: chunkSamples)
            let (enhanced, background) = try await processChunk(chunk)
            eval(enhanced, background)
            consume(
                enhanced.asArray(Float.self),
                startIdx: startIdx,
                chunkIndex: chunkIndex,
                strategy: strategy,
                chunkSamples: chunkSamples,
                stride: stride,
                totalLength: totalLength,
                incremental: &enhancedIncremental,
                output: &enhancedOutput,
                streamedOutput: &enhancedStreamed
            )
            consume(
                background.asArray(Float.self),
                startIdx: startIdx,
                chunkIndex: chunkIndex,
                strategy: strategy,
                chunkSamples: chunkSamples,
                stride: stride,
                totalLength: totalLength,
                incremental: &backgroundIncremental,
                output: &backgroundOutput,
                streamedOutput: &backgroundStreamed
            )
            MLXCachePolicy.trimIfNeeded(afterChunk: chunkIndex + 1)
            startIdx += stride
            chunkIndex += 1
        }

        if var enhancedIncremental, var backgroundIncremental {
            enhancedStreamed.append(contentsOf: enhancedIncremental.finish())
            backgroundStreamed.append(contentsOf: backgroundIncremental.finish())
            return (
                MLXArray(Array(enhancedStreamed.prefix(totalLength))),
                MLXArray(Array(backgroundStreamed.prefix(totalLength)))
            )
        }
        return (MLXArray(enhancedOutput), MLXArray(backgroundOutput))
    }

    private static func windowArray(for strategy: OverlapStrategy, length: Int) -> [Float]? {
        switch strategy {
        case .noOverlap, .discardEdges:
            return nil
        case .overlapAdd:
            return [Float](repeating: 1, count: length)
        case .triangular:
            return triangularWindow(length: length).asArray(Float.self)
        case .hann:
            return hannWindow(length: length).asArray(Float.self)
        }
    }

    private static func shouldProcessChunk(
        startIdx: Int,
        totalLength: Int,
        chunkSamples: Int,
        stride: Int,
        strategy: OverlapStrategy
    ) -> Bool {
        guard startIdx > 0 else { return totalLength > 0 }
        if strategy == .discardEdges {
            return startIdx + chunkSamples <= totalLength + stride
        }
        return startIdx < totalLength
    }

    private static func paddedChunk(_ audio: MLXArray, startIdx: Int, length: Int) -> MLXArray {
        let endIdx = min(startIdx + length, audio.shape[0])
        var chunk = audio[startIdx..<endIdx]
        if endIdx - startIdx < length {
            chunk = concatenated([chunk, MLXArray.zeros([length - (endIdx - startIdx)])], axis: 0)
        }
        eval(chunk)
        return chunk
    }

    private static func consume(
        _ samples: [Float],
        startIdx: Int,
        chunkIndex: Int,
        strategy: OverlapStrategy,
        chunkSamples: Int,
        stride: Int,
        totalLength: Int,
        incremental: inout IncrementalOverlapAdd?,
        output: inout [Float],
        streamedOutput: inout [Float]
    ) {
        if var assembler = incremental {
            streamedOutput.append(contentsOf: assembler.add(samples, startIdx: startIdx))
            incremental = assembler
            return
        }

        switch strategy {
        case .noOverlap:
            let count = min(samples.count, max(0, totalLength - startIdx))
            guard count > 0 else { return }
            output.replaceSubrange(startIdx..<(startIdx + count), with: samples.prefix(count))
        case .discardEdges:
            let giveUp = (chunkSamples - stride) / 2
            let keepStart = chunkIndex == 0 ? 0 : giveUp
            let outputStart = chunkIndex == 0 ? 0 : startIdx + giveUp
            let nominalEnd = startIdx + chunkSamples - giveUp
            let outputEnd = min(nominalEnd, totalLength)
            guard outputEnd > outputStart else { return }
            let count = min(outputEnd - outputStart, max(0, samples.count - keepStart))
            guard count > 0 else { return }
            output.replaceSubrange(
                outputStart..<(outputStart + count),
                with: samples[keepStart..<(keepStart + count)]
            )
        case .overlapAdd, .triangular, .hann:
            break
        }
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

    /// Weighted sums for the live window in circular storage. `head` is the
    /// physical slot corresponding to `emittedCount`.
    private var accumulator: [Float]
    /// Accumulated window weight for the same positions, the divisor on the way out.
    private var weights: [Float]
    private var head = 0
    private var emittedCount = 0

    /// Samples emitted so far. Equals `totalLength` once ``finish()`` has run.
    public var emittedSamples: Int { emittedCount }

    /// - Parameters:
    ///   - chunkSamples: Length of each processed chunk.
    ///   - stride: Hop between consecutive chunk starts.
    ///   - window: Per-sample weight, `chunkSamples` long.
    ///   - totalLength: Length of the finished output, used to trim padding.
    public init(chunkSamples: Int, stride: Int, window: [Float], totalLength: Int) {
        precondition(chunkSamples > 0, "Chunk length must be positive")
        precondition(totalLength >= 0, "Total length must not be negative")
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
        // The window applies unmodified at the edges, as it does everywhere else.
        //
        // Forcing the first and last stride to weight 1 looks like it should be
        // harmless - those regions have a single contributor, so `w * x / w == x`
        // either way - and it is, except at the one sample per end where the Hann
        // window is exactly zero. There the reference leaves silence (`sum_weight`
        // is 0, so its mask skips the divide) and forcing emits a full-scale
        // sample instead. One sample was worth 25 dB: it put SR parity at 69.5 dB
        // against a 95 dB floor.
        //
        // The Float32 cancellation that motivated the forcing was already fixed in
        // 53d723b, by building the window as sin^2 rather than 0.5*(1-cos).
        for i in 0..<contributionCount {
            let position = offset + i
            guard position >= 0, position < accumulator.count else { continue }
            let storageIndex = (head + position) % accumulator.count
            let weight = i < window.count ? window[i] : 1.0
            accumulator[storageIndex] += weight * chunk[i]
            weights[storageIndex] += weight
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
        let shift = min(readyCount, accumulator.count)
        for i in 0..<shift {
            let storageIndex = (head + i) % accumulator.count
            if weights[storageIndex] > 0 {
                ready[i] = accumulator[storageIndex] / weights[storageIndex]
            }
            accumulator[storageIndex] = 0
            weights[storageIndex] = 0
        }

        emittedCount = endIndex
        head = (head + shift) % accumulator.count
        return ready
    }
}
