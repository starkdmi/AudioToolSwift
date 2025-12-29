//
//  MLXOverlap.swift
//  ClearVoiceMLX
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
    public static func hannWindow(length: Int) -> MLXArray {
        var window = [Float](repeating: 0, count: length)
        for i in 0..<length {
            window[i] = 0.5 * (1 - cos(2 * Float.pi * Float(i) / Float(length - 1)))
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
            let keepEnd = chunkSamples - giveUp
            
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
        
        // Process each chunk
        var processedChunks: [(chunk: MLXArray, startIdx: Int)] = []
        for (chunk, startIdx) in inputChunks {
            let processed = try await processChunk(chunk)
            eval(processed)
            processedChunks.append((processed, startIdx))
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
