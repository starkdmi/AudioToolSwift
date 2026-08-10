//
//  AudioChunking.swift
//  AudioToolCore
//
//  Audio chunking utilities with various overlap strategies
//

import Foundation

// MARK: - Chunking Configuration

/// Overlap blending strategy
public enum ChunkBlendingStrategy: Sendable {
    /// No overlap - simple concatenation
    case none
    /// Hann window overlap-add
    case hann
    /// Triangular window crossfade
    case triangular
    /// Discard edges and use center
    case discardEdges
}

/// Configuration for chunked audio processing
public struct ChunkingConfig: Sendable {
    /// Chunk duration in seconds
    public let chunkDuration: Float
    
    /// Overlap ratio (0.0 = no overlap, 0.5 = 50% overlap)
    public let overlapRatio: Float
    
    /// Blending strategy for overlapping regions
    public let blendingStrategy: ChunkBlendingStrategy
    
    /// Sample rate
    public let sampleRate: Int
    
    public init(
        chunkDuration: Float,
        overlapRatio: Float = 0.25,
        blendingStrategy: ChunkBlendingStrategy = .triangular,
        sampleRate: Int
    ) {
        precondition(chunkDuration > 0, "Chunk duration must be positive")
        precondition(overlapRatio >= 0 && overlapRatio < 1, "Overlap ratio must be in [0, 1)")
        precondition(sampleRate > 0, "Sample rate must be positive")
        precondition(
            Int(chunkDuration * Float(sampleRate)) > 0,
            "Chunk duration must contain at least one sample"
        )
        self.chunkDuration = chunkDuration
        self.overlapRatio = overlapRatio
        self.blendingStrategy = blendingStrategy
        self.sampleRate = sampleRate
    }
    
    /// Chunk size in samples
    public var chunkSamples: Int {
        Int(chunkDuration * Float(sampleRate))
    }
    
    /// Overlap size in samples
    public var overlapSamples: Int {
        Int(Float(chunkSamples) * overlapRatio)
    }
    
    /// Stride between chunks (chunk size - overlap)
    public var strideSamples: Int {
        chunkSamples - overlapSamples
    }
}

// MARK: - Audio Chunker

/// Splits audio into overlapping chunks and reassembles after processing
public struct AudioChunker: Sendable {
    
    public let config: ChunkingConfig
    
    public init(config: ChunkingConfig) {
        self.config = config
    }
    
    /// Split audio into chunks
    /// - Parameter samples: Input audio samples
    /// - Returns: Array of chunks with metadata
    public func split(_ samples: [Float]) -> [AudioChunk] {
        guard samples.count > config.chunkSamples else {
            // Audio is shorter than chunk size
            return [AudioChunk(samples: samples, startIndex: 0, isFirst: true, isLast: true)]
        }
        
        var chunks: [AudioChunk] = []
        var position = 0
        let totalLength = samples.count
        
        while position < totalLength {
            let endPosition = min(position + config.chunkSamples, totalLength)
            let chunkLength = endPosition - position
            
            // A chunk's first sample always corresponds to `startIndex`. The old
            // tail path prepended earlier context but kept the later start index,
            // shifting that context forward during reconstruction and duplicating
            // the end of the recording.
            var chunkSamples = Array(samples[position..<endPosition])
            if chunkLength < config.chunkSamples {
                chunkSamples.append(contentsOf: [Float](
                    repeating: 0,
                    count: config.chunkSamples - chunkLength
                ))
            }
            
            let chunk = AudioChunk(
                samples: chunkSamples,
                startIndex: position,
                isFirst: position == 0,
                isLast: false  // Will set the true last chunk after loop
            )
            chunks.append(chunk)
            
            position += config.strideSamples
            
            // Prevent infinite loop for zero stride
            if config.strideSamples <= 0 {
                break
            }
        }
        
        // Mark the actual last chunk
        if !chunks.isEmpty {
            chunks[chunks.count - 1].isLast = true
        }
        
        return chunks
    }
    
    /// Reassemble processed chunks using configured blending strategy
    /// - Parameters:
    ///   - chunks: Processed chunks
    ///   - originalLength: Original audio length
    /// - Returns: Reassembled audio
    public func reassemble(_ chunks: [AudioChunk], originalLength: Int) -> [Float] {
        guard !chunks.isEmpty else { return [] }
        
        switch config.blendingStrategy {
        case .none:
            return reassembleNoOverlap(chunks, originalLength: originalLength)
        case .hann:
            return reassembleHann(chunks, originalLength: originalLength)
        case .triangular:
            return reassembleTriangular(chunks, originalLength: originalLength)
        case .discardEdges:
            return reassembleDiscardEdges(chunks, originalLength: originalLength)
        }
    }
    
    // MARK: - Reassembly Strategies
    
    private func reassembleNoOverlap(_ chunks: [AudioChunk], originalLength: Int) -> [Float] {
        var output: [Float] = []
        
        for (idx, chunk) in chunks.enumerated() {
            if idx == chunks.count - 1 {
                // Last chunk - only take what we need
                let remaining = originalLength - output.count
                output.append(contentsOf: chunk.samples.prefix(remaining))
            } else {
                // Take stride worth of samples
                output.append(contentsOf: chunk.samples.prefix(config.strideSamples))
            }
        }
        
        return Array(output.prefix(originalLength))
    }
    
    private func reassembleHann(_ chunks: [AudioChunk], originalLength: Int) -> [Float] {
        guard config.overlapSamples > 0 else {
            return reassembleNoOverlap(chunks, originalLength: originalLength)
        }
        
        let outputLength = (chunks.count - 1) * config.strideSamples + config.chunkSamples
        var output = [Float](repeating: 0, count: outputLength)
        var windowSum = [Float](repeating: 0, count: outputLength)
        
        // Create Hann window
        let window = hannWindow(config.chunkSamples)
        
        for chunk in chunks {
            let start = chunk.startIndex
            
            for i in 0..<chunk.samples.count {
                if start + i < outputLength {
                    var weight = window[i]
                    // A Hann window is zero at both ends. At the outer recording
                    // boundaries there is no neighbouring chunk to supply those
                    // samples, so use a flat edge there. Interior endpoints are
                    // covered by the adjacent chunk and retain the Hann blend.
                    if chunk.isFirst && i < config.strideSamples {
                        weight = 1
                    } else if chunk.isLast && i >= config.overlapSamples {
                        weight = 1
                    }
                    output[start + i] += chunk.samples[i] * weight
                    windowSum[start + i] += weight
                }
            }
        }
        
        // Normalize by window sum
        for i in 0..<output.count {
            if windowSum[i] > 1e-8 {
                output[i] /= windowSum[i]
            }
        }
        
        return Array(output.prefix(originalLength))
    }
    
    private func reassembleTriangular(_ chunks: [AudioChunk], originalLength: Int) -> [Float] {
        guard config.overlapSamples > 0 else {
            return reassembleNoOverlap(chunks, originalLength: originalLength)
        }
        
        // Use overlap-add style approach for triangular blending
        let outputLength = max(originalLength, (chunks.count - 1) * config.strideSamples + config.chunkSamples)
        var output = [Float](repeating: 0, count: outputLength)
        var weightSum = [Float](repeating: 0, count: outputLength)
        
        for chunk in chunks {
            let start = chunk.startIndex
            
            for i in 0..<chunk.samples.count {
                let pos = start + i
                if pos < outputLength {
                    // Triangular weight: ramps up in first half, down in second half
                    let halfLen = chunk.samples.count / 2
                    let weight: Float
                    if chunk.samples.count <= 1 {
                        weight = 1
                    } else if i < halfLen {
                        weight = Float(i + 1) / Float(halfLen)
                    } else {
                        weight = Float(chunk.samples.count - i) / Float(halfLen)
                    }
                    
                    output[pos] += chunk.samples[i] * weight
                    weightSum[pos] += weight
                }
            }
        }
        
        // Normalize
        for i in 0..<outputLength {
            if weightSum[i] > 1e-8 {
                output[i] /= weightSum[i]
            }
        }
        
        return Array(output.prefix(originalLength))
    }
    
    private func reassembleDiscardEdges(_ chunks: [AudioChunk], originalLength: Int) -> [Float] {
        guard config.overlapSamples > 0 else {
            return reassembleNoOverlap(chunks, originalLength: originalLength)
        }
        
        // Pre-allocate output buffer
        var output = [Float](repeating: 0, count: originalLength)
        let giveUp = config.overlapSamples / 2
        
        // Following Python implementation exactly:
        // First chunk: keep [0, chunkSamples - giveUp), write to [0, chunkSamples - giveUp)
        // Other chunks: keep [giveUp, chunkSamples - giveUp), write to [startIndex + giveUp, startIndex + chunkSamples - giveUp)
        
        for (idx, chunk) in chunks.enumerated() {
            let isFirst = idx == 0
            
            // Calculate what range of the output chunk to use
            let keepStart = isFirst ? 0 : giveUp
            let keepEnd = config.chunkSamples - giveUp  // NOT isLast dependent - always discard end except via capping
            
            // Calculate where in original audio this maps to
            let outputStart = chunk.startIndex + keepStart
            let outputEnd = min(chunk.startIndex + keepEnd, originalLength)
            
            // Copy samples
            for outPos in outputStart..<outputEnd {
                let sampleIdx = outPos - chunk.startIndex
                if sampleIdx >= 0 && sampleIdx < chunk.samples.count {
                    output[outPos] = chunk.samples[sampleIdx]
                }
            }
        }
        
        return output
    }
    
    // MARK: - Window Functions
    
    /// Uses `sin²(πi/(N-1))` rather than `0.5(1 - cos(2πi/(N-1)))`: the two are
    /// equal on paper, but in Float32 `cos(x)` rounds to exactly 1 for small `x`,
    /// so the subtraction cancels to zero over the opening samples of a long
    /// window. See ``MLXOverlap/hannWindow(length:)``.
    private func hannWindow(_ length: Int) -> [Float] {
        guard length > 1 else {
            return [Float](repeating: 1, count: max(0, length))
        }
        var window = [Float](repeating: 0, count: length)
        for i in 0..<length {
            let s = sin(Float.pi * Float(i) / Float(length - 1))
            window[i] = s * s
        }
        return window
    }
}

// MARK: - Audio Chunk

/// A chunk of audio with metadata
public struct AudioChunk: Sendable {
    /// Audio samples
    public var samples: [Float]
    
    /// Start index in original audio
    public let startIndex: Int
    
    /// Is this the first chunk
    public let isFirst: Bool
    
    /// Is this the last chunk
    public var isLast: Bool
    
    public init(samples: [Float], startIndex: Int, isFirst: Bool, isLast: Bool) {
        self.samples = samples
        self.startIndex = startIndex
        self.isFirst = isFirst
        self.isLast = isLast
    }
}

// MARK: - Predefined Configurations

public extension ChunkingConfig {
    
    /// Demucs: 7.8s chunks, 25% overlap, triangular blending
    static func demucs(sampleRate: Int = 44100) -> ChunkingConfig {
        ChunkingConfig(
            chunkDuration: 7.8,
            overlapRatio: 0.25,
            blendingStrategy: .triangular,
            sampleRate: sampleRate
        )
    }
    
    /// MossFormer2 SE 48K: 4s chunks, 25% overlap, discard-edges
    static func mossformer2SE48K(sampleRate: Int = 48000) -> ChunkingConfig {
        ChunkingConfig(
            chunkDuration: 4.0,
            overlapRatio: 0.25,
            blendingStrategy: .discardEdges,
            sampleRate: sampleRate
        )
    }
    
    /// MossFormer2 SR 48K: 4s chunks, 25% overlap, discard-edges.
    ///
    /// These are the reference's numbers, not a local choice. `generate.py` in
    /// `starkdmi/MossFormer2_SR_48K_MLX` sets `decode_window = 4.0`, advances by
    /// `stride = int(window * 0.75)` - so a 25% overlap - and assembles with
    /// `give_up_length = (window - stride) // 2`, discarding half the overlap at
    /// each edge with no blending at all.
    ///
    /// This was 50% overlap with a Hann overlap-add until it was checked against
    /// that file. Nothing recorded why, the comment said only "from benchmarks",
    /// and it made SR the only model here using a strategy no reference uses. It
    /// also cost throughput: halving the stride is 1.5x the model calls, and every
    /// output sample was a weighted sum of two model runs instead of one. End to
    /// end the correction is worth less than that - about 1.10x on a 30 s file -
    /// because roughly 80% of this provider's time is the 16 -> 48 kHz resample and
    /// `bandwidthSub`, neither of which chunking touches.
    static func mossformer2SR48K(sampleRate: Int = 48000) -> ChunkingConfig {
        ChunkingConfig(
            chunkDuration: 4.0,
            overlapRatio: 0.25,
            blendingStrategy: .discardEdges,
            sampleRate: sampleRate
        )
    }
    
    /// MossFormer2 SS: 4s chunks, 25% overlap, triangular
    static func mossformer2SS(sampleRate: Int = 16000) -> ChunkingConfig {
        ChunkingConfig(
            chunkDuration: 4.0,
            overlapRatio: 0.25,
            blendingStrategy: .triangular,
            sampleRate: sampleRate
        )
    }
    
    /// FRCRN SE 16K: 4s chunks, 25% overlap, discard-edges
    static func frcrnSE16K(sampleRate: Int = 16000) -> ChunkingConfig {
        ChunkingConfig(
            chunkDuration: 4.0,
            overlapRatio: 0.25,
            blendingStrategy: .discardEdges,
            sampleRate: sampleRate
        )
    }
    
    /// GAN SE CoreML: 1.594s chunks, NO overlap
    static func ganSECoreML(sampleRate: Int = 16000) -> ChunkingConfig {
        ChunkingConfig(
            chunkDuration: 1.594,
            overlapRatio: 0.0,
            blendingStrategy: .none,
            sampleRate: sampleRate
        )
    }
}
