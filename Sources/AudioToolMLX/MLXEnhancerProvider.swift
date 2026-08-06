//
//  MLXEnhancerProvider.swift
//  AudioToolMLX
//
//  MLX-based speech enhancement providers with chunking support
//

import Foundation
import AudioTool
import AudioToolCore
@preconcurrency import MLX
@preconcurrency import MLXNN
@preconcurrency import AudioUtils  // SwiftAudio - used for audio I/O
@preconcurrency import Mossformer2MLXSwift
@preconcurrency import FRCRNMLXSwift

/// Result type for background extraction (enhanced + background audio)
public struct MLXEnhancedWithBackground: Sendable {
    public let enhanced: AudioBuffer
    public let background: AudioBuffer
    
    public init(enhanced: AudioBuffer, background: AudioBuffer) {
        self.enhanced = enhanced
        self.background = background
    }
}

// MARK: - MossFormer2 SE 48K Provider

/// MLX MossFormer2 Speech Enhancement (48kHz)
/// Chunking: 4s chunks, 25% overlap, discard-edges (from benchmarks)
public actor MossFormer2SE48KProvider: SpeechEnhancer {
    
    /// HuggingFace repository for model weights
    public static let repo = "starkdmi/MossFormer2_SE_48K_MLX"
    
    /// Supported precisions for this model
    public static let supportedPrecisions: [ModelPrecision] = [.fp32, .fp16]
    
    public nonisolated let sampleRate: Int = 48000
    public nonisolated let inputChannels: Int = 1

    // No declaration on purpose. The reference resamples with `scipy.signal.resample`
    // (`Models/python/mossformer2_se_mlx/generate.py:271`) - FFT-based - and the Swift
    // standalone does not resample at all, so there is no Swift behaviour to inherit.
    // AVAudioConverter Mastering is anti-aliased too, but "also anti-aliased" is not
    // "the same filter": it differs from scipy's method in transition band, stopband
    // and phase. Substituting it here would be reasoning from first principles about
    // aliasing, which is exactly what `ResamplingQuality` warns against. Declare once
    // measured against Python, not before.
    public nonisolated let outputChannels: Int = 1
    public nonisolated let minChunkSize: Int = 9600   // 0.2s at 48kHz
    public nonisolated let recommendedChunkSize: Int = 192000  // 4s at 48kHz (optimal from benchmarks)
    
    private var pipeline: MossFormer2Pipeline?
    private let weightsPath: String?
    private let precision: ModelPrecision
    
    /// Chunking config: 4s chunks, 25% overlap, discard-edges
    private let chunkingConfig = ChunkingConfig.mossformer2SE48K()
    
    /// Max audio duration that can be processed without chunking (seconds)
    private let maxDirectDuration: Float = 4.0
    
    /// Initialize with precision (auto-downloads from HuggingFace)
    public init(precision: ModelPrecision = .fp32) {
        self.weightsPath = nil
        self.precision = precision
    }
    
    /// Initialize with explicit weights path (no download)
    public init(weightsPath: String) {
        self.weightsPath = weightsPath
        self.precision = .fp32
    }
    
    /// Load model weights (downloads if not cached)
    public func load() async throws {
        // Disable normalization to match Python behavior - critical for clean background extraction
        let config = Mossformer2MLXSwift.PipelineConfiguration(
            enableFloat16: precision == .fp16,
            normalizationMode: .disabled
        )
        pipeline = MossFormer2Pipeline(configuration: config)
        
        let resolvedPath: String
        if let path = weightsPath {
            resolvedPath = path
        } else {
            // Check if already downloaded
            if let cached = ModelDownloader.shared.localPath(for: Self.repo) {
                let modelPath = cached.appendingPathComponent(precision.weightsFilename).path
                if FileManager.default.fileExists(atPath: modelPath) {
                    resolvedPath = modelPath
                } else {
                    throw AudioToolError.modelNotFound("Precision \(precision.rawValue) not available for MossFormer2SE48K")
                }
            } else {
                // Auto-download from HuggingFace
                let modelDir = try await ModelDownloader.shared.downloadAndGetPath(
                    repo: Self.repo,
                    matching: [precision.weightsFilename, "config.json"]
                )
                resolvedPath = modelDir.appendingPathComponent(precision.weightsFilename).path
            }
        }
        
        try pipeline?.loadWeights(from: resolvedPath)
    }
    
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        guard let pipeline = pipeline else {
            throw AudioToolError.modelNotLoaded("MossFormer2SE48K")
        }
        
        try validateSampleRate(input)
        let durationSeconds = Float(input.samples.count) / Float(input.sampleRate)
        
        // Use chunking for longer audio
        if durationSeconds > maxDirectDuration {
            return try await processWithChunking(input, pipeline: pipeline)
        }
        
        // Direct processing for short audio
        return try await processChunk(input.samples, pipeline: pipeline)
    }
    
    /// Process with chunking - direct port from Python benchmark_chunking.py
    /// Uses discard-edges strategy: keep center of each chunk, discard edges where overlap occurs
    private func processWithChunking(_ input: AudioBuffer, pipeline: MossFormer2Pipeline) async throws -> AudioBuffer {
        let audio = input.samples
        let totalLength = audio.count
        let chunkSamples = chunkingConfig.chunkSamples
        let stride = chunkingConfig.strideSamples
        let giveUp = chunkingConfig.overlapSamples / 2
        
        
        // Pre-allocate output
        var result = [Float](repeating: 0, count: totalLength)
        
        var currentIdx = 0
        var chunkCount = 0
        
        while currentIdx + chunkSamples <= totalLength + stride {
            let endIdx = min(currentIdx + chunkSamples, totalLength)
            
            // Extract and pad chunk
            var chunk = Array(audio[currentIdx..<endIdx])
            if chunk.count < chunkSamples {
                chunk.append(contentsOf: [Float](repeating: 0, count: chunkSamples - chunk.count))
            }
            
            // Process through model
            let processed = try await processChunk(chunk, pipeline: pipeline)
            let output = processed.samples
            
            // Determine valid range
            let validStart = currentIdx == 0 ? 0 : giveUp
            // validEnd would be: chunkSamples - giveUp (used implicitly below)
            let outputRangeStart = currentIdx == 0 ? 0 : currentIdx + giveUp
            let outputRangeEnd = currentIdx == 0 ? chunkSamples - giveUp : currentIdx + chunkSamples - giveUp
            
            // Copy valid portion
            let actualEnd = min(outputRangeEnd, totalLength)
            for i in outputRangeStart..<actualEnd {
                let srcIdx = validStart + (i - outputRangeStart)
                if srcIdx < output.count {
                    result[i] = output[srcIdx]
                }
            }
            
            currentIdx += stride
            chunkCount += 1
            
            // Clear GPU cache between chunks to reduce peak memory
            GPU.clearCache()
        }
        
        
        return AudioBuffer(samples: result, sampleRate: sampleRate, channels: 1)
    }
    
    /// Process a single chunk using pure MLX (no file I/O)
    private func processChunk(_ samples: [Float], pipeline: MossFormer2Pipeline) async throws -> AudioBuffer {
        let inputMLX = MLXArray(samples)
        
        // Use batch method with single array for pure MLX processing
        let results = try pipeline.enhanceAudioArrayBatch([inputMLX])
        
        guard let enhanced = results.first else {
            throw AudioToolError.pipelineConfigurationInvalid("SE 48K enhancement returned empty")
        }
        
        eval(enhanced)
        return AudioBuffer(samples: enhanced.asArray(Float.self), sampleRate: sampleRate, channels: 1)
    }
    
    public nonisolated func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            Task {
                for await chunk in input {
                    do {
                        let processed = try await process(chunk)
                        continuation.yield(processed)
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
                continuation.finish()
            }
        }
    }
    
    public func reset() async {}
    
    // MARK: - Background Extraction
    
    /// Process audio with background extraction using soft inverse mask
    /// Background = pow(1 - mask, gamma) * spectrum
    /// - Parameters:
    ///   - input: Input audio buffer
    ///   - gamma: Power exponent for background mask (default 1.10)
    ///            1.10 = Closest to GAN backgrounds
    ///            1.20 = Slightly more aggressive
    ///            1.40 = More aggressive, less background bleed
    /// - Returns: Enhanced audio with background track
    public func processWithBackground(_ input: AudioBuffer, gamma: Float = 1.10) async throws -> MLXEnhancedWithBackground {
        guard let pipeline = pipeline else {
            throw AudioToolError.modelNotLoaded("MossFormer2SE48K")
        }
        
        // Use direct processing with background extraction (matches Python implementation)
        let inputMLX = MLXArray(input.samples)
        let result = try pipeline.enhanceAudioWithBackground(inputMLX, gamma: gamma)
        eval(result.enhanced, result.background)
        
        return MLXEnhancedWithBackground(
            enhanced: AudioBuffer(samples: result.enhanced.asArray(Float.self), sampleRate: sampleRate, channels: 1),
            background: AudioBuffer(samples: result.background.asArray(Float.self), sampleRate: sampleRate, channels: 1)
        )
    }
}

// MARK: - StreamableOutput Conformance

extension MossFormer2SE48KProvider: StreamableOutput {
    /// Process audio and stream output chunks as they're ready.
    /// Uses discardEdges strategy - each chunk is independent after edge trimming.
    /// - Parameter input: Complete input audio to process
    /// - Returns: Async stream of processed audio chunks (in sequential order)
    public nonisolated func processStream(_ input: AudioBuffer) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.processStreamImpl(input, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    /// Internal implementation for streaming - runs within actor context
    private func processStreamImpl(_ input: AudioBuffer, continuation: AsyncThrowingStream<AudioBuffer, Error>.Continuation) async throws {
        guard let pipeline = pipeline else {
            throw AudioToolError.modelNotLoaded("MossFormer2SE48K")
        }
        
        let audio = input.samples
        let totalLength = audio.count
        let durationSeconds = Float(totalLength) / Float(sampleRate)
        
        // For short audio, just yield single result
        if durationSeconds <= maxDirectDuration {
            let result = try await processChunk(audio, pipeline: pipeline)
            continuation.yield(result)
            continuation.finish()
            return
        }
        
        // Streaming with chunking
        let chunkSamples = chunkingConfig.chunkSamples
        let stride = chunkingConfig.strideSamples
        let giveUp = chunkingConfig.overlapSamples / 2
        
        var currentIdx = 0
        var isFirst = true
        
        while currentIdx + chunkSamples <= totalLength + stride {
            let endIdx = min(currentIdx + chunkSamples, totalLength)
            
            // Extract and pad chunk
            var chunk = Array(audio[currentIdx..<endIdx])
            if chunk.count < chunkSamples {
                chunk.append(contentsOf: [Float](repeating: 0, count: chunkSamples - chunk.count))
            }
            
            // Process through model
            let processed = try await processChunk(chunk, pipeline: pipeline)
            let output = processed.samples
            
            // Determine valid range (discard edges)
            let validStart = isFirst ? 0 : giveUp
            let validEnd = min(chunkSamples - giveUp, output.count)
            
            // Calculate how much of the original audio this chunk covers
            let outputRangeStart = isFirst ? 0 : currentIdx + giveUp
            let outputRangeEnd = min(currentIdx + chunkSamples - giveUp, totalLength)
            let actualLen = outputRangeEnd - outputRangeStart
            
            // Extract valid portion
            if validEnd > validStart && actualLen > 0 {
                let trimmedSamples = Array(output[validStart..<min(validStart + actualLen, validEnd)])
                let chunkBuffer = AudioBuffer(
                    samples: trimmedSamples,
                    sampleRate: sampleRate,
                    channels: 1
                )
                continuation.yield(chunkBuffer)
            }
            
            // Clear GPU cache between chunks
            GPU.clearCache()
            
            currentIdx += stride
            isFirst = false
        }
        
        continuation.finish()
    }
}

// MARK: - ManagedModel Conformance

extension MossFormer2SE48KProvider: ManagedModel {
    /// Unique identifier for this model instance
    public nonisolated var modelId: String { "mossformer2_se_48k" }
    
    /// Estimated memory footprint in bytes (VRAM + RAM)
    /// ~200MB for FP16, ~400MB for FP32
    public nonisolated var estimatedMemoryBytes: Int {
        precision == .fp16 ? 200_000_000 : 400_000_000
    }
    
    /// Check if the model is currently loaded
    public func checkIfLoaded() async -> Bool { pipeline != nil }
    
    /// Unload model from memory
    public func unload() async {
        pipeline = nil
        GPU.clearCache()
    }
}

// MARK: - FRCRN SE 16K Provider

/// MLX FRCRN Speech Enhancement (16kHz)
/// Chunking: 4s chunks, 25% overlap, discard-edges (auto-enabled by default)
public actor FRCRNSE16KProvider: SpeechEnhancer {
    
    public nonisolated let sampleRate: Int = 16000
    public nonisolated let inputChannels: Int = 1

    // No declaration on purpose. The reference takes 16 kHz input directly and never
    // resamples, so nothing here was validated against any resampler - which makes
    // this the weakest possible case for choosing one. What FRCRN should get for
    // non-16 kHz input is an open question to answer with measurements.
    public nonisolated let outputChannels: Int = 1
    public nonisolated let minChunkSize: Int = 3200   // 0.2s at 16kHz
    public nonisolated let recommendedChunkSize: Int = 64000  // 4s at 16kHz
    
    private var model: FRCRN_SE_16K?
    private let weightsPath: String
    
    /// Chunking config: 4s chunks, 25% overlap, discard-edges
    private let chunkingConfig: ChunkingConfig
    
    public init(weightsPath: String) {
        self.weightsPath = weightsPath
        self.chunkingConfig = ChunkingConfig.frcrnSE16K(sampleRate: 16000)
    }
    
    /// Load model weights
    public func load() async throws {
        model = FRCRN_SE_16K()
        try model?.loadWeights(from: weightsPath)
        model?.prepareForInference()
        
        // Prewarm (JIT compilation)
        let dummy = MLXArray.zeros([1, sampleRate])
        let output = model?(dummy)
        eval(output ?? dummy)
        GPU.clearCache()
    }
    
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("FRCRN_SE_16K")
        }
        try validateSampleRate(input)
        
        // Always use chunking for consistent quality
        return try await processWithChunking(input, model: model)
    }
    
    /// Process with chunking using MLXOverlap
    /// Uses discard-edges strategy: keep center of each chunk, discard edges
    private func processWithChunking(_ input: AudioBuffer, model: FRCRN_SE_16K) async throws -> AudioBuffer {
        let inputMLX = MLXArray(input.samples)
        
        let result = try await MLXOverlap.processWithChunking(
            audio: inputMLX,
            chunkSamples: chunkingConfig.chunkSamples,
            overlapRatio: chunkingConfig.overlapRatio,
            strategy: .discardEdges
        ) { chunk in
            // Process chunk through FRCRN model
            let batchedChunk = chunk.reshaped([1, -1])
            let output = model(batchedChunk)
            eval(output)
            return output[0]
        }
        eval(result)
        
        return AudioBuffer(samples: result.asArray(Float.self), sampleRate: sampleRate, channels: 1)
    }
    
    /// Process a single chunk
    private func processChunk(_ samples: [Float], model: FRCRN_SE_16K) async throws -> AudioBuffer {
        let inputMLX = MLXArray(samples).reshaped([1, -1])
        let outputMLX = model(inputMLX)
        eval(outputMLX)
        
        var outputSamples = outputMLX.asArray(Float.self)
        if outputMLX.shape[0] == 1 && outputMLX.ndim == 2 {
            outputSamples = outputMLX[0].asArray(Float.self)
        }
        
        return AudioBuffer(samples: outputSamples, sampleRate: sampleRate, channels: 1)
    }
    
    public nonisolated func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            Task {
                for await chunk in input {
                    do {
                        let processed = try await process(chunk)
                        continuation.yield(processed)
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
                continuation.finish()
            }
        }
    }
    
    public func reset() async {}
    
    // MARK: - Background Extraction
    
    /// Process audio with background extraction
    /// Uses simple time-domain subtraction: background = input - enhanced
    /// - Parameter input: Input audio buffer
    /// - Returns: Enhanced audio with background track
    public func processWithBackground(_ input: AudioBuffer) async throws -> MLXEnhancedWithBackground {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("FRCRN_SE_16K")
        }
        
        let inputMLX = MLXArray(input.samples).reshaped([1, -1])
        
        // Get enhanced audio using the existing working forward pass
        let enhanced = model(inputMLX)
        eval(enhanced)
        
        // FRCRN output is slightly shorter due to STFT/iSTFT - truncate input to match
        let enhancedLen = enhanced.shape[1]
        let inputTruncated = inputMLX[0..., 0..<enhancedLen]
        
        // Background = input - enhanced (time-domain subtraction)
        let background = inputTruncated - enhanced
        eval(background)
        
        // Extract samples from batched output
        let enhancedSamples = enhanced[0].asArray(Float.self)
        let backgroundSamples = background[0].asArray(Float.self)
        
        return MLXEnhancedWithBackground(
            enhanced: AudioBuffer(samples: enhancedSamples, sampleRate: sampleRate, channels: 1),
            background: AudioBuffer(samples: backgroundSamples, sampleRate: sampleRate, channels: 1)
        )
    }
}

// MARK: - StreamableOutput Conformance

extension FRCRNSE16KProvider: StreamableOutput {
    /// Process audio and stream output chunks as they're ready.
    /// Uses discardEdges strategy - each chunk is independent after edge trimming.
    public nonisolated func processStream(_ input: AudioBuffer) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.processStreamImpl(input, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    /// Internal implementation for streaming - runs within actor context
    private func processStreamImpl(_ input: AudioBuffer, continuation: AsyncThrowingStream<AudioBuffer, Error>.Continuation) async throws {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("FRCRN_SE_16K")
        }
        
        let audio = input.samples
        let totalLength = audio.count
        let chunkSamples = chunkingConfig.chunkSamples
        let overlapSamples = chunkingConfig.overlapSamples
        let stride = chunkSamples - overlapSamples
        let giveUp = overlapSamples / 2
        
        // Split into chunks
        let inputMLX = MLXArray(audio)
        var currentIdx = 0
        var isFirst = true
        
        while currentIdx + chunkSamples <= totalLength + stride {
            let endIdx = min(currentIdx + chunkSamples, totalLength)
            
            // Extract chunk
            var chunk = inputMLX[currentIdx..<endIdx]
            eval(chunk)
            
            // Pad if needed
            let chunkLength = endIdx - currentIdx
            if chunkLength < chunkSamples {
                let padding = MLXArray.zeros([chunkSamples - chunkLength])
                chunk = concatenated([chunk, padding], axis: 0)
                eval(chunk)
            }
            
            // Process through FRCRN model
            let batchedChunk = chunk.reshaped([1, -1])
            let output = model(batchedChunk)
            eval(output)
            let processed = output[0]
            eval(processed)
            
            // Determine valid range (discard edges)
            let keepStart = isFirst ? 0 : giveUp
            let keepEnd = chunkSamples - giveUp
            
            // Calculate actual output length for this chunk
            let outputRangeStart = isFirst ? 0 : currentIdx + giveUp
            let outputRangeEnd = min(currentIdx + chunkSamples - giveUp, totalLength)
            let actualLen = outputRangeEnd - outputRangeStart
            
            if keepEnd > keepStart && actualLen > 0 {
                let trimmed = processed[keepStart..<min(keepStart + actualLen, keepEnd)]
                eval(trimmed)
                let trimmedSamples = trimmed.asArray(Float.self)
                
                let chunkBuffer = AudioBuffer(
                    samples: trimmedSamples,
                    sampleRate: sampleRate,
                    channels: 1
                )
                continuation.yield(chunkBuffer)
            }
            
            GPU.clearCache()
            
            currentIdx += stride
            isFirst = false
        }
        
        continuation.finish()
    }
}

// Note: MossFormer GAN SE is now available via AudioToolCoreML (CoreML-based provider)
// Use CoreMLProviders.mossformerGANSE16K() instead

// MARK: - ManagedModel

extension FRCRNSE16KProvider: ManagedModel {
    public nonisolated var modelId: String { "frcrn_se_16k" }

    /// ~60 MB FP32. FRCRN is the smallest of the enhancement models.
    public nonisolated var estimatedMemoryBytes: Int { 60_000_000 }

    public func checkIfLoaded() async -> Bool { model != nil }

    public func unload() async {
        model = nil
        GPU.clearCache()
    }
}
