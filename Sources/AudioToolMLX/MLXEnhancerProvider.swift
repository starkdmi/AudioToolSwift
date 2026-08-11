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
    public static let repo = ModelRepository.mossFormer2SE48K
    
    /// Supported precisions for this model.
    ///
    /// The quantized widths became loadable when `MossFormer2Pipeline` learned to
    /// apply `quantize` before the parameters arrive; before that they were
    /// published, unreferenced, and would have loaded into an unquantized model
    /// without complaining. See ``QuantizationParameters``.
    public static let supportedPrecisions: [ModelPrecision] = [.fp32, .fp16, .int8, .int6, .int4]
    
    public nonisolated let sampleRate: Int = 48000
    public nonisolated let inputChannels: Int = 1

    /// Matches the standalone generator, which builds `AudioLoader` without naming a
    /// resampling method and so gets `.auto` - cubic up, AVAudioConverter Normal at
    /// `.medium` down. `starkdmi/mossformer_se_mlx`, `swift/Sources/Generate/main.swift`.
    ///
    /// Not Mastering: the Swift port was validated against the Python reference
    /// (`scipy.signal.resample`) with the loader's ordinary settings, and that is the
    /// behaviour to reproduce, not the best-sounding one available.
    public nonisolated var preferredResamplingQuality: AudioToolCore.ResamplingQuality { .auto }
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
        // Before the first allocation, not at the first chunk boundary.
        // `trimIfNeeded` used to be the only thing that applied these, so a
        // provider ran its load and its opening chunks under MLX's own default
        // ceiling and the cache had already grown past the cap by the time the
        // cap arrived. Measured on Demucs: 5.1 GB peak applying it late against
        // 3.2 GB applying it here.
        MLXCachePolicy.applyProcessLimits()
        // Disable normalization to match Python behavior - critical for clean background extraction
        let config = Mossformer2MLXSwift.PipelineConfiguration(
            enableFloat16: precision == .fp16,
            normalizationMode: .disabled
        )
        let candidate = MossFormer2Pipeline(configuration: config)
        
        let resolvedPath: String
        if let path = weightsPath {
            resolvedPath = path
        } else {
            // Two different questions, two different lists. What must be present
            // to load is the weights alone - this pipeline's configuration is
            // built above, not read from a config.json. What to fetch when a
            // download is needed stays the full manifest, so a snapshot arrives
            // complete. See ModelFiles.standardRequired.
            if let cached = ModelDownloader.shared.localPath(
                for: Self.repo,
                matching: ModelFiles.standardRequired(precision)
            ) {
                resolvedPath = cached.appendingPathComponent(precision.weightsFilename).path
            } else {
                // Auto-download from HuggingFace
                let modelDir = try await ModelDownloader.shared.downloadAndGetPath(
                    repo: Self.repo,
                    matching: ModelFiles.standard(precision)
                )
                resolvedPath = modelDir.appendingPathComponent(precision.weightsFilename).path
            }
        }

        try candidate.loadWeights(from: resolvedPath)
        try Task.checkCancellation()
        pipeline = candidate
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
            
            MLXCachePolicy.trimIfNeeded(afterChunk: chunkCount)
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
            let producer = Task {
                for await chunk in input {
                    do {
                        try Task.checkCancellation()
                        let processed = try await process(chunk)
                        if case .terminated = continuation.yield(processed) { return }
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
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
        try validateInputFormat(input)
        
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
            let producer = Task {
                do {
                    try await self.processStreamImpl(input, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
        }
    }
    
    /// Internal implementation for streaming - runs within actor context
    private func processStreamImpl(_ input: AudioBuffer, continuation: AsyncThrowingStream<AudioBuffer, Error>.Continuation) async throws {
        guard let pipeline = pipeline else {
            throw AudioToolError.modelNotLoaded("MossFormer2SE48K")
        }
        try validateInputFormat(input)
        
        let audio = input.samples
        let totalLength = audio.count
        let durationSeconds = Float(totalLength) / Float(sampleRate)
        
        // For short audio, just yield single result
        if durationSeconds <= maxDirectDuration {
            let result = try await processChunk(audio, pipeline: pipeline)
            if case .terminated = continuation.yield(result) { throw CancellationError() }
            continuation.finish()
            return
        }
        
        // Streaming with chunking
        let chunkSamples = chunkingConfig.chunkSamples
        let stride = chunkingConfig.strideSamples
        let giveUp = chunkingConfig.overlapSamples / 2
        
        var currentIdx = 0
        var isFirst = true
        var completedChunks = 0
        
        while currentIdx + chunkSamples <= totalLength + stride {
            try Task.checkCancellation()
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
                if case .terminated = continuation.yield(chunkBuffer) { throw CancellationError() }
            }
            
            completedChunks += 1
            MLXCachePolicy.trimIfNeeded(afterChunk: completedChunks)
            
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
    
    /// Estimated memory footprint in bytes (VRAM + RAM).
    ///
    /// Measured peak, 30 s at 48 kHz on an M1 Pro: 1091 MiB at fp32, 847 at fp16,
    /// 901 at int8, 956 at int6, 939 at int4. The read of 200/400 MB was the
    /// checkpoint size; see `ManagedModel.estimatedMemoryBytes`.
    ///
    /// Two figures rather than five, because the spread across the quantized widths
    /// is 100 MiB and the residency manager is choosing whether to keep a model, not
    /// billing for it. Chunked at 4 s, so this does not grow with input length.
    public nonisolated var estimatedMemoryBytes: Int {
        precision == .fp32 ? 1_150_000_000 : 1_000_000_000
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

    /// Matches the standalone generator, which builds `AudioLoader` with
    /// `normalizationMode: .none` - to match Python's soundfile - and leaves the
    /// resampling method at `.auto`.
    public nonisolated var preferredResamplingQuality: AudioToolCore.ResamplingQuality { .auto }
    public nonisolated let outputChannels: Int = 1
    public nonisolated let minChunkSize: Int = 3200   // 0.2s at 16kHz
    public nonisolated let recommendedChunkSize: Int = 64000  // 4s at 16kHz
    
    /// HuggingFace repository for model weights
    public static let repo = ModelRepository.frcrnSE16K

    /// Supported precisions
    public static let supportedPrecisions: [ModelPrecision] = [.fp32]

    private var model: FRCRN_SE_16K?
    private let weightsPath: String?
    private let precision: ModelPrecision

    /// Chunking config: 4s chunks, 25% overlap, discard-edges
    private let chunkingConfig: ChunkingConfig

    /// Initialize with precision, downloading from HuggingFace on first load.
    ///
    /// The catalog has advertised this repo since it was written, but there was no
    /// code that could fetch it: the only initializer required a path the caller had
    /// already obtained somehow. Every other MLX provider here downloads; this one
    /// silently did not.
    public init(precision: ModelPrecision = .fp32) {
        self.weightsPath = nil
        self.precision = precision
        self.chunkingConfig = ChunkingConfig.frcrnSE16K(sampleRate: 16000)
    }

    /// Initialize with an explicit weights path (no download)
    public init(weightsPath: String) {
        self.weightsPath = weightsPath
        self.precision = .fp32
        self.chunkingConfig = ChunkingConfig.frcrnSE16K(sampleRate: 16000)
    }

    /// Load model weights (downloads if not cached)
    public func load() async throws {
        // Before the first allocation, not at the first chunk boundary.
        // `trimIfNeeded` used to be the only thing that applied these, so a
        // provider ran its load and its opening chunks under MLX's own default
        // ceiling and the cache had already grown past the cap by the time the
        // cap arrived. Measured on Demucs: 5.1 GB peak applying it late against
        // 3.2 GB applying it here.
        MLXCachePolicy.applyProcessLimits()
        let resolvedPath = try await resolveWeightsPath()
        let candidate = FRCRN_SE_16K()
        try candidate.loadWeights(from: resolvedPath)
        candidate.prepareForInference()
        
        // Prewarm (JIT compilation)
        let dummy = MLXArray.zeros([1, sampleRate])
        let output = candidate(dummy)
        eval(output)
        try Task.checkCancellation()
        model = candidate
        GPU.clearCache()
    }

    /// Explicit path if given, otherwise the cache, otherwise HuggingFace.
    private func resolveWeightsPath() async throws -> String {
        if let path = weightsPath { return path }

        // Present-to-load is the weights alone; `load()` above calls nothing but
        // `loadWeights`. The download manifest stays complete. See
        // ModelFiles.standardRequired.
        let filename = precision.weightsFilename
        if let cached = ModelDownloader.shared.localPath(
            for: Self.repo,
            matching: ModelFiles.standardRequired(precision)
        ) {
            return cached.appendingPathComponent(filename).path
        }
        let modelDir = try await ModelDownloader.shared.downloadAndGetPath(
            repo: Self.repo,
            matching: ModelFiles.standard(precision)
        )
        return modelDir.appendingPathComponent(filename).path
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
        let audio = input.samples
        let totalLength = audio.count
        let chunkSamples = chunkingConfig.chunkSamples
        let stride = chunkingConfig.strideSamples
        let giveUp = chunkingConfig.overlapSamples / 2
        var result = [Float](repeating: 0, count: totalLength)
        var currentIdx = 0
        var chunkIndex = 0

        // Keep the recording in its existing CPU AudioBuffer. A single MLXArray for
        // the complete input is recording-sized device memory once evaluated; only
        // the active chunk belongs in the MLX graph.
        while currentIdx + chunkSamples <= totalLength + stride {
            try Task.checkCancellation()
            let endIdx = min(currentIdx + chunkSamples, totalLength)
            var chunk = Array(audio[currentIdx..<endIdx])
            if chunk.count < chunkSamples {
                chunk.append(contentsOf: repeatElement(0, count: chunkSamples - chunk.count))
            }

            let processed = try await processChunk(chunk, model: model).samples
            let keepStart = chunkIndex == 0 ? 0 : giveUp
            let outputStart = chunkIndex == 0 ? 0 : currentIdx + giveUp
            let outputEnd = min(currentIdx + chunkSamples - giveUp, totalLength)
            let count = min(
                max(0, outputEnd - outputStart),
                max(0, processed.count - keepStart)
            )
            if count > 0 {
                result.replaceSubrange(
                    outputStart..<(outputStart + count),
                    with: processed[keepStart..<(keepStart + count)]
                )
            }

            currentIdx += stride
            chunkIndex += 1
            MLXCachePolicy.trimIfNeeded(afterChunk: chunkIndex)
        }

        return AudioBuffer(samples: result, sampleRate: sampleRate, channels: 1)
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
            let producer = Task {
                for await chunk in input {
                    do {
                        try Task.checkCancellation()
                        let processed = try await process(chunk)
                        if case .terminated = continuation.yield(processed) { return }
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
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
        try validateInputFormat(input)
        
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
            let producer = Task {
                do {
                    try await self.processStreamImpl(input, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
        }
    }
    
    /// Internal implementation for streaming - runs within actor context
    private func processStreamImpl(_ input: AudioBuffer, continuation: AsyncThrowingStream<AudioBuffer, Error>.Continuation) async throws {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("FRCRN_SE_16K")
        }
        try validateInputFormat(input)
        
        let audio = input.samples
        let totalLength = audio.count
        let chunkSamples = chunkingConfig.chunkSamples
        let overlapSamples = chunkingConfig.overlapSamples
        let stride = chunkSamples - overlapSamples
        let giveUp = overlapSamples / 2
        
        var currentIdx = 0
        var isFirst = true
        var completedChunks = 0
        
        while currentIdx + chunkSamples <= totalLength + stride {
            try Task.checkCancellation()
            let endIdx = min(currentIdx + chunkSamples, totalLength)
            
            // Keep the complete recording on CPU and materialize just this chunk.
            var chunk = Array(audio[currentIdx..<endIdx])
            if chunk.count < chunkSamples {
                chunk.append(contentsOf: repeatElement(0, count: chunkSamples - chunk.count))
            }

            let processed = try await processChunk(chunk, model: model).samples
            
            // Determine valid range (discard edges)
            let keepStart = isFirst ? 0 : giveUp
            let keepEnd = chunkSamples - giveUp
            
            // Calculate actual output length for this chunk
            let outputRangeStart = isFirst ? 0 : currentIdx + giveUp
            let outputRangeEnd = min(currentIdx + chunkSamples - giveUp, totalLength)
            let actualLen = outputRangeEnd - outputRangeStart
            
            if keepEnd > keepStart && actualLen > 0 {
                let trimmedEnd = min(keepStart + actualLen, keepEnd, processed.count)
                if trimmedEnd > keepStart {
                    let chunkBuffer = AudioBuffer(
                        samples: Array(processed[keepStart..<trimmedEnd]),
                        sampleRate: sampleRate,
                        channels: 1
                    )
                    if case .terminated = continuation.yield(chunkBuffer) { throw CancellationError() }
                }
            }
            
            completedChunks += 1
            MLXCachePolicy.trimIfNeeded(afterChunk: completedChunks)
            
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
    /// Measured peak 2522 MiB, 30 s at 16 kHz on an M1 Pro - against the 60 MB
    /// this declared, which was the checkpoint size. The widest gap in the package:
    /// FRCRN's STFT/ISTFT buffers dwarf its 56 MB of weights. Chunked at 4 s, so it
    /// does not grow with input length.
    public nonisolated var estimatedMemoryBytes: Int { 2_600_000_000 }

    public func checkIfLoaded() async -> Bool { model != nil }

    public func unload() async {
        model = nil
        GPU.clearCache()
    }
}
