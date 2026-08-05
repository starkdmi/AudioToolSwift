//
//  MLXSeparatorProvider.swift
//  AudioToolMLX
//
//  MLX-based speech separation and source separation providers with chunking
//

import Foundation
import AudioTool
import AudioToolCore
@preconcurrency import MLX
@preconcurrency import MLXNN
@preconcurrency import AudioUtils
@preconcurrency import MossFormer2SS
@preconcurrency import DemucsMLXSwift

// MARK: - MossFormer2 Speaker Separation Provider

/// MLX MossFormer2 Speaker Separation - supports 2spk, 3spk, and WHAMR models
/// Chunking: 4s chunks, 25% overlap, triangular blending
///
/// Supports progress reporting during chunked processing via `separate(_:speakers:onProgress:)`.
/// For streaming output, use `separateStream(_:)` which yields per-speaker results as chunks complete.
public actor MossFormer2SSProvider: SpeechSeparator, ChunkedProgressProvider {
    
    // MARK: - ChunkedProgressProvider Conformance
    
    /// MossFormer2SS supports chunked progress during separation
    public nonisolated var supportsChunkedProgress: Bool { true }
    
    // SpeechSeparator conformance. Fixed by the loaded weights, which is why it is
    // not a per-call argument.
    public nonisolated var outputCount: Int { modelType.numSpeakers }
    
    /// Available speaker separation models
    public enum Model: String, CaseIterable, Sendable {
        case twoSpeaker = "2spk"          // 2 speakers at 16kHz
        case threeSpeaker = "3spk"         // 3 speakers at 8kHz
        case twoSpeakerWHAMR = "2spk-whamr" // 2 speakers at 8kHz with WHAMR
        
        public var numSpeakers: Int {
            switch self {
            case .twoSpeaker: return 2
            case .threeSpeaker: return 3
            case .twoSpeakerWHAMR: return 2
            }
        }
        
        public var sampleRate: Int {
            switch self {
            case .twoSpeaker: return 16000
            case .threeSpeaker: return 8000
            case .twoSpeakerWHAMR: return 8000
            }
        }
        
        public var skipMaskMultiplication: Bool {
            switch self {
            case .twoSpeakerWHAMR: return true
            default: return false
            }
        }
        
        public var huggingFaceRepo: String {
            switch self {
            case .twoSpeaker: return "starkdmi/MossFormer2_SS_2SPK_16K_MLX"
            case .threeSpeaker: return "starkdmi/MossFormer2_SS_3SPK_8K_MLX"
            case .twoSpeakerWHAMR: return "starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX"
            }
        }
        
        /// Supported precisions (currently fp32 only)
        public var supportedPrecisions: [ModelPrecision] { [.fp32] }
    }
    
    public let modelType: Model
    public nonisolated var sampleRate: Int { modelType.sampleRate }
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1
    
    public nonisolated var minChunkSize: Int { modelType.sampleRate / 5 }  // 0.2s
    public nonisolated var recommendedChunkSize: Int { modelType.sampleRate * 4 }  // 4s (from benchmarks)
    
    private var model: MossFormer2_SS_16K?
    private let weightsPath: String?
    private let precision: ModelPrecision
    
    /// Max audio without chunking
    private let maxDirectDuration: Float = 4.0
    
    /// Initialize with model type and precision (auto-downloads from HuggingFace)
    public init(model: Model, precision: ModelPrecision = .fp32) {
        self.modelType = model
        self.weightsPath = nil
        self.precision = precision
    }
    
    /// Initialize with explicit weights path (no download)
    public init(model: Model, weightsPath: String) {
        self.modelType = model
        self.weightsPath = weightsPath
        self.precision = .fp32
    }
    
    /// Load model weights (downloads if not cached)
    public func load() async throws {
        let config = MossFormer2Config(
            encoder_embedding_dim: 512,
            mossformer_sequence_dim: 512,
            num_mossformer_layer: 24,
            encoder_kernel_size: 16,
            num_spks: modelType.numSpeakers,
            skip_mask_multiplication: modelType.skipMaskMultiplication
        )
        
        model = MossFormer2_SS_16K(config: config)
        
        let resolvedPath: String
        if let path = weightsPath {
            resolvedPath = path
        } else {
            // Check if already downloaded
            if let cached = ModelDownloader.shared.localPath(for: modelType.huggingFaceRepo) {
                let modelPath = cached.appendingPathComponent(precision.weightsFilename).path
                if FileManager.default.fileExists(atPath: modelPath) {
                    resolvedPath = modelPath
                } else {
                    throw AudioToolError.modelNotFound("Precision \(precision.rawValue) not available for \(modelType.rawValue)")
                }
            } else {
                // Auto-download from HuggingFace
                let modelDir = try await ModelDownloader.shared.downloadAndGetPath(
                    repo: modelType.huggingFaceRepo,
                    matching: [precision.weightsFilename, "config.json"]
                )
                resolvedPath = modelDir.appendingPathComponent(precision.weightsFilename).path
            }
        }
        
        let weights = try loadArrays(url: URL(fileURLWithPath: resolvedPath))
        let nestedWeights = NestedDictionary<String, MLXArray>.unflattened(weights)
        try model?.update(parameters: nestedWeights, verify: .all)
    }
    
    /// Separate mixed audio into speaker streams
    /// Output is RMS-normalized to match input energy (AudioTool PyTorch behavior)
    public func separate(_ audio: AudioBuffer) async throws -> [AudioBuffer] {
        try await separate(audio, onProgress: nil)
    }
    
    /// Separate mixed audio with progress reporting
    /// - Parameters:
    ///   - audio: Input audio buffer
    ///   - onProgress: Progress callback (0.0 to 100.0) as chunks are processed
    /// - Returns: ``outputCount`` separated audio buffers, one per speaker
    public func separate(
        _ audio: AudioBuffer,
        onProgress: ProgressCallback?
    ) async throws -> [AudioBuffer] {
        try validateSampleRate(audio)
        await onProgress?(0.0)
        
        let durationSeconds = Float(audio.samples.count) / Float(audio.sampleRate)
        
        // Use chunking for longer audio
        var results: [AudioBuffer]
        if durationSeconds > maxDirectDuration {
            results = try await processWithChunking(audio, onProgress: onProgress)
        } else {
            results = try await processChunk(audio.samples)
        }
        
        // RMS normalization: match output energy to input energy (AudioTool PyTorch behavior)
        // This preserves the relative loudness of each separated source
        let inputRMS = sqrt(audio.samples.map { $0 * $0 }.reduce(0, +) / Float(audio.samples.count))
        
        let normalized = results.map { buffer in
            let samples = buffer.samples
            let outputRMS = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))
            
            // Scale output to match input RMS (energy preservation)
            let scale = outputRMS > 1e-8 ? inputRMS / outputRMS : 1.0
            var scaled = samples.map { $0 * scale }
            
            // Clip if needed to prevent overflow
            let maxVal = scaled.map { abs($0) }.max() ?? 0.0
            if maxVal > 1.0 {
                scaled = scaled.map { $0 / maxVal }
            }
            
            return AudioBuffer(
                samples: scaled,
                sampleRate: buffer.sampleRate,
                channels: buffer.channels
            )
        }
        
        await onProgress?(100.0)
        return normalized
    }
    
    
    /// Process with chunking using MLXOverlap
    /// Uses triangular weighted overlap-add for each speaker independently
    private func processWithChunking(_ input: AudioBuffer, onProgress: ProgressCallback? = nil) async throws -> [AudioBuffer] {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormer2_SS_\(modelType.rawValue)")
        }
        
        let chunkingConfig = ChunkingConfig.mossformer2SS(sampleRate: sampleRate)
        let inputMLX = MLXArray(input.samples)
        
        // Split into chunks using MLXOverlap
        let chunks = MLXOverlap.split(
            audio: inputMLX,
            chunkSamples: chunkingConfig.chunkSamples,
            stride: chunkingConfig.strideSamples
        )
        
        let totalChunks = chunks.count
        
        // Process each chunk through model - collect all speaker outputs
        // Each chunk produces [numSpeakers] outputs
        var speakerChunks: [[(chunk: MLXArray, startIdx: Int)]] = Array(
            repeating: [],
            count: modelType.numSpeakers
        )
        
        for (chunkIdx, (chunk, startIdx)) in chunks.enumerated() {
            // Run model: input [1, T] -> outputs [numSpeakers x [1, T]]
            let batchedChunk = chunk.expandedDimensions(axis: 0)
            let separatedSources = model(batchedChunk)
            eval(separatedSources)
            
            // Distribute outputs to per-speaker chunk arrays
            for (spkIdx, source) in separatedSources.enumerated() {
                let squeezed = source.squeezed(axis: 0)
                speakerChunks[spkIdx].append((squeezed, startIdx))
            }
            
            // Clear GPU cache between chunks to reduce peak memory
            GPU.clearCache()
            
            // Report progress (0-90% for chunking, 90-100% for reassembly)
            let percent = Double(chunkIdx + 1) / Double(totalChunks) * 90.0
            await onProgress?(percent)
        }
        
        // Reassemble each speaker's audio using triangular blending
        let window = MLXOverlap.triangularWindow(length: chunkingConfig.chunkSamples)
        var results: [AudioBuffer] = []
        for spkIdx in 0..<modelType.numSpeakers {
            let reassembled = MLXOverlap.reassembleOverlapAdd(
                processedChunks: speakerChunks[spkIdx],
                chunkSamples: chunkingConfig.chunkSamples,
                stride: chunkingConfig.strideSamples,
                window: window,
                originalLength: input.samples.count
            )
            eval(reassembled)
            results.append(AudioBuffer(
                samples: reassembled.asArray(Float.self),
                sampleRate: sampleRate,
                channels: 1
            ))
        }
        
        return results
    }
    
    /// Process a single chunk (direct, no chunking)
    private func processChunk(_ samples: [Float]) async throws -> [AudioBuffer] {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormer2_SS_\(modelType.rawValue)")
        }
        
        let inputMLX = MLXArray(samples).expandedDimensions(axis: 0)
        let separatedSources = model(inputMLX)
        eval(separatedSources)
        
        var results: [AudioBuffer] = []
        for source in separatedSources {
            let squeezed = source.squeezed(axis: 0)
            let samples = squeezed.asArray(Float.self)
            results.append(AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1))
        }
        
        return results
    }
    
    public func reset() async {}
}

// MARK: - Demucs Source Separation Provider

/// Demucs source separation - separates audio into drums, bass, vocals, other
/// Chunking: 7.8s chunks, 25% overlap, triangular blending (from benchmarks)
public actor DemucsProvider: MusicSeparator {
    
    public enum Stem: String, CaseIterable, Sendable {
        case drums = "drums"
        case bass = "bass"
        case vocals = "vocals"
        case other = "other"

        /// Position of this stem on axis 1 of the model output, which is
        /// (batch, source, channel, time). Fixed by the HTDemucs architecture; each
        /// per-stem weight file still emits all four, so the right one must be
        /// selected rather than reduced over.
        var sourceIndex: Int {
            switch self {
            case .drums: return 0
            case .bass: return 1
            case .other: return 2
            case .vocals: return 3
            }
        }
    }
    
    public nonisolated let sampleRate: Int = 44100
    public nonisolated let inputChannels: Int = 2
    public nonisolated let outputChannels: Int = 1
    
    /// Stems the loaded weights can produce. Each has its own weight file, so this
    /// reflects what is on disk rather than a fixed capability.
    public nonisolated var availableStems: [Stem] { Stem.allCases }
    
    private var models: [Stem: HTDemucs] = [:]
    private let weightsDirectory: String
    
    /// Max audio without chunking (7.8s is the limit from benchmarks)
    private let maxDirectDuration: Float = 7.8
    
    public init(weightsDirectory: String) {
        self.weightsDirectory = weightsDirectory
    }
    
    /// Load all source models
    public func loadAll() async throws {
        for stem in Stem.allCases {
            try loadSync(stem: stem)
        }
    }
    
    /// Load a specific source model
    public func loadSync(stem: Stem) throws {
        let config = HTDemucsConfig()
        let model = HTDemucs(config: config)
        
        // Use HTDemucs.loadWeights for proper Python→Swift key conversion
        let weightsPath = "\(weightsDirectory)/\(stem.rawValue).safetensors"
        let parameters = try HTDemucs.loadWeights(from: weightsPath)
        try model.update(parameters: parameters, verify: .all)
        
        // Set to eval mode and materialize weights
        model.train(false)
        MLX.eval(model)
        
        models[stem] = model
    }
    
    /// Load a specific source model
    public func load(stem: Stem) async throws {
        try loadSync(stem: stem)
    }
    
    /// Separate a specific source from the audio
    public func separate(_ input: AudioBuffer, stem: Stem) async throws -> AudioBuffer {
        try validateSampleRate(input)
        let durationSeconds = Float(input.samples.count) / Float(input.sampleRate)
        
        // Use chunking for longer audio
        if durationSeconds > maxDirectDuration {
            return try await separateWithChunking(input, stem: stem)
        }
        
        return try await separateChunk(input.samples, stem: stem, channels: input.channels)
    }
    
    /// Separate with chunking using MLXOverlap
    /// Uses triangular weighted overlap-add for seamless blending
    private func separateWithChunking(_ input: AudioBuffer, stem: Stem) async throws -> AudioBuffer {
        let chunkingConfig = ChunkingConfig.demucs(sampleRate: sampleRate)
        let inputMLX = MLXArray(input.samples)
        
        guard let model = models[stem] else {
            throw AudioToolError.modelNotLoaded("Demucs_\(stem.rawValue)")
        }
        
        let sourceIndex = stem.sourceIndex
        
        let result = try await MLXOverlap.processWithChunking(
            audio: inputMLX,
            chunkSamples: chunkingConfig.chunkSamples,
            overlapRatio: chunkingConfig.overlapRatio,
            strategy: .triangular
        ) { chunk in
            // Process chunk through Demucs model
            // Demucs expects [batch, channels, samples] input
            let mono = chunk  // [samples]
            let stereo = MLX.stacked([mono, mono], axis: 0)  // [2, samples]
            let batched = stereo.expandedDimensions(axis: 0)  // [1, 2, samples]
            
            // Model output is (B, S=4, C=2, T) where S = sources
            let output = model(batched)  // [1, 4, 2, samples]
            eval(output)
            
            // Extract correct source and average stereo to mono
            let sourceOutput = output[0, sourceIndex]  // [2, samples]
            let monoOutput = mean(sourceOutput, axis: 0)  // [samples]
            eval(monoOutput)
            
            return monoOutput
        }
        eval(result)
        
        return AudioBuffer(samples: result.asArray(Float.self), sampleRate: sampleRate, channels: 1)
    }
    
    /// Separate a single chunk
    private func separateChunk(_ samples: [Float], stem: Stem, channels: Int) async throws -> AudioBuffer {
        guard let model = models[stem] else {
            throw AudioToolError.modelNotLoaded("Demucs_\(stem.rawValue)")
        }
        
        var inputMLX: MLXArray
        if channels == 1 {
            let mono = MLXArray(samples)
            inputMLX = MLX.stacked([mono, mono], axis: 0)
        } else {
            inputMLX = MLXArray(samples).reshaped([2, -1])
        }
        
        inputMLX = inputMLX.expandedDimensions(axis: 0)
        
        // Model output is (batch, source, channel, time) - four sources, always.
        //
        // This used to squeeze the batch axis and take mean(axis: 0), which averaged
        // the four *stems* into each other rather than selecting one, and left the
        // result still stereo: a 4s mono request came back with 2x the samples,
        // labelled mono, containing a blend of drums, bass, vocals and other. The
        // chunked path indexed correctly, so which one ran depended only on whether
        // the input was longer than maxDirectDuration.
        let output = model(inputMLX)
        eval(output)
        
        let sourceOutput = output[0, stem.sourceIndex]   // (channel, time)
        let mono = mean(sourceOutput, axis: 0)           // (time)
        eval(mono)
        
        return AudioBuffer(samples: mono.asArray(Float.self), sampleRate: sampleRate, channels: 1)
    }
    
    /// Separate all sources
    public func separateAll(_ input: AudioBuffer) async throws -> [Stem: AudioBuffer] {
        var results: [Stem: AudioBuffer] = [:]
        for stem in models.keys {
            results[stem] = try await separate(input, stem: stem)
        }
        return results
    }
    
    // MARK: - Background / Accompaniment Extraction
    
    /// Result containing vocals and accompaniment (instrumental) tracks
    public struct VocalsWithAccompaniment: Sendable {
        public let vocals: AudioBuffer
        public let accompaniment: AudioBuffer  // drums + bass + other
    }
    
    /// Separate audio into vocals and accompaniment (instrumental) tracks
    /// Accompaniment = drums + bass + other (everything except vocals)
    /// - Parameter audio: Input audio buffer at 44.1kHz
    /// - Returns: Vocals and accompaniment (instrumental) audio buffers
    public func separateVocalsWithAccompaniment(_ audio: AudioBuffer) async throws -> VocalsWithAccompaniment {
        // Ensure all models are loaded
        for stem in Stem.allCases {
            if models[stem] == nil {
                try loadSync(stem: stem)
            }
        }
        
        // Get all 4 stems
        let allStems = try await separateAll(audio)
        
        guard let vocals = allStems[.vocals],
              let drums = allStems[.drums],
              let bass = allStems[.bass],
              let other = allStems[.other] else {
            throw AudioToolError.modelNotLoaded("Demucs - not all stems available")
        }
        
        // Combine non-vocal stems into accompaniment
        let minLen = min(drums.samples.count, bass.samples.count, other.samples.count)
        var accompanimentSamples = [Float](repeating: 0, count: minLen)
        for i in 0..<minLen {
            accompanimentSamples[i] = drums.samples[i] + bass.samples[i] + other.samples[i]
        }
        
        return VocalsWithAccompaniment(
            vocals: vocals,
            accompaniment: AudioBuffer(samples: accompanimentSamples, sampleRate: sampleRate, channels: 1)
        )
    }
    
}
