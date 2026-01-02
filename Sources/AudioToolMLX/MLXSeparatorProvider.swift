//
//  MLXSeparatorProvider.swift
//  ClearVoiceMLX
//
//  MLX-based speech separation and source separation providers with chunking
//

import Foundation
import ClearVoice
import ClearVoiceCore
@preconcurrency import MLX
@preconcurrency import MLXNN
@preconcurrency import AudioUtils
@preconcurrency import MossFormer2SS
@preconcurrency import DemucsMLXSwift

// MARK: - MossFormer2 Speaker Separation Provider

/// MLX MossFormer2 Speaker Separation - supports 2spk, 3spk, and WHAMR models
/// Chunking: 4s chunks, 25% overlap, triangular blending
public actor MossFormer2SSProvider: SpeechSeparator {
    
    // SpeechSeparator conformance
    public nonisolated var supportedSpeakerCounts: [Int] { [modelType.numSpeakers] }
    
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
                    throw ClearVoiceError.modelNotFound("Precision \(precision.rawValue) not available for \(modelType.rawValue)")
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
    /// Output is peak-normalized to 1.0 like the original model
    public func separate(_ audio: AudioBuffer, speakers: Int) async throws -> [AudioBuffer] {
        guard speakers == modelType.numSpeakers else {
            throw ClearVoiceError.pipelineConfigurationInvalid("Model supports \(modelType.numSpeakers) speakers, not \(speakers)")
        }
        
        let durationSeconds = Float(audio.samples.count) / Float(sampleRate)
        
        // Use chunking for longer audio
        var results: [AudioBuffer]
        if durationSeconds > maxDirectDuration {
            results = try await processWithChunking(audio)
        } else {
            results = try await processChunk(audio.samples)
        }
        
        // Normalize each speaker's output to peak 1.0 (like original model)
        return results.map { buffer in
            let audioMLX = MLXArray(buffer.samples)
            let normalized = normalizeToPeak(audioMLX, targetPeak: 1.0)
            eval(normalized)
            return AudioBuffer(
                samples: normalized.asArray(Float.self),
                sampleRate: buffer.sampleRate,
                channels: buffer.channels
            )
        }
    }
    
    /// AudioProcessor conformance - returns first separated track
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        let separated = try await separate(input, speakers: modelType.numSpeakers)
        return separated.first ?? input
    }
    
    /// Process with chunking using MLXOverlap
    /// Uses triangular weighted overlap-add for each speaker independently
    private func processWithChunking(_ input: AudioBuffer) async throws -> [AudioBuffer] {
        guard let model = model else {
            throw ClearVoiceError.modelNotLoaded("MossFormer2_SS_\(modelType.rawValue)")
        }
        
        let chunkingConfig = ChunkingConfig.mossformer2SS(sampleRate: sampleRate)
        let inputMLX = MLXArray(input.samples)
        
        // Split into chunks using MLXOverlap
        let chunks = MLXOverlap.split(
            audio: inputMLX,
            chunkSamples: chunkingConfig.chunkSamples,
            stride: chunkingConfig.strideSamples
        )
        
        // Process each chunk through model - collect all speaker outputs
        // Each chunk produces [numSpeakers] outputs
        var speakerChunks: [[(chunk: MLXArray, startIdx: Int)]] = Array(
            repeating: [],
            count: modelType.numSpeakers
        )
        
        for (chunk, startIdx) in chunks {
            // Run model: input [1, T] -> outputs [numSpeakers x [1, T]]
            let batchedChunk = chunk.expandedDimensions(axis: 0)
            let separatedSources = model(batchedChunk)
            eval(separatedSources)
            
            // Distribute outputs to per-speaker chunk arrays
            for (spkIdx, source) in separatedSources.enumerated() {
                let squeezed = source.squeezed(axis: 0)
                speakerChunks[spkIdx].append((squeezed, startIdx))
            }
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
            throw ClearVoiceError.modelNotLoaded("MossFormer2_SS_\(modelType.rawValue)")
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
public actor DemucsProvider: SpeechSeparator {
    
    public enum Source: String, CaseIterable, Sendable {
        case drums = "drums"
        case bass = "bass"
        case vocals = "vocals"
        case other = "other"
    }
    
    public nonisolated let sampleRate: Int = 44100
    public nonisolated let inputChannels: Int = 2
    public nonisolated let outputChannels: Int = 1
    
    public nonisolated var supportedSpeakerCounts: [Int] { [4] }
    
    private var models: [Source: HTDemucs] = [:]
    private let weightsDirectory: String
    
    /// Max audio without chunking (7.8s is the limit from benchmarks)
    private let maxDirectDuration: Float = 7.8
    
    public init(weightsDirectory: String) {
        self.weightsDirectory = weightsDirectory
    }
    
    /// Load all source models
    public func loadAll() async throws {
        for source in Source.allCases {
            try loadSync(source: source)
        }
    }
    
    /// Load a specific source model
    public func loadSync(source: Source) throws {
        let config = HTDemucsConfig()
        let model = HTDemucs(config: config)
        
        // Use HTDemucs.loadWeights for proper Python→Swift key conversion
        let weightsPath = "\(weightsDirectory)/\(source.rawValue).safetensors"
        let parameters = try HTDemucs.loadWeights(from: weightsPath)
        try model.update(parameters: parameters, verify: .all)
        
        // Set to eval mode and materialize weights
        model.train(false)
        MLX.eval(model)
        
        models[source] = model
    }
    
    /// Load a specific source model
    public func load(source: Source) async throws {
        try loadSync(source: source)
    }
    
    /// Separate a specific source from the audio
    public func separate(_ input: AudioBuffer, source: Source) async throws -> AudioBuffer {
        let durationSeconds = Float(input.samples.count) / Float(sampleRate)
        
        // Use chunking for longer audio
        if durationSeconds > maxDirectDuration {
            return try await separateWithChunking(input, source: source)
        }
        
        return try await separateChunk(input.samples, source: source, channels: input.channels)
    }
    
    /// Separate with chunking using MLXOverlap
    /// Uses triangular weighted overlap-add for seamless blending
    private func separateWithChunking(_ input: AudioBuffer, source: Source) async throws -> AudioBuffer {
        let chunkingConfig = ChunkingConfig.demucs(sampleRate: sampleRate)
        let inputMLX = MLXArray(input.samples)
        
        guard let model = models[source] else {
            throw ClearVoiceError.modelNotLoaded("Demucs_\(source.rawValue)")
        }
        
        // Source indices in Demucs output: drums=0, bass=1, other=2, vocals=3
        let sourceIndex: Int
        switch source {
        case .drums: sourceIndex = 0
        case .bass: sourceIndex = 1
        case .other: sourceIndex = 2
        case .vocals: sourceIndex = 3
        }
        
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
    private func separateChunk(_ samples: [Float], source: Source, channels: Int) async throws -> AudioBuffer {
        guard let model = models[source] else {
            throw ClearVoiceError.modelNotLoaded("Demucs_\(source.rawValue)")
        }
        
        var inputMLX: MLXArray
        if channels == 1 {
            let mono = MLXArray(samples)
            inputMLX = MLX.stacked([mono, mono], axis: 0)
        } else {
            inputMLX = MLXArray(samples).reshaped([2, -1])
        }
        
        inputMLX = inputMLX.expandedDimensions(axis: 0)
        
        print("DEBUG separateChunk: input shape=\(inputMLX.shape)")
        
        let output = model(inputMLX)
        eval(output)
        
        print("DEBUG separateChunk: output shape=\(output.shape)")
        
        let squeezed = output.squeezed(axis: 0)
        print("DEBUG separateChunk: squeezed shape=\(squeezed.shape)")
        
        // Average stereo channels to mono - mean along channel axis (axis 0)
        let mono = mean(squeezed, axis: 0)
        print("DEBUG separateChunk: mono shape=\(mono.shape)")
        
        return AudioBuffer(samples: mono.asArray(Float.self), sampleRate: sampleRate, channels: 1)
    }
    
    /// Separate all sources
    public func separateAll(_ input: AudioBuffer) async throws -> [Source: AudioBuffer] {
        var results: [Source: AudioBuffer] = [:]
        for source in models.keys {
            results[source] = try await separate(input, source: source)
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
        for source in Source.allCases {
            if models[source] == nil {
                try loadSync(source: source)
            }
        }
        
        // Get all 4 stems
        let allStems = try await separateAll(audio)
        
        guard let vocals = allStems[.vocals],
              let drums = allStems[.drums],
              let bass = allStems[.bass],
              let other = allStems[.other] else {
            throw ClearVoiceError.modelNotLoaded("Demucs - not all stems available")
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
    
    // MARK: - SpeechSeparator Conformance
    
    /// Separate all 4 sources
    public func separate(_ audio: AudioBuffer, speakers: Int) async throws -> [AudioBuffer] {
        let allResults = try await separateAll(audio)
        return [
            allResults[.drums],
            allResults[.bass],
            allResults[.vocals],
            allResults[.other]
        ].compactMap { $0 }
    }
    
    /// AudioProcessor conformance - returns vocals track
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        return try await separate(input, source: .vocals)
    }
}
