//
//  USSMLXProvider.swift
//  ClearVoiceUSS
//
//  Universal Speech Separation provider using USS MLX model
//

import Foundation
import ClearVoice
import ClearVoiceCore
@preconcurrency import USSMLXSwift
import AudioUtils
import MLX

// MARK: - USS MLX Provider

/// Universal Speech Separation provider using ResUNet30 with FiLM conditioning
/// Separates audio by sound type (speech, music, noise, etc.)
///
/// ## Embedding Swap Performance
///
/// Based on benchmarking with 30s audio (harry_potter_short.wav):
///
/// | Approach | Time | RTF | Notes |
/// |----------|------|-----|-------|
/// | Sequential `process(_:type:)` | 1.1s | 27x | **Recommended** |
/// | `processMultiple()` | 2.4s | 12x | Has logging overhead |
///
/// **Key findings:**
/// - Embedding swap is instant (~0ms) because all embeddings are cached
/// - JIT compilation happens once during prewarm, not per-embedding
/// - First call with ANY new embedding has only ~5% penalty after initial prewarm
/// - Sequential `process(_:type:)` is faster than `processMultiple()` for small batches
///
/// ## Usage for chunked multi-type separation:
/// ```swift
/// let uss = USSProviders.speechSeparation()
/// try await uss.load()
///
/// for chunk in audioChunks {
///     // Recommended: Use sequential process with type (fastest)
///     let music = try await uss.process(chunk, type: .music)
///     let animal = try await uss.process(chunk, type: .animal)
/// }
///
/// await uss.unload() // Release GPU memory when done
/// ```
public actor USSMLXProvider: AudioProcessor, ManagedModel {
    
    // MARK: - AudioProcessor Conformance
    
    public nonisolated let sampleRate: Int = 32000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1
    
    // MARK: - ManagedModel Conformance
    
    /// Model ID for lifecycle manager
    public nonisolated var modelId: String { "uss_mlx" }
    
    /// Estimated memory: ~53MB FP16 weights + ~14KB embeddings cache
    public nonisolated var estimatedMemoryBytes: Int {
        useFp16 ? 55_000_000 : 110_000_000
    }
    
    // MARK: - Private Properties
    
    private var inference: USSInference?
    private var conditioning: MLXArray?
    
    /// Cached embeddings for all types (loaded on first access)
    private var embeddingCache: [EmbeddingLoader.EmbeddingType: MLXArray] = [:]
    
    /// Current active embedding type
    private var currentEmbeddingType: EmbeddingLoader.EmbeddingType
    
    private let initialEmbeddingType: EmbeddingLoader.EmbeddingType
    private let segmentDuration: Float
    private let useFp16: Bool
    private let compile: Bool
    
    // MARK: - Initialization
    
    /// Initialize USS MLX provider
    /// - Parameters:
    ///   - embeddingType: Initial sound type to separate (default: .speech)
    ///   - segmentDuration: Chunk duration in seconds (default: 2.0)
    ///   - useFp16: Use FP16 weights (default: true, 53MB vs 106MB)
    ///   - compile: Enable MLX compilation (default: true)
    public init(
        embeddingType: EmbeddingLoader.EmbeddingType = .speech,
        segmentDuration: Float = 2.0,
        useFp16: Bool = true,
        compile: Bool = true
    ) {
        self.initialEmbeddingType = embeddingType
        self.currentEmbeddingType = embeddingType
        self.segmentDuration = segmentDuration
        self.useFp16 = useFp16
        self.compile = compile
    }
    
    // MARK: - Model Lifecycle
    
    /// Load USS model and cache all embeddings from USSMLXSwift bundle
    public func load() async throws {
        // Initialize model
        let model = ResUNet30()
        
        // Get model weights path from USSMLXSwift bundle using public accessor
        guard let weightsURL = USSBundle.weightsURL(fp16: useFp16) else {
            throw ClearVoiceError.modelNotFound("USS ResUNet30 weights (\(useFp16 ? "fp16" : "fp32"))")
        }
        
        // Load weights
        try WeightLoader.loadWeights(model: model, from: weightsURL.path)
        
        // Load and cache ALL embeddings upfront (~14KB total for 7 types)
        guard let embeddingsDir = USSBundle.embeddingsDirectory else {
            throw ClearVoiceError.modelNotFound("USS embeddings directory")
        }
        
        for type in EmbeddingLoader.EmbeddingType.allCases {
            let embedding = try EmbeddingLoader.loadEmbedding(type: type, from: embeddingsDir.path)
            embeddingCache[type] = embedding
        }
        
        // Set initial conditioning
        guard let initialConditioning = embeddingCache[initialEmbeddingType] else {
            throw ClearVoiceError.modelNotFound("USS embedding for \(initialEmbeddingType.rawValue)")
        }
        
        // Create inference pipeline with 2s chunks, no overlap (hopLength == segmentDuration)
        let inference = USSInference(
            model: model,
            sampleRate: sampleRate,
            segmentDuration: segmentDuration,
            hopLength: segmentDuration,  // No overlap
            compile: compile,
            segmentBatchSize: 1
        )
        
        // Prewarm model with initial conditioning
        inference.prewarm(conditioning: initialConditioning)
        
        self.inference = inference
        self.conditioning = initialConditioning
        self.currentEmbeddingType = initialEmbeddingType
    }
    
    /// Check if the model is currently loaded and ready for inference
    public func checkIfLoaded() async -> Bool {
        inference != nil
    }
    
    /// Unload model from memory and release GPU resources
    public func unload() async {
        inference = nil
        conditioning = nil
        embeddingCache.removeAll()
        GPU.clearCache()
    }
    
    // MARK: - Embedding Switching
    
    /// Current active embedding type
    public var activeEmbeddingType: EmbeddingLoader.EmbeddingType {
        currentEmbeddingType
    }
    
    /// Switch conditioning embedding type without reloading model weights.
    /// Embeddings are cached, so switching is instant (~0ms).
    /// - Parameter type: New embedding type to use for subsequent process() calls
    /// - Throws: ClearVoiceError.modelNotLoaded if model not loaded
    public func setConditioning(_ type: EmbeddingLoader.EmbeddingType) async throws {
        guard inference != nil else {
            throw ClearVoiceError.modelNotLoaded("USS MLX")
        }
        
        guard let cachedConditioning = embeddingCache[type] else {
            throw ClearVoiceError.modelNotFound("USS embedding for \(type.rawValue) not in cache")
        }
        
        self.conditioning = cachedConditioning
        self.currentEmbeddingType = type
    }
    
    /// Prewarm with additional embeddings for faster first-call performance.
    /// 
    /// **Note**: Based on benchmarking, this provides minimal benefit (~5% on first call)
    /// because the JIT compilation happens on model graph, not per-embedding.
    /// The main prewarm during `load()` handles most of the compilation.
    /// 
    /// - Parameter types: Embedding types to prewarm (runs tiny inference with each)
    public func prewarmEmbeddings(_ types: [EmbeddingLoader.EmbeddingType]) async throws {
        guard let inference = inference else {
            throw ClearVoiceError.modelNotLoaded("USS MLX")
        }
        
        // Create tiny audio for prewarming (0.1 second = 3200 samples)
        let tinySamples = [Float](repeating: 0, count: 3200)
        let tinyAudio = MLXArray(tinySamples).reshaped([1, -1])
        
        for type in types {
            guard let conditioning = embeddingCache[type] else { continue }
            
            let _ = inference.separate(
                audio: tinyAudio,
                conditioning: conditioning,
                compile: compile,
                useSimpleSegmentation: true
            )
        }
    }
    
    // MARK: - AudioProcessor Conformance
    
    /// Process audio - separate target sound type using current embedding
    /// - Parameter input: Input audio buffer at 32kHz
    /// - Returns: Separated audio buffer
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        guard let inference = inference, let conditioning = conditioning else {
            throw ClearVoiceError.modelNotLoaded("USS MLX")
        }
        
        return try separateWithConditioning(input, conditioning: conditioning, inference: inference)
    }
    
    /// Process audio with specific embedding type (one-off, doesn't change stored state).
    /// 
    /// **This is the recommended approach for multi-type separation.**
    /// Benchmarks show sequential `process(_:type:)` calls are faster than `processMultiple()`.
    /// 
    /// - Parameters:
    ///   - input: Input audio buffer at 32kHz
    ///   - type: Embedding type to use for this separation only
    /// - Returns: Separated audio buffer
    public func process(_ input: AudioBuffer, type: EmbeddingLoader.EmbeddingType) async throws -> AudioBuffer {
        guard let inference = inference else {
            throw ClearVoiceError.modelNotLoaded("USS MLX")
        }
        
        guard let cachedConditioning = embeddingCache[type] else {
            throw ClearVoiceError.modelNotFound("USS embedding for \(type.rawValue) not in cache")
        }
        
        // Use specified conditioning without modifying stored state
        return try separateWithConditioning(input, conditioning: cachedConditioning, inference: inference)
    }
    
    /// Process audio and separate multiple sound types.
    /// 
    /// **Note**: Benchmarks show this is ~2x slower than sequential `process(_:type:)` calls
    /// due to internal logging overhead. For best performance, use `process(_:type:)` instead.
    /// 
    /// - Parameters:
    ///   - input: Input audio buffer at 32kHz
    ///   - types: Array of embedding types to separate
    /// - Returns: Dictionary mapping type to separated audio
    public func processMultiple(
        _ input: AudioBuffer,
        types: [EmbeddingLoader.EmbeddingType]
    ) async throws -> [EmbeddingLoader.EmbeddingType: AudioBuffer] {
        guard let inference = inference else {
            throw ClearVoiceError.modelNotLoaded("USS MLX")
        }
        
        // Collect conditionings from cache
        var conditionings: [MLXArray] = []
        for type in types {
            guard let cachedConditioning = embeddingCache[type] else {
                throw ClearVoiceError.modelNotFound("USS embedding for \(type.rawValue) not in cache")
            }
            conditionings.append(cachedConditioning)
        }
        
        // Convert input to MLXArray
        let samples = MLXArray(input.samples)
        let audio = samples.reshaped([1, -1])
        
        // Use USSInference.separateMultipleEmbeddings for efficient batch processing
        let separatedArrays = inference.separateMultipleEmbeddings(
            audio: audio,
            conditionings: conditionings,
            useSimpleSegmentation: true
        )
        
        // Build result dictionary
        var results: [EmbeddingLoader.EmbeddingType: AudioBuffer] = [:]
        for (idx, type) in types.enumerated() {
            let separated = separatedArrays[idx]
            eval(separated)
            
            let outputSamples = extractOutputSamples(from: separated)
            results[type] = AudioBuffer(samples: outputSamples, sampleRate: sampleRate)
        }
        
        return results
    }
    
    // MARK: - Private Helpers
    
    /// Core separation logic shared by process methods
    private func separateWithConditioning(
        _ input: AudioBuffer,
        conditioning: MLXArray,
        inference: USSInference
    ) throws -> AudioBuffer {
        // Convert to MLXArray - (samples,) -> (batch=1, samples)
        let samples = MLXArray(input.samples)
        let audio = samples.reshaped([1, -1])
        
        // Run separation
        let separated = inference.separate(
            audio: audio,
            conditioning: conditioning,
            compile: compile,
            useSimpleSegmentation: true  // No overlap matches Python
        )
        eval(separated)
        
        // Extract output
        let outputSamples = extractOutputSamples(from: separated)
        return AudioBuffer(samples: outputSamples, sampleRate: sampleRate)
    }
    
    /// Extract output samples handling different output shapes
    private func extractOutputSamples(from separated: MLXArray) -> [Float] {
        // Extract output - (batch, channels, samples) -> (samples,)
        if separated.ndim == 3 {
            return separated[0, 0, 0...].asArray(Float.self)
        } else if separated.ndim == 2 {
            return separated[0, 0...].asArray(Float.self)
        } else {
            return separated.asArray(Float.self)
        }
    }
    
    // MARK: - Separation API
    
    /// Separate target sound from audio (alias for process)
    public func separate(_ audio: AudioBuffer) async throws -> AudioBuffer {
        return try await process(audio)
    }
    
    /// Separate specific sound type from audio (alias for process with type)
    public func separate(_ audio: AudioBuffer, type: EmbeddingLoader.EmbeddingType) async throws -> AudioBuffer {
        return try await process(audio, type: type)
    }
    
    // MARK: - Background Extraction
    
    /// Result containing both separated target sound and background/residual audio
    public struct SeparatedWithBackground: Sendable {
        public let separated: AudioBuffer
        public let background: AudioBuffer
    }
    
    /// Separate target sound and extract background (everything else)
    /// Uses simple subtraction: background = input - separated
    /// - Parameter audio: Input audio buffer at 32kHz
    /// - Returns: Separated target and background audio
    public func separateWithBackground(_ audio: AudioBuffer) async throws -> SeparatedWithBackground {
        guard let inference = inference, let conditioning = conditioning else {
            throw ClearVoiceError.modelNotLoaded("USS MLX")
        }
        
        return try separateWithBackgroundInternal(audio, conditioning: conditioning, inference: inference)
    }
    
    /// Separate specific sound type and extract background
    /// - Parameters:
    ///   - audio: Input audio buffer at 32kHz
    ///   - type: Embedding type to use for separation
    /// - Returns: Separated target and background audio
    public func separateWithBackground(_ audio: AudioBuffer, type: EmbeddingLoader.EmbeddingType) async throws -> SeparatedWithBackground {
        guard let inference = inference else {
            throw ClearVoiceError.modelNotLoaded("USS MLX")
        }
        
        guard let cachedConditioning = embeddingCache[type] else {
            throw ClearVoiceError.modelNotFound("USS embedding for \(type.rawValue) not in cache")
        }
        
        return try separateWithBackgroundInternal(audio, conditioning: cachedConditioning, inference: inference)
    }
    
    /// Internal helper for background extraction
    private func separateWithBackgroundInternal(
        _ audio: AudioBuffer,
        conditioning: MLXArray,
        inference: USSInference
    ) throws -> SeparatedWithBackground {
        // Convert to MLXArray
        let inputSamples = MLXArray(audio.samples)
        let inputAudio = inputSamples.reshaped([1, -1])
        
        // Run separation
        let separated = inference.separate(
            audio: inputAudio,
            conditioning: conditioning,
            compile: compile,
            useSimpleSegmentation: true
        )
        eval(separated)
        
        // Extract output samples
        let separatedSamples = extractOutputSamples(from: separated)
        
        // Compute background via subtraction: input - separated
        // Handle potential length mismatch (separated may be slightly different due to padding)
        let minLen = min(audio.samples.count, separatedSamples.count)
        var backgroundSamples = [Float](repeating: 0, count: minLen)
        for i in 0..<minLen {
            backgroundSamples[i] = audio.samples[i] - separatedSamples[i]
        }
        
        return SeparatedWithBackground(
            separated: AudioBuffer(samples: Array(separatedSamples.prefix(minLen)), sampleRate: sampleRate),
            background: AudioBuffer(samples: backgroundSamples, sampleRate: sampleRate)
        )
    }
}

// MARK: - Sound Type Alias

public extension USSMLXProvider {
    /// Available sound types for separation
    typealias SoundType = EmbeddingLoader.EmbeddingType
}

// MARK: - UniversalSoundSeparator Conformance

extension USSMLXProvider: UniversalSoundSeparator {
    
    /// Map USSSoundType to EmbeddingLoader.EmbeddingType
    private func embeddingType(for soundType: USSSoundType) -> EmbeddingLoader.EmbeddingType {
        switch soundType {
        case .speech: return .speech
        case .music: return .music
        case .animal: return .animal
        case .nature: return .nature
        case .noise: return .noise
        case .things: return .things
        case .human: return .human
        }
    }
    
    /// Separate a specific sound type from audio using USSSoundType
    /// This is the primary API for pipeline integration.
    public func separateSound(_ audio: AudioBuffer, type: USSSoundType) async throws -> AudioBuffer {
        let embedding = embeddingType(for: type)
        return try await process(audio, type: embedding)
    }
    
    /// Separate multiple sound types with progress reporting
    /// Reports progress per-embedding (e.g., 3 types = 33%, 66%, 100%)
    public func separateMultipleSounds(
        _ audio: AudioBuffer,
        types: [USSSoundType],
        onProgress: ProgressCallback?
    ) async throws -> [USSSoundType: AudioBuffer] {
        var results: [USSSoundType: AudioBuffer] = [:]
        for (idx, type) in types.enumerated() {
            let embedding = embeddingType(for: type)
            results[type] = try await process(audio, type: embedding)
            let percent = Double(idx + 1) / Double(types.count) * 100.0
            await onProgress?(percent)
        }
        return results
    }
    
    /// Separate sound type and return background residual using USSSoundType
    public func separateSoundWithBackground(
        _ audio: AudioBuffer,
        type: USSSoundType
    ) async throws -> (separated: AudioBuffer, background: AudioBuffer) {
        let embedding = embeddingType(for: type)
        let result = try await separateWithBackground(audio, type: embedding)
        return (separated: result.separated, background: result.background)
    }
}
