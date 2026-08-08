//
//  USSMLXProvider.swift
//  AudioToolUSS
//
//  Universal Speech Separation provider using USS MLX model
//

import Foundation
import AudioTool
import AudioToolCore
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
    /// Matches the reference conversion, which loaded audio through AVAudioConverter's
    /// Mastering algorithm at maximum quality (see `runUSS` in USS/Utils/Inference.swift).
    /// The shipped path previously used the generic cubic default instead, so provider
    /// output could differ from what the conversion was validated against.
    public nonisolated var preferredResamplingQuality: AudioToolCore.ResamplingQuality { .high }

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

    /// Explicit weights path, bypassing bundle lookup and download.
    private let weightsPath: String?
    
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
        self.weightsPath = nil
    }

    /// Initialize with explicit weights path (no bundle lookup, no download)
    ///
    /// Matches the escape hatch every other provider offers. USS lost its bundled
    /// weights when they moved to HuggingFace, and without this there was no way to
    /// run it against a local file - which made a private weights repo block local
    /// development rather than just distribution.
    ///
    /// - Parameter weightsPath: Path to a resunet30 .safetensors file
    public init(
        weightsPath: String,
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
        self.weightsPath = weightsPath
    }
    
    // MARK: - Model Lifecycle
    
    /// HuggingFace repository holding the ResUNet30 weights
    public static let repo = ModelRepository.uss

    /// Locate the ResUNet30 weights, downloading them if necessary.
    ///
    /// These used to be ~106 MB of SPM bundle resources. They are fetched at runtime
    /// now, like every other model's weights, so a clone stays small. The bundle is
    /// still checked first so an app that vendors them keeps working.
    private func resolveWeightsPath() async throws -> String {
        let filename = useFp16 ? "resunet30_fp16.safetensors" : "resunet30_fp32.safetensors"

        if let explicit = weightsPath {
            guard FileManager.default.fileExists(atPath: explicit) else {
                throw AudioToolError.modelNotFound("USS ResUNet30 weights at \(explicit)")
            }
            return explicit
        }

        if let bundled = USSBundle.weightsURL(fp16: useFp16) {
            return bundled.path
        }

        let requiredFiles = [filename]
        if let cached = ModelDownloader.shared.localPath(
            for: Self.repo,
            matching: requiredFiles
        ) {
            return cached.appendingPathComponent(filename).path
        }

        let directory = try await ModelDownloader.shared.downloadAndGetPath(
            repo: Self.repo,
            matching: requiredFiles
        )
        let path = directory.appendingPathComponent(filename).path
        guard FileManager.default.fileExists(atPath: path) else {
            throw AudioToolError.modelNotFound(
                "USS ResUNet30 weights (\(useFp16 ? "fp16" : "fp32")) not found in \(Self.repo)")
        }
        return path
    }

    /// Load USS model and cache all embeddings
    public func load() async throws {
        // Initialize model
        let model = ResUNet30()
        
        let weightsPath = try await resolveWeightsPath()
        
        // Load weights
        try WeightLoader.loadWeights(model: model, from: weightsPath)
        
        // Load and cache ALL embeddings upfront (~14KB total for 7 types)
        guard let embeddingsDir = USSBundle.embeddingsDirectory else {
            throw AudioToolError.modelNotFound("USS embeddings directory")
        }
        
        for type in EmbeddingLoader.EmbeddingType.allCases {
            let embedding = try EmbeddingLoader.loadEmbedding(type: type, from: embeddingsDir.path)
            embeddingCache[type] = embedding
        }
        
        // Set initial conditioning
        guard let initialConditioning = embeddingCache[initialEmbeddingType] else {
            throw AudioToolError.modelNotFound("USS embedding for \(initialEmbeddingType.rawValue)")
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
    /// - Throws: AudioToolError.modelNotLoaded if model not loaded
    public func setConditioning(_ type: EmbeddingLoader.EmbeddingType) async throws {
        guard inference != nil else {
            throw AudioToolError.modelNotLoaded("USS MLX")
        }
        
        guard let cachedConditioning = embeddingCache[type] else {
            throw AudioToolError.modelNotFound("USS embedding for \(type.rawValue) not in cache")
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
            throw AudioToolError.modelNotLoaded("USS MLX")
        }
        
        // Create tiny audio for prewarming (0.1 second = 3200 samples)
        let tinySamples = [Float](repeating: 0, count: 3200)
        let tinyAudio = MLXArray(tinySamples).reshaped([1, -1])
        
        for type in types {
            try Task.checkCancellation()
            guard let conditioning = embeddingCache[type] else { continue }
            
            let output = inference.separate(
                audio: tinyAudio,
                conditioning: conditioning,
                compile: compile,
                useSimpleSegmentation: true
            )
            // MLX is lazy: retaining or discarding the graph does not compile it.
            // Materialize the actual model output before reporting this embedding
            // as prewarmed.
            eval(output)
            _ = output.asArray(Float.self)
        }
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
            throw AudioToolError.modelNotLoaded("USS MLX")
        }
        try validateSampleRate(input)
        
        guard let cachedConditioning = embeddingCache[type] else {
            throw AudioToolError.modelNotFound("USS embedding for \(type.rawValue) not in cache")
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
            throw AudioToolError.modelNotLoaded("USS MLX")
        }
        try validateInputFormat(input)
        
        // Collect conditionings from cache
        var conditionings: [MLXArray] = []
        for type in types {
            guard let cachedConditioning = embeddingCache[type] else {
                throw AudioToolError.modelNotFound("USS embedding for \(type.rawValue) not in cache")
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
            throw AudioToolError.modelNotLoaded("USS MLX")
        }
        try validateInputFormat(audio)
        
        return try separateWithBackgroundInternal(audio, conditioning: conditioning, inference: inference)
    }
    
    /// Separate specific sound type and extract background
    /// - Parameters:
    ///   - audio: Input audio buffer at 32kHz
    ///   - type: Embedding type to use for separation
    /// - Returns: Separated target and background audio
    public func separateWithBackground(_ audio: AudioBuffer, type: EmbeddingLoader.EmbeddingType) async throws -> SeparatedWithBackground {
        guard let inference = inference else {
            throw AudioToolError.modelNotLoaded("USS MLX")
        }
        try validateInputFormat(audio)
        
        guard let cachedConditioning = embeddingCache[type] else {
            throw AudioToolError.modelNotFound("USS embedding for \(type.rawValue) not in cache")
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

    /// Conditioning tensor for an arbitrary target.
    ///
    /// The model takes any 527-d vector, so this is a straight reshape rather than a
    /// lookup - which is the whole point of `SoundEmbedding`. Presets go through the
    /// cache because they are the common case and were already loaded at `load()`;
    /// custom targets are built on the spot.
    private func conditioningTensor(for target: SoundEmbedding) -> MLXArray {
        // Built from the weights, always. An earlier version keyed the preset cache
        // off `target.label`, which is free-form public metadata - so a custom vector
        // labelled "speech" silently separated using the bundled speech conditioning
        // and ignored every weight the caller supplied.
        MLXArray(target.weights).reshaped([1, SoundEmbedding.dimension])
    }

    /// Separate a target sound from audio.
    public func separateSound(_ audio: AudioBuffer, target: SoundEmbedding) async throws -> AudioBuffer {
        guard let inference = inference else {
            throw AudioToolError.modelNotLoaded("USS MLX")
        }
        try validateSampleRate(audio)
        return try separateWithConditioning(
            audio,
            conditioning: conditioningTensor(for: target),
            inference: inference
        )
    }

    /// Separate several targets, reporting progress per target.
    ///
    /// Returns pairs rather than a dictionary so that duplicate or unlabelled targets
    /// do not collide and the caller's ordering is preserved.
    public func separateMultipleSounds(
        _ audio: AudioBuffer,
        targets: [SoundEmbedding],
        onProgress: ProgressCallback?
    ) async throws -> [(target: SoundEmbedding, audio: AudioBuffer)] {
        var results: [(target: SoundEmbedding, audio: AudioBuffer)] = []
        results.reserveCapacity(targets.count)
        for (index, target) in targets.enumerated() {
            results.append((target, try await separateSound(audio, target: target)))
            await onProgress?(Double(index + 1) / Double(targets.count) * 100.0)
        }
        return results
    }

    /// Separate a target and return the residual background alongside it.
    public func separateSoundWithBackground(
        _ audio: AudioBuffer,
        target: SoundEmbedding
    ) async throws -> (separated: AudioBuffer, background: AudioBuffer) {
        guard let inference = inference else {
            throw AudioToolError.modelNotLoaded("USS MLX")
        }
        try validateSampleRate(audio)
        let result = try separateWithBackgroundInternal(
            audio,
            conditioning: conditioningTensor(for: target),
            inference: inference
        )
        return (separated: result.separated, background: result.background)
    }
}
