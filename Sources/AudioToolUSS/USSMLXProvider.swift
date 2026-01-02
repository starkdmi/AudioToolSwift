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
public actor USSMLXProvider: AudioProcessor {
    
    // MARK: - AudioProcessor Conformance
    
    public nonisolated let sampleRate: Int = 32000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1
    
    // MARK: - Private Properties
    
    private var inference: USSInference?
    private var conditioning: MLXArray?
    
    private let embeddingType: EmbeddingLoader.EmbeddingType
    private let segmentDuration: Float
    private let useFp16: Bool
    private let compile: Bool
    
    // MARK: - Initialization
    
    /// Initialize USS MLX provider
    /// - Parameters:
    ///   - embeddingType: Sound type to separate (default: .speech)
    ///   - segmentDuration: Chunk duration in seconds (default: 2.0)
    ///   - useFp16: Use FP16 weights (default: true, 53MB vs 106MB)
    ///   - compile: Enable MLX compilation (default: true)
    public init(
        embeddingType: EmbeddingLoader.EmbeddingType = .speech,
        segmentDuration: Float = 2.0,
        useFp16: Bool = true,
        compile: Bool = true
    ) {
        self.embeddingType = embeddingType
        self.segmentDuration = segmentDuration
        self.useFp16 = useFp16
        self.compile = compile
    }
    
    // MARK: - AudioProcessor Conformance
    
    /// Load USS model and embeddings from USSMLXSwift bundle
    public func load() async throws {
        // Initialize model
        let model = ResUNet30()
        
        // Get model weights path from USSMLXSwift bundle using public accessor
        guard let weightsURL = USSBundle.weightsURL(fp16: useFp16) else {
            throw ClearVoiceError.modelNotFound("USS ResUNet30 weights (\(useFp16 ? "fp16" : "fp32"))")
        }
        
        // Load weights
        try WeightLoader.loadWeights(model: model, from: weightsURL.path)
        
        // Load conditioning embedding from USSMLXSwift bundle
        guard let embeddingsDir = USSBundle.embeddingsDirectory else {
            throw ClearVoiceError.modelNotFound("USS embeddings directory")
        }
        
        let conditioning = try EmbeddingLoader.loadEmbedding(type: embeddingType, from: embeddingsDir.path)
        
        // Create inference pipeline with 2s chunks, no overlap (hopLength == segmentDuration)
        let inference = USSInference(
            model: model,
            sampleRate: sampleRate,
            segmentDuration: segmentDuration,
            hopLength: segmentDuration,  // No overlap
            compile: compile,
            segmentBatchSize: 1
        )
        
        // Prewarm model
        inference.prewarm(conditioning: conditioning)
        
        self.inference = inference
        self.conditioning = conditioning
    }
    
    /// Process audio - separate target sound type
    /// - Parameter input: Input audio buffer
    /// - Returns: Separated audio buffer
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        guard let inference = inference, let conditioning = conditioning else {
            throw ClearVoiceError.modelNotLoaded("USS MLX")
        }
        
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
        
        // Extract output - (batch, channels, samples) -> (samples,)
        let outputSamples: [Float]
        if separated.ndim == 3 {
            outputSamples = separated[0, 0, 0...].asArray(Float.self)
        } else if separated.ndim == 2 {
            outputSamples = separated[0, 0...].asArray(Float.self)
        } else {
            outputSamples = separated.asArray(Float.self)
        }
        
        return AudioBuffer(samples: outputSamples, sampleRate: sampleRate)
    }
    
    // MARK: - Separation API
    
    /// Separate target sound from audio (alias for process)
    public func separate(_ audio: AudioBuffer) async throws -> AudioBuffer {
        return try await process(audio)
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
        let separatedSamples: [Float]
        if separated.ndim == 3 {
            separatedSamples = separated[0, 0, 0...].asArray(Float.self)
        } else if separated.ndim == 2 {
            separatedSamples = separated[0, 0...].asArray(Float.self)
        } else {
            separatedSamples = separated.asArray(Float.self)
        }
        
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
