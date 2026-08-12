//
//  SpeakerEmbeddingProvider.swift
//  AudioToolFluidAudio
//
//  Speaker embedding extraction using FluidAudio's WeSpeaker model
//

import Foundation
import AudioTool
import AudioToolCore
@preconcurrency import FluidAudio

// MARK: - Speaker Embedding Provider

/// Provider for extracting speaker embeddings from audio
///
/// Uses FluidAudio's WeSpeaker model to extract 256-dimensional speaker embeddings.
/// These embeddings can be used for voice matching, speaker verification, and
/// voice similarity comparisons.
///
/// ## Improved Embedding Extraction
/// - **10s input with repetition**: Short audio is repeated to fill 10 seconds
///   (instead of zero-padding) for more consistent embeddings
/// - **Audio normalization**: RMS normalization ensures consistent input level
/// - **Direct embedding extraction**: Uses EmbeddingExtractor directly instead
///   of full diarization pipeline for deterministic results
///
/// ## Usage
/// ```swift
/// let provider = SpeakerEmbeddingProvider()
/// try await provider.load()
///
/// // Extract embedding from reference audio (16kHz mono)
/// let embedding = try await provider.extractEmbedding(referenceAudio)
/// print("Embedding dimension: \(embedding.count)")  // 256
/// ```
///
/// ## Thread Safety
/// This actor provides thread-safe access to embedding extraction.
public actor SpeakerEmbeddingProvider: SpeakerEmbeddingExtractor {
    
    // MARK: - Configuration
    
    /// Configuration for embedding extraction
    public struct Config: Sendable {
        /// Target input length for WeSpeaker (10 seconds = 160,000 samples at 16kHz)
        public let targetSamples: Int
        
        /// Target RMS level for audio normalization
        public let targetRMS: Float
        
        /// Whether to use repetition (true) or zero-padding (false) for short audio
        public let useRepetition: Bool
        
        /// Whether to use VAD for trimming silence and building accurate mask
        public let useVAD: Bool
        
        /// WeSpeaker frame count for 10 seconds (used for mask)
        public let wespeakerFrames: Int
        
        /// VAD speech probability threshold (0.0-1.0)
        public let vadThreshold: Float
        
        public static let `default` = Config(
            targetSamples: 160_000,   // 10 seconds at 16kHz
            targetRMS: 0.1,           // Target RMS for normalization
            useRepetition: true,      // Repetition is better than zero-padding
            useVAD: true,             // Use VAD for trimming and mask
            wespeakerFrames: 589,     // WeSpeaker frames for 10s
            vadThreshold: 0.5         // Speech probability threshold
        )
        
        public init(
            targetSamples: Int = 160_000,
            targetRMS: Float = 0.1,
            useRepetition: Bool = true,
            useVAD: Bool = true,
            wespeakerFrames: Int = 589,
            vadThreshold: Float = 0.5
        ) {
            self.targetSamples = targetSamples
            self.targetRMS = targetRMS
            self.useRepetition = useRepetition
            self.useVAD = useVAD
            self.wespeakerFrames = wespeakerFrames
            self.vadThreshold = vadThreshold
        }
    }
    
    // MARK: - Public Properties
    
    /// Expected sample rate for input audio
    public nonisolated let sampleRate: Int = 16000
    
    /// Input channels (mono)
    public nonisolated let inputChannels: Int = 1
    
    /// Output channels (mono)
    public nonisolated let outputChannels: Int = 1
    
    /// Embedding dimension (WeSpeaker produces 256-dim embeddings)
    public nonisolated let embeddingDimension: Int = 256
    
    /// Configuration for embedding extraction
    public nonisolated let config: Config
    
    /// Whether the provider is loaded and ready
    public var isReady: Bool { embeddingExtractor != nil }
    
    // MARK: - Private Properties
    
    private var embeddingExtractor: EmbeddingExtractor?
    private var vadManager: VadManager?
    
    // MARK: - Initialization
    
    public init(config: Config = .default) {
        self.config = config
    }
    
    /// Load the embedding model
    ///
    /// Downloads the WeSpeaker model if needed and prepares for inference.
    /// This must be called before `extractEmbedding`.
    public func load() async throws {
        // Download models (uses FluidAudio's DiarizerModels which includes WeSpeaker)
        let models = try await DiarizerModels.downloadIfNeeded()
        
        // Create direct embedding extractor
        let candidateExtractor = EmbeddingExtractor(embeddingModel: models.embeddingModel)
        
        // Initialize VAD if enabled
        let candidateVAD: VadManager?
        if config.useVAD {
            let vadConfig = VadConfig(defaultThreshold: config.vadThreshold)
            candidateVAD = try await PinnedVADModel.makeManager(config: vadConfig)
        } else {
            candidateVAD = nil
        }
        try Task.checkCancellation()
        embeddingExtractor = candidateExtractor
        vadManager = candidateVAD
    }
    
    // MARK: - Embedding Extraction
    
    /// Extract speaker embedding from audio samples
    ///
    /// Uses improved embedding extraction with:
    /// - VAD-based trimming of leading/trailing silence
    /// - Accurate mask marking 0 for mid-audio silences
    /// - Audio repetition to fill 10 seconds (better than zero-padding)
    /// - RMS normalization for consistent input levels
    /// - Direct embedding extraction (bypasses diarization pipeline)
    ///
    /// - Parameter audio: Audio samples at 16kHz mono
    /// - Returns: 256-dimensional L2-normalized embedding vector
    /// - Throws: `AudioToolError.modelNotLoaded` if not loaded
    ///
    /// ## Performance
    /// - ~50-70ms on Apple Silicon (VAD adds ~5-10ms)
    /// - Memory: ~100MB peak during inference
    public func extractEmbedding(_ audio: [Float]) async throws -> [Float] {
        guard let extractor = embeddingExtractor else {
            throw AudioToolError.modelNotLoaded("SpeakerEmbeddingProvider")
        }
        
        guard !audio.isEmpty else {
            throw AudioToolError.resourceUnavailable("Empty audio provided for embedding extraction")
        }
        
        // Step 1: Run VAD to detect speech regions (if enabled)
        var processedAudio = audio
        var vadMask: [Float]? = nil
        
        if config.useVAD, let vad = vadManager {
            let vadResult = try await runVADAndTrim(audio: audio, vad: vad)
            processedAudio = vadResult.trimmedAudio
            vadMask = vadResult.speechMask  // Mask for trimmed audio (1=speech, 0=silence)
        }
        
        // Step 2: Normalize audio (RMS normalization)
        let normalizedAudio = normalizeRMS(processedAudio, targetRMS: config.targetRMS)
        
        // Step 3: Prepare 10s audio using repetition or zero-padding
        let paddedAudio: [Float]
        let mask: [Float]
        
        if config.useRepetition {
            (paddedAudio, mask) = prepareAudioWithRepetition(normalizedAudio, vadMask: vadMask)
        } else {
            (paddedAudio, mask) = prepareAudioWithZeroPadding(normalizedAudio)
        }
        
        // Step 4: Create masks array (speaker 0 = active, speakers 1,2 = empty)
        let emptyMask = [Float](repeating: 0, count: config.wespeakerFrames)
        let masks = [mask, emptyMask, emptyMask]
        
        // Step 5: Extract embedding directly
        let embeddings = try extractor.getEmbeddings(
            audio: paddedAudio,
            masks: masks,
            minActivityThreshold: 0.0  // Accept all activity
        )
        
        guard let embedding = embeddings.first, embedding.count == embeddingDimension else {
            // Fallback: return zero embedding
            return [Float](repeating: 0, count: embeddingDimension)
        }
        
        return embedding
    }
    
    /// VAD result containing trimmed audio and speech mask
    private struct VADResult {
        let trimmedAudio: [Float]
        let speechMask: [Float]  // 1.0 = speech, 0.0 = silence (per frame)
    }
    
    /// Run VAD to trim leading/trailing silence and build speech mask
    private func runVADAndTrim(audio: [Float], vad: VadManager) async throws -> VADResult {
        // Run VAD to get speech probabilities for each chunk
        let vadChunks = try await vad.process(audio)
        
        // VAD processes in 256ms chunks at 16kHz = 4096 samples per chunk
        // (Note: chunk size used for documentation, actual timing is from vadChunks)
        
        // Build frame-level speech mask (1 = speech, 0 = silence)
        // WeSpeaker uses ~17ms frames, so we need to map VAD chunks to frames
        let frameStep: Float = 0.017  // 17ms per WeSpeaker frame
        let audioDuration = Float(audio.count) / Float(sampleRate)
        let numFrames = Int(audioDuration / frameStep)
        
        var frameMask = [Float](repeating: 0, count: numFrames)
        
        for (chunkIdx, chunk) in vadChunks.enumerated() {
            let isSpeech = chunk.probability >= config.vadThreshold
            
            // Map chunk to frames
            let chunkStartTime = Float(chunkIdx) * 0.256  // 256ms per VAD chunk
            let chunkEndTime = chunkStartTime + 0.256
            
            let startFrame = Int(chunkStartTime / frameStep)
            let endFrame = min(Int(chunkEndTime / frameStep), numFrames)
            
            for frame in startFrame..<endFrame {
                frameMask[frame] = isSpeech ? 1.0 : 0.0
            }
        }
        
        // Find first and last speech frames for trimming.
        //
        // Left as optionals rather than seeded with the full input bounds. Seeding
        // made "no speech anywhere" indistinguishable from "speech spans the whole
        // recording": both produced 0...numFrames-1, the range passed the validity
        // check below, and the fallback underneath it could never run. A silent
        // clip came back untrimmed with a mask that was zero everywhere, which is
        // the one mask an embedding cannot be pooled over.
        var firstSpeechFrame: Int?
        var lastSpeechFrame: Int?

        for i in 0..<numFrames {
            if frameMask[i] > 0.5 {
                firstSpeechFrame = i
                break
            }
        }

        for i in stride(from: numFrames - 1, through: 0, by: -1) {
            if frameMask[i] > 0.5 {
                lastSpeechFrame = i
                break
            }
        }

        guard let firstSpeechFrame, let lastSpeechFrame else {
            // No speech detected - use original audio with all-active mask
            return VADResult(
                trimmedAudio: audio,
                speechMask: [Float](repeating: 1.0, count: numFrames)
            )
        }

        // Convert frames to samples for trimming
        let firstSpeechSample = max(0, Int(Float(firstSpeechFrame) * frameStep * Float(sampleRate)))
        let lastSpeechSample = min(audio.count, Int(Float(lastSpeechFrame + 1) * frameStep * Float(sampleRate)))

        // Speech frames were found, but frame-to-sample rounding can still collapse
        // the range on very short input.
        guard firstSpeechSample < lastSpeechSample else {
            return VADResult(
                trimmedAudio: audio,
                speechMask: [Float](repeating: 1.0, count: numFrames)
            )
        }

        return VADResult(
            trimmedAudio: Array(audio[firstSpeechSample..<lastSpeechSample]),
            speechMask: Array(frameMask[firstSpeechFrame...lastSpeechFrame])
        )
    }
    
    /// Extract speaker embedding from audio file URL
    ///
    /// Convenience method that loads audio from file, resamples if needed,
    /// and extracts the embedding.
    ///
    /// - Parameter url: Path to audio file (WAV, CAF, MP3, etc.)
    /// - Returns: 256-dimensional embedding vector
    public func extractEmbedding(from url: URL) async throws -> [Float] {
        // Load and resample audio to 16kHz mono
        let audio = try loadAndResampleAudio(from: url)
        return try await extractEmbedding(audio)
    }
    
    /// Extract embeddings for multiple audio samples in batch
    ///
    /// More efficient than calling `extractEmbedding` multiple times
    /// as it can potentially batch model inference.
    ///
    /// - Parameter audioSamples: Array of audio sample arrays
    /// - Returns: Array of embedding vectors
    public func extractEmbeddings(_ audioSamples: [[Float]]) async throws -> [[Float]] {
        var embeddings: [[Float]] = []
        embeddings.reserveCapacity(audioSamples.count)
        
        for audio in audioSamples {
            let embedding = try await extractEmbedding(audio)
            embeddings.append(embedding)
        }
        
        return embeddings
    }
    
    // MARK: - SpeakerEmbeddingExtractor Protocol Conformance
    
    /// Extract speaker embedding from AudioBuffer
    ///
    /// Conforms to `SpeakerEmbeddingExtractor` protocol.
    /// Delegates to the `extractEmbedding(_ audio: [Float])` method.
    ///
    /// - Parameter audio: Audio buffer (16kHz mono expected)
    /// - Returns: 256-dimensional L2-normalized embedding vector
    public func extractEmbedding(_ audio: AudioToolCore.AudioBuffer) async throws -> [Float] {
        try validateInputChannels(audio)
        // Fluid models are trained behind an ordinary band-limited resampler.
        // Linear interpolation aliases frequencies above the target Nyquist into
        // the speaker band and can materially change an embedding.
        let samples = try audio.resampled(
            to: sampleRate,
            quality: .auto
        ).samples
        
        return try await extractEmbedding(samples)
    }
    
    /// Process audio buffer (AudioProcessor protocol requirement)
    ///
    /// Note: Speaker embedding extraction is not a typical audio processing operation.
    /// This method throws an error as embeddings should be extracted via `extractEmbedding()`.
    ///
    /// - Parameter input: Input audio buffer
    /// - Throws: `AudioToolError.resourceUnavailable` - use `extractEmbedding()` instead
    public func process(_ input: AudioToolCore.AudioBuffer) async throws -> AudioToolCore.AudioBuffer {
        throw AudioToolError.resourceUnavailable(
            "SpeakerEmbeddingProvider does not process audio. Use extractEmbedding() instead."
        )
    }
    
    // MARK: - Audio Preprocessing
    
    /// Normalize audio to target RMS level
    private func normalizeRMS(_ audio: [Float], targetRMS: Float) -> [Float] {
        // Calculate current RMS
        let sumSquares = audio.reduce(0.0) { $0 + $1 * $1 }
        let rms = sqrt(sumSquares / Float(audio.count))
        
        guard rms > 1e-6 else { return audio }  // Avoid division by zero
        
        // Scale to target RMS
        if rms < targetRMS {
            let scale = targetRMS / rms
            return audio.map { $0 * scale }
        }
        
        return audio
    }
    
    /// Prepare audio with repetition (recommended for better embeddings)
    ///
    /// Repeats short audio cyclically to fill 10 seconds.
    /// Also builds a corresponding mask that repeats with the audio.
    /// If VAD mask is provided, uses it to mark silence frames as 0.
    private func prepareAudioWithRepetition(_ audio: [Float], vadMask: [Float]? = nil) -> (audio: [Float], mask: [Float]) {
        let targetSamples = config.targetSamples
        let targetFrames = config.wespeakerFrames
        
        var paddedAudio: [Float]
        
        if audio.count < targetSamples {
            // Repeat audio cyclically to fill 10 seconds
            paddedAudio = []
            paddedAudio.reserveCapacity(targetSamples)
            
            while paddedAudio.count < targetSamples {
                let remaining = targetSamples - paddedAudio.count
                if remaining >= audio.count {
                    paddedAudio.append(contentsOf: audio)
                } else {
                    paddedAudio.append(contentsOf: audio.prefix(remaining))
                }
            }
        } else {
            // Audio is already 10s or longer - just truncate
            paddedAudio = Array(audio.prefix(targetSamples))
        }
        
        // Build mask that matches the repeated audio pattern
        let frameStep: Float = 0.017  // WeSpeaker frame step in seconds
        let originalDuration = Float(audio.count) / Float(sampleRate)
        let originalFrames = Int(originalDuration / frameStep)
        
        // Use VAD mask if provided, otherwise use all-active mask
        let originalMask: [Float]
        if let vadMask = vadMask, !vadMask.isEmpty {
            // Resample VAD mask to match expected frame count if needed
            if vadMask.count == originalFrames {
                originalMask = vadMask
            } else {
                // Resample mask to match frame count
                originalMask = resampleMask(vadMask, toLength: max(1, originalFrames))
            }
        } else {
            // No VAD mask - use all-active
            originalMask = [Float](repeating: 1.0, count: max(1, originalFrames))
        }
        
        // Repeat mask pattern to fill target frames
        var mask: [Float] = []
        while mask.count < targetFrames {
            let remaining = targetFrames - mask.count
            if remaining >= originalMask.count {
                mask.append(contentsOf: originalMask)
            } else {
                mask.append(contentsOf: originalMask.prefix(remaining))
            }
        }
        
        // Ensure exactly targetFrames
        if mask.count > targetFrames {
            mask = Array(mask.prefix(targetFrames))
        }
        
        return (paddedAudio, mask)
    }
    
    /// Resample mask to target length using nearest-neighbor
    private func resampleMask(_ mask: [Float], toLength targetLength: Int) -> [Float] {
        guard !mask.isEmpty, targetLength > 0 else {
            return [Float](repeating: 1.0, count: targetLength)
        }
        
        var resampled = [Float](repeating: 0, count: targetLength)
        let ratio = Float(mask.count) / Float(targetLength)
        
        for i in 0..<targetLength {
            let sourceIndex = min(Int(Float(i) * ratio), mask.count - 1)
            resampled[i] = mask[sourceIndex]
        }
        
        return resampled
    }
    
    /// Prepare audio with zero-padding (for comparison)
    private func prepareAudioWithZeroPadding(_ audio: [Float]) -> (audio: [Float], mask: [Float]) {
        let targetSamples = config.targetSamples
        let targetFrames = config.wespeakerFrames
        
        var paddedAudio: [Float]
        
        if audio.count < targetSamples {
            paddedAudio = [Float](repeating: 0, count: targetSamples)
            for i in 0..<audio.count {
                paddedAudio[i] = audio[i]
            }
        } else {
            paddedAudio = Array(audio.prefix(targetSamples))
        }
        
        // Full mask for entire 10s (even though zero-padded portion is silent)
        let mask = [Float](repeating: 1.0, count: targetFrames)
        
        return (paddedAudio, mask)
    }
    
    // MARK: - Audio Loading
    
    /// Load and resample audio from file to 16kHz mono
    private func loadAndResampleAudio(from url: URL) throws -> [Float] {
        let converter = AudioConverter()
        return try converter.resampleAudioFile(path: url.path)
    }
}

// MARK: - FluidAudioProviders Extension

extension FluidAudioProviders {
    
    /// Create speaker embedding provider with default configuration
    ///
    /// Uses WeSpeaker model for 256-dimensional embeddings suitable for
    /// voice matching, speaker verification, and similarity comparisons.
    ///
    /// - Returns: Configured provider (must call `load()` before use)
    public static func speakerEmbedding() -> SpeakerEmbeddingProvider {
        SpeakerEmbeddingProvider()
    }
    
    /// Create speaker embedding provider with custom configuration
    ///
    /// - Parameter config: Custom configuration for embedding extraction
    /// - Returns: Configured provider (must call `load()` before use)
    public static func speakerEmbedding(config: SpeakerEmbeddingProvider.Config) -> SpeakerEmbeddingProvider {
        SpeakerEmbeddingProvider(config: config)
    }
}
