//
//  AudioProcessor.swift
//  AudioTool
//
//  Core processing protocols for audio ML models
//

import Foundation

// MARK: - Base Protocol

/// Format contract shared by everything that consumes audio.
///
/// This describes only what a processor expects and produces. It deliberately does
/// **not** require a `process` method: most audio models are not one-buffer-in,
/// one-buffer-out. A separator returns N tracks, a transcriber returns text, a
/// classifier returns labels. Forcing them all through a single `AudioBuffer ->
/// AudioBuffer` signature meant those types had to invent a return value, and the
/// values they invented silently discarded results - a separator returning only its
/// first speaker, a music separator returning only vocals.
///
/// Conform to ``AudioTransform`` instead when a type genuinely is 1-to-1.
public protocol AudioProcessor: Sendable {
    /// Preferred sample rate (Hz)
    var sampleRate: Int { get }

    /// Expected input channels
    var inputChannels: Int { get }

    /// Output channels produced
    var outputChannels: Int { get }
}

// MARK: - Audio Transform

/// A processor that genuinely maps one audio buffer to one audio buffer.
///
/// Enhancement, denoising and super-resolution belong here. Anything whose natural
/// output is a list, a transcript, or a set of labels should conform to
/// ``AudioProcessor`` and expose its own honestly-typed method instead.
public protocol AudioTransform: AudioProcessor {
    /// Process audio buffer
    func process(_ input: AudioBuffer) async throws -> AudioBuffer
}

// MARK: - Streamable Processor

/// Processor supporting streaming/chunked processing
public protocol StreamableProcessor: AudioTransform {
    /// Minimum chunk size for streaming (samples)
    var minChunkSize: Int { get }
    
    /// Recommended chunk size for optimal performance
    var recommendedChunkSize: Int { get }
    
    /// Process stream of chunks
    func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error>
    
    /// Reset internal state between streams
    func reset() async
}

// MARK: - Progress Reporting

/// Progress callback type for long-running operations
/// - Parameter percent: Progress percentage (0.0 to 100.0)
public typealias ProgressCallback = @Sendable (Double) async -> Void

/// Protocol for providers that support chunked progress reporting during batch processing.
/// Renamed from ProgressReporting to avoid conflict with Foundation's NSProgressReporting.
public protocol ChunkedProgressProvider: Sendable {
    /// Whether this provider supports progress callbacks during processing
    var supportsChunkedProgress: Bool { get }
}

// MARK: - Output Streaming

/// Protocol for processors that can stream output chunks during processing.
/// This allows yielding processed audio chunks as they become ready,
/// rather than waiting for complete processing.
public protocol StreamableOutput: AudioTransform {
    /// Process audio and stream output chunks as they're ready.
    /// Each yielded chunk represents a portion of the processed audio.
    /// Consumers can play/save chunks immediately for lower latency.
    /// - Parameter input: Complete input audio to process
    /// - Returns: Async stream of processed audio chunks
    nonisolated func processStream(_ input: AudioBuffer) -> AsyncThrowingStream<AudioBuffer, Error>
}

// MARK: - Specialized Protocols

/// Voice Activity Detection
///
/// Refines ``AudioProcessor`` rather than ``StreamableProcessor``: VAD reports where
/// speech is, it does not alter audio. Inheriting the transform protocol forced two
/// meaningless members on every conformer - a `process` that returned its input and
/// a `stream` that yielded its input back unchanged - which is the same defect that
/// splitting ``AudioTransform`` out of ``AudioProcessor`` exists to remove. The real
/// streaming entry point is ``streamDetection(_:)``.
public protocol VADProvider: AudioProcessor {
    /// Minimum chunk size the detector accepts (samples).
    /// Silero requires exactly 512-sample frames at 16 kHz, so this is a hard
    /// constraint on callers rather than a performance hint.
    var minChunkSize: Int { get }

    /// Recommended chunk size for optimal performance
    var recommendedChunkSize: Int { get }

    /// Reset internal state between streams
    func reset() async

    /// Detect speech segments in audio
    func detect(_ audio: AudioBuffer) async throws -> [VADSegment]

    /// Detect speech segments with progress reporting
    /// - Parameters:
    ///   - audio: Input audio buffer
    ///   - onProgress: Callback with progress percentage (0.0 to 100.0)
    /// - Returns: Array of VAD segments
    func detect(_ audio: AudioBuffer, onProgress: ProgressCallback?) async throws -> [VADSegment]
    
    /// Stream VAD segments as detected
    func streamDetection(_ audio: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<VADSegment, Error>
}

/// Default implementation for progress-aware detection
public extension VADProvider {
    /// Default implementation calls detect without progress
    func detect(_ audio: AudioBuffer, onProgress: ProgressCallback?) async throws -> [VADSegment] {
        await onProgress?(0.0)
        let result = try await detect(audio)
        await onProgress?(100.0)
        return result
    }
}

/// Speaker Diarization
public protocol DiarizationProvider: AudioProcessor {
    /// Identify speakers in audio
    func diarize(_ audio: AudioBuffer) async throws -> SpeakerTimeline
    
    /// Diarize with VAD hint for efficiency
    func diarize(_ audio: AudioBuffer, vadHint: [VADSegment]) async throws -> SpeakerTimeline
    
    /// Diarize with progress reporting
    /// - Parameters:
    ///   - audio: Input audio buffer
    ///   - onProgress: Callback with progress percentage (0.0 to 100.0)
    /// - Returns: Speaker timeline
    func diarize(_ audio: AudioBuffer, onProgress: ProgressCallback?) async throws -> SpeakerTimeline
}

/// Default implementation for progress-aware diarization
public extension DiarizationProvider {
    /// Default implementation calls diarize without progress
    func diarize(_ audio: AudioBuffer, onProgress: ProgressCallback?) async throws -> SpeakerTimeline {
        await onProgress?(0.0)
        let result = try await diarize(audio)
        await onProgress?(100.0)
        return result
    }
}

/// Speaker Identification (for re-identification after separation)
///
/// Providers conforming to this protocol can identify which speaker slot
/// a separated audio track belongs to, using preserved internal state.
/// This is used for re-identifying speakers after source separation.
public protocol SpeakerIdentifier: Sendable {
    /// Identify which speaker slot an audio segment belongs to
    ///
    /// Uses preserved internal state (e.g., spkcache in Sortformer) to determine
    /// which speaker produced the audio.
    ///
    /// - Parameter audio: Audio to identify (will be resampled if needed)
    /// - Returns: Speaker identification with slot and confidence
    func identifySpeaker(_ audio: AudioBuffer) async throws -> SpeakerIdentification
    
    /// Identify multiple separated tracks
    ///
    /// - Parameter tracks: Array of separated audio tracks
    /// - Returns: Array of (track, identification) pairs sorted by speaker slot
    func identifySpeakers(_ tracks: [AudioBuffer]) async throws -> [(audio: AudioBuffer, identification: SpeakerIdentification)]
}

/// Default implementation for batch speaker identification
public extension SpeakerIdentifier {
    func identifySpeakers(_ tracks: [AudioBuffer]) async throws -> [(audio: AudioBuffer, identification: SpeakerIdentification)] {
        var results: [(audio: AudioBuffer, identification: SpeakerIdentification)] = []
        for track in tracks {
            let identification = try await identifySpeaker(track)
            results.append((track, identification))
        }
        return results.sorted { $0.identification.speakerSlot < $1.identification.speakerSlot }
    }
}

/// Speech Enhancement (denoising, cleanup)
public protocol SpeechEnhancer: StreamableProcessor {
    // Inherits process() from AudioTransform: enhancement is genuinely 1-to-1.
}

/// Speech Separation (multi-speaker)
public protocol SpeechSeparator: AudioProcessor {
    /// Supported speaker counts
    var supportedSpeakerCounts: [Int] { get }
    
    /// Separate speakers from mixed audio
    func separate(_ audio: AudioBuffer, speakers: Int) async throws -> [AudioBuffer]
    
    /// Separate speakers with progress reporting
    /// - Parameters:
    ///   - audio: Input audio buffer
    ///   - speakers: Number of speakers to separate
    ///   - onProgress: Progress callback (0.0 to 100.0) as chunks are processed
    /// - Returns: Array of separated audio buffers, one per speaker
    func separate(_ audio: AudioBuffer, speakers: Int, onProgress: ProgressCallback?) async throws -> [AudioBuffer]
}

/// Default implementation for progress-aware separation
public extension SpeechSeparator {
    func separate(_ audio: AudioBuffer, speakers: Int, onProgress: ProgressCallback?) async throws -> [AudioBuffer] {
        // Default: report 0% at start, separate, then report 100% at end
        await onProgress?(0.0)
        let result = try await separate(audio, speakers: speakers)
        await onProgress?(100.0)
        return result
    }
}

/// Audio Super-Resolution (upscaling)
public protocol AudioUpscaler: AudioTransform {
    /// Input sample rate
    var inputSampleRate: Int { get }
    
    /// Output sample rate after upscaling
    var outputSampleRate: Int { get }
}

/// Speech-to-Text
public protocol Transcriber: AudioProcessor {
    /// Transcribe audio to text
    func transcribe(_ audio: AudioBuffer) async throws -> Transcription
    
    /// Transcribe audio with progress callback
    /// Reports progress as segments are recognized
    func transcribe(_ audio: AudioBuffer, onProgress: ProgressCallback?) async throws -> Transcription
    
    /// Stream transcription segments as recognized
    func streamTranscription(_ audio: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<TranscriptionSegment, Error>
}

/// Default implementation for transcribe with progress
public extension Transcriber {
    func transcribe(_ audio: AudioBuffer, onProgress: ProgressCallback?) async throws -> Transcription {
        // Default: report 0% at start, transcribe, then report 100% at end
        await onProgress?(0.0)
        let result = try await transcribe(audio)
        await onProgress?(100.0)
        return result
    }
}

/// Text-to-Speech
public protocol SpeechSynthesizer: Sendable {
    /// Output sample rate
    var sampleRate: Int { get }
    
    /// Available voice identifiers
    var availableVoices: [String] { get }
    
    /// Synthesize text to speech
    func synthesize(_ text: String, voice: String) async throws -> AudioBuffer
    
    /// Stream synthesized audio chunks
    func streamSynthesis(_ text: String, voice: String) -> AsyncThrowingStream<AudioBuffer, Error>
}

/// Sound Classification (USS, background detection)
public protocol SoundClassifier: AudioProcessor {
    /// Classify sounds in audio
    func classify(_ audio: AudioBuffer) async throws -> [SoundClassification]
}

/// Universal Sound Separation (USS)
///
/// Separates a target sound from mixed audio using FiLM conditioning on a
/// ``SoundEmbedding``. The target is an arbitrary 527-d AudioSet class vector, not a
/// fixed menu: ``SoundEmbedding/speech`` and friends are presets over that space,
/// and callers can build their own.
public protocol UniversalSoundSeparator: AudioProcessor {
    /// Separate a target sound from audio
    /// - Parameters:
    ///   - audio: Input audio buffer (32kHz expected)
    ///   - target: Conditioning vector selecting what to extract
    /// - Returns: Separated audio containing only the target sound
    func separateSound(_ audio: AudioBuffer, target: SoundEmbedding) async throws -> AudioBuffer

    /// Separate several targets from the same audio
    /// - Parameters:
    ///   - audio: Input audio buffer (32kHz expected)
    ///   - targets: Conditioning vectors to extract, in order
    ///   - onProgress: Optional callback with progress (0.0 to 100.0) per target
    /// - Returns: Separated audio per target, in the same order as `targets`
    func separateMultipleSounds(
        _ audio: AudioBuffer,
        targets: [SoundEmbedding],
        onProgress: ProgressCallback?
    ) async throws -> [(target: SoundEmbedding, audio: AudioBuffer)]

    /// Separate a target and also return the background (residual)
    /// - Parameters:
    ///   - audio: Input audio buffer
    ///   - target: Conditioning vector selecting what to extract
    /// - Returns: Tuple of (separated target, background residual)
    func separateSoundWithBackground(
        _ audio: AudioBuffer,
        target: SoundEmbedding
    ) async throws -> (separated: AudioBuffer, background: AudioBuffer)
}

/// Default implementation for separateMultipleSounds with progress
public extension UniversalSoundSeparator {
    func separateMultipleSounds(
        _ audio: AudioBuffer,
        targets: [SoundEmbedding],
        onProgress: ProgressCallback?
    ) async throws -> [(target: SoundEmbedding, audio: AudioBuffer)] {
        var results: [(target: SoundEmbedding, audio: AudioBuffer)] = []
        results.reserveCapacity(targets.count)
        for (idx, target) in targets.enumerated() {
            results.append((target, try await separateSound(audio, target: target)))
            await onProgress?(Double(idx + 1) / Double(targets.count) * 100.0)
        }
        return results
    }
}

// MARK: - Text Translation

/// Text translation provider
/// Note: Translation is text-only (not AudioProcessor) but follows same patterns
public protocol TextTranslator: Sendable {
    /// Translate single text string
    /// - Parameters:
    ///   - text: Source text to translate
    ///   - source: Source language code (BCP-47) or nil for auto-detect
    ///   - target: Target language code (BCP-47)
    /// - Returns: Translation result
    func translate(
        _ text: String,
        from source: String?,
        to target: String
    ) async throws -> TranslationResult
    
    /// Translate batch of strings
    /// - Parameters:
    ///   - texts: Array of source texts
    ///   - source: Source language code (BCP-47) or nil for auto-detect
    ///   - target: Target language code (BCP-47)
    /// - Returns: Batch translation result
    func translateBatch(
        _ texts: [String],
        from source: String?,
        to target: String
    ) async throws -> BatchTranslationResult
    
    /// Check if language pair is available (models downloaded)
    /// - Parameters:
    ///   - source: Source language code (BCP-47)
    ///   - target: Target language code (BCP-47)
    /// - Returns: true if translation is available offline
    func isAvailable(from source: String, to target: String) async -> Bool
    
    /// Prepare language pair (trigger download prompt if needed)
    /// - Parameters:
    ///   - source: Source language code
    ///   - target: Target language code
    func prepareLanguagePair(from source: String, to target: String) async throws
}

// MARK: - Model Identifier Protocol

/// Protocol for identifying models (for preloading/caching)
public protocol ModelIdentifier: Sendable, Hashable {
    var modelName: String { get }
}

extension VADModel: ModelIdentifier {
    public var modelName: String {
        switch self {
        case .silero: return "silero_vad"
        case .sileroManual: return "silero_vad_manual"
        }
    }
}

extension EnhancementModel: ModelIdentifier {
    public var modelName: String {
        switch self {
        case .mossformerSE16k: return "mossformer2_se_16k"
        case .mossformerSE48k: return "mossformer2_se_48k"
        case .mossformerGAN: return "mossformer_gan_se"
        case .frcrn: return "frcrn_se"
        }
    }
}

extension SeparationModel: ModelIdentifier {
    public var modelName: String {
        switch self {
        case .mossformer2spk: return "mossformer2_ss_2spk"
        case .mossformer3spk: return "mossformer2_ss_3spk"
        case .mossformerWhamr: return "mossformer2_ss_whamr"
        case .demucs: return "demucs"
        case .uss: return "uss"
        }
    }
}

extension TranscriptionModel: ModelIdentifier {
    public var modelName: String {
        switch self {
        case .parakeet: return "parakeet_v3"
        case .whisperTiny: return "whisper_tiny"
        case .whisperBase: return "whisper_base"
        case .whisperSmall: return "whisper_small"
        case .whisperLarge: return "whisper_large_v3"
        case .appleSpeech: return "apple_speech"
        }
    }
}

extension TranslationModel: ModelIdentifier {
    public var modelName: String {
        switch self {
        case .appleTranslation: return "apple_translation"
        case .translateGemma: return "translate_gemma"
        }
    }
}

// MARK: - Speaker Embedding Extraction

/// Protocol for extracting speaker embeddings from audio
///
/// Used for speaker verification, voice matching, and re-identification
/// when Sortformer is not available (e.g., 5+ speakers).
public protocol SpeakerEmbeddingExtractor: AudioProcessor {
    /// Embedding dimension (typically 256 for WeSpeaker)
    var embeddingDimension: Int { get }
    
    /// Extract speaker embedding from audio samples
    /// - Parameter audio: Audio buffer (16kHz mono expected)
    /// - Returns: L2-normalized embedding vector
    func extractEmbedding(_ audio: AudioBuffer) async throws -> [Float]
    
    /// Extract embeddings from multiple audio segments
    /// - Parameter segments: Array of audio buffers
    /// - Returns: Array of embedding vectors
    func extractEmbeddings(_ segments: [AudioBuffer]) async throws -> [[Float]]
}

/// Default implementation for batch embedding extraction
public extension SpeakerEmbeddingExtractor {
    func extractEmbeddings(_ segments: [AudioBuffer]) async throws -> [[Float]] {
        var embeddings: [[Float]] = []
        embeddings.reserveCapacity(segments.count)
        for segment in segments {
            let embedding = try await extractEmbedding(segment)
            embeddings.append(embedding)
        }
        return embeddings
    }
}

/// Result of speaker identification using embeddings
public struct EmbeddingIdentificationResult: Sendable, Equatable {
    /// The matched speaker ID
    public let speakerID: SpeakerID
    
    /// Cosine similarity score (0.0 - 1.0, higher is better match)
    public let similarity: Float
    
    /// The embedding vector of the identified track
    public let embedding: [Float]
    
    public init(speakerID: SpeakerID, similarity: Float, embedding: [Float]) {
        self.speakerID = speakerID
        self.similarity = similarity
        self.embedding = embedding
    }
    
    /// Check if the match is confident enough
    /// - Parameter threshold: Minimum similarity threshold (default 0.7 for cosine similarity)
    public func isConfident(threshold: Float = 0.7) -> Bool {
        similarity >= threshold
    }
}
