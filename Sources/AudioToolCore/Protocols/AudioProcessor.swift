//
//  AudioProcessor.swift
//  ClearVoice
//
//  Core processing protocols for audio ML models
//

import Foundation

// MARK: - Base Protocol

/// Base protocol for all audio processors
public protocol AudioProcessor: Sendable {
    /// Preferred sample rate (Hz)
    var sampleRate: Int { get }
    
    /// Expected input channels
    var inputChannels: Int { get }
    
    /// Output channels produced
    var outputChannels: Int { get }
    
    /// Process audio buffer
    func process(_ input: AudioBuffer) async throws -> AudioBuffer
}

// MARK: - Streamable Processor

/// Processor supporting streaming/chunked processing
public protocol StreamableProcessor: AudioProcessor {
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
public protocol StreamableOutput: AudioProcessor {
    /// Process audio and stream output chunks as they're ready.
    /// Each yielded chunk represents a portion of the processed audio.
    /// Consumers can play/save chunks immediately for lower latency.
    /// - Parameter input: Complete input audio to process
    /// - Returns: Async stream of processed audio chunks
    nonisolated func processStream(_ input: AudioBuffer) -> AsyncThrowingStream<AudioBuffer, Error>
}

// MARK: - Specialized Protocols

/// Voice Activity Detection
public protocol VADProvider: StreamableProcessor {
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

/// Speech Enhancement (denoising, cleanup)
public protocol SpeechEnhancer: StreamableProcessor {
    // Inherits process() from AudioProcessor
}

/// Speech Separation (multi-speaker)
public protocol SpeechSeparator: AudioProcessor {
    /// Supported speaker counts
    var supportedSpeakerCounts: [Int] { get }
    
    /// Separate speakers from mixed audio
    func separate(_ audio: AudioBuffer, speakers: Int) async throws -> [AudioBuffer]
}

/// Audio Super-Resolution (upscaling)
public protocol AudioUpscaler: AudioProcessor {
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
/// Separates specific sound types from mixed audio using FiLM conditioning.
/// Supports efficient embedding switching for multi-type separation.
public protocol UniversalSoundSeparator: AudioProcessor {
    /// Separate a specific sound type from audio
    /// - Parameters:
    ///   - audio: Input audio buffer (32kHz expected)
    ///   - type: Target sound type to extract
    /// - Returns: Separated audio containing only the target sound type
    func separateSound(_ audio: AudioBuffer, type: USSSoundType) async throws -> AudioBuffer
    
    /// Separate multiple sound types from audio
    /// - Parameters:
    ///   - audio: Input audio buffer (32kHz expected)
    ///   - types: Array of target sound types to extract
    ///   - onProgress: Optional callback with progress (0.0 to 100.0) per embedding
    /// - Returns: Dictionary mapping sound type to separated audio
    func separateMultipleSounds(
        _ audio: AudioBuffer,
        types: [USSSoundType],
        onProgress: ProgressCallback?
    ) async throws -> [USSSoundType: AudioBuffer]
    
    /// Separate sound type and also return background (residual)
    /// - Parameters:
    ///   - audio: Input audio buffer
    ///   - type: Target sound type to extract
    /// - Returns: Tuple of (separated target, background residual)
    func separateSoundWithBackground(
        _ audio: AudioBuffer,
        type: USSSoundType
    ) async throws -> (separated: AudioBuffer, background: AudioBuffer)
}

/// Default implementation for separateMultipleSounds with progress
public extension UniversalSoundSeparator {
    func separateMultipleSounds(
        _ audio: AudioBuffer,
        types: [USSSoundType],
        onProgress: ProgressCallback?
    ) async throws -> [USSSoundType: AudioBuffer] {
        var results: [USSSoundType: AudioBuffer] = [:]
        for (idx, type) in types.enumerated() {
            results[type] = try await separateSound(audio, type: type)
            let percent = Double(idx + 1) / Double(types.count) * 100.0
            await onProgress?(percent)
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
