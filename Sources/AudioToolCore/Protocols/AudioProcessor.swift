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

// MARK: - Specialized Protocols

/// Voice Activity Detection
public protocol VADProvider: StreamableProcessor {
    /// Detect speech segments in audio
    func detect(_ audio: AudioBuffer) async throws -> [VADSegment]
    
    /// Stream VAD segments as detected
    func streamDetection(_ audio: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<VADSegment, Error>
}

/// Speaker Diarization
public protocol DiarizationProvider: AudioProcessor {
    /// Identify speakers in audio
    func diarize(_ audio: AudioBuffer) async throws -> SpeakerTimeline
    
    /// Diarize with VAD hint for efficiency
    func diarize(_ audio: AudioBuffer, vadHint: [VADSegment]) async throws -> SpeakerTimeline
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
    
    /// Stream transcription segments as recognized
    func streamTranscription(_ audio: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<TranscriptionSegment, Error>
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
