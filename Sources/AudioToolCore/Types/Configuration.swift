//
//  Configuration.swift
//  ClearVoice
//
//  Framework configuration and model enums
//

import Foundation

// MARK: - Configuration

/// Global framework configuration
public struct ClearVoiceConfiguration: Sendable {
    
    /// Compute precision
    public var precision: Precision
    
    /// Maximum memory for loaded models (bytes)
    public var modelMemoryLimit: Int
    
    /// Enable STFT result caching
    public var enableSTFTCache: Bool
    
    /// Custom model repository base URL
    public var modelRepository: URL?
    
    /// Segment pool size for streaming
    public var segmentPoolSize: Int
    
    /// Backpressure channel capacity
    public var channelCapacity: Int
    
    public init(
        precision: Precision = .float16,
        modelMemoryLimit: Int = 2_000_000_000,
        enableSTFTCache: Bool = true,
        modelRepository: URL? = nil,
        segmentPoolSize: Int = 32,
        channelCapacity: Int = 16
    ) {
        self.precision = precision
        self.modelMemoryLimit = modelMemoryLimit
        self.enableSTFTCache = enableSTFTCache
        self.modelRepository = modelRepository
        self.segmentPoolSize = segmentPoolSize
        self.channelCapacity = channelCapacity
    }
    
    public static let `default` = ClearVoiceConfiguration()
    
    public enum Precision: Sendable {
        case float16
        case float32
    }
}

// MARK: - Model Enums

/// Available VAD implementations
public enum VADModel: Sendable, Hashable {
    case silero           // FluidAudio (default)
    case sileroManual     // Direct CoreML
}

/// Available enhancement models
public enum EnhancementModel: Sendable, Hashable {
    case mossformerSE16k  // MLX, 16kHz, ~30x RTF
    case mossformerSE48k  // MLX, 48kHz, ~30x RTF
    case mossformerGAN    // CoreML, 16kHz, ~3.5x RTF
    case frcrn            // MLX/CoreML, 16kHz, 5-11x RTF
}

/// Available separation models
public enum SeparationModel: Sendable, Hashable {
    case mossformer2spk   // MLX, 2 speakers
    case mossformer3spk   // MLX, 3 speakers
    case mossformerWhamr  // MLX, 2 speakers, noisy
    case demucs           // MLX, music stems
    case uss              // CoreML/MLX, universal
}

/// Available upscaling models
public enum UpscalingModel: Sendable, Hashable {
    case mossformerSR     // MLX, 16k → 48k
}

/// Available transcription models
public enum TranscriptionModel: Sendable, Hashable {
    case parakeet         // FluidAudio, ~200x RTF
    case whisperTiny      // MLX
    case whisperBase      // MLX
    case whisperSmall     // MLX
    case whisperLarge     // MLX, supports translation
    case appleSpeech      // Native, iOS 26+
}

/// Available TTS models
public enum SynthesisModel: Sendable, Hashable {
    case kokoro(voice: String)
    case f5tts
    case cosyVoice        // Voice cloning
    case outeTTS          // Voice cloning
}

/// Audio format for export
public enum AudioFormat: Sendable, Hashable {
    case wav
    case mp3
    case m4a
    case flac
}

// MARK: - Audio Source

/// Input audio source
public enum AudioSource: Sendable {
    case file(URL)
    case buffer(AudioBuffer)
    
    // TODO: Add streaming sources
    // case microphone
    // case stream(AsyncStream<AudioBuffer>)
}
