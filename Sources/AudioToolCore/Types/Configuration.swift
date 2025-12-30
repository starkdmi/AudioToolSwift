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

// MARK: - Model Precision

/// Model weight precision for loading
///
/// Supports two patterns:
/// - **Single-repo**: Different weight files (e.g., model_fp32.safetensors, model_fp16.safetensors)
/// - **Multi-repo**: Different HF repos per precision (e.g., Kokoro-82M-bf16, Kokoro-82M-4bit)
///
/// Usage:
/// ```swift
/// // Single-repo pattern (starkdmi models)
/// let provider = MLXProviders.mossformerGANSE16K(precision: .fp16)
///
/// // Multi-repo pattern (Kokoro)
/// let tts = TTSProviders.kokoro(precision: .int4)
/// ```
public enum ModelPrecision: String, Sendable, CaseIterable, Hashable {
    /// Full precision (32-bit floating point)
    case fp32 = "fp32"
    
    /// Half precision (16-bit floating point)
    case fp16 = "fp16"
    
    /// Brain float 16 (bfloat16)
    case bf16 = "bf16"
    
    /// 8-bit integer quantization
    case int8 = "int8"
    
    /// 4-bit integer quantization
    case int4 = "int4"
    
    /// 8-bit quantization (mlx-community style)
    case bit8 = "8bit"
    
    /// 6-bit quantization (mlx-community style)
    case bit6 = "6bit"
    
    /// 4-bit quantization (mlx-community style)
    case bit4 = "4bit"
    
    // MARK: - Single-Repo Pattern (file-based)
    
    /// Weight filename for single-repo pattern
    /// Example: "model_fp32.safetensors", "model_fp16.safetensors"
    public var weightsFilename: String {
        "model_\(rawValue).safetensors"
    }
    
    // MARK: - Multi-Repo Pattern (suffix-based)
    
    /// Repo suffix for multi-repo pattern
    /// Example: "-bf16", "-4bit"
    public var repoSuffix: String {
        "-\(rawValue)"
    }
    
    /// Construct full repo name with precision suffix
    /// Example: "mlx-community/Kokoro-82M" + .bf16 → "mlx-community/Kokoro-82M-bf16"
    public func repo(base: String) -> String {
        "\(base)\(repoSuffix)"
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

/// Available translation models
public enum TranslationModel: Sendable, Hashable {
    case appleTranslation  // Apple Translation framework (iOS 18+, macOS 15+)
    // Future: .marian, .nllb, .seamless
}

/// Kokoro TTS supported languages
public enum KokoroLanguage: String, Sendable, Hashable, CaseIterable {
    case americanEnglish = "a"  // 🇺🇸 en-US
    case britishEnglish = "b"   // 🇬🇧 en-GB
    case japanese = "j"         // 🇯🇵 ja
    case chinese = "z"          // 🇨🇳 zh
}

/// Available TTS models
public enum SynthesisModel: Sendable, Hashable {
    case kokoro(language: KokoroLanguage, voice: String)
    case appleTTS(language: String)  // AVSpeechSynthesizer, 60+ languages
    case f5tts
    case cosyVoice        // Voice cloning
    case outeTTS          // Voice cloning
}

extension SynthesisModel: ModelIdentifier {
    public var modelName: String {
        switch self {
        case .kokoro: return "kokoro_tts"
        case .appleTTS: return "apple_tts"
        case .f5tts: return "f5_tts"
        case .cosyVoice: return "cosy_voice"
        case .outeTTS: return "oute_tts"
        }
    }
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
