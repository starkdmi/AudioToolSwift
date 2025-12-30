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
///
/// All languages use misaki[en] except Japanese (misaki[ja]) and Chinese (misaki[zh])
public enum KokoroLanguage: String, Sendable, Hashable, CaseIterable {
    case americanEnglish = "a"  // 🇺🇸 en-US (misaki[en])
    case britishEnglish = "b"   // 🇬🇧 en-GB (misaki[en])
    case japanese = "j"         // 🇯🇵 ja (misaki[ja] - not yet supported)
    case chinese = "z"          // 🇨🇳 zh (misaki[zh] - not yet supported)
    case spanish = "e"          // 🇪🇸 es (misaki[en])
    case french = "f"           // 🇫🇷 fr (misaki[en])
    case hindi = "h"            // 🇮🇳 hi (misaki[en])
    case italian = "i"          // 🇮🇹 it (misaki[en])
    case portuguese = "p"       // 🇧🇷 pt-BR (misaki[en])
}

// MARK: - Kokoro Voices

/// All available Kokoro TTS voices
///
/// Voice naming convention:
/// - First letter: Language code (a=American English, b=British English, etc.)
/// - Second letter: Gender (f=female, m=male)
/// - Rest: Voice name
///
/// Quality grades based on training data:
/// - **A**: Best quality (e.g., af_heart, af_bella)
/// - **B-**: Good quality
/// - **C/C+**: Average quality
/// - **D**: Lower quality
///
/// For detailed quality info, see: https://huggingface.co/mlx-community/Kokoro-82M-bf16/blob/main/VOICES.md
public enum KokoroVoice: String, Sendable, Hashable, CaseIterable {
    // MARK: - 🇺🇸 American English (11F 9M)
    /// American English female - heart voice (A grade, best quality)
    case af_heart
    /// American English female - alloy
    case af_alloy
    /// American English female - aoede
    case af_aoede
    /// American English female - bella (A- grade, high quality)
    case af_bella
    /// American English female - jessica
    case af_jessica
    /// American English female - kore
    case af_kore
    /// American English female - nicole
    case af_nicole
    /// American English female - nova
    case af_nova
    /// American English female - river
    case af_river
    /// American English female - sarah
    case af_sarah
    /// American English female - sky
    case af_sky
    /// American English male - adam
    case am_adam
    /// American English male - echo
    case am_echo
    /// American English male - eric
    case am_eric
    /// American English male - fenrir
    case am_fenrir
    /// American English male - liam
    case am_liam
    /// American English male - michael
    case am_michael
    /// American English male - onyx
    case am_onyx
    /// American English male - puck
    case am_puck
    /// American English male - santa
    case am_santa
    
    // MARK: - 🇬🇧 British English (4F 4M)
    /// British English female - alice
    case bf_alice
    /// British English female - emma (B- grade)
    case bf_emma
    /// British English female - isabella
    case bf_isabella
    /// British English female - lily
    case bf_lily
    /// British English male - daniel
    case bm_daniel
    /// British English male - fable
    case bm_fable
    /// British English male - george
    case bm_george
    /// British English male - lewis
    case bm_lewis
    
    // MARK: - 🇯🇵 Japanese (4F 1M)
    /// Japanese female - alpha
    case jf_alpha
    /// Japanese female - gongitsune
    case jf_gongitsune
    /// Japanese female - nezumi
    case jf_nezumi
    /// Japanese female - tebukuro
    case jf_tebukuro
    /// Japanese male - kumo
    case jm_kumo
    
    // MARK: - 🇨🇳 Mandarin Chinese (4F 4M)
    /// Chinese female - xiaobei
    case zf_xiaobei
    /// Chinese female - xiaoni
    case zf_xiaoni
    /// Chinese female - xiaoxiao
    case zf_xiaoxiao
    /// Chinese female - xiaoyi
    case zf_xiaoyi
    /// Chinese male - yunjian
    case zm_yunjian
    /// Chinese male - yunxi
    case zm_yunxi
    /// Chinese male - yunxia
    case zm_yunxia
    /// Chinese male - yunyang
    case zm_yunyang
    
    // MARK: - 🇪🇸 Spanish (1F 2M)
    /// Spanish female - dora
    case ef_dora
    /// Spanish male - alex
    case em_alex
    /// Spanish male - santa
    case em_santa
    
    // MARK: - 🇫🇷 French (1F)
    /// French female - siwis (B- grade, only French voice)
    case ff_siwis
    
    // MARK: - 🇮🇳 Hindi (2F 2M)
    /// Hindi female - alpha
    case hf_alpha
    /// Hindi female - beta
    case hf_beta
    /// Hindi male - omega
    case hm_omega
    /// Hindi male - psi
    case hm_psi
    
    // MARK: - 🇮🇹 Italian (1F 1M)
    /// Italian female - sara
    case if_sara
    /// Italian male - nicola
    case im_nicola
    
    // MARK: - 🇧🇷 Brazilian Portuguese (1F 2M)
    /// Brazilian Portuguese female - dora
    case pf_dora
    /// Brazilian Portuguese male - alex
    case pm_alex
    /// Brazilian Portuguese male - santa
    case pm_santa
    
    // MARK: - Properties
    
    /// The language this voice belongs to
    public var language: KokoroLanguage {
        let prefix = String(rawValue.prefix(1))
        return KokoroLanguage(rawValue: prefix) ?? .americanEnglish
    }
    
    /// Whether this voice is female
    public var isFemale: Bool {
        rawValue.dropFirst().first == "f"
    }
    
    /// Whether this voice is male
    public var isMale: Bool {
        rawValue.dropFirst().first == "m"
    }
    
    /// Display name without prefix (e.g., "heart" from "af_heart")
    public var displayName: String {
        String(rawValue.dropFirst(3))
    }
}

// MARK: - KokoroLanguage Extensions

extension KokoroLanguage {
    /// All available voices for this language
    public var availableVoices: [KokoroVoice] {
        KokoroVoice.allCases.filter { $0.language == self }
    }
    
    /// Female voices for this language
    public var femaleVoices: [KokoroVoice] {
        availableVoices.filter { $0.isFemale }
    }
    
    /// Male voices for this language
    public var maleVoices: [KokoroVoice] {
        availableVoices.filter { $0.isMale }
    }
    
    /// Default/recommended voice for this language
    public var defaultVoice: KokoroVoice {
        switch self {
        case .americanEnglish: return .af_heart
        case .britishEnglish: return .bf_emma
        case .japanese: return .jf_alpha
        case .chinese: return .zf_xiaoxiao
        case .spanish: return .ef_dora
        case .french: return .ff_siwis
        case .hindi: return .hf_alpha
        case .italian: return .if_sara
        case .portuguese: return .pf_dora
        }
    }
    
    /// Human-readable language name
    public var displayName: String {
        switch self {
        case .americanEnglish: return "American English"
        case .britishEnglish: return "British English"
        case .japanese: return "Japanese"
        case .chinese: return "Mandarin Chinese"
        case .spanish: return "Spanish"
        case .french: return "French"
        case .hindi: return "Hindi"
        case .italian: return "Italian"
        case .portuguese: return "Brazilian Portuguese"
        }
    }
    
    /// Whether this language is fully supported (has misaki G2P)
    ///
    /// Japanese and Chinese require their own misaki packages which aren't
    /// bundled by default. Other languages use misaki[en].
    public var isFullySupported: Bool {
        switch self {
        case .japanese, .chinese:
            return false  // Requires misaki[ja]/misaki[zh]
        default:
            return true   // Uses misaki[en]
        }
    }
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
