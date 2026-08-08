//
//  ModelVariant.swift
//  AudioToolCore
//
//  Model variant and definition types for download management
//

import Foundation

// MARK: - Quantization

/// Model quantization precision level
public enum Quantization: String, Codable, CaseIterable, Sendable, Hashable {
    case fp32
    case fp16
    case bf16
    case int8
    case int4
    
    /// Human-readable display name
    public var displayName: String {
        switch self {
        case .fp32: return "Full Precision (FP32)"
        case .fp16: return "Half Precision (FP16)"
        case .bf16: return "Brain Float 16 (BF16)"
        case .int8: return "8-bit Quantized"
        case .int4: return "4-bit Quantized"
        }
    }
    
    /// Short label for compact display
    public var shortName: String {
        rawValue.uppercased()
    }
    
    /// Approximate memory multiplier relative to fp32
    public var memoryMultiplier: Double {
        switch self {
        case .fp32: return 1.0
        case .fp16, .bf16: return 0.5
        case .int8: return 0.25
        case .int4: return 0.125
        }
    }
}

// MARK: - Model Category

/// Category of ML model functionality
public enum ModelCategory: String, Codable, CaseIterable, Sendable, Hashable {
    case speechEnhancement = "enhancement"
    case speechSeparation = "separation"
    case superResolution = "super_resolution"
    case textToSpeech = "tts"
    case transcription = "transcription"
    case vad = "vad"
    case diarization = "diarization"
    case translation = "translation"
    case uss = "uss"
    
    /// Human-readable display name
    public var displayName: String {
        switch self {
        case .speechEnhancement: return "Speech Enhancement"
        case .speechSeparation: return "Speech Separation"
        case .superResolution: return "Super Resolution"
        case .textToSpeech: return "Text to Speech"
        case .transcription: return "Transcription"
        case .vad: return "Voice Activity Detection"
        case .diarization: return "Speaker Diarization"
        case .translation: return "Translation"
        case .uss: return "Universal Source Separation"
        }
    }
    
    /// SF Symbol icon name
    public var iconName: String {
        switch self {
        case .speechEnhancement: return "waveform.badge.plus"
        case .speechSeparation: return "person.2.wave.2"
        case .superResolution: return "arrow.up.right.circle"
        case .textToSpeech: return "text.bubble"
        case .transcription: return "text.badge.checkmark"
        case .vad: return "waveform.badge.mic"
        case .diarization: return "person.3"
        case .translation: return "globe"
        case .uss: return "music.note.list"
        }
    }
}

// MARK: - Model Variant

/// A specific model variant (quantization level, file set)
///
/// Example:
/// ```swift
/// let variant = ModelVariant(
///     id: "mossformer2_se_48k_int8",
///     name: "MossFormer2 SE 48K (Int8)",
///     quantization: .int8,
///     sizeBytes: 45_000_000,
///     repo: "starkdmi/MossFormer2_SE_48K_MLX",
///     files: ["model_int8.safetensors", "config.json"],
///     requiredFiles: ["model_int8.safetensors"]
/// )
/// ```
public struct ModelVariant: Identifiable, Codable, Hashable, Sendable {
    /// Unique identifier (e.g., "mossformer2_se_48k_int8")
    public let id: String
    
    /// Human-readable name (e.g., "MossFormer2 SE 48K (Int8)")
    public let name: String
    
    /// Quantization precision
    public let quantization: Quantization
    
    /// Approximate size in bytes for UI display and storage checks
    public let sizeBytes: Int64
    
    /// HuggingFace repository ID
    public let repo: String
    
    /// Files owned by this variant and requested from the repository.
    ///
    /// These may include optional runtime assets and alternative layouts. A Hub
    /// download treats the patterns as a selection list, so a pattern does not
    /// need to match for the download itself to succeed.
    public let files: [String]

    /// Files that must all exist before this variant is considered installed.
    ///
    /// This defaults to ``files``. Keeping it separate lets a variant fetch
    /// optional or alternative runtime assets without treating every download
    /// pattern as mandatory proof that the resulting snapshot is loadable.
    public let requiredFiles: [String]
    
    public init(
        id: String,
        name: String,
        quantization: Quantization,
        sizeBytes: Int64,
        repo: String,
        files: [String],
        requiredFiles: [String]? = nil
    ) {
        self.id = id
        self.name = name
        self.quantization = quantization
        self.sizeBytes = sizeBytes
        self.repo = repo
        self.files = files
        self.requiredFiles = requiredFiles ?? files
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case quantization
        case sizeBytes
        case repo
        case files
        case requiredFiles
    }

    /// Decode older persisted variants by treating their download manifest as
    /// the verification manifest, matching the behavior before the two concepts
    /// were represented separately.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        quantization = try container.decode(Quantization.self, forKey: .quantization)
        sizeBytes = try container.decode(Int64.self, forKey: .sizeBytes)
        repo = try container.decode(String.self, forKey: .repo)
        files = try container.decode([String].self, forKey: .files)
        requiredFiles = try container.decodeIfPresent(
            [String].self,
            forKey: .requiredFiles
        ) ?? files
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(name, forKey: .name)
        try container.encode(quantization, forKey: .quantization)
        try container.encode(sizeBytes, forKey: .sizeBytes)
        try container.encode(repo, forKey: .repo)
        try container.encode(files, forKey: .files)
        try container.encode(requiredFiles, forKey: .requiredFiles)
    }
    
    /// Human-readable size string
    public var sizeString: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }
}

// MARK: - Model Definition

/// A model with multiple variant options
///
/// Example:
/// ```swift
/// let model = ModelDefinition(
///     id: "mossformer2_se",
///     name: "MossFormer2 Speech Enhancement",
///     category: .speechEnhancement,
///     description: "High-quality speech enhancement for 48kHz audio",
///     variants: [fp32Variant, int8Variant]
/// )
/// ```
public struct ModelDefinition: Identifiable, Codable, Hashable, Sendable {
    /// Unique identifier (e.g., "mossformer2_se")
    public let id: String
    
    /// Human-readable name
    public let name: String
    
    /// Model category
    public let category: ModelCategory
    
    /// Description of the model's functionality
    public let description: String
    
    /// Available variants (different quantizations)
    public let variants: [ModelVariant]
    
    /// ID of another model required together (nil if standalone)
    public let requiredPeer: String?
    
    public init(
        id: String,
        name: String,
        category: ModelCategory,
        description: String,
        variants: [ModelVariant],
        requiredPeer: String? = nil
    ) {
        self.id = id
        self.name = name
        self.category = category
        self.description = description
        self.variants = variants
        self.requiredPeer = requiredPeer
    }
    
    /// Default variant (first in list)
    public var defaultVariant: ModelVariant? {
        variants.first
    }
    
    /// Smallest variant by size
    public var smallestVariant: ModelVariant? {
        variants.min(by: { $0.sizeBytes < $1.sizeBytes })
    }
}

// MARK: - Model Package

/// A bundle of multiple model variants for batch download
///
/// Example:
/// ```swift
/// let package = ModelPackage(
///     id: "speech_studio_pro",
///     name: "Speech Studio Pro",
///     description: "Complete toolkit: Enhancement, Separation, TTS",
///     variantIds: ["mossformer2_se_fp16", "mossformer2_ss_2spk_16k_fp32", "kokoro_tts_bf16"],
///     totalSizeBytes: 250_000_000
/// )
/// ```
public struct ModelPackage: Identifiable, Codable, Hashable, Sendable {
    /// Unique identifier
    public let id: String
    
    /// Human-readable name
    public let name: String
    
    /// Description of what's included
    public let description: String
    
    /// Variant IDs included in package
    public let variantIds: [String]
    
    /// Pre-calculated total size
    public let totalSizeBytes: Int64
    
    public init(
        id: String,
        name: String,
        description: String,
        variantIds: [String],
        totalSizeBytes: Int64
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.variantIds = variantIds
        self.totalSizeBytes = totalSizeBytes
    }
    
    /// Human-readable size string
    public var sizeString: String {
        ByteCountFormatter.string(fromByteCount: totalSizeBytes, countStyle: .file)
    }
    
    /// Number of models in package
    public var modelCount: Int {
        variantIds.count
    }
}
