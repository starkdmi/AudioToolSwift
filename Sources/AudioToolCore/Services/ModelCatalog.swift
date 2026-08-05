//
//  ModelCatalog.swift
//  AudioToolCore
//
//  Static catalog of available models, variants, and packages
//

import Foundation

/// Static catalog of available models and packages
///
/// Provides lookup for model definitions, variants, and pre-defined packages.
/// Customize this to add your own models.
///
/// Usage:
/// ```swift
/// let registry = ModelCatalog.shared
///
/// // Get all models
/// for model in registry.models {
///     print(model.name)
/// }
///
/// // Look up a variant
/// if let variant = registry.variant(id: "mossformer2_se_int8") {
///     print(variant.sizeString)
/// }
/// ```
public final class ModelCatalog: @unchecked Sendable {
    /// Shared instance
    public static let shared = ModelCatalog()
    
    /// All available model definitions
    public let models: [ModelDefinition]
    
    /// All available packages
    public let packages: [ModelPackage]
    
    private init() {
        // Initialize with your model catalog
        self.models = Self.createModelCatalog()
        self.packages = Self.createPackageCatalog()
    }
    
    // MARK: - Lookup
    
    /// All variants across all models
    public var allVariants: [ModelVariant] {
        models.flatMap { $0.variants }
    }
    
    /// Look up a model by ID
    /// - Parameter id: Model ID
    /// - Returns: ModelDefinition or nil
    public func model(id: String) -> ModelDefinition? {
        models.first { $0.id == id }
    }
    
    /// Look up a variant by ID
    /// - Parameter id: Variant ID
    /// - Returns: ModelVariant or nil
    public func variant(id: String) -> ModelVariant? {
        allVariants.first { $0.id == id }
    }
    
    /// Look up a package by ID
    /// - Parameter id: Package ID
    /// - Returns: ModelPackage or nil
    public func package(id: String) -> ModelPackage? {
        packages.first { $0.id == id }
    }
    
    /// Get models by category
    /// - Parameter category: Category to filter by
    /// - Returns: Array of matching models
    public func models(in category: ModelCategory) -> [ModelDefinition] {
        models.filter { $0.category == category }
    }
    
    /// Get variants for a specific model
    /// - Parameter modelId: Model ID
    /// - Returns: Array of variants for that model
    public func variants(for modelId: String) -> [ModelVariant] {
        model(id: modelId)?.variants ?? []
    }
    
    // MARK: - Model Catalog
    
    private static func createModelCatalog() -> [ModelDefinition] {
        [
            // MARK: Speech Enhancement
            ModelDefinition(
                id: "mossformer2_se",
                name: "MossFormer2 Speech Enhancement",
                category: .speechEnhancement,
                description: "High-quality speech enhancement for 48kHz audio with advanced noise reduction",
                variants: [
                    ModelVariant(
                        id: "mossformer2_se_fp32",
                        name: "MossFormer2 SE (FP32)",
                        quantization: .fp32,
                        sizeBytes: 180_000_000,
                        repo: "starkdmi/MossFormer2_SE_48K_MLX",
                        files: ["model_fp32.safetensors", "config.json"]
                    ),
                    ModelVariant(
                        id: "mossformer2_se_fp16",
                        name: "MossFormer2 SE (FP16)",
                        quantization: .fp16,
                        sizeBytes: 90_000_000,
                        repo: "starkdmi/MossFormer2_SE_48K_MLX",
                        files: ["model_fp16.safetensors", "config.json"]
                    ),
                    ModelVariant(
                        id: "mossformer2_se_int8",
                        name: "MossFormer2 SE (Int8)",
                        quantization: .int8,
                        sizeBytes: 45_000_000,
                        repo: "starkdmi/MossFormer2_SE_48K_MLX",
                        files: ["model_int8.safetensors", "config.json"]
                    ),
                    ModelVariant(
                        id: "mossformer2_se_int4",
                        name: "MossFormer2 SE (Int4)",
                        quantization: .int4,
                        sizeBytes: 23_000_000,
                        repo: "starkdmi/MossFormer2_SE_48K_MLX",
                        files: ["model_int4.safetensors", "config.json"]
                    ),
                ]
            ),
            
            ModelDefinition(
                id: "frcrn_se",
                name: "FRCRN Speech Enhancement",
                category: .speechEnhancement,
                description: "Fast and efficient speech enhancement model",
                variants: [
                    ModelVariant(
                        id: "frcrn_se_fp32",
                        name: "FRCRN SE (FP32)",
                        quantization: .fp32,
                        sizeBytes: 150_000_000,
                        repo: "starkdmi/FRCRN_SE_MLX",
                        files: ["model.safetensors", "config.json"]
                    ),
                ]
            ),
            
            // MARK: Speech Separation
            ModelDefinition(
                id: "mossformer2_ss",
                name: "MossFormer2 Speech Separation",
                category: .speechSeparation,
                description: "Separate overlapping speakers into individual audio tracks",
                variants: [
                    ModelVariant(
                        id: "mossformer2_ss_fp32",
                        name: "MossFormer2 SS (FP32)",
                        quantization: .fp32,
                        sizeBytes: 200_000_000,
                        repo: "starkdmi/MossFormer2_SS_MLX",
                        files: ["model.safetensors", "config.json"]
                    ),
                ]
            ),
            
            // MARK: Super Resolution
            ModelDefinition(
                id: "mossformer2_sr",
                name: "MossFormer2 Super Resolution",
                category: .superResolution,
                description: "Upscale audio from 8kHz/16kHz to 48kHz",
                variants: [
                    ModelVariant(
                        id: "mossformer2_sr_fp32",
                        name: "MossFormer2 SR (FP32)",
                        quantization: .fp32,
                        sizeBytes: 180_000_000,
                        repo: "starkdmi/MossFormer2_SR_MLX",
                        files: ["model.safetensors", "config.json"]
                    ),
                ]
            ),
            
            // MARK: Universal Source Separation
            ModelDefinition(
                id: "uss",
                name: "Universal Source Separation",
                category: .uss,
                description: "Separate music, speech, and sound effects",
                variants: [
                    ModelVariant(
                        id: "uss_fp32",
                        name: "USS (FP32)",
                        quantization: .fp32,
                        sizeBytes: 120_000_000,
                        repo: "starkdmi/USS_MLX",
                        files: ["model.safetensors", "config.json"]
                    ),
                ]
            ),
            
            ModelDefinition(
                id: "demucs",
                name: "Demucs Music Separation",
                category: .uss,
                description: "Separate vocals, drums, bass, and other from music",
                variants: [
                    ModelVariant(
                        id: "demucs_fp32",
                        name: "Demucs (FP32)",
                        quantization: .fp32,
                        sizeBytes: 320_000_000,
                        repo: "starkdmi/Demucs_MLX",
                        files: ["model.safetensors", "config.json"]
                    ),
                ]
            ),
            
            // MARK: Text to Speech
            ModelDefinition(
                id: "kokoro_tts",
                name: "Kokoro TTS",
                category: .textToSpeech,
                description: "High-quality text-to-speech with multiple voices",
                variants: [
                    ModelVariant(
                        id: "kokoro_tts_fp16",
                        name: "Kokoro TTS (FP16)",
                        quantization: .fp16,
                        sizeBytes: 80_000_000,
                        repo: "hexgrad/Kokoro-82M",
                        files: ["kokoro-v1.0-fp16.onnx", "voices-v1.0.bin"]
                    ),
                ]
            ),
            
            ModelDefinition(
                id: "chatterbox_tts",
                name: "ChatterBox TTS",
                category: .textToSpeech,
                description: "Multilingual TTS supporting 25+ languages with voice cloning",
                variants: [
                    ModelVariant(
                        id: "chatterbox_tts_fp32",
                        name: "ChatterBox TTS (FP32)",
                        quantization: .fp32,
                        sizeBytes: 450_000_000,
                        repo: "ResembleAI/chatterbox",
                        files: ["*.safetensors", "config.json", "tokenizer.json"]
                    ),
                ]
            ),
            
            // MARK: Transcription
            ModelDefinition(
                id: "whisper",
                name: "Whisper Transcription",
                category: .transcription,
                description: "Accurate speech-to-text transcription",
                variants: [
                    ModelVariant(
                        id: "whisper_large_v3",
                        name: "Whisper Large v3",
                        quantization: .fp16,
                        sizeBytes: 1_500_000_000,
                        repo: "mlx-community/whisper-large-v3-mlx",
                        files: ["*.safetensors", "config.json"]
                    ),
                    ModelVariant(
                        id: "whisper_small",
                        name: "Whisper Small",
                        quantization: .fp16,
                        sizeBytes: 250_000_000,
                        repo: "mlx-community/whisper-small-mlx",
                        files: ["*.safetensors", "config.json"]
                    ),
                ]
            ),
            
            // MARK: VAD
            ModelDefinition(
                id: "silero_vad",
                name: "Silero VAD",
                category: .vad,
                description: "Fast voice activity detection",
                variants: [
                    ModelVariant(
                        id: "silero_vad_coreml",
                        name: "Silero VAD (CoreML)",
                        quantization: .fp16,
                        sizeBytes: 2_000_000,
                        repo: "FluidInference/SileroVAD",
                        files: ["*.mlpackage"]
                    ),
                ]
            ),
        ]
    }
    
    // MARK: - Package Catalog
    
    private static func createPackageCatalog() -> [ModelPackage] {
        [
            ModelPackage(
                id: "speech_studio_essentials",
                name: "Speech Studio Essentials",
                description: "Core tools for speech processing: enhancement, VAD",
                variantIds: ["mossformer2_se_int8", "silero_vad_coreml"],
                totalSizeBytes: 47_000_000
            ),
            
            ModelPackage(
                id: "speech_studio_pro",
                name: "Speech Studio Pro",
                description: "Complete toolkit: Enhancement, Separation, TTS, Transcription",
                variantIds: [
                    "mossformer2_se_int8",
                    "mossformer2_ss_fp32",
                    "kokoro_tts_fp16",
                    "whisper_small",
                    "silero_vad_coreml"
                ],
                totalSizeBytes: 577_000_000
            ),
            
            ModelPackage(
                id: "music_production",
                name: "Music Production",
                description: "Tools for music: Source separation, Super resolution",
                variantIds: ["demucs_fp32", "uss_fp32", "mossformer2_sr_fp32"],
                totalSizeBytes: 620_000_000
            ),
            
            ModelPackage(
                id: "podcaster",
                name: "Podcaster Pack",
                description: "Perfect for podcast production: Enhancement, Transcription",
                variantIds: [
                    "mossformer2_se_fp16",
                    "whisper_large_v3",
                    "silero_vad_coreml"
                ],
                totalSizeBytes: 1_592_000_000
            ),
        ]
    }
}

// MARK: - Summary Statistics

extension ModelCatalog {
    /// Total size of all models in registry
    public var totalCatalogSize: Int64 {
        allVariants.reduce(0) { $0 + $1.sizeBytes }
    }
    
    /// Total number of models
    public var modelCount: Int {
        models.count
    }
    
    /// Total number of variants
    public var variantCount: Int {
        allVariants.count
    }
    
    /// Total number of packages
    public var packageCount: Int {
        packages.count
    }
}
