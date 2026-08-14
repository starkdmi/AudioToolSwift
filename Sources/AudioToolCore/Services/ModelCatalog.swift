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
/// if let variant = registry.variant(id: "mossformer2_se_fp16") {
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
        let models = Self.createModelCatalog()
        self.models = models
        self.packages = Self.createPackageCatalog(models: models)
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

    /// Every `sizeBytes` below is the sum of the blobs that variant's own `files`
    /// manifest matches on the Hub, measured 2026-08-12 — not the repository total,
    /// and not an estimate.
    ///
    /// They were round numbers before, and round numbers drift out of contact with
    /// the thing they describe: super-resolution was declared at 180 MB against a
    /// real 439 MB, having been copied from the SE entry beside it, so a host
    /// showing that figure before a download understated it by 2.4x. Kokoro was out
    /// by 1.8x and the SS trio by 12% each. A wrong size is
    /// worse than no size: it is shown to a user deciding whether to download on a
    /// metered connection.
    ///
    /// `Scripts/catalog-sizes.py` re-derives the whole table and prints a diff
    /// against what is written here. Re-run it whenever a checkpoint is re-uploaded.
    private static func createModelCatalog() -> [ModelDefinition] {
        [
            // MARK: Speech Enhancement
            ModelDefinition(
                id: "mossformer2_se",
                name: "MossFormer2 Speech Enhancement",
                category: .speechEnhancement,
                description: "High-quality speech enhancement for 48kHz audio with advanced noise reduction",
                variants: [
                    // requiredFiles is narrower than files here, and for every
                    // `standard` variant below except super-resolution: those
                    // providers read the safetensors and nothing else, so a
                    // snapshot without the config is loadable and must not be
                    // reported as missing. Letting it default to `files` made the
                    // catalog disagree with the loader about what "installed"
                    // means. See ModelFiles.standardRequired.
                    ModelVariant(
                        id: "mossformer2_se_fp32",
                        name: "MossFormer2 SE (FP32)",
                        quantization: .fp32,
                        sizeBytes: 221_178_088,
                        repo: ModelRepository.mossFormer2SE48K,
                        files: ModelFiles.standard(.fp32),
                        requiredFiles: ModelFiles.standardRequired(.fp32)
                    ),
                    ModelVariant(
                        id: "mossformer2_se_fp16",
                        name: "MossFormer2 SE (FP16)",
                        quantization: .fp16,
                        sizeBytes: 110_652_628,
                        repo: ModelRepository.mossFormer2SE48K,
                        files: ModelFiles.standard(.fp16),
                        requiredFiles: ModelFiles.standardRequired(.fp16)
                    ),
                ]
            ),
            
            ModelDefinition(
                id: "frcrn_se",
                name: "FRCRN Speech Enhancement",
                category: .speechEnhancement,
                description: "Fast and efficient speech enhancement model",
                variants: [
                    // No FRCRN `config.json` exists and the provider never opens one -
                    // it calls `loadWeights` and stops. That is not a reason to drop
                    // the request: neither `MossFormer2_SE_48K_MLX` nor any of the
                    // three SS repos publishes one either, so every `standard` variant
                    // here asks for a file that is not there. An unmatched glob is not
                    // an error, `requiredFiles` is what decides whether a snapshot is
                    // loadable, and keeping the request means a repo that later gains
                    // a config gets it. So nothing needs inventing at upload time.
                    // Size is the real 56 MB checkpoint, not the old 150 MB guess.
                    ModelVariant(
                        id: "frcrn_se_fp32",
                        name: "FRCRN SE (FP32)",
                        quantization: .fp32,
                        sizeBytes: 56_002_160,
                        repo: ModelRepository.frcrnSE16K,
                        files: ModelFiles.standard(.fp32),
                        requiredFiles: ModelFiles.standardRequired(.fp32)
                    ),
                ]
            ),
            
            ModelDefinition(
                id: "mossformer_gan_se_coreml",
                name: "MossFormerGAN Speech Enhancement (CoreML)",
                category: .speechEnhancement,
                description: "Speech enhancement for 16kHz audio, running on CoreML",
                // The only Core ML model this package downloads for itself; the
                // Silero VAD entry below is listed for pre-download but fetched at
                // runtime by FluidAudio. Core ML fixes precision at conversion time,
                // so these are two compiled packages rather than one variant with a
                // quantization switch - which is also why `files` names a package
                // instead of globbing the repository root.
                //
                // Sizes are the compiled packages as uploaded, not estimates.
                variants: [
                    ModelVariant(
                        id: "mossformer_gan_se_coreml_fp16",
                        name: "MossFormerGAN SE (CoreML FP16)",
                        quantization: .fp16,
                        sizeBytes: 7_938_742,
                        repo: ModelRepository.mossFormerGANSE16KCoreML,
                        files: ModelFiles.mossFormerGANCoreML(.fp16),
                        requiredFiles: ModelFiles.mossFormerGANCoreMLRequired(.fp16)
                    ),
                    ModelVariant(
                        id: "mossformer_gan_se_coreml_fp32",
                        name: "MossFormerGAN SE (CoreML FP32)",
                        quantization: .fp32,
                        sizeBytes: 14_126_289,
                        repo: ModelRepository.mossFormerGANSE16KCoreML,
                        files: ModelFiles.mossFormerGANCoreML(.fp32),
                        requiredFiles: ModelFiles.mossFormerGANCoreMLRequired(.fp32)
                    ),
                ]
            ),

            // MARK: Speech Separation
            ModelDefinition(
                id: "mossformer2_ss",
                name: "MossFormer2 Speech Separation",
                category: .speechSeparation,
                description: "Separate overlapping speakers into individual audio tracks",
                // Three separately trained models, not three quantizations of one:
                // different speaker counts, different sample rates, different repos.
                // A single "MossFormer2_SS_MLX" entry described none of them.
                variants: [
                    ModelVariant(
                        id: "mossformer2_ss_2spk_16k_fp32",
                        name: "MossFormer2 SS 2-speaker 16 kHz (FP32)",
                        quantization: .fp32,
                        sizeBytes: 223_095_144,
                        repo: ModelRepository.mossFormer2SS2Spk16K,
                        files: ModelFiles.standard(.fp32),
                        requiredFiles: ModelFiles.standardRequired(.fp32)
                    ),
                    ModelVariant(
                        id: "mossformer2_ss_3spk_8k_fp32",
                        name: "MossFormer2 SS 3-speaker 8 kHz (FP32)",
                        quantization: .fp32,
                        sizeBytes: 224_145_816,
                        repo: ModelRepository.mossFormer2SS3Spk8K,
                        files: ModelFiles.standard(.fp32),
                        requiredFiles: ModelFiles.standardRequired(.fp32)
                    ),
                    ModelVariant(
                        id: "mossformer2_ss_2spk_whamr_8k_fp32",
                        name: "MossFormer2 SS 2-speaker WHAMR 8 kHz (FP32)",
                        quantization: .fp32,
                        sizeBytes: 223_095_144,
                        repo: ModelRepository.mossFormer2SS2SpkWHAMR8K,
                        files: ModelFiles.standard(.fp32),
                        requiredFiles: ModelFiles.standardRequired(.fp32)
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
                        sizeBytes: 438_668_788,
                        repo: ModelRepository.mossFormer2SR48K,
                        files: ModelFiles.standard(.fp32)
                    ),
                    // int8 was published and pinned on 2026-08-12 and is the second
                    // half of `MLXSuperResolutionProvider.supportedPrecisions`, so a
                    // caller can already select it - it just had no catalog entry to
                    // be downloaded or sized through. fp16 is absent because its
                    // forward pass returns NaN, int6 and int4 because they are
                    // strictly dominated by int8 and were never uploaded; the
                    // provider documents all three exclusions.
                    //
                    // No `requiredFiles` narrowing here, unlike every other
                    // `standard` variant: SR is the one provider that opens the
                    // config.json, so both files are genuinely required.
                    ModelVariant(
                        id: "mossformer2_sr_int8",
                        name: "MossFormer2 SR (INT8)",
                        quantization: .int8,
                        sizeBytes: 307_581_821,
                        repo: ModelRepository.mossFormer2SR48K,
                        files: ModelFiles.standard(.int8)
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
                        sizeBytes: 106_522_912,
                        repo: ModelRepository.uss,
                        files: ModelFiles.uss(.fp32)
                    ),
                    ModelVariant(
                        id: "uss_fp16",
                        name: "USS (FP16)",
                        quantization: .fp16,
                        sizeBytes: 53_279_412,
                        repo: ModelRepository.uss,
                        files: ModelFiles.uss(.fp16)
                    ),
                ]
            ),
            
            ModelDefinition(
                id: "demucs",
                name: "Demucs Music Separation",
                category: .uss,
                description: "Separate vocals, drums, bass, and other from music",
                variants: [
                    // Four weight files, one per stem - HTDemucs ships a separate
                    // checkpoint for drums, bass, other and vocals, and each emits
                    // all four sources. A single "model.safetensors" was never what
                    // this repo contains.
                    ModelVariant(
                        id: "demucs_fp32",
                        name: "Demucs (FP32)",
                        quantization: .fp32,
                        sizeBytes: 4 * 84_030_924,
                        repo: ModelRepository.demucs,
                        files: ModelFiles.demucsAll
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
                        id: "kokoro_tts_bf16",
                        name: "Kokoro TTS (BF16)",
                        quantization: .bf16,
                        sizeBytes: 327_117_503,
                        repo: ModelRepository.kokoroBF16,
                        files: ModelFiles.kokoro
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
                        sizeBytes: 2_711_415_628,
                        repo: ModelRepository.chatterboxFP32,
                        files: ModelFiles.chatterboxDownload(for: .fp32),
                        requiredFiles: ModelFiles.chatterboxRequired(for: .fp32)
                    ),
                ]
            ),
            
            // MARK: Transcription
            //
            // Nothing here. Transcription runs through `FluidAudioTranscriber`
            // (Parakeet) and `AppleSpeechTranscriber`, neither of which downloads
            // through this catalog: FluidAudio fetches its own weights in
            // `AsrModels.downloadAndLoad`, and Apple's is part of the OS.
            //
            // Whisper Large v3 and Small used to be listed here, and were the only
            // transcription entries. No `Transcriber` ever loaded them, so the two
            // packages that bundled them promised a model that could not run -
            // 3.08 GB of the podcaster pack's 3.20 GB was weights nothing could
            // open. A catalog entry is a download offer, so an entry without a
            // provider is worse than no entry. The `Transcriber` protocol is the
            // extension point; add the provider first, then the catalog rows.

            // MARK: VAD
            ModelDefinition(
                id: "silero_vad",
                name: "Silero VAD",
                category: .vad,
                description: "Fast voice activity detection",
                variants: [
                    // The repository holds several conversions side by side, so the
                    // pattern names the one FluidAudio actually loads
                    // (`ModelNames.VAD.sileroVad`) rather than every `.mlpackage` in
                    // the repo - which would have pulled four unused models, and in
                    // any case matched none of them: a compiled model is a directory,
                    // and `*` does not cross a path separator.
                    ModelVariant(
                        id: "silero_vad_coreml",
                        name: "Silero VAD (CoreML)",
                        quantization: .fp16,
                        sizeBytes: 1_063_425,
                        repo: ModelRepository.sileroVADCoreML,
                        files: ["silero-vad-unified-256ms-v6.2.1.mlmodelc/**"],
                        // `requiredFiles` defaults to `files`, and a single recursive
                        // wildcard is satisfied by *any one* entry inside the compiled
                        // model - so an interrupted download that left one file behind
                        // reported the variant as installed. A compiled Core ML model
                        // is a directory of parts that are individually useless; these
                        // are the ones it cannot load without.
                        requiredFiles: [
                            "silero-vad-unified-256ms-v6.2.1.mlmodelc/coremldata.bin",
                            "silero-vad-unified-256ms-v6.2.1.mlmodelc/model.mil",
                            "silero-vad-unified-256ms-v6.2.1.mlmodelc/weights/weight.bin",
                        ]
                    ),
                ]
            ),
        ]
    }
    
    // MARK: - Package Catalog
    
    /// A package's `totalSizeBytes` is summed from its variants, never written by
    /// hand.
    ///
    /// The figures here used to be hand-rounded and every one understated:
    /// speech_studio_pro advertised 721 MB against 1.14 GB of actual weights, and
    /// the since-removed podcaster pack 1.59 GB against 3.20 GB. They were guesses from before
    /// the per-variant sizes were measured, and nothing updated them afterwards -
    /// the same failure `createModelCatalog`'s note describes, one level up.
    ///
    /// Summing here makes that class of drift impossible: add a variant to a
    /// package, or re-measure a checkpoint, and the total follows. The stored
    /// property stays on ``ModelPackage`` because it is public, `Codable` API.
    private static func createPackageCatalog(models: [ModelDefinition]) -> [ModelPackage] {
        let sizes = Dictionary(
            models.flatMap(\.variants).map { ($0.id, $0.sizeBytes) },
            uniquingKeysWith: { first, _ in first }
        )

        func package(
            id: String,
            name: String,
            description: String,
            variantIds: [String]
        ) -> ModelPackage {
            // A missing id contributes nothing rather than trapping: the catalog is
            // read at launch, and `ModelDownloadTests` already fails the build on a
            // package that names a variant which does not exist.
            ModelPackage(
                id: id,
                name: name,
                description: description,
                variantIds: variantIds,
                totalSizeBytes: variantIds.reduce(0) { $0 + (sizes[$1] ?? 0) }
            )
        }

        return [
            package(
                id: "speech_studio_essentials",
                name: "Speech Studio Essentials",
                description: "Core tools for speech processing: enhancement, VAD",
                variantIds: ["mossformer2_se_fp16", "silero_vad_coreml"]
            ),

            package(
                id: "speech_studio_pro",
                name: "Speech Studio Pro",
                description: "Complete toolkit: Enhancement, Separation, TTS",
                variantIds: [
                    "mossformer2_se_fp16",
                    "mossformer2_ss_2spk_16k_fp32",
                    "kokoro_tts_bf16",
                    "silero_vad_coreml",
                ]
            ),

            package(
                id: "music_production",
                name: "Music Production",
                description: "Tools for music: Source separation, Super resolution",
                variantIds: ["demucs_fp32", "uss_fp32", "mossformer2_sr_fp32"]
            ),

            // The podcaster pack was enhancement + Whisper + VAD. Without the
            // Whisper variant it was `speech_studio_essentials` under a second
            // name, so it is gone rather than duplicated: a podcast host wants
            // essentials, and transcription arrives with a provider, not a
            // download.
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
