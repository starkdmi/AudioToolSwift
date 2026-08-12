//
//  ModelRepositories.swift
//  AudioToolCore
//
//  Canonical HuggingFace coordinates, shared by the catalog and the providers.
//

import Foundation

// MARK: - Repository Coordinates

/// Where each model's weights live, named once.
///
/// The catalog and the providers both need these strings, and until now each kept
/// its own copy. They disagreed, which is worse than either being wrong alone: the
/// catalog is what a host app shows a user and what a pre-download step fetches,
/// while the provider constant is what actually runs at load time. A user could
/// download from one repo and have inference fail looking in another.
///
/// What had drifted:
///
/// | Model  | Catalog said              | Provider used                        |
/// | ------ | ------------------------- | ------------------------------------ |
/// | SR     | `MossFormer2_SR_MLX`      | `MossFormer2_SR_48K_MLX`             |
/// | SS     | `MossFormer2_SS_MLX`      | three repos, one per speaker config  |
/// | USS    | `model.safetensors`       | `resunet30_fp32.safetensors`         |
/// | Demucs | one `model.safetensors`   | four files, one per stem             |
/// | FRCRN  | `FRCRN_SE_MLX`            | nothing - no download path existed   |
///
/// Living in `AudioToolCore` puts them below both consumers, so neither can drift
/// from the other without the compiler noticing.
public enum ModelRepository {

    /// MossFormer2 speech enhancement, 48 kHz.
    public static let mossFormer2SE48K = "starkdmi/MossFormer2_SE_48K_MLX"

    /// MossFormer2 super-resolution, 16 kHz in, 48 kHz out.
    public static let mossFormer2SR48K = "starkdmi/MossFormer2_SR_48K_MLX"

    /// MossFormer2 speaker separation, two speakers at 16 kHz.
    public static let mossFormer2SS2Spk16K = "starkdmi/MossFormer2_SS_2SPK_16K_MLX"

    /// MossFormer2 speaker separation, three speakers at 8 kHz.
    public static let mossFormer2SS3Spk8K = "starkdmi/MossFormer2_SS_3SPK_8K_MLX"

    /// MossFormer2 speaker separation, two speakers at 8 kHz, WHAMR-trained.
    public static let mossFormer2SS2SpkWHAMR8K = "starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX"

    /// Universal source separation.
    public static let uss = "starkdmi/USS_MLX"

    /// The S3 speech tokenizer Chatterbox needs when its own repo does not ship one.
    ///
    /// Named here rather than spelled out at the use site, so it resolves through
    /// `ModelDownloader` and its pin like every other repository.
    public static let s3Tokenizer = "mlx-community/S3TokenizerV2"

    /// FRCRN speech enhancement, 16 kHz.
    ///
    /// The rate is in the name to match its neighbours - the catalog previously said
    /// `FRCRN_SE_MLX`, which matched nothing because no download path existed to
    /// disagree with it. Rename here if the published repo ends up called something
    /// else; both the catalog and the provider follow this constant.
    public static let frcrnSE16K = "starkdmi/FRCRN_SE_16K_MLX"

    /// Demucs music separation.
    public static let demucs = "starkdmi/Demucs_MLX"

    /// MossFormerGAN speech enhancement, 16 kHz, compiled for Core ML.
    ///
    /// The only Core ML model this package downloads itself. It was the last
    /// conversion without a repository - the provider took a filesystem path and
    /// nothing else, so the benchmark needed `--coreml-gan` and skipped the case
    /// without it. Published 2026-08-12.
    public static let mossFormerGANSE16KCoreML = "starkdmi/MossFormer_GAN_SE_16K_CoreML"

    /// Base for Kokoro's precision-specific MLX conversions.
    public static let kokoroBase = "mlx-community/Kokoro-82M"

    /// Kokoro's default bfloat16 MLX conversion.
    public static let kokoroBF16 = "\(kokoroBase)-bf16"

    /// Chatterbox's default full-precision MLX checkpoint.
    public static let chatterboxFP32 = "starkdmi/chatterbox"

    /// Silero VAD, as converted for Core ML by FluidAudio.
    ///
    /// The catalog previously named `FluidInference/SileroVAD`, which does not exist
    /// - so a pre-download of the VAD entry could only ever 404. This is the
    /// repository FluidAudio itself fetches from (`ModelNames.Repo.vad`).
    ///
    /// Listed here for the catalog's pre-download and size display only. The runtime
    /// fetch goes through FluidAudio's own downloader, which this package does not
    /// call and cannot pin; that one is fixed by the FluidAudio dependency version.
    public static let sileroVADCoreML = "FluidInference/silero-vad-coreml"
}

// MARK: - Core ML Precisions

/// Which of the two compiled MossFormerGAN packages to load.
///
/// A dedicated two-case type rather than ``ModelPrecision``, because Core ML fixes
/// precision when the package is compiled: these are two separate files, not one
/// model with a switch, and there is no int8 package for a wider type to name.
///
/// FP16 is the recommendation - measured faster (4.2x against 3.7x) and 5.5x lower
/// peak memory, at 61 dB against the fp32 conversion. See `MODEL-PRECISIONS.md`.
public enum CoreMLGANPrecision: String, Sendable, CaseIterable, Hashable {
    case fp32
    case fp16

    /// The package as it was compiled and uploaded.
    public var packageName: String {
        switch self {
        case .fp32: "MossFormerGAN_256frames.mlpackage"
        case .fp16: "MossFormerGAN_256frames_FP16.mlpackage"
        }
    }

    /// How the catalog labels this conversion.
    ///
    /// ``Quantization`` rather than ``ModelPrecision``: the catalog's type is the
    /// display one, and these two never name a file - the package does that.
    public var quantization: Quantization {
        switch self {
        case .fp32: .fp32
        case .fp16: .fp16
        }
    }
}

// MARK: - File Layouts

/// Which files each repository holds.
public enum ModelFiles {

    /// Repos following the single-repo, precision-suffixed convention:
    /// `model_fp32.safetensors`, `model_fp16.safetensors`, and a shared config.
    ///
    /// This is the **download** manifest. For deciding whether a snapshot on disk
    /// is loadable, use ``standardRequired(_:)`` - see the note there.
    ///
    /// - Parameter precision: which weights to fetch.
    public static func standard(_ precision: ModelPrecision) -> [String] {
        [precision.weightsFilename, "config.json"]
    }

    /// What a `standard` repo's model must have on disk before it can load.
    ///
    /// The weights, and nothing else. Of the providers using this layout, only
    /// `MossFormer2SR48KProvider` opens the `config.json`: `MossFormer2SE48KProvider`
    /// builds its `PipelineConfiguration` in code, `MossFormer2SSProvider` hardcodes
    /// its `MossFormer2Config`, and `FRCRNSE16KProvider` calls `loadWeights` and
    /// nothing more. SR keeps ``standard(_:)`` for both purposes.
    ///
    /// Load paths used ``standard(_:)`` for the presence check as well as the
    /// download, and `localPath(for:matching:)` requires *every* glob to match. So
    /// a machine holding the safetensors but not the config was treated as having
    /// nothing: every load took a repository round-trip to fetch a file it would
    /// never open, and would have failed outright with no network. Measured on a
    /// cache holding only `model_fp32.safetensors` - SE 48K reported uncached,
    /// then loaded in 0.6 s and transferred nothing.
    ///
    /// This is the same split ``ModelVariant/requiredFiles`` already draws against
    /// ``ModelVariant/files``, and whose documentation uses this exact repository
    /// as its example. The concept was there; the load paths were not using it.
    public static func standardRequired(_ precision: ModelPrecision) -> [String] {
        [precision.weightsFilename]
    }

    /// USS names its checkpoints after the architecture rather than "model".
    public static func uss(_ precision: ModelPrecision) -> [String] {
        ["resunet30_\(precision.rawValue).safetensors"]
    }

    /// Demucs stems, in the order `DemucsProvider.Stem.allCases` uses.
    ///
    /// HTDemucs ships one checkpoint per stem and each emits all four sources, so a
    /// caller who wants only vocals still downloads a single 84 MB file rather than a
    /// quarter of a combined one.
    public static let demucsStems = ["drums", "bass", "other", "vocals"]

    /// Every Demucs weight file.
    public static var demucsAll: [String] {
        demucsStems.map { "\($0).safetensors" }
    }

    /// The weight file for one Demucs stem.
    public static func demucsStem(_ stem: String) -> String {
        "\(stem).safetensors"
    }

    /// One compiled MossFormerGAN package, whole.
    ///
    /// A recursive glob because an `.mlpackage` is a directory, and `*` does not
    /// cross a path separator - the same reason the Silero VAD entry names
    /// `.mlmodelc/**`. Naming the package rather than globbing the repository root
    /// matters here: both precisions live side by side, so `*.mlpackage/**` would
    /// fetch 21 MB to load 8.
    public static func mossFormerGANCoreML(_ precision: CoreMLGANPrecision) -> [String] {
        ["\(precision.packageName)/**"]
    }

    /// What a MossFormerGAN package must have on disk before Core ML can compile it.
    ///
    /// A single `**` is satisfied by *any one* entry inside the directory, so an
    /// interrupted download that left one file behind would otherwise report the
    /// variant as installed. These three are what `MLModel.compileModel(at:)`
    /// cannot proceed without.
    public static func mossFormerGANCoreMLRequired(
        _ precision: CoreMLGANPrecision
    ) -> [String] {
        let package = precision.packageName
        return [
            "\(package)/Manifest.json",
            "\(package)/Data/com.apple.CoreML/model.mlmodel",
            "\(package)/Data/com.apple.CoreML/weights/weight.bin",
        ]
    }

    /// Kokoro weights, voice embeddings, and architecture configuration.
    public static let kokoro = ["*.safetensors", "voices/*.npy", "config.json"]

    /// Files every canonical Chatterbox checkpoint needs in order to load.
    /// Optional voice conditioning, tokenizer shards, and quantization metadata
    /// must not let an interrupted snapshot satisfy this manifest by accident.
    public static let chatterbox = [
        "model.safetensors",
        "grapheme_mtl_merged_expanded_v1.json",
    ]

    /// Additional files worth fetching when a repository provides them. They are
    /// deliberately separate from the common loadability manifest above.
    public static let chatterboxOptional = [
        "config.json",
        "conds.safetensors",
        "s3tokenizer.safetensors",
        "s3_tokenizer.safetensors",
        "s3tokenizer/model.safetensors",
        "s3_tokenizer/model.safetensors",
    ]

    /// Quantized checkpoints cannot reconstruct their module layout without the
    /// quantization metadata in `config.json`.
    public static func chatterboxRequired(for precision: ModelPrecision) -> [String] {
        switch precision {
        case .int8, .int6, .int4, .bit8, .bit6, .bit4:
            chatterbox + ["config.json"]
        case .fp32, .fp16, .bf16:
            chatterbox
        }
    }

    /// Download filters include optional assets without making them evidence that
    /// the snapshot itself is complete.
    public static func chatterboxDownload(for precision: ModelPrecision) -> [String] {
        let required = chatterboxRequired(for: precision)
        return required + chatterboxOptional.filter { !required.contains($0) }
    }
}
