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

    /// FRCRN speech enhancement, 16 kHz.
    ///
    /// The rate is in the name to match its neighbours - the catalog previously said
    /// `FRCRN_SE_MLX`, which matched nothing because no download path existed to
    /// disagree with it. Rename here if the published repo ends up called something
    /// else; both the catalog and the provider follow this constant.
    public static let frcrnSE16K = "starkdmi/FRCRN_SE_16K_MLX"

    /// Demucs music separation.
    public static let demucs = "starkdmi/Demucs_MLX"
}

// MARK: - File Layouts

/// Which files each repository holds.
public enum ModelFiles {

    /// Repos following the single-repo, precision-suffixed convention:
    /// `model_fp32.safetensors`, `model_fp16.safetensors`, and a shared config.
    ///
    /// - Parameter precision: which weights to fetch.
    public static func standard(_ precision: ModelPrecision) -> [String] {
        [precision.weightsFilename, "config.json"]
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
}
