//
//  LocalWeights.swift
//  AudioToolBenchmark
//
//  Using weights that are already on the machine instead of fetching them again.
//

import Foundation

/// Weights found in the research checkout this package was extracted from.
///
/// Every provider here takes an explicit `weightsPath:` alongside its downloading
/// initialiser, and on a development machine the weights usually already exist -
/// under `Models/` in the sibling checkout, or staged under `Parity/weights`. Making
/// the benchmark fetch its own copy from HuggingFace when a usable one is sitting
/// next to it wastes gigabytes and, worse, makes the whole suite unrunnable offline
/// or on a machine whose HuggingFace cache is cold.
///
/// Absent in a standalone clone, which is the normal case: ``autodetect()`` returns
/// nil, nothing resolves, and every case falls back to the HuggingFace path exactly
/// as before. Point `--weights-root` at any directory with this layout to override.
///
/// The layout is the research checkout's, not a convention this package invents -
/// see the table in ``Layout``. It is deliberately a lookup of known relative paths
/// rather than a search: a search would eventually find a half-converted checkpoint
/// somebody left in a scratch directory and benchmark that instead.
public struct LocalWeights: Sendable {

    public let root: URL?

    public init(root: URL?) {
        self.root = root
    }

    /// The sibling research checkout, if this binary is still beside it.
    ///
    /// Same test `TestGate.checkoutRoot` uses - the presence of a `Models/`
    /// directory above the package - so the benchmark and the integration suites
    /// agree about what "the checkout is present" means.
    public static func autodetect() -> URL? {
        // LocalWeights.swift -> AudioToolBenchmark -> Sources -> AudioToolSwift -> root
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        var isDirectory: ObjCBool = false
        let models = url.appendingPathComponent("Models")
        guard FileManager.default.fileExists(atPath: models.path, isDirectory: &isDirectory),
              isDirectory.boolValue else { return nil }
        return url
    }

    /// Where each model's weights sit under the checkout root.
    ///
    /// | case | path |
    /// | --- | --- |
    /// | FRCRN | `Models/frcrn_se_mlx_swift/Weights/frcrn_se_16k.safetensors` |
    /// | SE 48K FP32 | `Models/mossformer2_se_mlx_swift/model_fp32.safetensors` |
    /// | Demucs | `Models/demucs_mlx_swift/Weights` (a directory of four stems) |
    /// | USS | `Models/uss_mlx_swift/USSSwift/Models/resunet30_<precision>.safetensors` |
    /// | MossFormerGAN | `Models/mossformer_gan_se_coreml/MossFormerGAN_256frames.mlpackage` |
    ///
    /// Super-resolution and speaker separation are absent on purpose: the checkout
    /// has no copy of either, and inventing a path for them would produce a silent
    /// fallback that looks like a lookup.
    public enum Layout {
        public static let frcrnSE16K = "Models/frcrn_se_mlx_swift/Weights/frcrn_se_16k.safetensors"
        public static let mossFormer2SE48KFP32 = "Models/mossformer2_se_mlx_swift/model_fp32.safetensors"
        public static let demucsDirectory = "Models/demucs_mlx_swift/Weights"
        public static let mossFormerGANCoreML =
            "Models/mossformer_gan_se_coreml/MossFormerGAN_256frames.mlpackage"

        /// The FP16-weight conversion of the same graph, alongside it in the
        /// checkout at 7.6 MB against 13 MB. CoreML precision is a property of the
        /// compiled `.mlpackage`, not a runtime flag, so the two are separate files
        /// and separate benchmark cases rather than one case with a parameter.
        public static let mossFormerGANCoreMLFP16 =
            "Models/mossformer_gan_se_coreml/MossFormerGAN_256frames_FP16.mlpackage"

        /// `USSSwift/Models`, not `USSSwiftTests`.
        ///
        /// The test directory holds one real symlink and one macOS *alias*:
        /// `resunet30_fp32.safetensors` is a 45-byte symlink that resolves, and
        /// `resunet30_fp16.safetensors` is a 1152-byte alias file beginning
        /// `book....mark....` that nothing below Finder can follow. `file(1)` calls
        /// it "MacOS Alias file". So fp16 was rejected by the size floor below and
        /// every fp16 benchmark fell through to HuggingFace - where `USS_MLX` was at
        /// the time unpublished, making the default USS configuration unrunnable. The
        /// repository has been public since 2026-08-12, so that fallback now works;
        /// the local path is still preferred, to keep a benchmark run off the network.
        ///
        /// `USSSwift/Models` holds both as ordinary files, 53 MB and 106 MB.
        public static func uss(_ precision: ModelPrecisionName) -> String {
            "Models/uss_mlx_swift/USSSwift/Models/resunet30_\(precision.rawValue).safetensors"
        }
    }

    /// Precision names as the USS checkpoints spell them.
    ///
    /// A tiny local enum rather than `AudioToolCore.ModelPrecision`, so this file
    /// does not have to decide what a local path for `int4` would mean.
    public enum ModelPrecisionName: String, Sendable {
        case fp16, fp32
    }

    /// An existing, plausibly-real path under the root, or nil.
    ///
    /// Checked here rather than at the point of use, so a stale or broken entry
    /// degrades to a HuggingFace fetch instead of failing a case halfway through a
    /// long run.
    ///
    /// "Plausibly real" is a size floor, and it is not paranoia: the checkout's
    /// `resunet30_fp16.safetensors` is a 1152-byte macOS bookmark alias whose
    /// contents begin `book....mark....`. Existence alone accepted it, and the case
    /// failed inside MLX with "Invalid json header length" - a message about the
    /// wrong layer of the system entirely. Git LFS pointers and interrupted
    /// downloads fail the same way. A directory (Demucs, an `.mlpackage`) is taken
    /// as-is; only files are size-checked.
    public func path(_ relative: String) -> String? {
        guard let root else { return nil }
        let candidate = root.appendingPathComponent(relative)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDirectory)
        else { return nil }
        if isDirectory.boolValue { return candidate.path }

        // Resolved first, because the checkout stores `resunet30_fp32.safetensors`
        // as a symlink and `attributesOfItem` reports the link's own size - 45
        // bytes - rather than the target's. Measuring the link failed the floor
        // below and sent a perfectly good local checkpoint back to HuggingFace.
        let resolved = candidate.resolvingSymlinksInPath()
        var resolvedIsDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: resolved.path, isDirectory: &resolvedIsDirectory)
        else { return nil }
        if resolvedIsDirectory.boolValue { return candidate.path }

        // The smallest real checkpoint in this catalog is USS FP16 at ~53 MB, and
        // every stub form is a few kilobytes. A megabyte separates them with three
        // orders of magnitude to spare.
        let size = (try? FileManager.default.attributesOfItem(atPath: resolved.path)[.size])
            .flatMap { $0 as? Int } ?? 0
        guard size >= 1024 * 1024 else { return nil }
        return candidate.path
    }

    // MARK: Convenience

    public var frcrnSE16K: String? { path(Layout.frcrnSE16K) }
    public var mossFormer2SE48KFP32: String? { path(Layout.mossFormer2SE48KFP32) }
    public var demucsDirectory: String? { path(Layout.demucsDirectory) }
    public var mossFormerGANCoreML: String? { path(Layout.mossFormerGANCoreML) }
    public var mossFormerGANCoreMLFP16: String? { path(Layout.mossFormerGANCoreMLFP16) }

    public func uss(_ precision: ModelPrecisionName) -> String? {
        path(Layout.uss(precision))
    }
}

/// Where a case's weights came from, recorded per result.
///
/// Without this a report cannot be read: a load time from a local file, a warm
/// HuggingFace cache and a cold download are three different measurements, and
/// only one of them belongs in a comparison.
public enum WeightsSource: String, Codable, Sendable {
    /// An explicit path on this machine - the research checkout or `--weights-root`.
    case local
    /// Resolved from the HuggingFace cache without a transfer.
    case cache
    /// Not present when the case started; `loadSeconds` includes a download.
    case network
    /// The case supplies no weights through either route.
    case none
}
