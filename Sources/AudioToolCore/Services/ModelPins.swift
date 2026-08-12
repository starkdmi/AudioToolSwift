//
//  ModelPins.swift
//  AudioToolCore
//
//  Immutable revisions and file hashes for every repository this package downloads
//

import Foundation

// MARK: - Pin

/// The exact upstream revision a model was audited at, and the hashes of the files
/// taken from it.
///
/// Without this, every download follows its repository's default branch. Upstream can
/// then change both the bytes and the licence terms under a caller who never asked for
/// a new version, and a published benchmark number names no revision it can be checked
/// against.
public struct ModelPin: Sendable, Hashable {

    /// Git commit on the Hub. A commit determines the whole tree, so this - not the
    /// hash table - is what fixes the terms and the contents.
    public let revision: String

    /// SHA-256 by repository-relative path, for files stored in Git LFS.
    ///
    /// Deliberately partial. The Hub's model API exposes an object id only for
    /// LFS-tracked files, so small text assets - `config.json`, tokenizer JSON,
    /// `README.md` - have no hash here. That is not a gap in coverage: ``revision``
    /// already covers them upstream, and what these hashes add is local verification,
    /// catching a truncated download or a cache directory populated from some other
    /// revision. Weights are always LFS, so the large files are always covered.
    public let fileHashes: [String: String]

    public init(revision: String, fileHashes: [String: String] = [:]) {
        self.revision = revision
        self.fileHashes = fileHashes
    }
}

// MARK: - Registry

/// Where each repository is pinned.
///
/// ``ModelDownloader`` consults this on every download, so pinning applies to the
/// providers without any of them naming a revision. Pass an explicit `revision` to
/// the downloader to override it.
public enum ModelPins {

    /// The revision to fetch from `repo`, or `"main"` when it is not pinned.
    public static func revision(for repo: String) -> String {
        all[repo]?.revision ?? unpinnedRevision
    }

    /// The pin for `repo`, if it has one.
    public static func pin(for repo: String) -> ModelPin? {
        all[repo]
    }

    /// What an unpinned repository resolves to. The Hub's own default.
    public static let unpinnedRevision = "main"

    /// Repositories that are deliberately unpinned, with the reason.
    ///
    /// Empty since 2026-08-12: USS, FRCRN and Demucs were published that day and
    /// pinned below, in line with the rule this list existed to enforce. Kept rather
    /// than deleted because "absent from `all`" and "known to be unpublished" remain
    /// different states, and the next unpublished conversion belongs here until it
    /// ships.
    public static let unpublished: Set<String> = []

    /// Fetched from the Hub on 2026-08-12 and frozen here.
    ///
    /// Two repositories in the catalog are absent by design rather than oversight:
    /// FluidAudio downloads its own ASR, diarization and Sortformer models through
    /// its own code, which this downloader never sees, so those are pinned by the
    /// FluidAudio dependency version instead. The Silero VAD entry below is pinned
    /// because the catalog also offers it as a pre-download, but the runtime fetch
    /// is still FluidAudio's.
    static let all: [String: ModelPin] = [
        "starkdmi/MossFormer2_SE_48K_MLX": ModelPin(
            revision: "ccd0ded00e26f38e9f5b0ba21608aa6a0bcd6434",
            fileHashes: [
                "model_fp32.safetensors": "8e47b75ca25dc402db5420c45c868544da8d2ac43b21a919197da113d4d81313",
                "model_fp16.safetensors": "61e63484df9c2be7e1111ca0346d431422a98b263331021a67c2d7ddb2f67a85",
                "model_int8.safetensors": "89a0a7fef6de4a7b25bac7365ea60e9b490e978d2ad2fc95c381092f06a5315f",
                "model_int6.safetensors": "2412aabb29dea11af8cc4a8546b5d938e1bee4bacac2606b0d78132acb0d55fe",
                "model_int4.safetensors": "dfe3a01376e6a3985bf9638f07b78cf30f06159534a48e485d51e15ecc81d358",
            ]
        ),
        // Re-pinned 2026-08-12, when `model_int8.safetensors` was uploaded.
        // `model_fp32.safetensors` is unchanged across the move - same hash as at
        // revision f568847f.
        //
        // Three precisions `MODEL-PRECISIONS.md` measures are deliberately absent
        // from the repository, so this table pins two files against the doc's four
        // rows. A measurement is worth publishing whether or not the checkpoint is;
        // a checkpoint is worth publishing only if something should download it.
        //
        // - `model_fp16`: forward pass is all-NaN.
        // - `model_int6`, `model_int4`: strictly dominated by int8. Quantization
        //   reaches only the linear layers here, so on this convolution-heavy model
        //   it changes neither speed nor memory - int6 and int4 buy 11 MiB and
        //   21 MiB over int8 while giving up 12 dB and 24 dB. No caller should pick
        //   either, and the catalog exposes fp32 alone.
        "starkdmi/MossFormer2_SR_48K_MLX": ModelPin(
            revision: "e0d987da4b084ae03cf8e14cd35d9b778068fd79",
            fileHashes: [
                "model_fp32.safetensors": "6061573a4ccd41731afdcb94bb65bf019c2058988caa4472ae396cd03e8d704a",
                "model_int8.safetensors": "086a8ae3d8349d1a9ecc6bee4346106f0f16860701fb9e4ef15fb35b5dbe248c",
            ]
        ),
        // Published 2026-08-12. USS names its checkpoints after the architecture
        // rather than "model" - see `ModelFiles.uss(_:)`.
        "starkdmi/USS_MLX": ModelPin(
            revision: "8a14820edde1baa00402cf716b41c485913382e1",
            fileHashes: [
                "resunet30_fp32.safetensors": "28132369445c3196ce0237cd13db52f72d724af4b99f6b08390897a3cdacb1dc",
                "resunet30_fp16.safetensors": "8dfc858b4f343d0dabff807668adea08fa80bd408123b412c01819571a224555",
            ]
        ),
        // Published 2026-08-12. One checkpoint per stem, each emitting all four
        // sources - see `ModelFiles.demucsStems`. The four sibling `.json` are
        // per-stem architecture configs; no load path opens them, so they are
        // uploaded for provenance and left out of this table like every other
        // non-LFS asset.
        "starkdmi/Demucs_MLX": ModelPin(
            revision: "57b2a970b084ff3693bd92c2dcc8b30ab7b11057",
            fileHashes: [
                "drums.safetensors": "cc77206f79e543deffc92f3f93167e8d05cc90fd2d5285622ece2c5d64159fdd",
                "bass.safetensors": "d71a63b45171e773f486c375d5079c37b65ff4787d80b3d9b4815ef6b327945b",
                "other.safetensors": "65144751ac6661ffc26bf6fe7857853c90a8d2bea077061f0b7927197ecef5e7",
                "vocals.safetensors": "3a92bc147648589974c9be14808fde04d5440fe24a66d8dc363da0e96302d21d",
            ]
        ),
        // Published 2026-08-12, by making an existing private repository public and
        // adding the safetensors conversion. Uploaded as `model_fp32.safetensors` to
        // match `ModelFiles.standard(_:)`, which is what the provider resolves
        // through; the local file is named `frcrn_se_16k.safetensors`.
        //
        // The repository still carries a `weights.npz` from the private era. Nothing
        // downloads it - the manifest names the safetensors alone - and it is left in
        // place rather than deleted.
        "starkdmi/FRCRN_SE_16K_MLX": ModelPin(
            revision: "f0b77c3e2f681bef215005154dcb057932694be5",
            fileHashes: [
                "model_fp32.safetensors": "487b2a5846db5421f554d6e446fc121b71c1bd526ae963bb51b287924825933f",
            ]
        ),
        "starkdmi/MossFormer2_SS_2SPK_16K_MLX": ModelPin(
            revision: "3d404393efb7e60ccefa8822db0a943e5830b55d",
            fileHashes: [
                "model_fp32.safetensors": "7a7aaeb0e482786f8f1eed33bb76bd1e2eb0dbfe554637c1bec00f6be1affb7c",
            ]
        ),
        "starkdmi/MossFormer2_SS_3SPK_8K_MLX": ModelPin(
            revision: "1468938256ef70ae984062cb042d2780655b2d0c",
            fileHashes: [
                "model_fp32.safetensors": "26b12f1aaf8528d0f8b63d58a5e9c3203042e9135a751f91fddbb0f756700e28",
            ]
        ),
        "starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX": ModelPin(
            revision: "33aa38abf8bbe1f88942ec083927852df30c4dea",
            fileHashes: [
                "model_fp32.safetensors": "9c638aed6d414577e38adbbe3947ba388e00518be254095a398f392b315c5159",
            ]
        ),
        "starkdmi/chatterbox": ModelPin(
            revision: "5ff46ddabb704ddf20590a1841be924cbb9ef4b3",
            fileHashes: [
                "model.safetensors": "cca8522deb59e28e805e8005ed587937bc22831397a6f341092d1a427ccddf76",
                "conds.safetensors": "709e5a7fa80e010a011c8244f553853aed7a49c106fff54008fbd89a0f5a6148",
            ]
        ),
        "starkdmi/chatterbox-fp16": ModelPin(
            revision: "e68aaba8ef36cb9a8a8c9a5807c1b1004b113c70",
            fileHashes: [
                "model.safetensors": "9be84ebbf3cd5fa4304b64cb949c57706842e99b9b4c7465b5d3f1733dd5c06a",
                "conds.safetensors": "709e5a7fa80e010a011c8244f553853aed7a49c106fff54008fbd89a0f5a6148",
            ]
        ),
        "starkdmi/chatterbox-8bit": ModelPin(
            revision: "c2b45027c280570e1c1ec5ed80f6f60c5910666b",
            fileHashes: [
                "model.safetensors": "45ce4b49deea45e0fd49d24c83385b890d31df686d3b324e7c0ffe80b62008d8",
                "conds.safetensors": "709e5a7fa80e010a011c8244f553853aed7a49c106fff54008fbd89a0f5a6148",
            ]
        ),
        "starkdmi/chatterbox-6bit": ModelPin(
            revision: "94df0fb857060cfef12ba675687a70ed55c520b3",
            fileHashes: [
                "model.safetensors": "f2d6452ff319caf91e515b0eb2e83a32f069e76f78a3ad0bc240e7ba3edb5c33",
                "conds.safetensors": "709e5a7fa80e010a011c8244f553853aed7a49c106fff54008fbd89a0f5a6148",
            ]
        ),
        "starkdmi/chatterbox-4bit": ModelPin(
            revision: "e871120133964086d19d25785095dd74ff04dde1",
            fileHashes: [
                "model.safetensors": "8957d56724a27182c702f9d9de33ffd462688fe068bd2bfdd33fb630cbaf1f5e",
                "conds.safetensors": "709e5a7fa80e010a011c8244f553853aed7a49c106fff54008fbd89a0f5a6148",
            ]
        ),
        "mlx-community/Kokoro-82M-bf16": ModelPin(
            revision: "a71e4d38b236d968966a2002c4c895dbd12b1c3c",
            fileHashes: [
                "kokoro-v1_0.safetensors": "4e9ecdf03b8b6cf906070390237feda473dc13327cb8d56a43deaa374c02acd8",
            ]
        ),
        // The quantized Kokoro repositories. `TTSProviders.kokoro(precision:)` derives
        // a repository name from any `ModelPrecision`, so these three were reachable
        // from public API while following mutable `main` - the one thing this registry
        // exists to prevent. Pinned at the revision each was serving when they were
        // added here, which is what `main` already resolved to, so nothing about what
        // gets downloaded changes; only whether it can change underneath a caller.
        "mlx-community/Kokoro-82M-4bit": ModelPin(
            revision: "e4468a460f6f70b9125a003e0adb1ab7d4904bbd",
            fileHashes: [
                "kokoro-v1_0.safetensors": "5a5cb3d87478f2e74dfca208ee52209ccfce024095e137097fd276026506e45f",
            ]
        ),
        "mlx-community/Kokoro-82M-6bit": ModelPin(
            revision: "9fb47e2969f4d6f05cefd3a3de99e33387080fbc",
            fileHashes: [
                "kokoro-v1_0.safetensors": "eac5a5e2c11a791a1e7f70ffb470ca4caca0b877926bc0a8997fe3e7d8934743",
            ]
        ),
        "mlx-community/Kokoro-82M-8bit": ModelPin(
            revision: "7e173a214392b1e0cb397e71c9745cbd14c06063",
            fileHashes: [
                "kokoro-v1_0.safetensors": "fb2b8f22b906f2e70e7dae4c337457784b079fb44142d8f4da93d8a9ace905ed",
            ]
        ),
        "mlx-community/whisper-large-v3-mlx": ModelPin(
            revision: "49e6aa286ad60c14352c404340ded53710378a11",
            fileHashes: [
                "weights.npz": "05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d",
            ]
        ),
        "mlx-community/whisper-small-mlx": ModelPin(
            revision: "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e",
            fileHashes: [
                "weights.npz": "55b6674c9b339702d486e2b1573839a66f8ec8f821ed2886993ef717a86b09f5",
            ]
        ),
        "mlx-community/S3TokenizerV2": ModelPin(
            revision: "e0c9886f0e1c35ae85b1f27277416fb19fc72bec",
            fileHashes: [
                "model.safetensors": "928726bc1f206a613d36b8f49e297eae9c5593a21bf9b92ddfe2c23f85eb92cc",
            ]
        ),
        "mlx-community/translategemma-4b-it-4bit": ModelPin(
            revision: "5788ec08c047f3f2e17808101b8d9566ac930d58",
            fileHashes: [
                "model.safetensors": "113acb0c29997a3015af84bec2c8f967cb7b15f8959d1c26b9628b921e324c40",
                "tokenizer.json": "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
                "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
            ]
        ),
        // Published 2026-08-12. Two compiled packages side by side; a caller loads
        // one. `Manifest.json` is small enough that the Hub does not store it in
        // LFS, so it has no hash here for the reason `fileHashes` documents - the
        // revision covers it.
        "starkdmi/MossFormer_GAN_SE_16K_CoreML": ModelPin(
            revision: "8961405a395be865ca3fd629816868ac8906a4a4",
            fileHashes: [
                "MossFormerGAN_256frames.mlpackage/Data/com.apple.CoreML/model.mlmodel":
                    "32dc37b3905d1d4a202ec8dba94d15ec0704fec6a0f27f838cd5bdb95e93396a",
                "MossFormerGAN_256frames.mlpackage/Data/com.apple.CoreML/weights/weight.bin":
                    "ab5450f48d508006135f7953897a61e2cef6aa6717c219b53107331ad5375105",
                "MossFormerGAN_256frames_FP16.mlpackage/Data/com.apple.CoreML/model.mlmodel":
                    "f4d86febdc1579f05f614b0e29d8c88b0a520765c7a11804fc090e237735b453",
                "MossFormerGAN_256frames_FP16.mlpackage/Data/com.apple.CoreML/weights/weight.bin":
                    "49225b4a0089c977b0be44a4cb29804d42db2aae6672d10a054e1d7642b38e43",
            ]
        ),
        "FluidInference/silero-vad-coreml": ModelPin(
            revision: "b419383c55c110e2c9271fa6ee0ea83d03c70d96",
            fileHashes: [
                "silero-vad-unified-256ms-v6.2.1.mlmodelc/weights/weight.bin": "53ecc8b5081146140ab654c89109cf001f2183abddd7a2411c5081feeffff063",
                "silero-vad-unified-256ms-v6.2.1.mlmodelc/coremldata.bin": "7db35a4fd995222a7fb0129713473b15d1462572ab4a2e5e4d56bcaad9e40f41",
            ]
        ),
    ]
}
