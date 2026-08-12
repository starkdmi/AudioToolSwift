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
    /// These are the three conversions that are not published yet, so there is no
    /// revision to pin to. They are listed rather than omitted because "absent from
    /// `all`" and "known to be unpublished" are different states, and the release
    /// checklist needs to tell them apart. Pin them in the same commit that makes
    /// them public.
    public static let unpublished: Set<String> = [
        ModelRepository.uss,
        ModelRepository.frcrnSE16K,
        ModelRepository.demucs,
    ]

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
        "starkdmi/MossFormer2_SR_48K_MLX": ModelPin(
            revision: "f568847f16d23af4720b302b3dfea0db6558c8f9",
            fileHashes: [
                "model_fp32.safetensors": "6061573a4ccd41731afdcb94bb65bf019c2058988caa4472ae396cd03e8d704a",
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
        "FluidInference/silero-vad-coreml": ModelPin(
            revision: "b419383c55c110e2c9271fa6ee0ea83d03c70d96",
            fileHashes: [
                "silero-vad-unified-256ms-v6.2.1.mlmodelc/weights/weight.bin": "53ecc8b5081146140ab654c89109cf001f2183abddd7a2411c5081feeffff063",
                "silero-vad-unified-256ms-v6.2.1.mlmodelc/coremldata.bin": "7db35a4fd995222a7fb0129713473b15d1462572ab4a2e5e4d56bcaad9e40f41",
            ]
        ),
    ]
}
