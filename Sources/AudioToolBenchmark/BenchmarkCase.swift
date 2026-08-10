//
//  BenchmarkCase.swift
//  AudioToolBenchmark
//
//  What gets measured: one entry per model configuration worth a number.
//

import Foundation
import AudioTool
import AudioToolCore
import AudioToolMLX
import AudioToolCoreML
import AudioToolUSS

// MARK: - Workload

/// What one measured run produced, so an empty result is visible in the report.
public struct WorkloadOutput: Sendable {
    public var streams: Int
    public var frames: Int
    public var sampleRate: Int

    public init(streams: Int, frames: Int, sampleRate: Int) {
        self.streams = streams
        self.frames = frames
        self.sampleRate = sampleRate
    }
}

/// A model, reduced to the three operations the runner times.
///
/// Closure-based rather than generic over the provider protocols, because the
/// providers do not share one: `SpeechEnhancer` has `process`, `SpeechSeparator`
/// has `separate`, `DemucsProvider` takes a stem, `USSMLXProvider` takes a target
/// embedding, and `load()` is on each concrete type rather than any protocol.
/// Three closures express all of that without a protocol that exists only to be
/// conformed to once each.
public struct BenchmarkWorkload: Sendable {
    public var load: @Sendable () async throws -> Void
    public var run: @Sendable (AudioBuffer) async throws -> WorkloadOutput
    public var unload: @Sendable () async -> Void

    public init(
        load: @escaping @Sendable () async throws -> Void,
        run: @escaping @Sendable (AudioBuffer) async throws -> WorkloadOutput,
        unload: @escaping @Sendable () async -> Void
    ) {
        self.load = load
        self.run = run
        self.unload = unload
    }
}

// MARK: - Case

/// One measurable model configuration.
public struct BenchmarkCase: Sendable {

    /// Stable identifier, used by `--case` and as the JSON key. Renaming one
    /// breaks comparison with older reports, so treat these as an interface.
    public let id: String
    public let label: String
    public let category: String
    /// `"mlx"` or `"coreml"`.
    public let backend: String

    /// Rate the model consumes. The runner generates input at this rate rather
    /// than resampling a common buffer, so no case is charged for a conversion
    /// that its caller would not perform.
    public let sampleRate: Int

    /// Channels the model consumes. `DemucsProvider` declares 2 and validates it,
    /// so this is not cosmetic - a mono buffer is rejected before any inference.
    public let inputChannels: Int

    /// Rough working set, used only for the pre-flight headroom check.
    public let estimatedMemoryBytes: Int

    /// Per-case caps, when this model needs something other than the run default.
    public let gpuCacheLimitBytes: Int?
    public let gpuMemoryLimitBytes: Int?

    /// Whether the weights are already on this machine. Nil when the case does
    /// not resolve weights through `ModelDownloader` - the CoreML model is a file
    /// the caller supplies, not a repository.
    public let weightsPreCached: @Sendable () -> Bool?

    /// Which of the three routes the weights will take. See ``WeightsSource``.
    public let weightsSource: @Sendable () -> WeightsSource

    /// Why this case cannot run here, or nil if it can.
    ///
    /// Distinct from a failure. A machine without the CoreML `.mlpackage` is not
    /// broken, and reporting that as a red result trains people to ignore red.
    public let unavailableReason: @Sendable () -> String?

    /// Fresh provider per run. Never a shared instance: the point of this harness
    /// is that a case starts from nothing.
    public let makeWorkload: @Sendable () -> BenchmarkWorkload

    public init(
        id: String,
        label: String,
        category: String,
        backend: String,
        sampleRate: Int,
        inputChannels: Int = 1,
        estimatedMemoryBytes: Int,
        gpuCacheLimitBytes: Int? = nil,
        gpuMemoryLimitBytes: Int? = nil,
        weightsPreCached: @escaping @Sendable () -> Bool? = { nil },
        weightsSource: @escaping @Sendable () -> WeightsSource = { .none },
        unavailableReason: @escaping @Sendable () -> String? = { nil },
        makeWorkload: @escaping @Sendable () -> BenchmarkWorkload
    ) {
        self.id = id
        self.label = label
        self.category = category
        self.backend = backend
        self.sampleRate = sampleRate
        self.inputChannels = inputChannels
        self.estimatedMemoryBytes = estimatedMemoryBytes
        self.gpuCacheLimitBytes = gpuCacheLimitBytes
        self.gpuMemoryLimitBytes = gpuMemoryLimitBytes
        self.weightsPreCached = weightsPreCached
        self.weightsSource = weightsSource
        self.unavailableReason = unavailableReason
        self.makeWorkload = makeWorkload
    }
}

// MARK: - Catalog

/// Every case this binary can run.
///
/// Scope is MLX and CoreML audio models - the ones this package implements. The
/// FluidAudio-backed VAD, transcription and diarization providers are excluded
/// deliberately: they wrap a third-party pipeline whose numbers describe that
/// project rather than this one, and mixing them into the same table invites a
/// comparison that is not being made.
///
/// TTS (Kokoro, Chatterbox) is also absent. Not for any dependency reason - it
/// links fine - but because a text-to-speech RTF is a different measurement with a
/// different denominator, and folding it into a table of audio-to-audio RTFs would
/// produce a column where two rows mean different things. Adding a `tts` category
/// here is a small change when someone wants it.
public enum BenchmarkCatalog {

    public static func allCases(options: CatalogOptions = CatalogOptions()) -> [BenchmarkCase] {
        enhancement(options) + separation(options) + superResolution(options)
            + music(options) + universal(options)
    }

    /// Inputs the catalog needs that cannot be discovered.
    public struct CatalogOptions: Sendable {
        /// Path to `MossFormerGAN_*.mlpackage`. The CoreML enhancer takes a file
        /// rather than a repository, so without one that case reports as skipped.
        ///
        /// Falls back to the research checkout's copy when one is present, which is
        /// why this case now runs unattended on a development machine.
        public var coreMLGANModelPath: String?

        /// Weights already on this machine. See ``LocalWeights``.
        public var localWeights: LocalWeights

        public init(
            coreMLGANModelPath: String? = nil,
            localWeights: LocalWeights = LocalWeights(root: LocalWeights.autodetect())
        ) {
            self.localWeights = localWeights
            self.coreMLGANModelPath = coreMLGANModelPath ?? localWeights.mossFormerGANCoreML
        }
    }

    // MARK: Speech enhancement

    private static func enhancement(_ options: CatalogOptions) -> [BenchmarkCase] {
        let frcrnLocal = options.localWeights.frcrnSE16K
        var cases: [BenchmarkCase] = [
            BenchmarkCase(
                id: "mlx.frcrn_se_16k",
                label: "FRCRN SE 16 kHz",
                category: "enhancement",
                backend: "mlx",
                sampleRate: 16000,
                estimatedMemoryBytes: 400_000_000,
                weightsPreCached: {
                    frcrnLocal != nil
                        || cached(FRCRNSE16KProvider.repo, ModelFiles.standardRequired(.fp32))
                },
                weightsSource: {
                    source(
                        local: frcrnLocal,
                        repo: FRCRNSE16KProvider.repo,
                        files: ModelFiles.standardRequired(.fp32)
                    )
                },
                makeWorkload: {
                    let provider = frcrnLocal.map { FRCRNSE16KProvider(weightsPath: $0) }
                        ?? FRCRNSE16KProvider(precision: .fp32)
                    return transform(
                        load: { try await provider.load() },
                        process: { try await provider.process($0) },
                        unload: { await provider.unload() }
                    )
                }
            )
        ]

        for precision in [ModelPrecision.fp32, .fp16] {
            // Only FP32 exists in the checkout; FP16 resolves through HuggingFace.
            let local = precision == .fp32 ? options.localWeights.mossFormer2SE48KFP32 : nil
            cases.append(BenchmarkCase(
                id: "mlx.mossformer2_se_48k.\(precision.rawValue)",
                label: "MossFormer2 SE 48 kHz (\(precision.rawValue.uppercased()))",
                category: "enhancement",
                backend: "mlx",
                sampleRate: 48000,
                estimatedMemoryBytes: 900_000_000,
                weightsPreCached: {
                    local != nil
                        || cached(MossFormer2SE48KProvider.repo, ModelFiles.standardRequired(precision))
                },
                weightsSource: {
                    source(
                        local: local,
                        repo: MossFormer2SE48KProvider.repo,
                        files: ModelFiles.standardRequired(precision)
                    )
                },
                makeWorkload: {
                    let provider = local.map { MossFormer2SE48KProvider(weightsPath: $0) }
                        ?? MossFormer2SE48KProvider(precision: precision)
                    return transform(
                        load: { try await provider.load() },
                        process: { try await provider.process($0) },
                        unload: { await provider.unload() }
                    )
                }
            ))
        }

        if let modelPath = options.coreMLGANModelPath {
            cases.append(BenchmarkCase(
                id: "coreml.mossformer_gan_se_16k",
                label: "MossFormerGAN SE 16 kHz (CoreML)",
                category: "enhancement",
                backend: "coreml",
                sampleRate: 16000,
                estimatedMemoryBytes: 700_000_000,
                weightsPreCached: { FileManager.default.fileExists(atPath: modelPath) },
                weightsSource: { .local },
                unavailableReason: {
                    FileManager.default.fileExists(atPath: modelPath)
                        ? nil
                        : "CoreML model not found at \(modelPath)"
                },
                makeWorkload: {
                    let provider = MossFormerGANCoreMLProvider(modelPath: modelPath)
                    return transform(
                        load: { try await provider.load() },
                        process: { try await provider.process($0) },
                        unload: { await provider.unload() }
                    )
                }
            ))
        } else {
            cases.append(BenchmarkCase(
                id: "coreml.mossformer_gan_se_16k",
                label: "MossFormerGAN SE 16 kHz (CoreML)",
                category: "enhancement",
                backend: "coreml",
                sampleRate: 16000,
                estimatedMemoryBytes: 700_000_000,
                weightsPreCached: { nil },
                unavailableReason: {
                    "no .mlpackage supplied - pass --coreml-gan <path> or set "
                        + "AUDIOTOOL_BENCH_COREML_GAN. Unlike the MLX models this one "
                        + "has no HuggingFace repository to download from."
                },
                makeWorkload: {
                    // Unreachable: the runner checks availability first. A workload
                    // that throws is still better than a fatalError in a benchmark
                    // that is meant to survive one bad case.
                    BenchmarkWorkload(
                        load: { throw CaseUnavailable.noCoreMLModel },
                        run: { _ in throw CaseUnavailable.noCoreMLModel },
                        unload: {}
                    )
                }
            ))
        }

        return cases
    }

    // MARK: Speaker separation

    private static func separation(_ options: CatalogOptions) -> [BenchmarkCase] {
        _ = options
        return MossFormer2SSProvider.Model.allCases.map { model in
            BenchmarkCase(
                id: "mlx.mossformer2_ss.\(model.rawValue.replacingOccurrences(of: "-", with: "_"))",
                label: "MossFormer2 SS \(model.rawValue) \(model.sampleRate / 1000) kHz",
                category: "separation",
                backend: "mlx",
                sampleRate: model.sampleRate,
                estimatedMemoryBytes: 1_000_000_000,
                // No copy in the checkout; these resolve through the HuggingFace
                // cache, which after the `standardRequired` fix now recognises the
                // safetensors that were already there.
                weightsPreCached: {
                    cached(model.huggingFaceRepo, ModelFiles.standardRequired(.fp32))
                },
                weightsSource: {
                    source(
                        local: nil,
                        repo: model.huggingFaceRepo,
                        files: ModelFiles.standardRequired(.fp32)
                    )
                },
                makeWorkload: {
                    let provider = MossFormer2SSProvider(model: model, precision: .fp32)
                    return BenchmarkWorkload(
                        load: { try await provider.load() },
                        run: { input in
                            let outputs = try await provider.separate(input)
                            return WorkloadOutput(
                                streams: outputs.count,
                                frames: outputs.first?.frameCount ?? 0,
                                sampleRate: outputs.first?.sampleRate ?? model.sampleRate
                            )
                        },
                        unload: { await provider.unload() }
                    )
                }
            )
        }
    }

    // MARK: Super resolution

    private static func superResolution(_ options: CatalogOptions) -> [BenchmarkCase] {
        _ = options
        return [
            BenchmarkCase(
                id: "mlx.mossformer2_sr_48k",
                label: "MossFormer2 SR 16 -> 48 kHz",
                category: "superResolution",
                backend: "mlx",
                // Consumes 16 kHz and emits 48 kHz. Feeding it 48 kHz would be
                // self-defeating - see the note on `sampleRate` in the provider -
                // and would also make its RTF incomparable to its own docs.
                sampleRate: 16000,
                estimatedMemoryBytes: 1_200_000_000,
                // The one case where `config.json` is not decoration: this provider
                // reads its architecture out of it rather than hardcoding one, so
                // both files have to be present for a load to succeed.
                weightsPreCached: {
                    cached(MossFormer2SR48KProvider.repo, ModelFiles.standard(.fp32))
                },
                weightsSource: {
                    source(
                        local: nil,
                        repo: MossFormer2SR48KProvider.repo,
                        files: ModelFiles.standard(.fp32)
                    )
                },
                makeWorkload: {
                    let provider = MossFormer2SR48KProvider(precision: .fp32)
                    return transform(
                        load: { try await provider.load() },
                        process: { try await provider.process($0) },
                        unload: { await provider.unload() }
                    )
                }
            )
        ]
    }

    // MARK: Music separation

    private static func music(_ options: CatalogOptions) -> [BenchmarkCase] {
        let demucsLocal = options.localWeights.demucsDirectory
        return [
            BenchmarkCase(
                id: "mlx.demucs.vocals",
                label: "Demucs (vocals stem only)",
                category: "music",
                backend: "mlx",
                sampleRate: 44100,
                inputChannels: 2,
                estimatedMemoryBytes: 1_500_000_000,
                weightsPreCached: {
                    demucsLocal != nil
                        || cached(DemucsProvider.repo, [ModelFiles.demucsStem("vocals")])
                },
                weightsSource: {
                    source(
                        local: demucsLocal,
                        repo: DemucsProvider.repo,
                        files: [ModelFiles.demucsStem("vocals")]
                    )
                },
                makeWorkload: {
                    let provider = demucsLocal.map { DemucsProvider(weightsDirectory: $0) }
                        ?? DemucsProvider()
                    return BenchmarkWorkload(
                        load: { try await provider.load(stem: .vocals) },
                        run: { input in
                            let output = try await provider.separate(input, stem: .vocals)
                            return WorkloadOutput(
                                streams: 1,
                                frames: output.frameCount,
                                sampleRate: output.sampleRate
                            )
                        },
                        unload: { await provider.unload() }
                    )
                }
            ),
            BenchmarkCase(
                id: "mlx.demucs.all_stems",
                label: "Demucs (all four stems)",
                category: "music",
                backend: "mlx",
                sampleRate: 44100,
                inputChannels: 2,
                // Four HTDemucs checkpoints resident at once, each run over the
                // whole input. The largest thing in this catalog by a wide margin,
                // and the reason the per-case override exists at all.
                estimatedMemoryBytes: 4_000_000_000,
                // No cache override. This used to ask for 1 GB, tighter than the
                // 3 GB production ceiling of the time. Production is 512 MB now and
                // `MLXCachePolicy` applies it during `load()`, so an override here
                // would be both looser and ignored.
                weightsPreCached: {
                    demucsLocal != nil || cached(DemucsProvider.repo, ModelFiles.demucsAll)
                },
                weightsSource: {
                    source(
                        local: demucsLocal,
                        repo: DemucsProvider.repo,
                        files: ModelFiles.demucsAll
                    )
                },
                makeWorkload: {
                    let provider = demucsLocal.map { DemucsProvider(weightsDirectory: $0) }
                        ?? DemucsProvider()
                    return BenchmarkWorkload(
                        load: { try await provider.loadAll() },
                        run: { input in
                            let outputs = try await provider.separateAll(input)
                            return WorkloadOutput(
                                streams: outputs.count,
                                frames: outputs.values.first?.frameCount ?? 0,
                                sampleRate: outputs.values.first?.sampleRate ?? 44100
                            )
                        },
                        unload: { await provider.unload() }
                    )
                }
            ),
        ]
    }

    // MARK: Universal source separation

    private static func universal(_ options: CatalogOptions) -> [BenchmarkCase] {
        [true, false].map { useFp16 in
            let precision: ModelPrecision = useFp16 ? .fp16 : .fp32
            let local = options.localWeights.uss(useFp16 ? .fp16 : .fp32)
            return BenchmarkCase(
                id: "mlx.uss.\(precision.rawValue)",
                label: "USS ResUNet30 (\(precision.rawValue.uppercased()))",
                category: "universal",
                backend: "mlx",
                sampleRate: 32000,
                estimatedMemoryBytes: 800_000_000,
                // USS is the case that motivated an explicit cap in the first
                // place: it segments a whole file into two-second pieces and, with
                // MLX's default cache ceiling, its resident size tracks the machine
                // instead of the work. `USSInference` applies the same cap itself at
                // construction; naming it here keeps the report honest about what
                // was in force rather than implying the run-wide default was.
                gpuCacheLimitBytes: MemoryBudget.productionCacheLimitBytes,
                weightsPreCached: {
                    local != nil || cached(USSMLXProvider.repo, ModelFiles.uss(precision))
                },
                weightsSource: {
                    source(local: local, repo: USSMLXProvider.repo, files: ModelFiles.uss(precision))
                },
                makeWorkload: {
                    let provider = local.map {
                        USSProviders.separation(
                            weightsPath: $0,
                            type: .speech,
                            segmentDuration: 2.0,
                            useFp16: useFp16
                        )
                    } ?? USSProviders.separation(
                        type: .speech,
                        segmentDuration: 2.0,
                        useFp16: useFp16
                    )
                    return BenchmarkWorkload(
                        load: { try await provider.load() },
                        run: { input in
                            let output = try await provider.separateSound(input, target: .speech)
                            return WorkloadOutput(
                                streams: 1,
                                frames: output.frameCount,
                                sampleRate: output.sampleRate
                            )
                        },
                        unload: { await provider.unload() }
                    )
                }
            )
        }
    }

    // MARK: Helpers

    /// The common shape: an `AudioTransform`-style `process(_:)`.
    private static func transform(
        load: @escaping @Sendable () async throws -> Void,
        process: @escaping @Sendable (AudioBuffer) async throws -> AudioBuffer,
        unload: @escaping @Sendable () async -> Void
    ) -> BenchmarkWorkload {
        BenchmarkWorkload(
            load: load,
            run: { input in
                let output = try await process(input)
                return WorkloadOutput(
                    streams: 1,
                    frames: output.frameCount,
                    sampleRate: output.sampleRate
                )
            },
            unload: unload
        )
    }

    /// Whether the files this model actually opens are already on disk.
    ///
    /// Load time with a download in it is not a load time, so the report records
    /// this rather than leaving a reader to wonder why one row is forty seconds.
    ///
    /// The lists passed here are the providers' own present-to-load manifests -
    /// `ModelFiles.standardRequired` for the models that read nothing but their
    /// safetensors, `ModelFiles.standard` for super-resolution, which really does
    /// read a `config.json`. Sharing the constant is the point: if a provider's
    /// requirements change, this report cannot go on describing the old ones.
    ///
    /// Writing this benchmark is what found the mismatch. The providers used to
    /// check for `[weights, "config.json"]` before deciding a model was cached, so
    /// a machine holding the weights and not the unused config was treated as
    /// having nothing - measured: SE 48K reported uncached, then loaded in 0.6 s
    /// and transferred nothing. Both sides now ask the same narrower question.
    private static func cached(_ repo: String, _ files: [String]) -> Bool {
        ModelDownloader.shared.localPath(for: repo, matching: files) != nil
    }

    /// Which of the three routes a case's weights will take.
    ///
    /// Order matters and matches what the workload does: an explicit local path
    /// wins, then the cache, then the network.
    private static func source(local: String?, repo: String, files: [String]) -> WeightsSource {
        if local != nil { return .local }
        return cached(repo, files) ? .cache : .network
    }
}

enum CaseUnavailable: Error, CustomStringConvertible {
    case noCoreMLModel

    var description: String {
        switch self {
        case .noCoreMLModel: "no CoreML model path was supplied"
        }
    }
}

// MARK: - Selection

extension BenchmarkCatalog {

    /// Cases matching any of `patterns`, or all of them when empty.
    ///
    /// Substring match on the id, and on the category, so `--filter enhancement`
    /// and `--filter demucs` both do what they look like.
    public static func select(
        _ patterns: [String],
        options: CatalogOptions = CatalogOptions()
    ) -> [BenchmarkCase] {
        let all = allCases(options: options)
        guard !patterns.isEmpty else { return all }
        let needles = patterns.map { $0.lowercased() }
        return all.filter { benchmarkCase in
            needles.contains { needle in
                benchmarkCase.id.lowercased().contains(needle)
                    || benchmarkCase.category.lowercased().contains(needle)
                    || benchmarkCase.backend.lowercased().contains(needle)
            }
        }
    }
}
