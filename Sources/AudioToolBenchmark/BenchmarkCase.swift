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
import AudioToolTTS
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

/// Which side of a case the real-time factor divides by.
///
/// Every audio-to-audio model here consumes and produces the same duration, so
/// "seconds of audio per second of wall time" is unambiguous and the input is the
/// convenient place to read it. Text-to-speech consumes no audio at all: the
/// runner still hands it a buffer, the model ignores it, and dividing by that
/// buffer's duration would report the length of a signal nothing looked at.
///
/// Naming the basis per case rather than inferring it from the category keeps the
/// two kinds of row from silently sharing a column - a `tts` RTF and an
/// `enhancement` RTF are both "x realtime" and are not the same measurement. The
/// renderer groups them separately for the same reason.
public enum RateBasis: String, Codable, Sendable {
    /// Audio in per second of wall time. Every transform case.
    case input
    /// Audio out per second of wall time. Synthesis.
    case output
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

    /// Which duration the real-time factor divides by. See ``RateBasis``.
    public let rateBasis: RateBasis

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
        rateBasis: RateBasis = .input,
        makeWorkload: @escaping @Sendable () -> BenchmarkWorkload
    ) {
        self.rateBasis = rateBasis
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
/// TTS was absent for a long time, and the objection was right: a text-to-speech
/// RTF has a different denominator, and folding it into a table of audio-to-audio
/// RTFs produces a column where two rows mean different things. Chatterbox is here
/// now because ``RateBasis`` answers that - each case names which duration its rate
/// divides by, and the renderer groups the categories separately - rather than
/// because the objection stopped applying. Kokoro is still absent.
public enum BenchmarkCatalog {

    public static func allCases(options: CatalogOptions = CatalogOptions()) -> [BenchmarkCase] {
        enhancement(options) + separation(options) + superResolution(options)
            + music(options) + universal(options) + tts(options)
    }

    // MARK: Text to speech

    /// The sentence every TTS case synthesizes.
    ///
    /// Fixed, and long enough that per-call overhead is not most of the
    /// measurement, but the absolute duration does not matter: the rate divides by
    /// what came out, so a case that generates more audio is not thereby faster.
    /// What does matter is that every precision synthesizes the *same* text -
    /// Chatterbox is autoregressive, so a longer sentence is more decoder steps,
    /// and comparing precisions on different text would compare sentence lengths.
    private static let ttsPrompt =
        "The quick brown fox jumps over the lazy dog, and then it does so again."

    /// Chatterbox at each published precision.
    ///
    /// This category is reported apart from the rest and its rate means something
    /// different: `RateBasis.output`, seconds generated per second of wall time,
    /// against `.input` everywhere else. Both render as "x realtime" and they are
    /// not comparable - a TTS row and an enhancement row in one sorted column
    /// would invite exactly the comparison this split exists to prevent.
    ///
    /// Quality is deliberately not here. Everything downstream of conditioning
    /// samples, so two precisions do not produce comparable waveforms even with a
    /// fixed seed; what a precision costs is measured on the conditioning
    /// embeddings instead. See `Scripts/quantization-report.py`.
    ///
    /// **Read the cross-precision rates with care.** Within one precision the
    /// measurement is tight - standard deviation of 0.02 s over three runs. Across
    /// precisions it is not controlled: sampling means each one draws a different
    /// token sequence for the same sentence, so they generate different amounts of
    /// audio - measured 3.18 s at 4bit against 3.95 s at 6bit. Dividing by the
    /// output duration removes most of that, but not the part where per-token cost
    /// differs between prefill and decode, and the ratio between the two moves with
    /// length. A gap of a few percent here is not a result; the 4bit-to-6bit gap of
    /// ~20% is larger than that and survives, which is worth knowing but is still
    /// one sentence on one machine.
    private static func tts(_ options: CatalogOptions) -> [BenchmarkCase] {
        let precisions: [ModelPrecision] = [.fp32, .fp16, .bit8, .bit6, .bit4]

        return precisions.map { precision in
            let repo = ChatterboxTTSProvider.repository(for: precision)
            let files = ModelFiles.chatterboxRequired(for: precision)

            return BenchmarkCase(
                id: "mlx.chatterbox.\(precision.rawValue)",
                label: "Chatterbox TTS (\(precision.rawValue.uppercased()))",
                category: "tts",
                backend: "mlx",
                // The model's own output rate. Nothing resamples the result.
                sampleRate: 24000,
                estimatedMemoryBytes: ChatterboxTTSProvider.estimatedMemoryBytes(for: precision),
                weightsPreCached: { cached(repo, files) },
                weightsSource: { source(local: nil, repo: repo, files: files) },
                rateBasis: .output,
                makeWorkload: {
                    let tts = ChatterboxTTSProvider(precision: precision)
                    return synthesis(
                        load: { try await tts.load() },
                        generate: { try await tts.synthesize(ttsPrompt, voice: "default") },
                        unload: { await tts.unload() }
                    )
                }
            )
        }
    }

    /// Inputs the catalog needs that cannot be discovered.
    public struct CatalogOptions: Sendable {
        /// An `.mlpackage` to measure instead of the published FP32 one.
        ///
        /// An override rather than a requirement since the packages were published:
        /// the case downloads without it. Falls back to the research checkout's copy
        /// when one is present, which keeps a development machine off the network.
        public var coreMLGANModelPath: String?

        /// The FP16 conversion, when the checkout has one beside the fp32 file.
        /// Not settable from `--coreml-gan`, which names one file.
        public var coreMLGANFP16ModelPath: String?

        /// The local override for one conversion, if there is one.
        public func coreMLGANPath(for precision: CoreMLGANPrecision) -> String? {
            switch precision {
            case .fp32: coreMLGANModelPath
            case .fp16: coreMLGANFP16ModelPath
            }
        }

        /// Weights already on this machine. See ``LocalWeights``.
        public var localWeights: LocalWeights

        public init(
            coreMLGANModelPath: String? = nil,
            localWeights: LocalWeights = LocalWeights(root: LocalWeights.autodetect())
        ) {
            self.localWeights = localWeights
            self.coreMLGANModelPath = coreMLGANModelPath ?? localWeights.mossFormerGANCoreML
            self.coreMLGANFP16ModelPath = localWeights.mossFormerGANCoreMLFP16
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

        // Every precision the repository publishes, which is also
        // `MossFormer2SE48KProvider.supportedPrecisions`. The quantized widths are
        // here to be compared against fp32 rather than merely to run: this is the
        // only model in the catalog offering a precision ladder, so it is the only
        // one that can answer what a user gives up by picking a smaller one.
        for precision in MossFormer2SE48KProvider.supportedPrecisions {
            // Only FP32 exists in the checkout; everything else resolves through
            // HuggingFace.
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

        // One case per compiled `.mlpackage`. CoreML fixes precision at conversion
        // time rather than at load, so fp32 and fp16 are two files and cannot be one
        // case with a parameter - which also means the comparison here is between two
        // conversions of one graph, not between two ways of running it.
        //
        // Both run unconditionally now that the packages are published. They used to
        // be gated on `--coreml-gan`, which names one file: the whole case was
        // skipped without it, and fp16 ran only when its package happened to sit
        // beside the fp32 one in a checkout. That made the recommended precision the
        // one most likely to go unmeasured.
        for precision in MossFormerGANCoreMLProvider.supportedPrecisions {
            let local = options.coreMLGANPath(for: precision)
            let files = ModelFiles.mossFormerGANCoreMLRequired(precision)
            cases.append(BenchmarkCase(
                id: "coreml.mossformer_gan_se_16k.\(precision.rawValue)",
                label: "MossFormerGAN SE 16 kHz (CoreML \(precision.rawValue.uppercased()))",
                category: "enhancement",
                backend: "coreml",
                sampleRate: 16000,
                // Measured peaks, 30 s at 16 kHz: 1631 MiB for the fp32 conversion
                // and 302 MiB for fp16. Almost none of either is CoreML -
                // mlxPeakDuringRunBytes is 5 MiB - because the STFT, ISTFT and
                // segment stitching either side of the model run in MLX. Which is
                // also why one estimate cannot serve both: the gap between them is
                // the model, and it is 5.5x.
                estimatedMemoryBytes: precision == .fp32 ? 1_700_000_000 : 400_000_000,
                weightsPreCached: {
                    local != nil || cached(MossFormerGANCoreMLProvider.repo, files)
                },
                weightsSource: {
                    source(local: local, repo: MossFormerGANCoreMLProvider.repo, files: files)
                },
                makeWorkload: {
                    let provider = MossFormerGANCoreMLProvider(
                        modelPath: local,
                        precision: precision
                    )
                    return transform(
                        load: { try await provider.load() },
                        process: { try await provider.process($0) },
                        unload: { await provider.unload() }
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
        return MossFormer2SR48KProvider.supportedPrecisions.map { precision in
                BenchmarkCase(
                    id: "mlx.mossformer2_sr_48k.\(precision.rawValue)",
                    label: "MossFormer2 SR 16 -> 48 kHz (\(precision.rawValue.uppercased()))",
                    category: "superResolution",
                    backend: "mlx",
                    // Consumes 16 kHz and emits 48 kHz. Feeding it 48 kHz would be
                    // self-defeating - see the note on `sampleRate` in the provider -
                    // and would also make its RTF incomparable to its own docs.
                    sampleRate: 16000,
                    // Measured peak 4017 MiB at fp32, the largest of the enhancement
                    // models. The precisions barely differ: only 168 `Linear` modules
                    // quantize against a Generator of convolutions.
                    estimatedMemoryBytes: 4_200_000_000,
                    // The one case where `config.json` is not decoration: this provider
                    // reads its architecture out of it rather than hardcoding one, so
                    // both files have to be present for a load to succeed. The
                    // quantized checkpoint shares the fp32 config - the bit width comes
                    // off the filename.
                    weightsPreCached: {
                        cached(MossFormer2SR48KProvider.repo, ModelFiles.standard(precision))
                    },
                    weightsSource: {
                        source(
                            local: nil,
                            repo: MossFormer2SR48KProvider.repo,
                            files: ModelFiles.standard(precision)
                        )
                    },
                    makeWorkload: {
                        let provider = MossFormer2SR48KProvider(precision: precision)
                        return transform(
                            load: { try await provider.load() },
                            process: { try await provider.process($0) },
                            unload: { await provider.unload() }
                        )
                    }
                )
            }
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
    /// A workload that generates rather than transforms.
    ///
    /// The buffer the runner supplies is discarded on purpose - synthesis has no
    /// audio input, and the case declares `rateBasis: .output` so the real-time
    /// factor divides by what came out instead of what was ignored.
    private static func synthesis(
        load: @escaping @Sendable () async throws -> Void,
        generate: @escaping @Sendable () async throws -> AudioBuffer,
        unload: @escaping @Sendable () async -> Void
    ) -> BenchmarkWorkload {
        BenchmarkWorkload(
            load: load,
            run: { _ in
                let output = try await generate()
                return WorkloadOutput(
                    streams: 1,
                    frames: output.frameCount,
                    sampleRate: output.sampleRate
                )
            },
            unload: unload
        )
    }

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
