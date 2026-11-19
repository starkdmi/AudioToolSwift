//
//  VoiceIsolatorProvider.swift
//  AudioToolMLX
//
//  Inworld Voice Isolator (48 kHz): the offline checkpoint and the causal one
//

import Foundation
import AudioTool
import AudioToolCore
@preconcurrency import MLX
@preconcurrency import VoiceIsolatorMLX

// MARK: - Shared

/// What both providers share: a repository, a precision, and a checkpoint that
/// `VoiceIsolator` opens directly.
///
/// The two models are separately trained - one non-causal, one causal - and
/// `VoiceIsolator.init` rejects the wrong mode for either, so they cannot be one
/// provider with a flag. What they do share is every line of the loading path, and
/// that is what lives here.
enum VoiceIsolatorWeights {

    /// Explicit path if given, otherwise the cache, otherwise HuggingFace.
    ///
    /// Present-to-load is the weights alone. These repositories publish no
    /// `config.json` - the architecture is read from the checkpoint's own tensor
    /// shapes - so asking for one in the *download* manifest is harmless (an
    /// unmatched glob is not an error) while asking for one in the *presence* check
    /// would report every complete install as missing. Same split as FRCRN and the
    /// three SS repositories; see `ModelFiles.standardRequired`.
    static func resolve(
        path: String?,
        repo: String,
        precision: ModelPrecision
    ) async throws -> URL {
        if let path { return URL(fileURLWithPath: path) }

        if let cached = ModelDownloader.shared.localPath(
            for: repo,
            matching: ModelFiles.standardRequired(precision)
        ) {
            return cached.appendingPathComponent(precision.weightsFilename)
        }
        let directory = try await ModelDownloader.shared.downloadAndGetPath(
            repo: repo,
            matching: ModelFiles.standard(precision)
        )
        return directory.appendingPathComponent(precision.weightsFilename)
    }

    /// The compute precision to ask `VoiceIsolator` for: whatever is stored.
    ///
    /// `nil` rather than a value derived from ``ModelPrecision``, because the file
    /// already says it - `model_fp16.safetensors` holds float16 and
    /// `model_fp32.safetensors` float32, and `Network` follows the weights. The
    /// library's own default is `.float16`, which would load the fp32 checkpoint
    /// and then compute in half precision: a third configuration, slower than
    /// fp16 and less accurate than fp32, that nobody measured and no catalog
    /// entry names.
    static let storedDType: DType? = nil

    /// `input - voice`, the part the model removed.
    ///
    /// Time-domain subtraction, as `FRCRNSE16KProvider` does it: the isolator emits
    /// a waveform rather than a mask, so there is no spectral residual to take
    /// instead. `isolate` fits its output to the input length, so the two line up
    /// exactly and the pair sums back to the input.
    ///
    /// Sized to the shorter of the two rather than to `voice`, which is defensive
    /// about an invariant this file does not own. `fitLength` makes the lengths
    /// equal today; if a future version of the library stopped doing that, indexing
    /// `input` by `voice`'s count would trap on an out-of-range read inside an
    /// actor - the least diagnosable failure available. This way the mismatch
    /// surfaces as a short buffer, which
    /// `testOfflineBackgroundAndVoiceSumToTheInput` already checks for.
    static func background(_ input: [Float], voice: [Float]) -> [Float] {
        let count = min(input.count, voice.count)
        var residual = [Float](repeating: 0, count: count)
        for index in 0..<count {
            residual[index] = input[index] - voice[index]
        }
        return residual
    }
}

// MARK: - Progress

/// Bridges the library's synchronous progress closure to this package's
/// `ProgressCallback`, which is async.
///
/// `isolate` reports progress from inside a blocking call, so the callback cannot
/// be awaited where it is produced. An `AsyncStream` carries the fractions out to
/// a sibling task instead: its continuation is `Sendable`, so the sync closure may
/// capture it, and `bufferingNewest(1)` means a slow callback falls behind rather
/// than backing up - what it eventually sees is the latest fraction, and the final
/// one is never the one dropped.
///
/// Deliberately not a helper that takes the work as a closure. That shape reads
/// better and does not compile: the closure would capture the provider's isolator
/// and buffer, so handing it to a non-isolated function sends actor-isolated state
/// across an isolation boundary. Keeping the bridge inert - it starts a task and
/// hands back a closure, it never runs the work - lets the model call stay inside
/// the actor where it belongs.
struct VoiceIsolatorProgressBridge {

    /// Pass to `isolate(progress:)`. `nil` when there is no callback, so the
    /// library skips progress accounting entirely.
    let yield: (@Sendable (Double) -> Void)?

    private let continuation: AsyncStream<Double>.Continuation?
    private let reporter: Task<Void, Never>?

    init(_ onProgress: ProgressCallback?) {
        guard let onProgress else {
            self.yield = nil
            self.continuation = nil
            self.reporter = nil
            return
        }
        let (stream, continuation) = AsyncStream<Double>.makeStream(
            bufferingPolicy: .bufferingNewest(1)
        )
        self.continuation = continuation
        // The library reports 0...1; `ProgressCallback` is documented in percent.
        self.yield = { fraction in continuation.yield(fraction * 100.0) }
        self.reporter = Task {
            for await percent in stream { await onProgress(percent) }
        }
    }

    /// Deliver whatever is outstanding, then stop.
    ///
    /// Awaited before the result is returned, so no progress callback can arrive
    /// after the value it was describing.
    func finish() async {
        continuation?.finish()
        await reporter?.value
    }
}

// MARK: - Offline Provider

/// Inworld Voice Isolator, 48 kHz, offline.
///
/// Non-causal: every frame sees its whole 8-second window, and inputs longer than
/// one window render in windows crossfaded over 2 s. That windowing is the
/// library's, not this package's - which is why there is no `ChunkingConfig` here
/// and why `recommendedChunkSize` describes the model's window rather than a
/// stitching strategy this file implements.
///
/// Defaults to fp16, unlike every other provider in this package. It is what
/// `VoiceIsolator` computes in by default and what the port's parity figures were
/// measured at; fp32 is the reference and is 19% slower for 2.4e-4 of waveform
/// difference. See `Docs/model-precisions.md`.
public actor VoiceIsolator48KProvider: SpeechEnhancer {

    /// HuggingFace repository for model weights
    public static let repo = ModelRepository.voiceIsolator48K

    /// Supported precisions for this model.
    public static let supportedPrecisions: [ModelPrecision] = [.fp16, .fp32]

    public nonisolated let sampleRate: Int = 48000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1

    /// Band-limited, which matters more here than for most providers.
    ///
    /// The model scales energy that is present; it does not decide that some of
    /// that energy is a resampler's artifact. An interpolating upsample (``fast``,
    /// ``balanced``, and ``auto`` on its upsampling side) leaves spectral images at
    /// -25 to -32 dB above the source band, and 16 kHz material brought to 48 kHz
    /// that way arrives with image energy across the whole top two thirds of the
    /// spectrum. The isolator passes it through.
    ///
    /// ``high`` is the nearest band-limited option the platform offers. It is not
    /// the library's own filter: `isolate` resamples with torchaudio's
    /// `kaiser_best`, and the way to get that one is ``isolate(_:onProgress:)``
    /// below, which hands the model the buffer at its own rate and lets the library
    /// convert in and out with a matched filter. Prefer it when the input is not
    /// already 48 kHz.
    public nonisolated var preferredResamplingQuality: AudioToolCore.ResamplingQuality { .high }

    /// 1 s at 48 kHz. Shorter inputs are legal - the network takes any length -
    /// but a window this short is mostly analysis edge.
    public nonisolated let minChunkSize: Int = 48000

    /// 8 s at 48 kHz: the model's own render window, so a caller chunking at this
    /// size never splits a window.
    public nonisolated let recommendedChunkSize: Int = 384000

    private var isolator: VoiceIsolator?
    private let weightsPath: String?
    private let precision: ModelPrecision

    /// One load at a time; see ``ModelLoadGate``.
    private let loadGate = ModelLoadGate()

    /// Initialize with precision (auto-downloads from HuggingFace)
    public init(precision: ModelPrecision = .fp16) {
        self.weightsPath = nil
        self.precision = precision
    }

    /// Initialize with explicit weights path (no download)
    public init(weightsPath: String, precision: ModelPrecision = .fp16) {
        self.weightsPath = weightsPath
        self.precision = precision
    }

    /// Load model weights (downloads if not cached)
    ///
    /// Concurrent calls share one load; see ``ModelLoadGate``.
    public func load() async throws {
        try await loadGate.run { [self] in try await performLoad() }
    }

    private func performLoad() async throws {
        // Before the first allocation, not at the first window boundary - see
        // `MLXCachePolicy`.
        MLXCachePolicy.applyProcessLimits()

        let url = try await VoiceIsolatorWeights.resolve(
            path: weightsPath,
            repo: Self.repo,
            precision: precision
        )
        let candidate = try VoiceIsolator(
            contentsOf: url,
            mode: .offline,
            dtype: VoiceIsolatorWeights.storedDType
        )
        try Task.checkCancellation()
        isolator = candidate
    }

    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        try await process(input, onProgress: nil)
    }

    /// Enhance 48 kHz audio, reporting progress after every render window.
    public func process(
        _ input: AudioBuffer,
        onProgress: ProgressCallback?
    ) async throws -> AudioBuffer {
        guard let isolator else {
            throw AudioToolError.modelNotLoaded("VoiceIsolator_48K")
        }
        try validateSampleRate(input)

        let voice = try await isolateSamples(
            isolator,
            input.samples,
            sampleRate: sampleRate,
            onProgress: onProgress
        )
        return AudioBuffer(samples: voice, sampleRate: sampleRate, channels: 1)
    }

    /// One `isolate` call, with progress bridged and the cache trimmed after.
    ///
    /// Isolated to this actor, which is the point: `isolate` is synchronous and
    /// touches the network's mutable state, so it must not be handed to anything
    /// that could run it elsewhere.
    ///
    /// Trimmed *after the call*, not per window, which is the opposite of what
    /// `MossFormer2SE48KProvider` and `FRCRNSE16KProvider` do with
    /// `MLXCachePolicy.trimIfNeeded(afterChunk:)`. Two reasons, and the second is
    /// the one that matters:
    ///
    /// - The window loop is inside `enhance()` in VoiceIsolatorMLX, so this
    ///   package has no per-window hook. The only candidate is the `progress`
    ///   closure, which `VoiceIsolatorProgressBridge` deliberately makes
    ///   asynchronous - so it does not run between windows anyway.
    /// - There is nothing to reclaim. Every window is `[1, chunk]`, the same
    ///   shape each time, so the blocks one window frees are exactly what the
    ///   next one asks for and the allocator reuses them in place. Measured:
    ///   MLX's activation high-water is 503, 545 and 555 MiB for 8, 30 and 60 s
    ///   of input - a 7.5x longer render for 10% more memory. A per-window
    ///   `GPU.clearCache()` would return those blocks to the driver and buy them
    ///   back immediately, every window, for no reduction in peak: the peak is
    ///   live activations inside one forward pass, which no cache policy can
    ///   release. `MLXCachePolicy` records the same finding from a 24x sweep of
    ///   the cache cap on the other models here.
    ///
    /// Read those figures to no better than ~10%: two runs of the identical
    /// configuration measured 545 and 610 MiB.
    private func isolateSamples(
        _ isolator: VoiceIsolator,
        _ samples: [Float],
        sampleRate: Int,
        onProgress: ProgressCallback?
    ) async throws -> [Float] {
        let bridge = VoiceIsolatorProgressBridge(onProgress)
        do {
            let voice = try isolator.isolate(
                samples,
                sampleRate: sampleRate,
                progress: bridge.yield
            )
            await bridge.finish()
            MLXCachePolicy.trimIfCacheGrew()
            return voice
        } catch {
            await bridge.finish()
            MLXCachePolicy.trimIfCacheGrew()
            throw error
        }
    }

    /// Enhance audio at **its own** sample rate, converting with the model's filter.
    ///
    /// The one place in this package where a provider resamples rather than
    /// validating, and it is deliberate. The general rule exists because a chain of
    /// stages each silently converting compounds into round trips - 48 to 16 to 48
    /// to 16 - and because the cost is then hidden from the caller. Neither applies
    /// to a single explicit call whose entire purpose is the conversion: `isolate`
    /// resamples to 48 kHz and back with torchaudio's `kaiser_best` sinc filter, so
    /// the trip in and the trip out use the same band-limited filter, which is
    /// better than anything the facade's edge conversion can do for this model and
    /// is what the port was validated against.
    ///
    /// ``process(_:)`` stays strict, so the facade, the pipeline and
    /// ``AudioEngine/enhance(_:model:preservingSampleRate:)`` behave exactly as
    /// they do for every other provider. Reach for this one directly when the input
    /// is not already 48 kHz and nothing downstream needs it at 48 kHz either.
    ///
    /// - Returns: Isolated voice at `input.sampleRate`, the same length as `input`.
    public func isolate(
        _ input: AudioBuffer,
        onProgress: ProgressCallback? = nil
    ) async throws -> AudioBuffer {
        guard let isolator else {
            throw AudioToolError.modelNotLoaded("VoiceIsolator_48K")
        }
        try validateInputChannels(input)

        let voice = try await isolateSamples(
            isolator,
            input.samples,
            sampleRate: input.sampleRate,
            onProgress: onProgress
        )
        return AudioBuffer(samples: voice, sampleRate: input.sampleRate, channels: 1)
    }

    public nonisolated func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                for await chunk in input {
                    do {
                        try Task.checkCancellation()
                        let processed = try await process(chunk)
                        if case .terminated = continuation.yield(processed) { return }
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
        }
    }

    public func reset() async {}

    // MARK: - Background Extraction

    /// Isolate the voice and keep what was removed.
    ///
    /// - Returns: The voice, and `input - voice`. The two sum back to the input
    ///   sample for sample.
    public func processWithBackground(
        _ input: AudioBuffer,
        onProgress: ProgressCallback? = nil
    ) async throws -> MLXEnhancedWithBackground {
        let enhanced = try await process(input, onProgress: onProgress)
        return MLXEnhancedWithBackground(
            enhanced: enhanced,
            background: AudioBuffer(
                samples: VoiceIsolatorWeights.background(input.samples, voice: enhanced.samples),
                sampleRate: sampleRate,
                channels: 1
            )
        )
    }
}

// MARK: - ManagedModel Conformance

extension VoiceIsolator48KProvider: ManagedModel {
    public nonisolated var modelId: String { "voice_isolator_48k" }

    /// Measured peak, 30 s at 48 kHz on an M1 Pro (16 GB): 1094 MiB at fp16,
    /// 1472 MiB at fp32.
    ///
    /// Flat in input length, as the library's windowing implies and a check
    /// confirms: at 60 s the same two cases peak at 1141 MiB and the RTF does not
    /// move. The render window is 8 s and `Network.maxTokens` bounds activations
    /// within a window, so a ten-minute file peaks where a ten-second one does.
    public nonisolated var estimatedMemoryBytes: Int {
        precision == .fp32 ? 1_550_000_000 : 1_150_000_000
    }

    public func checkIfLoaded() async -> Bool { isolator != nil }

    public func unload() async {
        // Bracketed, so a load in flight cannot publish a model into the gap and
        // have it wiped by the lines below.
        let teardown = await loadGate.beginTeardown()
        defer { loadGate.endTeardown(teardown) }
        isolator = nil
        GPU.clearCache()
    }
}

// MARK: - Realtime Provider

/// Inworld Voice Isolator, 48 kHz, causal.
///
/// A separately trained checkpoint, not the offline one run differently. It keeps
/// KV-cached attention, carried Mamba state and a windowed two-lane crossfade
/// across pushes, so a live session is genuinely incremental rather than a
/// sequence of independent chunks.
///
/// Three entry points, and the difference between them is what carries state:
///
/// | call | session | use |
/// | --- | --- | --- |
/// | ``process(_:)`` | one, internal, flushed | a whole buffer, no streaming |
/// | ``processStream(_:)`` | one, across the yields | a file, played as it renders |
/// | ``stream(_:)`` | one, across the inputs | live audio |
///
/// ``stream(_:)`` is the one that distinguishes this provider from the offline
/// one. Every other `SpeechEnhancer` here implements it by calling `process` per
/// chunk, which restarts the model at each buffer; this one pushes into a single
/// `IsolationStream`, so there is no boundary artifact between buffers and the
/// cost is one session's worth of state rather than one per chunk.
public actor VoiceIsolatorRealtime48KProvider: SpeechEnhancer {

    /// HuggingFace repository for model weights
    public static let repo = ModelRepository.voiceIsolatorRealtime48K

    /// Supported precisions for this model.
    public static let supportedPrecisions: [ModelPrecision] = [.fp16, .fp32]

    /// Look-ahead positions the checkpoint was trained at, in frames.
    ///
    /// One frame is `hopLength` samples - 480 at 48 kHz, so 10 ms. These are not a
    /// dial: the network has separate trained behaviour at each position, and
    /// `VoiceIsolator` rejects anything else.
    public static let supportedLookaheadFrames: [Int] = [0, 2, 4]

    public nonisolated let sampleRate: Int = 48000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1

    /// Band-limited, for the reason given on ``VoiceIsolator48KProvider``.
    public nonisolated var preferredResamplingQuality: AudioToolCore.ResamplingQuality { .high }

    /// One block at the configured block size. Pushes may be any size - the
    /// session buffers what it cannot yet finalize - so this is the smallest push
    /// that can produce output rather than a hard floor.
    public nonisolated let minChunkSize: Int = 480 * 4

    /// Half a second at 48 kHz, which is the piece size `VoiceIsolator` itself
    /// uses when it drives a session over a whole clip.
    public nonisolated let recommendedChunkSize: Int = 24000

    private var isolator: VoiceIsolator?

    /// The live session for ``stream(_:)``, held across input buffers.
    ///
    /// `IsolationStream` is not reentrant and neither is the `Network` behind it,
    /// so there is exactly one, and actor isolation is what enforces that. A second
    /// concurrent `stream(_:)` would otherwise interleave pushes into one session
    /// and get both callers' audio mixed into both results.
    private var session: IsolationStream?

    private let weightsPath: String?
    private let precision: ModelPrecision
    private let lookaheadFrames: Int?
    private let blockFrames: Int

    /// One load at a time; see ``ModelLoadGate``.
    private let loadGate = ModelLoadGate()

    /// Initialize with precision (auto-downloads from HuggingFace)
    ///
    /// - Parameters:
    ///   - precision: Checkpoint to load. fp16 by default, as upstream.
    ///   - lookaheadFrames: One of ``supportedLookaheadFrames``, or `nil` for the
    ///     checkpoint's own default. More look-ahead is more latency and better
    ///     separation; see ``latency``.
    ///   - blockFrames: Frames per forward pass in a live session. Smaller is
    ///     lower latency per push and more dispatches per second of audio; the
    ///     output does not depend on it.
    public init(
        precision: ModelPrecision = .fp16,
        lookaheadFrames: Int? = nil,
        blockFrames: Int = 4
    ) {
        self.weightsPath = nil
        self.precision = precision
        self.lookaheadFrames = lookaheadFrames
        self.blockFrames = blockFrames
    }

    /// Initialize with explicit weights path (no download)
    public init(
        weightsPath: String,
        precision: ModelPrecision = .fp16,
        lookaheadFrames: Int? = nil,
        blockFrames: Int = 4
    ) {
        self.weightsPath = weightsPath
        self.precision = precision
        self.lookaheadFrames = lookaheadFrames
        self.blockFrames = blockFrames
    }

    /// Load model weights (downloads if not cached)
    ///
    /// Concurrent calls share one load; see ``ModelLoadGate``.
    public func load() async throws {
        try await loadGate.run { [self] in try await performLoad() }
    }

    private func performLoad() async throws {
        MLXCachePolicy.applyProcessLimits()

        let url = try await VoiceIsolatorWeights.resolve(
            path: weightsPath,
            repo: Self.repo,
            precision: precision
        )
        let candidate = try VoiceIsolator(
            contentsOf: url,
            mode: .streaming,
            dtype: VoiceIsolatorWeights.storedDType,
            lookaheadFrames: lookaheadFrames
        )
        candidate.blockFrames = blockFrames
        try Task.checkCancellation()
        isolator = candidate
    }

    /// Algorithmic latency of a live session, in samples at ``sampleRate``: one
    /// analysis window plus the look-ahead.
    ///
    /// Throws rather than returning zero when the model is not loaded, because the
    /// figure depends on the checkpoint's own `nFFT` and hop and a caller sizing a
    /// playout buffer from a placeholder would underrun.
    public func latencySamples() throws -> Int {
        guard let isolator else {
            throw AudioToolError.modelNotLoaded("VoiceIsolator_Realtime_48K")
        }
        return isolator.latencySamples
    }

    /// ``latencySamples()`` as a duration.
    public func latency() throws -> Duration {
        .seconds(Double(try latencySamples()) / Double(sampleRate))
    }

    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        try await process(input, onProgress: nil)
    }

    /// Run a whole buffer through a session of its own, flushed at the end.
    ///
    /// Equivalent in output to pushing the same samples through ``stream(_:)`` in
    /// any piece sizes and concatenating, which is the property that makes the
    /// realtime model testable against itself.
    public func process(
        _ input: AudioBuffer,
        onProgress: ProgressCallback?
    ) async throws -> AudioBuffer {
        guard let isolator else {
            throw AudioToolError.modelNotLoaded("VoiceIsolator_Realtime_48K")
        }
        try validateSampleRate(input)

        let voice = try await isolateSamples(
            isolator,
            input.samples,
            sampleRate: sampleRate,
            onProgress: onProgress
        )
        return AudioBuffer(samples: voice, sampleRate: sampleRate, channels: 1)
    }

    /// One `isolate` call, with progress bridged and the cache trimmed after.
    ///
    /// Isolated to this actor; see the note on
    /// ``VoiceIsolator48KProvider``'s equivalent.
    private func isolateSamples(
        _ isolator: VoiceIsolator,
        _ samples: [Float],
        sampleRate: Int,
        onProgress: ProgressCallback?
    ) async throws -> [Float] {
        let bridge = VoiceIsolatorProgressBridge(onProgress)
        do {
            let voice = try isolator.isolate(
                samples,
                sampleRate: sampleRate,
                progress: bridge.yield
            )
            await bridge.finish()
            MLXCachePolicy.trimIfCacheGrew()
            return voice
        } catch {
            await bridge.finish()
            MLXCachePolicy.trimIfCacheGrew()
            throw error
        }
    }

    /// Enhance audio at **its own** sample rate; see
    /// ``VoiceIsolator48KProvider/isolate(_:onProgress:)``.
    public func isolate(
        _ input: AudioBuffer,
        onProgress: ProgressCallback? = nil
    ) async throws -> AudioBuffer {
        guard let isolator else {
            throw AudioToolError.modelNotLoaded("VoiceIsolator_Realtime_48K")
        }
        try validateInputChannels(input)

        let voice = try await isolateSamples(
            isolator,
            input.samples,
            sampleRate: input.sampleRate,
            onProgress: onProgress
        )
        return AudioBuffer(samples: voice, sampleRate: input.sampleRate, channels: 1)
    }

    // MARK: - Live Session

    /// Start a session, discarding any in progress.
    private func beginSession() throws {
        guard let isolator else {
            throw AudioToolError.modelNotLoaded("VoiceIsolator_Realtime_48K")
        }
        session = try isolator.stream()
    }

    /// Push one buffer and return the samples that became final.
    ///
    /// May be empty: the first pushes of a session fill the analysis window and
    /// the look-ahead before anything can be finalized.
    private func pushSession(_ input: AudioBuffer) throws -> [Float] {
        guard let session else {
            throw AudioToolError.modelNotLoaded("VoiceIsolator_Realtime_48K")
        }
        try validateSampleRate(input)
        return try session.push(input.samples)
    }

    /// Drain the session's tail and end it.
    private func endSession() throws -> [Float] {
        guard let session else { return [] }
        self.session = nil
        defer { MLXCachePolicy.trimIfCacheGrew() }
        return try session.flush()
    }

    /// Enhance live audio, carrying model state across the input buffers.
    ///
    /// The yielded buffers are the samples that have become final, so their
    /// lengths do not match the input's: the first few pushes yield nothing while
    /// the analysis window and look-ahead fill, and the tail arrives in a final
    /// buffer when the input stream ends. Concatenated, they are the whole
    /// enhanced signal, ``latency()`` behind the input.
    ///
    /// One session at a time per provider. A second `stream(_:)` started while one
    /// is running takes the session over, and the first then finishes early rather
    /// than interleaving - make a second provider instead.
    public nonisolated func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    try await beginSession()
                    for await chunk in input {
                        try Task.checkCancellation()
                        let ready = try await pushSession(chunk)
                        guard !ready.isEmpty else { continue }
                        let buffer = AudioBuffer(
                            samples: ready,
                            sampleRate: sampleRate,
                            channels: 1
                        )
                        if case .terminated = continuation.yield(buffer) {
                            _ = try? await endSession()
                            return
                        }
                    }
                    let tail = try await endSession()
                    if !tail.isEmpty {
                        continuation.yield(
                            AudioBuffer(samples: tail, sampleRate: sampleRate, channels: 1)
                        )
                    }
                    continuation.finish()
                } catch {
                    _ = try? await endSession()
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
        }
    }

    public func reset() async {
        session = nil
        MLXCachePolicy.trimIfCacheGrew()
    }

    // MARK: - Background Extraction

    /// Isolate the voice and keep what was removed; see
    /// ``VoiceIsolator48KProvider/processWithBackground(_:onProgress:)``.
    public func processWithBackground(
        _ input: AudioBuffer,
        onProgress: ProgressCallback? = nil
    ) async throws -> MLXEnhancedWithBackground {
        let enhanced = try await process(input, onProgress: onProgress)
        return MLXEnhancedWithBackground(
            enhanced: enhanced,
            background: AudioBuffer(
                samples: VoiceIsolatorWeights.background(input.samples, voice: enhanced.samples),
                sampleRate: sampleRate,
                channels: 1
            )
        )
    }
}

// MARK: - StreamableOutput Conformance

extension VoiceIsolatorRealtime48KProvider: StreamableOutput {
    /// Render a complete buffer through one session, yielding each piece as it is
    /// finalized.
    ///
    /// For playing a file back while it renders. The session spans the whole call,
    /// so the result is identical to ``process(_:)`` on the same input - unlike
    /// the `processStream` of the chunked providers here, where the yields are
    /// independent chunks with their edges discarded.
    public nonisolated func processStream(_ input: AudioBuffer) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    try await self.processStreamImpl(input, continuation: continuation)
                } catch {
                    _ = try? await self.endSession()
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
        }
    }

    private func processStreamImpl(
        _ input: AudioBuffer,
        continuation: AsyncThrowingStream<AudioBuffer, Error>.Continuation
    ) async throws {
        guard let isolator else {
            throw AudioToolError.modelNotLoaded("VoiceIsolator_Realtime_48K")
        }
        try validateSampleRate(input)

        // Larger blocks than a live session uses. The output does not depend on the
        // block size, and nothing here is waiting on a microphone, so the extra
        // frames per dispatch are free throughput.
        session = try isolator.stream(
            blockFrames: max(blockFrames, VoiceIsolator.renderBlockFrames)
        )

        let piece = Int(VoiceIsolator.streamPieceSeconds * Double(sampleRate))
        var index = 0
        while index < input.samples.count {
            try Task.checkCancellation()
            let end = min(input.samples.count, index + piece)
            let ready = try session?.push(Array(input.samples[index..<end])) ?? []
            index = end

            if !ready.isEmpty {
                let buffer = AudioBuffer(samples: ready, sampleRate: sampleRate, channels: 1)
                if case .terminated = continuation.yield(buffer) {
                    _ = try? endSession()
                    throw CancellationError()
                }
            }
        }

        let tail = try endSession()
        if !tail.isEmpty {
            continuation.yield(AudioBuffer(samples: tail, sampleRate: sampleRate, channels: 1))
        }
        continuation.finish()
    }
}

// MARK: - ManagedModel Conformance

extension VoiceIsolatorRealtime48KProvider: ManagedModel {
    public nonisolated var modelId: String { "voice_isolator_realtime_48k" }

    /// Measured peak, 30 s at 48 kHz on an M1 Pro (16 GB): 1700 MiB at fp16,
    /// 2614 MiB at fp32.
    ///
    /// **Higher than the offline model's, not lower** - 1.6x at fp16 - which is
    /// the opposite of what "causal, for live audio" suggests and the reason these
    /// numbers are measured rather than reasoned about.
    ///
    /// It is the reference's two-lane windowed crossfade, and it is inherent
    /// rather than a leak. Every `windowFrames` (600, so 6 s) the session clones
    /// its per-layer states and runs the *whole network at `batch: 2`* for the
    /// `crossfadeFrames` (200, so 2 s) that follow, holding two attention KV
    /// caches at once, before swapping to the fresh lane and dropping the old
    /// one. Double the forward pass for a quarter of the time, plus a second
    /// cache, is 2.26x the offline model's MLX activations measured over the same
    /// 8-second window: 1136 MiB against 503.
    ///
    /// Bounded, which is what matters for a live session: the crossfade is
    /// periodic and the retired lane is released at each boundary, so at 60 s the
    /// fp16 case peaks at 1721 MiB against 1700 at 30 s.
    ///
    /// `IsolationStream` does expose `windowFrames: 0`, which skips the second
    /// lane entirely and would roughly halve this. It is deliberately not
    /// surfaced here: the windowed crossfade is part of what makes the stream
    /// match the reference, and trading that for memory is not a default this
    /// package should pick on a caller's behalf.
    public nonisolated var estimatedMemoryBytes: Int {
        precision == .fp32 ? 2_750_000_000 : 1_800_000_000
    }

    public func checkIfLoaded() async -> Bool { isolator != nil }

    public func unload() async {
        let teardown = await loadGate.beginTeardown()
        defer { loadGate.endTeardown(teardown) }
        session = nil
        isolator = nil
        GPU.clearCache()
    }
}
