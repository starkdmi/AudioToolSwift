//
//  MLXSuperResolutionProvider.swift
//  AudioToolMLX
//
//  MLX-based super resolution provider with chunking support
//

import Foundation
import AudioTool
import AudioToolCore
import MLX
import MLXNN
import AudioUtils
import MossFormer2SR
// For `QuantizationParameters`, which describes how a MossFormer2 checkpoint was
// quantized. It lives beside the SE pipeline because that is where it was first
// needed; the concept is not SE-specific and the two providers share this file's
// notion of a checkpoint. Duplicating it so this target could avoid one import
// would let the two copies disagree about a group size.
@preconcurrency import Mossformer2MLXSwift

// MARK: - MossFormer2 SR 48K Provider (Super Resolution)

/// MLX MossFormer2 Super Resolution - upsamples audio to 48kHz
/// Chunking: 4s chunks, 25% overlap, discard-edges - the reference's numbers.
public actor MossFormer2SR48KProvider: AudioUpscaler {
    
    /// HuggingFace repository for model weights
    public static let repo = ModelRepository.mossFormer2SR48K
    
    /// Supported precisions.
    ///
    /// fp32 and int8 are published; the other three conversions exist only locally,
    /// and the exclusions below say why each stays that way. The measurements are
    /// kept in `MODEL-PRECISIONS.md` regardless, because the answer they produce is
    /// worth having written down: on this model quantization is close to pointless.
    /// Only 168 `Linear` modules quantize against a Generator that is almost entirely
    /// convolutions, so int4 is 0.65x the fp32 size where SE reaches 0.30x, and fp16
    /// at 0.50x is smaller than every quantized variant.
    ///
    /// **fp16 is excluded because it does not work.** The fp16 conversion returns
    /// NaN for every sample - 143872 of 143872 - while its checkpoint contains no
    /// non-finite weight, so it is the forward pass that overflows and not the
    /// cast. The integer widths are unaffected: they pack only the `Linear` weights
    /// and every activation stays fp32. Nothing here reports that at run time, which
    /// is why it is a supported-precision decision rather than a caller's problem.
    ///
    /// **int6 and int4 are excluded because they are not published.** Both were
    /// measured - the numbers are in `MODEL-PRECISIONS.md` - and both are strictly
    /// dominated by int8, which is 11 MiB and 21 MiB larger for 12 dB and 24 dB
    /// better at identical speed and memory. Since quantization changes nothing but
    /// download size on this model, a width that is bigger *and* worse than int8 has
    /// no use, so neither was uploaded. Listing them here would name a repository
    /// path that 404s for anyone without a local conversion.
    public static let supportedPrecisions: [ModelPrecision] = [.fp32, .int8]
    
    // AudioUpscaler conformance
    public nonisolated var inputSampleRate: Int { 16000 }
    public nonisolated var outputSampleRate: Int { 48000 }
    
    // AudioProcessor conformance.
    //
    // sampleRate is the rate this provider *consumes*, so it is 16 kHz - the same as
    // inputSampleRate. It used to report 48000, which made the facade upsample input
    // to 48 kHz before handing it over, only for the provider to be asked to
    // reconstruct detail that upsampling cannot restore. Feeding a super-resolution
    // model its own output rate is self-defeating; it wants the real 16 kHz signal.
    public nonisolated var sampleRate: Int { inputSampleRate }
    public nonisolated let inputChannels: Int = 1

    /// Matches both references: the Python path resamples with `librosa.resample`
    /// (`Models/python/mossformer2_sr_mlx/generate.py:27`) and the standalone Swift
    /// generator asks AVAudioConverter for Mastering at maximum quality
    /// (`Models/mossformer2_sr_mlx_swift/Sources/Generate/main.swift:29`), which is
    /// exactly what `.high` now does.
    public nonisolated var preferredResamplingQuality: AudioToolCore.ResamplingQuality { .high }
    public nonisolated let outputChannels: Int = 1
    
    private var model: MossFormer2_SR_48K?
    private var args: AttrDict = AttrDict()

    /// One load at a time; see ``ModelLoadGate``.
    private let loadGate = ModelLoadGate()
    private let weightsPath: String?
    private let configPath: String?
    private let precision: ModelPrecision
    
    /// Max audio processed in one pass, in seconds.
    ///
    /// The reference's equivalent is `one_time_decode_length = 20.0`, and this
    /// deliberately does not match it. Measured on an M1 Pro, 16 GB, with the
    /// process MLX caps this package applies (3 GB cache, 8.8 GB memory):
    ///
    /// | direct input | RTF  | peak footprint |
    /// | ------------ | ---- | -------------- |
    /// |  6 s         | 2.6x | 5.6 GB         |
    /// | 10 s         | 2.6x | 8.2 GB         |
    /// | 12 s         | 2.7x | 8.8 GB         |
    /// | 19 s         | 2.7x | 8.9 GB         |
    ///
    /// The direct path's throughput is flat - all of its advantage is simply not
    /// chunking - while its peak grows about linearly and reaches the memory limit
    /// by twelve seconds. At the reference's twenty it is past it, and on this
    /// machine that means swapping rather than failing, which is worse.
    ///
    /// So the divergence here is a memory decision and is not the same kind of
    /// thing as the chunking strategy that used to sit alongside it: 50% Hann
    /// overlap-add differed from the reference for no recorded reason and cost
    /// throughput, and has been corrected. This costs throughput to stay inside a
    /// budget, on purpose.
    ///
    /// Chunked processing holds a flat 4.9 GB at any length, so the trade is
    /// roughly 2.7x at up to `maxDirectDuration` against 1.4x and bounded memory
    /// beyond it. Raising this is safe on a machine with more headroom; it wants
    /// to become a parameter rather than a constant before anyone does that.
    private let maxDirectDuration: Float = 4.0
    
    /// Initialize with precision (auto-downloads from HuggingFace)
    public init(precision: ModelPrecision = .fp32) {
        self.weightsPath = nil
        self.configPath = nil
        self.precision = precision
    }
    
    /// Initialize with explicit weights path (no download)
    ///
    /// `precision` is not taken here: the checkpoint carries it. `load()` reads the
    /// quantization off the file - a sibling `config.json` if it names one, the
    /// `model_int<n>` filename otherwise - so pointing this at
    /// `model_int8.safetensors` is how you ask for int8 without a download.
    public init(weightsPath: String, configPath: String) {
        self.weightsPath = weightsPath
        self.configPath = configPath
        self.precision = .fp32
    }
    
    /// Load model with config and weights (downloads if not cached)
    ///
    /// Concurrent calls share one load; see ``ModelLoadGate``.
    public func load() async throws {
        try await loadGate.run { [self] in try await performLoad() }
    }

    private func performLoad() async throws {
        // Before the first allocation, not at the first chunk boundary.
        // `trimIfNeeded` used to be the only thing that applied these, so a
        // provider ran its load and its opening chunks under MLX's own default
        // ceiling and the cache had already grown past the cap by the time the
        // cap arrived. Measured on Demucs: 5.1 GB peak applying it late against
        // 3.2 GB applying it here.
        MLXCachePolicy.applyProcessLimits()
        let resolvedWeightsPath: String
        let resolvedConfigPath: String
        
        if let wPath = weightsPath, let cPath = configPath {
            resolvedWeightsPath = wPath
            resolvedConfigPath = cPath
        } else {
            let requiredFiles = ModelFiles.standard(precision)
            // Check if already downloaded
            if let cached = ModelDownloader.shared.localPath(
                for: Self.repo,
                matching: requiredFiles
            ) {
                resolvedWeightsPath = cached
                    .appendingPathComponent(precision.weightsFilename).path
                resolvedConfigPath = cached.appendingPathComponent("config.json").path
            } else {
                // Auto-download from HuggingFace
                let modelDir = try await ModelDownloader.shared.downloadAndGetPath(
                    repo: Self.repo,
                    matching: requiredFiles
                )
                resolvedWeightsPath = modelDir.appendingPathComponent(precision.weightsFilename).path
                resolvedConfigPath = modelDir.appendingPathComponent("config.json").path
            }
        }
        
        let configData = try Data(contentsOf: URL(fileURLWithPath: resolvedConfigPath))
        let configObject = try JSONSerialization.jsonObject(with: configData)
        guard let modelConfig = configObject as? [String: Any] else {
            throw AudioToolError.incompatibleModelVersion(
                expected: "a JSON object at the config root",
                found: String(describing: type(of: configObject))
            )
        }
        
        let candidateArgs = AttrDict(modelConfig)
        candidateArgs["one_time_decode_length"] = 20.0
        candidateArgs["decode_window"] = 4.0
        
        let candidate = MossFormer2_SR_48K(args: candidateArgs)
        
        let weights = try MLX.loadArrays(url: URL(fileURLWithPath: resolvedWeightsPath))
        let filteredWeights = weights.filter { !$0.key.contains("num_batches_tracked") }
        let parameters = ModuleParameters.unflattened(filteredWeights)

        // Quantized checkpoints need their modules replaced before the parameters
        // arrive - see `QuantizationParameters`.
        //
        // Filtered on the checkpoint rather than quantized wholesale, unlike the SE
        // pipeline, because the two ports disagree about one module: Python's
        // `UniDeepFsmn.__init__` returns early when `lorder is None` and builds no
        // `Linear` at all, while the Swift initialiser constructs a dummy
        // `Linear(1, 1)` on that branch. An unconditional `quantize` would therefore
        // quantize a layer on this side that has no counterpart in the checkpoint.
        // Asking the checkpoint which modules it holds `scales` for makes the two
        // sets agree by construction - the same predicate
        // `ChatterboxTTSProvider.updateModule` uses.
        if let quantization = QuantizationParameters.resolve(forWeightsAt: resolvedWeightsPath) {
            let quantizedPaths = Set(
                filteredWeights.keys.compactMap { key -> String? in
                    key.hasSuffix(".scales") ? String(key.dropLast(".scales".count)) : nil
                }
            )
            quantize(
                model: candidate,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                filter: { path, _ in quantizedPaths.contains(path) }
            )
        }

        // `.shapeMismatch` rather than the default `.none`. `update(parameters:)`
        // without a verify argument calls through to `verify: .none`, which would
        // assign a packed integer weight into a float parameter of a different shape
        // and report nothing - the same silent path the SE pipeline had.
        try candidate.update(parameters: parameters, verify: .shapeMismatch)
        eval(candidate)
        try Task.checkCancellation()

        args = candidateArgs
        model = candidate
    }
    
    /// Upsample audio to 48kHz
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormer2_SR_48K")
        }

        try validateSampleRate(input)

        // Duration is invariant under resampling. Decide before allocating the
        // 48 kHz representation so long-form inference can convert incrementally.
        let durationSeconds = Float(input.frameCount) / Float(input.sampleRate)
        if durationSeconds > maxDirectDuration {
            return try await processWithChunking(input, model: model)
        }

        // Short inputs can be converted in one allocation. The chunk *is* the
        // whole signal here, so detecting on it is detecting globally - this path
        // keeps calling `bandwidthSub` directly and is unchanged.
        let audio48k = try await resampleTo48k(input)
        return try await processChunk(audio48k, model: model)
    }
    
    /// Process with chunking, discarding overlap rather than blending it.
    ///
    /// Matches `generate.py`'s `give_up_length` assembly: each chunk contributes
    /// its centre, the overlap either side is context for the model and is thrown
    /// away, and the first chunk keeps its leading edge. See
    /// ``ChunkingConfig/mossformer2SR48K(sampleRate:)`` for why this replaced a
    /// Hann overlap-add.
    private func processWithChunking(_ input: AudioBuffer, model: MossFormer2_SR_48K) async throws -> AudioBuffer {
        let chunkingConfig = ChunkingConfig.mossformer2SR48K(sampleRate: outputSampleRate)
        let chunkSamples = chunkingConfig.chunkSamples
        let stride = chunkingConfig.strideSamples
        let crossover = try await detectCrossover(input)
        var source = try SuperResolutionChunkSource(
            input: input,
            targetRate: outputSampleRate
        )
        let totalLength = source.totalLength
        var assembler = IncrementalDiscardEdges(
            chunkSamples: chunkSamples,
            stride: stride,
            totalLength: totalLength
        )
        var output: [Float] = []
        output.reserveCapacity(totalLength)
        var completedChunks = 0

        // One chunk of lookahead. Substituting a window needs the next chunk's
        // raw output for its trailing `giveUp` samples, so each chunk is held
        // back until its successor exists. Only three raw chunks are ever
        // resident, which is what keeps this bounded for long inputs.
        var previousRaw: [Float]?
        var pending: (startIdx: Int, input: [Float], raw: [Float])?

        while let chunk = try source.nextChunk(chunkSamples: chunkSamples, stride: stride) {
            try Task.checkCancellation()
            let raw = try await generateChunk(chunk.samples, model: model, paddedTo: chunkSamples)

            if let ready = pending {
                let window = assembledWindow(
                    previous: previousRaw, current: ready.raw, next: raw,
                    chunkSamples: chunkSamples, stride: stride
                )
                let processed = try substituteWindow(
                    input: ready.input, assembledRaw: window,
                    realLength: totalLength - ready.startIdx,
                    crossover: crossover, fadeIn: ready.startIdx == 0
                )
                output.append(contentsOf: assembler.add(processed.samples, startIdx: ready.startIdx))
                previousRaw = ready.raw
            }
            pending = (chunk.startIdx, chunk.samples, raw)

            completedChunks += 1
            MLXCachePolicy.trimIfNeeded(afterChunk: completedChunks)
        }

        if let ready = pending {
            let window = assembledWindow(
                previous: previousRaw, current: ready.raw, next: nil,
                chunkSamples: chunkSamples, stride: stride
            )
            let processed = try substituteWindow(
                input: ready.input, assembledRaw: window,
                realLength: totalLength - ready.startIdx,
                crossover: crossover, fadeIn: ready.startIdx == 0
            )
            output.append(contentsOf: assembler.add(processed.samples, startIdx: ready.startIdx))
        }
        output.append(contentsOf: assembler.finish())
        return AudioBuffer(
            samples: Array(output.prefix(totalLength)),
            sampleRate: outputSampleRate,
            channels: 1
        )
    }
    
    /// Where the upsampled original hands over to the model's reconstruction.
    ///
    /// One frequency for the whole signal, because that is what the reference
    /// produces: `generate.py` assembles every chunk first and calls
    /// `bandwidth_sub` once, on the finished signal, so `detect_bandwidth` sees
    /// all of it and returns a single answer.
    ///
    /// Calling `bandwidthSub` per chunk instead - which this provider did until
    /// now, and which `Parity/adapters/mossformer2_sr.py` still mirrored, so the
    /// suite could not see it - redetects that handover every `stride` samples.
    /// Measured over 136 s of speech it ranged from 187.5 Hz to 6937.5 Hz against
    /// a single global 4875 Hz, and the assembled output differed from the
    /// reference algorithm by 24.5 dB SNR. Two seams apart it would source the
    /// 2-5 kHz band from the model on one side and from the upsampled original on
    /// the other.
    ///
    /// The low end of that range is worse than a mismatch. `filtfilt`'s float32
    /// recurrence degrades as the cutoff falls - a narrower Butterworth puts its
    /// poles nearer z=1, so round-off is amplified harder - and at 187.5 Hz it
    /// sits 14 dB from the same filter computed in float64. A near-silent chunk
    /// was enough to drive it there. Detecting globally keeps the cutoff in the
    /// kilohertz range, where the same filter measures 100-125 dB.
    /// Internal rather than private so ``SuperResolutionCrossoverTests`` can hold
    /// the bounded accumulation against whole-signal `detectBandwidth`. That
    /// equivalence is the whole basis for streaming it, so it needs a test that
    /// names it rather than inference from an end-to-end number.
    struct Crossover {
        /// The signal already reaches Nyquist, so there is nothing to substitute
        /// and the model output passes through untouched. Mirrors the guard in
        /// AudioUtils' `bandwidthSub`.
        let passthrough: Bool
        let fHigh: Float
    }

    /// Detect the crossover on the whole 48 kHz signal, once, before chunking -
    /// without ever holding that signal.
    ///
    /// `detectBandwidth` reduces its spectrogram to one number per frequency bin,
    /// `sum(psd, axis: 1)`, and reads both edges off the cumulative sum of those
    /// 129 values. Nothing else about the STFT survives, so the reduction can be
    /// accumulated block by block over the streamed resample.
    ///
    /// That matters because the obvious version does not scale. Materializing the
    /// full 48 kHz array and calling `detectBandwidth` on it makes `stft` build
    /// full-length real and imaginary tensors - for an hour of audio roughly
    /// 0.7 GB for the signal and about 1.4 GB more for the spectrogram - which
    /// grows linearly with duration and defeats the whole point of
    /// ``SuperResolutionChunkSource``. Here the resident set is one block: a few
    /// megabytes, whatever the duration.
    ///
    /// Accumulating in `Double` is deliberate. Summing millions of per-block
    /// float32 totals in sequence drifts, where MLX's whole-array reduction gets
    /// to use a tree; the wider accumulator removes the difference rather than
    /// hoping it stays below a bin boundary.
    func detectCrossover(_ input: AudioBuffer) async throws -> Crossover {
        let nFFT = 256
        let hop = 128
        let winLength = 256
        let padAmount = nFFT / 2
        let bins = nFFT / 2 + 1
        let framesPerBlock = 2048

        // Hann, periodic: false - the window detectBandwidth builds.
        let indices = MLXArray(0..<winLength).asType(.float32)
        let window = 0.5 * (1 - MLX.cos(2 * Float.pi * indices / Float(winLength - 1)))
        eval(window)

        var energy = [Double](repeating: 0, count: bins)
        var buffer: [Float] = []

        /// Consume every whole frame the buffer can supply, keeping the
        /// remainder. Frames sit at fixed absolute offsets, so draining greedily
        /// cannot shift them.
        func drain(flush: Bool) {
            while buffer.count >= winLength {
                let available = (buffer.count - winLength) / hop + 1
                let take = min(available, framesPerBlock)
                guard take > 0 else { break }
                let span = (take - 1) * hop + winLength
                let block = MLXArray(Array(buffer[0..<span])).expandedDimensions(axis: 0)
                let (real, imag) = stft(
                    block, nFFT: nFFT, hopLength: hop,
                    winLength: winLength, window: window, center: false
                )
                let psd = real[0] * real[0] + imag[0] * imag[0]
                let perBin = MLX.sum(psd, axis: 1)
                eval(perBin)
                let values = perBin.asArray(Float.self)
                for i in 0..<min(bins, values.count) { energy[i] += Double(values[i]) }
                buffer.removeFirst(take * hop)
                if !flush && available <= framesPerBlock { break }
            }
        }

        let stream = try SuperResolutionResampler.Stream(
            input.samples, from: input.sampleRate, to: outputSampleRate
        )
        var head: [Float] = []
        var tail: [Float] = []
        var startedBody = false
        var totalRead = 0

        func begin() {
            // Left pad: signal[1 ..< padAmount + 1], reversed. STFT's `center`
            // reflects without repeating the edge sample.
            let end = min(padAmount + 1, head.count)
            if end > 1 { buffer.append(contentsOf: head[1..<end].reversed()) }
            buffer.append(contentsOf: head)
            startedBody = true
        }

        while let piece = try stream.next(maxFrames: 65_536) {
            try Task.checkCancellation()
            totalRead += piece.count
            if !startedBody {
                head.append(contentsOf: piece)
                guard head.count >= padAmount + 1 else { continue }
                begin()
            } else {
                buffer.append(contentsOf: piece)
            }
            tail.append(contentsOf: piece)
            if tail.count > padAmount + 1 { tail.removeFirst(tail.count - padAmount - 1) }
            drain(flush: false)
        }
        // A signal shorter than the pad never triggered `begin`.
        if !startedBody {
            tail = head
            begin()
        }

        // Right pad: signal[len - padAmount - 1 ..< len - 1], reversed.
        if tail.count > 1 { buffer.append(contentsOf: tail[0..<(tail.count - 1)].reversed()) }
        drain(flush: true)

        let total = energy.reduce(0, +)
        let nyquist = Float(outputSampleRate) / 2.0
        // detectBandwidth's own guard: silence has no bandwidth to detect, and it
        // reports Nyquist, which lands on the passthrough below.
        guard total >= 1e-10 else {
            return Crossover(passthrough: true, fHigh: nyquist)
        }

        let binHz = Float(outputSampleRate) / Float(nFFT)
        var cumulative = 0.0
        var fHigh = Float(bins - 1) * binHz
        for i in 0..<bins {
            cumulative += energy[i]
            if cumulative / total >= 0.99 {
                fHigh = Float(i) * binHz
                break
            }
        }

        // Matches AudioUtils' bandwidthSub: at Nyquist there is no high band to
        // graft on, and filtering there is what the margin exists to avoid.
        let nyquistMarginHz: Float = 100
        return Crossover(passthrough: fHigh >= nyquist - nyquistMarginHz, fHigh: fHigh)
    }

    /// The substitution itself, with the crossover already decided.
    ///
    /// Transcribes AudioUtils' `replaceBandwidth` and `smoothTransition` rather
    /// than calling `bandwidthSub`, which would redetect. The only difference is
    /// where `fHigh` comes from.
    ///
    /// `modelOutput` must be a window of the *assembled* signal, not one chunk's
    /// raw output. Those are not interchangeable: assembly leaves a small step
    /// wherever two chunks meet, and the reference highpasses straight across it.
    /// Filtering each chunk's own output instead never sees that step, which
    /// sounds harmless and is not - it read 107-112 dB everywhere except the
    /// seams, where it collapsed to 57.5 dB and dragged the case to 72.9 dB
    /// overall. The window has to carry its neighbours' contribution so the
    /// filter crosses the same discontinuity the reference does.
    ///
    /// `fadeIn` is the reference's 100 ms crossfade out of the unprocessed input.
    /// It belongs at sample 0 of the *signal*, not of every window, so only the
    /// first one gets it.
    private func substitute(
        input: MLXArray,
        modelOutput: MLXArray,
        crossover: Crossover,
        fadeIn: Bool
    ) throws -> MLXArray {
        guard !crossover.passthrough else { return modelOutput }

        let fs = Float(outputSampleRate)
        let effectiveBand = try lowpassFilter(input, cutoff: crossover.fHigh, fs: fs)
        let highBand = try highpassFilter(modelOutput, cutoff: crossover.fHigh, fs: fs)
        let length = min(effectiveBand.shape[0], highBand.shape[0])
        let substituted = highBand[0..<length] + effectiveBand[0..<length]

        guard fadeIn else { return substituted }

        // transitionBand = 100 ms, as in AudioUtils' smoothTransition. Literals
        // must be Float: untyped `0, 1` infer Int and yield an int64 ramp that
        // truncates to zeros except the final element.
        let fadeLength = min(100 * outputSampleRate / 1000, length)
        let fade = MLX.linspace(Float(0), Float(1), count: fadeLength)
        let head = (1 - fade) * input[0..<fadeLength] + fade * substituted[0..<fadeLength]
        guard fadeLength < length else { return head }
        return MLX.concatenated([head, substituted[fadeLength..<length]], axis: 0)
    }

    /// The assembled raw signal over one chunk's span.
    ///
    /// Positions below `giveUp` belong to the previous chunk and those at or
    /// above `chunkSamples - giveUp` to the next, which is exactly how
    /// ``IncrementalDiscardEdges`` splices them. Where a neighbour is missing -
    /// the first and last chunks - the chunk's own samples stand, as they do in
    /// the assembler.
    ///
    /// The point is the two joins this reintroduces. They are what the reference
    /// filters across, and the kept region is `giveUp` samples away from either
    /// window edge, so the window's own edge transients never reach it: measured,
    /// a 4th-order Butterworth here settles within about 500 samples against the
    /// 24000 of margin.
    private nonisolated func assembledWindow(
        previous: [Float]?,
        current: [Float],
        next: [Float]?,
        chunkSamples: Int,
        stride: Int
    ) -> [Float] {
        let giveUp = (chunkSamples - stride) / 2
        var window = current
        if let previous {
            for i in 0..<min(giveUp, window.count) where i + stride < previous.count {
                window[i] = previous[i + stride]
            }
        }
        if let next {
            for i in max(0, chunkSamples - giveUp)..<min(chunkSamples, window.count)
            where i - stride >= 0 && i - stride < next.count {
                window[i] = next[i - stride]
            }
        }
        return window
    }

    /// Mel, forward pass, squeeze - the model alone, padded to `length`.
    private func generateChunk(
        _ samples: [Float],
        model: MossFormer2_SR_48K,
        paddedTo length: Int
    ) async throws -> [Float] {
        var raw = try await rawModelOutput(samples, model: model).asArray(Float.self)
        if raw.count < length {
            raw.append(contentsOf: repeatElement(0, count: length - raw.count))
        }
        return Array(raw.prefix(length))
    }

    /// Mel spectrogram and one forward pass. No substitution.
    private func rawModelOutput(
        _ samples: [Float],
        model: MossFormer2_SR_48K
    ) async throws -> MLXArray {
        let inputs = MLXArray(samples)

        // Config values
        let hopSize = args["hop_size"] as? Int ?? 256
        let nFFT = args["n_fft"] as? Int ?? 1024
        let numMels = args["num_mels"] as? Int ?? 80
        let winSize = args["win_size"] as? Int ?? 1024
        let fmin = args["fmin"] as? Float ?? 0
        let fmax = args["fmax"] as? Float ?? 8000

        let melSpec = try melSpectrogram(
            inputs.expandedDimensions(axis: 0),
            nFFT: nFFT,
            numMels: numMels,
            samplingRate: 48000,
            hopSize: hopSize,
            winSize: winSize,
            fmin: fmin,
            fmax: fmax
        )
        eval(melSpec)

        let output = model(melSpec)
        eval(output)
        return output.squeezed()
    }

    /// Substitute over one window, given the raw signal already assembled across
    /// it. Used by the chunked paths, which supply the neighbours' contribution.
    ///
    /// `realLength` trims the window to the samples the signal actually has. It
    /// matters only for the last chunk, and it matters a lot: the source
    /// zero-pads that chunk out to `chunkSamples`, so filtering the padded window
    /// makes `filtfilt` reflect about a step down to silence, where the reference
    /// - whose assembled array simply ends - reflects about the last real sample.
    /// Leaving the pad in place cost the tail of every file, and only the tail,
    /// which is why it survived a seam-by-seam reading of the error.
    private func substituteWindow(
        input: [Float],
        assembledRaw: [Float],
        realLength: Int,
        crossover: Crossover,
        fadeIn: Bool
    ) throws -> AudioBuffer {
        let usable = max(0, min(realLength, min(input.count, assembledRaw.count)))
        guard usable > 0 else {
            return AudioBuffer(samples: [], sampleRate: outputSampleRate, channels: 1)
        }
        let inputs = MLXArray(Array(input.prefix(usable)))
        let outputs = try substitute(
            input: inputs,
            modelOutput: MLXArray(Array(assembledRaw.prefix(usable))),
            crossover: crossover,
            fadeIn: fadeIn
        )
        eval(outputs)
        let trimmed = outputs[0..<min(outputs.shape[0], input.count)]
        return AudioBuffer(
            samples: trimmed.asArray(Float.self),
            sampleRate: outputSampleRate,
            channels: 1
        )
    }

    /// The direct path: the chunk is the whole signal, so `bandwidthSub`
    /// detecting on it *is* detecting globally. Unchanged.
    private func processChunk(
        _ samples: [Float],
        model: MossFormer2_SR_48K
    ) async throws -> AudioBuffer {
        let inputs = MLXArray(samples)
        let inputLen = inputs.shape[0]
        let raw = try await rawModelOutput(samples, model: model)
        var outputs = try bandwidthSub(inputs, raw, fs: 48000)
        eval(outputs)
        outputs = outputs[0..<inputLen]
        return AudioBuffer(
            samples: outputs.asArray(Float.self),
            sampleRate: outputSampleRate,
            channels: 1
        )
    }
    
    /// Resample input to 48 kHz - the first step of inference, not plumbing.
    ///
    /// Delegates to ``SuperResolutionResampler``, which lives in its own file
    /// because importing AVFoundation here makes `AudioBuffer` ambiguous with
    /// CoreAudioTypes'.
    private func resampleTo48k(_ input: AudioBuffer) async throws -> [Float] {
        guard input.sampleRate != outputSampleRate else { return input.samples }
        return try SuperResolutionResampler.upsample(
            input.samples, from: input.sampleRate, to: outputSampleRate
        )
    }
}

// MARK: - StreamableOutput Conformance

extension MossFormer2SR48KProvider: StreamableOutput {
    /// Process audio and stream output chunks as they're ready.
    /// Uses Hann-window overlap-add with buffered blending.
    public nonisolated func processStream(_ input: AudioBuffer) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    try await self.processStreamImpl(input, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                producer.cancel()
            }
        }
    }
    
    /// Internal implementation for streaming with overlap buffer
    private func processStreamImpl(_ input: AudioBuffer, continuation: AsyncThrowingStream<AudioBuffer, Error>.Continuation) async throws {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormer2_SR_48K")
        }
        // Same contract as the batch path. Without this, enabling progress reporting
        // switched the pipeline to streaming and quietly changed the result: 48 kHz
        // input went straight through 48 -> 48 instead of being super-resolved from 16.
        try validateSampleRate(input)
        
        let durationSeconds = Float(input.frameCount) / Float(input.sampleRate)
        
        // For short audio, just yield single result
        if durationSeconds <= maxDirectDuration {
            let samples = try await resampleTo48k(input)
            let result = try await processChunk(samples, model: model)
            if case .terminated = continuation.yield(result) {
                throw CancellationError()
            }
            continuation.finish()
            return
        }
        
        // Streaming with chunking and overlap buffer
        let chunkingConfig = ChunkingConfig.mossformer2SR48K(sampleRate: outputSampleRate)
        let chunkSamples = chunkingConfig.chunkSamples
        let stride = chunkingConfig.strideSamples
        // Detected before the first chunk is yielded, on the whole signal, so a
        // stream produces the samples the batch path produces. The input is a
        // complete buffer, not a live feed, so this is available up front.
        let crossover = try await detectCrossover(input)
        var source = try SuperResolutionChunkSource(input: input, targetRate: outputSampleRate)
        let totalLength = source.totalLength

        // Streaming must produce the samples batch would, so it uses the same
        // assembler. That contract is what previously went wrong here in a
        // different way: an earlier version multiplied the first chunk by the
        // rising half of a Hann window and never divided by the accumulated
        // weight, so every stream opened with a `stride`-long fade-in from silence
        // and attaching a progress handler was enough to get that instead of the
        // real output. Discard-edges has no weights to normalize, which removes
        // that class of bug rather than fixing an instance of it.
        var assembler = IncrementalDiscardEdges(
            chunkSamples: chunkSamples,
            stride: stride,
            totalLength: totalLength
        )
        var completedChunks = 0

        // Same one-chunk lookahead as the batch path, and for the same reason:
        // a window cannot be substituted until its successor's raw output exists.
        // Streaming therefore lags by one chunk. It still emits progressively -
        // this is a delay, not a buffer-everything - and the alternative is
        // yielding samples the batch path would not produce.
        var previousRaw: [Float]?
        var pending: (startIdx: Int, input: [Float], raw: [Float])?

        func emit(_ ready: (startIdx: Int, input: [Float], raw: [Float]), next: [Float]?) throws {
            let window = assembledWindow(
                previous: previousRaw, current: ready.raw, next: next,
                chunkSamples: chunkSamples, stride: stride
            )
            let processed = try substituteWindow(
                input: ready.input, assembledRaw: window,
                realLength: totalLength - ready.startIdx,
                crossover: crossover, fadeIn: ready.startIdx == 0
            )
            let out = assembler.add(processed.samples, startIdx: ready.startIdx)
            if !out.isEmpty {
                if case .terminated = continuation.yield(AudioBuffer(
                    samples: out,
                    sampleRate: outputSampleRate,
                    channels: 1
                )) {
                    throw CancellationError()
                }
            }
        }

        while let chunk = try source.nextChunk(chunkSamples: chunkSamples, stride: stride) {
            try Task.checkCancellation()
            let raw = try await generateChunk(chunk.samples, model: model, paddedTo: chunkSamples)

            if let ready = pending {
                try emit(ready, next: raw)
                previousRaw = ready.raw
            }
            pending = (chunk.startIdx, chunk.samples, raw)

            completedChunks += 1
            MLXCachePolicy.trimIfNeeded(afterChunk: completedChunks)
        }

        if let ready = pending {
            try emit(ready, next: nil)
        }

        let tail = assembler.finish()
        if !tail.isEmpty {
            if case .terminated = continuation.yield(AudioBuffer(
                samples: tail,
                sampleRate: outputSampleRate,
                channels: 1
            )) {
                throw CancellationError()
            }
        }

        continuation.finish()
    }
}

/// Generates overlapping 48 kHz chunks while retaining only the previous chunk
/// and the converter's bounded input/output buffers.
private struct SuperResolutionChunkSource {
    let totalLength: Int

    private let converter: SuperResolutionResampler.Stream
    private var previousChunk: [Float]?
    private var nextStart = 0

    init(input: AudioBuffer, targetRate: Int) throws {
        converter = try SuperResolutionResampler.Stream(
            input.samples,
            from: input.sampleRate,
            to: targetRate
        )
        totalLength = converter.expectedFrameCount
    }

    mutating func nextChunk(
        chunkSamples: Int,
        stride: Int
    ) throws -> (samples: [Float], startIdx: Int)? {
        guard nextStart < totalLength else { return nil }
        let overlap = max(0, chunkSamples - stride)
        var chunk: [Float]
        if let previousChunk, overlap > 0 {
            chunk = Array(previousChunk.suffix(overlap))
        } else {
            chunk = []
        }

        let needed = max(0, chunkSamples - chunk.count)
        chunk.append(contentsOf: try read(upTo: needed))
        if chunk.count < chunkSamples {
            chunk.append(contentsOf: repeatElement(0, count: chunkSamples - chunk.count))
        }

        let start = nextStart
        nextStart += max(1, stride)
        previousChunk = chunk
        return (chunk, start)
    }

    private func read(upTo count: Int) throws -> [Float] {
        guard count > 0 else { return [] }
        var result: [Float] = []
        result.reserveCapacity(count)
        while result.count < count,
              let part = try converter.next(maxFrames: count - result.count) {
            result.append(contentsOf: part)
        }
        return result
    }
}

// MARK: - ManagedModel

extension MossFormer2SR48KProvider: ManagedModel {
    public nonisolated var modelId: String { "mossformer2_sr_48k" }

    /// Measured peak 4017 MiB, 30 s of 16 kHz input on an M1 Pro.
    ///
    /// This read 300 MB - below even the 418 MiB checkpoint - and I replaced it with
    /// 1.6 GB scaled from SE's measurement before an SR benchmark run existed. Both
    /// were wrong by more than a factor of two in the direction that matters, which
    /// is the whole argument for measuring: SR has the largest footprint of the
    /// enhancement models and nothing about its checkpoint says so.
    public nonisolated var estimatedMemoryBytes: Int { 4_200_000_000 }

    public func checkIfLoaded() async -> Bool { model != nil }

    public func unload() async {
        // Cancel first: a load mid-download must not publish over this unload.
        // Bracketed: the gate stays shut until this method's state reset is
        // done, so a concurrent load cannot publish a model into the gap and
        // have it wiped by the lines below.
        let teardown = await loadGate.beginTeardown()
        defer { loadGate.endTeardown(teardown) }
        model = nil
        GPU.clearCache()
    }
}
