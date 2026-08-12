//
//  MLXSeparatorProvider.swift
//  AudioToolMLX
//
//  MLX-based speech separation and source separation providers with chunking
//

import Foundation
import AudioTool
import AudioToolCore
@preconcurrency import MLX
@preconcurrency import MLXNN
@preconcurrency import AudioUtils
@preconcurrency import MossFormer2SS
@preconcurrency import DemucsMLXSwift

// MARK: - MossFormer2 Speaker Separation Provider

/// MLX MossFormer2 Speaker Separation - supports 2spk, 3spk, and WHAMR models
/// Chunking: 4s chunks, 25% overlap, triangular blending
///
/// Supports progress reporting during chunked processing via `separate(_:speakers:onProgress:)`.
/// For streaming output, use `separateStream(_:)` which yields per-speaker results as chunks complete.
public actor MossFormer2SSProvider: SpeechSeparator, ChunkedProgressProvider {
    
    // MARK: - ChunkedProgressProvider Conformance
    
    /// MossFormer2SS supports chunked progress during separation
    public nonisolated var supportsChunkedProgress: Bool { true }
    
    // SpeechSeparator conformance. Fixed by the loaded weights, which is why it is
    // not a per-call argument.
    public nonisolated var outputCount: Int { modelType.numSpeakers }
    
    /// Available speaker separation models
    public enum Model: String, CaseIterable, Sendable {
        case twoSpeaker = "2spk"          // 2 speakers at 16kHz
        case threeSpeaker = "3spk"         // 3 speakers at 8kHz
        case twoSpeakerWHAMR = "2spk-whamr" // 2 speakers at 8kHz with WHAMR
        
        public var numSpeakers: Int {
            switch self {
            case .twoSpeaker: return 2
            case .threeSpeaker: return 3
            case .twoSpeakerWHAMR: return 2
            }
        }
        
        public var sampleRate: Int {
            switch self {
            case .twoSpeaker: return 16000
            case .threeSpeaker: return 8000
            case .twoSpeakerWHAMR: return 8000
            }
        }
        
        public var skipMaskMultiplication: Bool {
            switch self {
            case .twoSpeakerWHAMR: return true
            default: return false
            }
        }
        
        public var huggingFaceRepo: String {
            switch self {
            case .twoSpeaker: return ModelRepository.mossFormer2SS2Spk16K
            case .threeSpeaker: return ModelRepository.mossFormer2SS3Spk8K
            case .twoSpeakerWHAMR: return ModelRepository.mossFormer2SS2SpkWHAMR8K
            }
        }
        
        /// Supported precisions (currently fp32 only)
        public var supportedPrecisions: [ModelPrecision] { [.fp32] }
    }
    
    public let modelType: Model
    public nonisolated var sampleRate: Int { modelType.sampleRate }
    public nonisolated let inputChannels: Int = 1

    /// Matches the standalone Swift generator, which asks AVAudioConverter for
    /// Mastering at maximum quality
    /// (`Models/mosforrmer2_ss_mlx_swift/Sources/Generate/main.swift:155`).
    public nonisolated var preferredResamplingQuality: AudioToolCore.ResamplingQuality { .high }
    public nonisolated let outputChannels: Int = 1
    
    public nonisolated var minChunkSize: Int { modelType.sampleRate / 5 }  // 0.2s
    public nonisolated var recommendedChunkSize: Int { modelType.sampleRate * 4 }  // 4s (from benchmarks)
    
    private var model: MossFormer2_SS_16K?
    private let weightsPath: String?
    private let precision: ModelPrecision

    /// One load at a time; see ``ModelLoadGate``.
    private let loadGate = ModelLoadGate()
    
    /// Max audio without chunking
    private let maxDirectDuration: Float = 4.0
    
    /// Initialize with model type and precision (auto-downloads from HuggingFace)
    public init(model: Model, precision: ModelPrecision = .fp32) {
        self.modelType = model
        self.weightsPath = nil
        self.precision = precision
    }
    
    /// Initialize with explicit weights path (no download)
    public init(model: Model, weightsPath: String) {
        self.modelType = model
        self.weightsPath = weightsPath
        self.precision = .fp32
    }
    
    /// Load model weights (downloads if not cached)
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
        let config = MossFormer2Config(
            encoder_embedding_dim: 512,
            mossformer_sequence_dim: 512,
            num_mossformer_layer: 24,
            encoder_kernel_size: 16,
            num_spks: modelType.numSpeakers,
            skip_mask_multiplication: modelType.skipMaskMultiplication
        )
        
        let candidate = MossFormer2_SS_16K(config: config)
        
        let resolvedPath: String
        if let path = weightsPath {
            resolvedPath = path
        } else {
            // Present-to-load is the weights alone: the MossFormer2Config above is
            // constructed here, not read from a config.json. The download manifest
            // stays complete. See ModelFiles.standardRequired.
            if let cached = ModelDownloader.shared.localPath(
                for: modelType.huggingFaceRepo,
                matching: ModelFiles.standardRequired(precision)
            ) {
                resolvedPath = cached.appendingPathComponent(precision.weightsFilename).path
            } else {
                // Auto-download from HuggingFace
                let modelDir = try await ModelDownloader.shared.downloadAndGetPath(
                    repo: modelType.huggingFaceRepo,
                    matching: ModelFiles.standard(precision)
                )
                resolvedPath = modelDir.appendingPathComponent(precision.weightsFilename).path
            }
        }
        
        let weights = try loadArrays(url: URL(fileURLWithPath: resolvedPath))
        let nestedWeights = NestedDictionary<String, MLXArray>.unflattened(weights)
        try candidate.update(parameters: nestedWeights, verify: .all)
        eval(candidate)
        try Task.checkCancellation()
        model = candidate
    }
    
    /// Separate mixed audio into speaker streams.
    ///
    /// Each output is RMS-matched to the input and peak-limited, the same
    /// normalisation the Python reference applies.
    public func separate(_ audio: AudioBuffer) async throws -> [AudioBuffer] {
        try await separate(audio, onProgress: nil)
    }
    
    /// Separate mixed audio with progress reporting
    /// - Parameters:
    ///   - audio: Input audio buffer
    ///   - onProgress: Progress callback (0.0 to 100.0) as chunks are processed
    /// - Returns: ``outputCount`` separated audio buffers, one per speaker
    public func separate(
        _ audio: AudioBuffer,
        onProgress: ProgressCallback?
    ) async throws -> [AudioBuffer] {
        try validateSampleRate(audio)
        await onProgress?(0.0)
        
        let durationSeconds = Float(audio.samples.count) / Float(audio.sampleRate)
        
        // Use chunking for longer audio
        var results: [AudioBuffer]
        if durationSeconds > maxDirectDuration {
            results = try await processWithChunking(audio, onProgress: onProgress)
        } else {
            results = try await processChunk(audio.samples)
        }
        
        // Match each source's RMS to the mixture's, then de-clip. SS estimates come
        // out at an arbitrary and often very loud level, and this is the step that
        // makes them usable.
        //
        // This mirrors `_normalize` in Parity/adapters/mossformer2_ss.py, which is
        // why the published tensors are named `speaker_N_normalized`. Mixture
        // consistency - sharing the residual so sum(sources) == mixture - is a
        // defensible alternative that keeps relative gains, but it is a *different*
        // output contract: swapping it in here alone put every SS parity case at
        // -19 to -44 dB. If it is wanted, the reference has to move first.
        let inputRMS = Self.rootMeanSquare(audio.samples)
        let normalized = results.map { buffer in
            AudioBuffer(
                samples: Self.normalizeToInputRMS(buffer.samples, inputRMS: inputRMS),
                sampleRate: buffer.sampleRate,
                channels: buffer.channels
            )
        }

        await onProgress?(100.0)
        return normalized
    }

    /// Accumulated in `Double` because the reference computes it as
    /// `np.mean(x.astype(np.float64) ** 2)`; a Float32 sum over a long recording
    /// drifts from that by more than the parity floor allows.
    internal static func rootMeanSquare(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum = 0.0
        for sample in samples {
            sum += Double(sample) * Double(sample)
        }
        return Float((sum / Double(samples.count)).squareRoot())
    }

    /// `_normalize` from the Python adapter: scale to the mixture's RMS, then divide
    /// out any peak above full scale.
    internal static func normalizeToInputRMS(_ samples: [Float], inputRMS: Float) -> [Float] {
        let outputRMS = rootMeanSquare(samples)
        var scaled = outputRMS > 1e-8
            ? samples.map { $0 * (inputRMS / outputRMS) }
            : samples
        let peak = scaled.reduce(Float(0)) { Swift.max($0, abs($1)) }
        if peak > 1.0 {
            scaled = scaled.map { $0 / peak }
        }
        return scaled
    }
    
    
    /// Process with chunking using MLXOverlap
    /// Uses triangular weighted overlap-add for each speaker independently
    private func processWithChunking(_ input: AudioBuffer, onProgress: ProgressCallback? = nil) async throws -> [AudioBuffer] {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormer2_SS_\(modelType.rawValue)")
        }
        
        let chunkingConfig = ChunkingConfig.mossformer2SS(sampleRate: sampleRate)
        let chunkSamples = chunkingConfig.chunkSamples
        let stride = chunkingConfig.strideSamples
        let totalLength = input.samples.count
        let totalChunks = max(1, (totalLength + stride - 1) / stride)
        let window = MLXOverlap.triangularWindow(length: chunkSamples).asArray(Float.self)
        var assemblers = (0..<modelType.numSpeakers).map { _ in
            IncrementalOverlapAdd(
                chunkSamples: chunkSamples,
                stride: stride,
                window: window,
                totalLength: totalLength
            )
        }
        var speakerOutputs = Array(
            repeating: [Float](),
            count: modelType.numSpeakers
        )
        for index in speakerOutputs.indices {
            speakerOutputs[index].reserveCapacity(totalLength)
        }

        var chunkIdx = 0
        var startIdx = 0
        while startIdx < totalLength {
            try Task.checkCancellation()
            let endIdx = min(startIdx + chunkSamples, totalLength)
            var chunkArray = Array(input.samples[startIdx..<endIdx])
            if chunkArray.count < chunkSamples {
                chunkArray.append(contentsOf: repeatElement(0, count: chunkSamples - chunkArray.count))
            }
            let chunk = MLXArray(chunkArray)
            // Run model: input [1, T] -> outputs [numSpeakers x [1, T]]
            let batchedChunk = chunk.expandedDimensions(axis: 0)
            let separatedSources = model(batchedChunk)
            eval(separatedSources)
            
            for (spkIdx, source) in separatedSources.enumerated() {
                guard spkIdx < assemblers.count else { break }
                let samples = source.squeezed(axis: 0).asArray(Float.self)
                let ready = assemblers[spkIdx].add(samples, startIdx: startIdx)
                speakerOutputs[spkIdx].append(contentsOf: ready)
            }
            
            MLXCachePolicy.trimIfNeeded(afterChunk: chunkIdx + 1)
            
            // Report progress (0-90% for chunking, 90-100% for reassembly)
            let percent = Double(chunkIdx + 1) / Double(totalChunks) * 90.0
            await onProgress?(percent)
            startIdx += stride
            chunkIdx += 1
        }
        
        var results: [AudioBuffer] = []
        for spkIdx in 0..<modelType.numSpeakers {
            speakerOutputs[spkIdx].append(contentsOf: assemblers[spkIdx].finish())
            if speakerOutputs[spkIdx].count < totalLength {
                speakerOutputs[spkIdx].append(contentsOf: repeatElement(
                    0,
                    count: totalLength - speakerOutputs[spkIdx].count
                ))
            }
            results.append(AudioBuffer(
                samples: Array(speakerOutputs[spkIdx].prefix(totalLength)),
                sampleRate: sampleRate,
                channels: 1
            ))
        }
        
        return results
    }
    
    /// Process a single chunk (direct, no chunking)
    private func processChunk(_ samples: [Float]) async throws -> [AudioBuffer] {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormer2_SS_\(modelType.rawValue)")
        }
        
        let inputMLX = MLXArray(samples).expandedDimensions(axis: 0)
        let separatedSources = model(inputMLX)
        eval(separatedSources)
        
        var results: [AudioBuffer] = []
        for source in separatedSources {
            let squeezed = source.squeezed(axis: 0)
            let samples = squeezed.asArray(Float.self)
            results.append(AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1))
        }
        
        return results
    }
    
    public func reset() async {}
}

// MARK: - Demucs Source Separation Provider

/// Demucs source separation - separates audio into drums, bass, vocals, other
/// Chunking: 7.8s chunks, 25% overlap, triangular blending (from benchmarks)
public actor DemucsProvider: MusicSeparator {
    
    public enum Stem: String, CaseIterable, Sendable {
        case drums = "drums"
        case bass = "bass"
        case vocals = "vocals"
        case other = "other"

        /// Position of this stem on axis 1 of the model output, which is
        /// (batch, source, channel, time). Fixed by the HTDemucs architecture; each
        /// per-stem weight file still emits all four, so the right one must be
        /// selected rather than reduced over.
        var sourceIndex: Int {
            switch self {
            case .drums: return 0
            case .bass: return 1
            case .other: return 2
            case .vocals: return 3
            }
        }
    }
    
    public nonisolated let sampleRate: Int = 44100

    /// Matches the standalone generator, which builds `AudioLoader` with
    /// `targetSampleRate: 44100` and leaves the resampling method at `.auto`.
    ///
    /// Not Mastering. The Python reference uses `torchaudio.transforms.Resample`, and
    /// the Swift port was validated against it through the loader's ordinary
    /// settings - so `.auto` is the measured answer and Mastering would be a guess
    /// that happens to sound better.
    public nonisolated var preferredResamplingQuality: AudioToolCore.ResamplingQuality { .auto }
    /// The model's input width, and the layout the facade converts to - not a
    /// restriction on what this provider accepts.
    ///
    /// HTDemucs takes a `(2, frames)` tensor and separates partly on stereo cues, so
    /// stereo in is the best input and `AudioEngine` adapts to it. But ``separate(_:stem:)``
    /// takes any layout: ``stereoPair(from:)`` duplicates mono to both channels, the
    /// way HTDemucs itself does, and downmixes anything wider.
    public nonisolated let inputChannels: Int = 2

    /// Always mono: the separation uses both channels, and only the result is folded
    /// down - see ``separateDirect(_:stem:)``.
    public nonisolated let outputChannels: Int = 1
    
    /// Stems the loaded weights can produce. Each has its own weight file, so this
    /// reflects what is on disk rather than a fixed capability.
    public nonisolated var availableStems: [Stem] { Stem.allCases }
    
    /// HuggingFace repository for model weights
    public static let repo = ModelRepository.demucs

    private var models: [Stem: HTDemucs] = [:]
    private let weightsDirectory: String?

    /// One residency-managed load at a time; see ``ModelLoadGate``. Per-stem loads
    /// go through ``load(stem:)`` and are not gated - they are additive, and a
    /// duplicate costs one stem rather than the whole provider.
    private let loadGate = ModelLoadGate()

    /// Max audio without chunking (7.8s is the limit from benchmarks)
    private let maxDirectDuration: Float = 7.8

    /// Initialize to download from HuggingFace on first load.
    ///
    /// Like FRCRN, this repo was in the catalog with no code able to fetch it - the
    /// only initializer took a directory the caller had to fill themselves.
    public init() {
        self.weightsDirectory = nil
    }

    /// Initialize with a directory of `<stem>.safetensors` files (no download)
    public init(weightsDirectory: String) {
        self.weightsDirectory = weightsDirectory
    }

    /// Load all four source models
    public func loadAll() async throws {
        // Before the first allocation, not at the first chunk boundary.
        // `trimIfNeeded` used to be the only thing that applied these, so a
        // provider ran its load and its opening chunks under MLX's own default
        // ceiling and the cache had already grown past the cap by the time the
        // cap arrived. Measured on Demucs: 5.1 GB peak applying it late against
        // 3.2 GB applying it here.
        MLXCachePolicy.applyProcessLimits()
        let directory = try await resolveWeightsDirectory(for: Stem.allCases)
        var candidates: [Stem: HTDemucs] = [:]
        for stem in Stem.allCases {
            try Task.checkCancellation()
            candidates[stem] = try Self.loadModel(
                stem: stem,
                weightsDirectory: directory
            )
        }
        // An `unload()` that landed while the weights were downloading cancels this
        // load; publishing here anyway would resurrect the provider behind it.
        try Task.checkCancellation()
        models = candidates
    }

    /// Directory holding the stem weights, downloading only what is asked for.
    ///
    /// One 84 MB file per stem, so a caller who only wants vocals fetches a quarter
    /// of what `loadAll` does. `downloadAndGetPath` returns the repo directory either
    /// way; `matching` is what decides how much of it arrives.
    private func resolveWeightsDirectory(for stems: [Stem]) async throws -> String {
        if let directory = weightsDirectory { return directory }

        let wanted = stems.map { ModelFiles.demucsStem($0.rawValue) }
        if let cached = ModelDownloader.shared.localPath(
            for: Self.repo,
            matching: wanted
        ) {
            return cached.path
        }
        let modelDir = try await ModelDownloader.shared.downloadAndGetPath(
            repo: Self.repo,
            matching: wanted
        )
        return modelDir.path
    }

    /// Load a specific source model, fetching its weights if needed
    public func load(stem: Stem) async throws {
        // Before the first allocation, not at the first chunk boundary.
        // `trimIfNeeded` used to be the only thing that applied these, so a
        // provider ran its load and its opening chunks under MLX's own default
        // ceiling and the cache had already grown past the cap by the time the
        // cap arrived. Measured on Demucs: 5.1 GB peak applying it late against
        // 3.2 GB applying it here.
        MLXCachePolicy.applyProcessLimits()
        let directory = try await resolveWeightsDirectory(for: [stem])
        try loadSync(stem: stem, weightsDirectory: directory)
    }

    /// Load a specific source model from a directory already on disk
    public func loadSync(stem: Stem, weightsDirectory: String) throws {
        models[stem] = try Self.loadModel(
            stem: stem,
            weightsDirectory: weightsDirectory
        )
    }

    private nonisolated static func loadModel(
        stem: Stem,
        weightsDirectory: String
    ) throws -> HTDemucs {
        let config = HTDemucsConfig()
        let model = HTDemucs(config: config)
        
        // Use HTDemucs.loadWeights for proper Python→Swift key conversion
        let weightsPath = "\(weightsDirectory)/\(ModelFiles.demucsStem(stem.rawValue))"
        let parameters = try HTDemucs.loadWeights(from: weightsPath)
        try model.update(parameters: parameters, verify: .all)
        
        // Set to eval mode and materialize weights
        model.train(false)
        MLX.eval(model)
        
        return model
    }
    
    /// Separate a specific source from the audio.
    ///
    /// Mono, stereo and wider inputs are all accepted; ``stereoPair(from:)`` converts.
    /// The rate is not negotiable - 44.1 kHz is what the model was trained at.
    ///
    /// This validated the channel count too, against `inputChannels == 2`, which made
    /// the mono and downmix branches of that conversion unreachable for anyone calling
    /// the provider directly - and mono is the ordinary case here, since `AudioLoader`
    /// exposes `loadMono` and the CLI uses it. Two things had already bent around the
    /// guard rather than through it: the benchmark harness carried a per-case channel
    /// count so Demucs would not report `channelCountMismatch` where nine other models
    /// measured fine, and `BackgroundExtractionTests.testDemucsBackground` simply
    /// failed. The conversion was written, documented and correct; only this line
    /// stopped it running.
    public func separate(_ input: AudioBuffer, stem: Stem) async throws -> AudioBuffer {
        try validateInputRate(input)

        guard models[stem] != nil else {
            throw AudioToolError.modelNotLoaded("Demucs_\(stem.rawValue)")
        }

        let frameCount = input.frameCount
        let durationSeconds = Float(frameCount) / Float(input.sampleRate)

        // Use chunking for longer audio
        if durationSeconds > maxDirectDuration {
            return try await separateWithChunking(
                input,
                frameCount: frameCount,
                stem: stem
            )
        }

        let stereo = Self.stereoPair(from: input)
        eval(stereo)
        return try await separateDirect(stereo, stem: stem)
    }

    /// Turn a buffer into the `(2, frames)` planar pair HTDemucs expects.
    ///
    /// ``AudioBuffer`` stores multichannel audio interleaved. The stereo branch here
    /// used to be `MLXArray(samples).reshaped([2, -1])`, which reads an interleaved
    /// buffer as if it were planar: channel 0 became the first half of the recording
    /// and channel 1 the second half, so the model was handed two time-shifted copies
    /// of different halves of the song rather than a left/right pair. Nothing in the
    /// package fed it stereo - the CLI loads mono and the chunked path duplicated a
    /// mono stream - so it produced no visible symptom while making Demucs's stereo
    /// cues, the thing it separates with, meaningless for any caller who did.
    ///
    /// Mono is duplicated to both channels; more than two channels are downmixed,
    /// since the model has exactly two input channels.
    internal static func stereoPair(from input: AudioBuffer) -> MLXArray {
        stereoPair(from: input, frames: 0..<input.frameCount)
    }

    /// Convert only the requested frames. Long recordings use this overload so
    /// chunking never materializes a second full-recording MLX tensor.
    internal static func stereoPair(
        from input: AudioBuffer,
        frames requestedFrames: Range<Int>
    ) -> MLXArray {
        let channels = max(1, input.channels)
        let lower = min(input.frameCount, max(0, requestedFrames.lowerBound))
        let upper = min(input.frameCount, max(lower, requestedFrames.upperBound))
        let frameCount = upper - lower
        var left = [Float]()
        var right = [Float]()
        left.reserveCapacity(frameCount)
        right.reserveCapacity(frameCount)

        if channels == 1 {
            left.append(contentsOf: input.samples[lower..<upper])
            right = left
        } else if channels == 2 {
            for frame in lower..<upper {
                left.append(input.samples[frame * channels])
                right.append(input.samples[frame * channels + 1])
            }
        } else {
            for frame in lower..<upper {
                let base = frame * channels
                var sum: Float = 0
                for channel in 0..<channels {
                    sum += input.samples[base + channel]
                }
                let mono = sum / Float(channels)
                left.append(mono)
                right.append(mono)
            }
        }
        return MLX.stacked([MLXArray(left), MLXArray(right)], axis: 0)
    }

    /// Run one `(2, frames)` block through the model and take the requested stem.
    ///
    /// Model output is (batch, source, channel, time) - four sources, always.
    ///
    /// The direct path used to squeeze the batch axis and take mean(axis: 0), which
    /// averaged the four *stems* into each other rather than selecting one, and left
    /// the result still stereo: a 4s mono request came back with 2x the samples,
    /// labelled mono, containing a blend of drums, bass, vocals and other. The
    /// chunked path indexed correctly, so which one ran depended only on whether the
    /// input was longer than maxDirectDuration.
    private func stemOutput(_ stereo: MLXArray, stem: Stem, model: HTDemucs) -> MLXArray {
        let batched = stereo.expandedDimensions(axis: 0)   // (1, 2, time)
        let output = model(batched)                        // (1, 4, 2, time)
        eval(output)
        return output[0, stem.sourceIndex]                 // (2, time)
    }

    /// Separate audio short enough to go through the model in one pass.
    private func separateDirect(_ stereo: MLXArray, stem: Stem) async throws -> AudioBuffer {
        guard let model = models[stem] else {
            throw AudioToolError.modelNotLoaded("Demucs_\(stem.rawValue)")
        }
        // Threshold-based rather than unconditional: `GPU.clearCache()` is
        // process-global, and purging after every separation both discards other
        // MLX users' buffers and gives up the allocator reuse the next call wants.
        defer { MLXCachePolicy.trimIfCacheGrew() }

        let sourceOutput = stemOutput(stereo, stem: stem, model: model)
        // Downmixed to mono, matching ``outputChannels``. The separation itself used
        // both channels; only the result is folded down.
        let mono = mean(sourceOutput, axis: 0)
        eval(mono)

        return AudioBuffer(samples: mono.asArray(Float.self), sampleRate: sampleRate, channels: 1)
    }

    /// Separate with triangular weighted overlap-add across chunks.
    ///
    /// Hand-rolled rather than delegating to ``MLXOverlap/processWithChunking``,
    /// which is strictly one-dimensional: routing stereo through it meant flattening
    /// to mono first, so long audio silently lost the stereo information that short
    /// audio kept. Same windowing and normalization as
    /// ``MLXOverlap/reassembleOverlapAdd``, applied per channel.
    private func separateWithChunking(
        _ input: AudioBuffer,
        frameCount: Int,
        stem: Stem
    ) async throws -> AudioBuffer {
        guard let model = models[stem] else {
            throw AudioToolError.modelNotLoaded("Demucs_\(stem.rawValue)")
        }
        // Threshold-based rather than unconditional: `GPU.clearCache()` is
        // process-global, and purging after every separation both discards other
        // MLX users' buffers and gives up the allocator reuse the next call wants.
        defer { MLXCachePolicy.trimIfCacheGrew() }

        let chunkingConfig = ChunkingConfig.demucs(sampleRate: sampleRate)
        let chunkSamples = chunkingConfig.chunkSamples
        let overlapSamples = Int(Float(chunkSamples) * chunkingConfig.overlapRatio)
        let stride = max(1, chunkSamples - overlapSamples)
        let window = MLXOverlap.triangularWindow(length: chunkSamples).asArray(Float.self)

        var assembler = IncrementalOverlapAdd(
            chunkSamples: chunkSamples,
            stride: stride,
            window: window,
            totalLength: frameCount
        )
        var output: [Float] = []
        output.reserveCapacity(frameCount)

        var startIdx = 0
        while startIdx < frameCount {
            try Task.checkCancellation()
            let endIdx = min(startIdx + chunkSamples, frameCount)
            var chunk = Self.stereoPair(
                from: input,
                frames: startIdx..<endIdx
            )
            if endIdx - startIdx < chunkSamples {
                let padding = MLXArray.zeros([2, chunkSamples - (endIdx - startIdx)])
                chunk = concatenated([chunk, padding], axis: 1)
            }
            eval(chunk)

            let sourceOutput = stemOutput(chunk, stem: stem, model: model)
            let monoOutput = mean(sourceOutput, axis: 0)
            eval(monoOutput)
            let chunkResult = monoOutput.asArray(Float.self)
            output.append(contentsOf: assembler.add(chunkResult, startIdx: startIdx))
            startIdx += stride
        }
        output.append(contentsOf: assembler.finish())
        if output.count > frameCount {
            output.removeLast(output.count - frameCount)
        }
        return AudioBuffer(samples: output, sampleRate: sampleRate, channels: 1)
    }
    
    /// Separate all sources
    public func separateAll(_ input: AudioBuffer) async throws -> [Stem: AudioBuffer] {
        var results: [Stem: AudioBuffer] = [:]
        for stem in Stem.allCases where models[stem] != nil {
            try Task.checkCancellation()
            results[stem] = try await separate(input, stem: stem)
        }
        return results
    }
    
    // MARK: - Background / Accompaniment Extraction
    
    /// Result containing vocals and accompaniment (instrumental) tracks
    public struct VocalsWithAccompaniment: Sendable {
        public let vocals: AudioBuffer
        public let accompaniment: AudioBuffer  // drums + bass + other
    }
    
    /// Separate audio into vocals and accompaniment (instrumental) tracks
    /// Accompaniment = drums + bass + other (everything except vocals)
    /// - Parameter audio: Input audio buffer at 44.1kHz
    /// - Returns: Vocals and accompaniment (instrumental) audio buffers
    public func separateVocalsWithAccompaniment(_ audio: AudioBuffer) async throws -> VocalsWithAccompaniment {
        // Ensure all models are loaded. Resolve the directory once so a cold cache
        // fetches the four stems in a single call rather than four.
        let missing = Stem.allCases.filter { models[$0] == nil }
        if !missing.isEmpty {
            let directory = try await resolveWeightsDirectory(for: missing)
            var candidates: [Stem: HTDemucs] = [:]
            for stem in missing {
                try Task.checkCancellation()
                candidates[stem] = try Self.loadModel(
                    stem: stem,
                    weightsDirectory: directory
                )
            }
            for (stem, model) in candidates {
                models[stem] = model
            }
        }
        
        // Get all 4 stems
        let allStems = try await separateAll(audio)
        
        guard let vocals = allStems[.vocals],
              let drums = allStems[.drums],
              let bass = allStems[.bass],
              let other = allStems[.other] else {
            throw AudioToolError.modelNotLoaded("Demucs - not all stems available")
        }
        
        // Combine non-vocal stems into accompaniment
        let minLen = min(drums.samples.count, bass.samples.count, other.samples.count)
        var accompanimentSamples = [Float](repeating: 0, count: minLen)
        for i in 0..<minLen {
            accompanimentSamples[i] = drums.samples[i] + bass.samples[i] + other.samples[i]
        }
        
        return VocalsWithAccompaniment(
            vocals: vocals,
            accompaniment: AudioBuffer(samples: accompanimentSamples, sampleRate: sampleRate, channels: 1)
        )
    }
    
}

// MARK: - ManagedModel

extension MossFormer2SSProvider: ManagedModel {
    /// Includes the variant: the 2-speaker, 3-speaker and WHAMR builds are separate
    /// weights and must not share a residency slot.
    public nonisolated var modelId: String { "mossformer2_ss_\(modelType.rawValue)" }

    /// Measured peak, 30 s at the model's rate on an M1 Pro: 5426 MiB for 2-speaker
    /// 16 kHz, 4514 for 3-speaker 8 kHz, 3939 for 2-speaker WHAMR. The 250 MB this
    /// declared was the checkpoint size - see `ManagedModel.estimatedMemoryBytes`.
    ///
    /// The largest footprint in the package, and by a distance: separation holds a
    /// full-length output per speaker alongside the attention state that produced
    /// them. One figure, taken from the worst of the three, because the residency
    /// manager decides whether to admit *this provider* and cannot know which
    /// configuration it will be asked for.
    public nonisolated var estimatedMemoryBytes: Int { 5_600_000_000 }

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

extension DemucsProvider: ManagedModel {
    public nonisolated var modelId: String { "demucs" }

    /// Measured peak, 30 s at 44.1 kHz on an M1 Pro: 3791 MiB for all four stems,
    /// 3231 for vocals alone. The 320 MB this declared was four checkpoints' worth
    /// of file size - see `ManagedModel.estimatedMemoryBytes`.
    ///
    /// Still deliberately the all-stems figure, and the original reasoning stands:
    /// stems load independently and a caller asking only for vocals holds less, but
    /// this property is nonisolated and cannot read how many are resident, so
    /// over-estimating is the safe direction. Note how little the two differ - 3231
    /// against 3791 - because each stem's checkpoint emits all four sources and the
    /// forward pass, not the weights, is what costs.
    public nonisolated var estimatedMemoryBytes: Int { 3_900_000_000 }

    /// Load every stem.
    ///
    /// ManagedModel treats a provider as one loadable unit, so this is all four. Use
    /// ``load(stem:)`` to bring up a single stem, which is what you want when you
    /// only need vocals. Residency-managed loading deliberately requires all four
    /// stems because `ManagedModel` treats this provider as one loadable unit.
    public func load() async throws {
        try await loadGate.run { [self] in
            // Before the first allocation, not at the first chunk boundary.
            // `trimIfNeeded` used to be the only thing that applied these, so a
            // provider ran its load and its opening chunks under MLX's own default
            // ceiling and the cache had already grown past the cap by the time the
            // cap arrived. Measured on Demucs: 5.1 GB peak applying it late against
            // 3.2 GB applying it here.
            MLXCachePolicy.applyProcessLimits()
            try await loadAll()
        }
    }

    public func checkIfLoaded() async -> Bool {
        Stem.allCases.allSatisfy { models[$0] != nil }
    }

    public func unload() async {
        // Cancel first: a load mid-download must not publish over this unload.
        // Bracketed: the gate stays shut until this method's state reset is
        // done, so a concurrent load cannot publish a model into the gap and
        // have it wiped by the lines below.
        let teardown = await loadGate.beginTeardown()
        defer { loadGate.endTeardown(teardown) }
        models.removeAll()
        GPU.clearCache()
    }
}
