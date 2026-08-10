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

// MARK: - MossFormer2 SR 48K Provider (Super Resolution)

/// MLX MossFormer2 Super Resolution - upsamples audio to 48kHz
/// Chunking: 4s chunks, 25% overlap, discard-edges - the reference's numbers.
public actor MossFormer2SR48KProvider: AudioUpscaler {
    
    /// HuggingFace repository for model weights
    public static let repo = ModelRepository.mossFormer2SR48K
    
    /// Supported precisions (fp32 only)
    public static let supportedPrecisions: [ModelPrecision] = [.fp32]
    
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
    public init(weightsPath: String, configPath: String) {
        self.weightsPath = weightsPath
        self.configPath = configPath
        self.precision = .fp32
    }
    
    /// Load model with config and weights (downloads if not cached)
    public func load() async throws {
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
        candidate.update(parameters: parameters)
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

        // Short inputs can be converted in one allocation.
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

        while let chunk = try source.nextChunk(chunkSamples: chunkSamples, stride: stride) {
            try Task.checkCancellation()
            let processed = try await processChunk(chunk.samples, model: model)
            output.append(contentsOf: assembler.add(processed.samples, startIdx: chunk.startIdx))
            completedChunks += 1
            MLXCachePolicy.trimIfNeeded(afterChunk: completedChunks)
        }
        output.append(contentsOf: assembler.finish())
        return AudioBuffer(
            samples: Array(output.prefix(totalLength)),
            sampleRate: outputSampleRate,
            channels: 1
        )
    }
    
    /// Process a single chunk
    private func processChunk(_ samples: [Float], model: MossFormer2_SR_48K) async throws -> AudioBuffer {
        let inputs = MLXArray(samples)
        let inputLen = inputs.shape[0]
        
        // Config values
        let hopSize = args["hop_size"] as? Int ?? 256
        let nFFT = args["n_fft"] as? Int ?? 1024
        let numMels = args["num_mels"] as? Int ?? 80
        let winSize = args["win_size"] as? Int ?? 1024
        let fmin = args["fmin"] as? Float ?? 0
        let fmax = args["fmax"] as? Float ?? 8000
        
        // Compute mel spectrogram
        let inputs2D = inputs.expandedDimensions(axis: 0)
        let melSpec = try melSpectrogram(
            inputs2D,
            nFFT: nFFT,
            numMels: numMels,
            samplingRate: 48000,
            hopSize: hopSize,
            winSize: winSize,
            fmin: fmin,
            fmax: fmax
        )
        eval(melSpec)
        
        // Run model
        let output = model(melSpec)
        eval(output)
        
        var outputs = output.squeezed()
        
        // Apply bandwidth substitution
        outputs = try bandwidthSub(inputs, outputs, fs: 48000)
        eval(outputs)
        
        // Trim to original length
        outputs = outputs[0..<inputLen]
        
        return AudioBuffer(samples: outputs.asArray(Float.self), sampleRate: outputSampleRate, channels: 1)
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

        while let chunk = try source.nextChunk(chunkSamples: chunkSamples, stride: stride) {
            try Task.checkCancellation()
            let processed = try await processChunk(chunk.samples, model: model)

            let ready = assembler.add(processed.samples, startIdx: chunk.startIdx)
            if !ready.isEmpty {
                if case .terminated = continuation.yield(AudioBuffer(
                    samples: ready,
                    sampleRate: outputSampleRate,
                    channels: 1
                )) {
                    throw CancellationError()
                }
            }

            completedChunks += 1
            MLXCachePolicy.trimIfNeeded(afterChunk: completedChunks)
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

    /// ~300 MB FP32.
    public nonisolated var estimatedMemoryBytes: Int { 300_000_000 }

    public func checkIfLoaded() async -> Bool { model != nil }

    public func unload() async {
        model = nil
        GPU.clearCache()
    }
}
