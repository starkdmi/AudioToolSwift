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
/// Chunking: 4s chunks, 50% overlap, Hann window (from benchmarks)
public actor MossFormer2SR48KProvider: AudioUpscaler {
    
    /// HuggingFace repository for model weights
    public static let repo = "starkdmi/MossFormer2_SR_48K_MLX"
    
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
    public nonisolated let outputChannels: Int = 1
    
    private var model: MossFormer2_SR_48K?
    private var args: AttrDict = AttrDict()
    private let weightsPath: String?
    private let configPath: String?
    private let precision: ModelPrecision
    
    /// Max audio without chunking (seconds at 48kHz)
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
        let resolvedWeightsPath: String
        let resolvedConfigPath: String
        
        if let wPath = weightsPath, let cPath = configPath {
            resolvedWeightsPath = wPath
            resolvedConfigPath = cPath
        } else {
            // Check if already downloaded
            if let cached = ModelDownloader.shared.localPath(for: Self.repo) {
                let modelPath = cached.appendingPathComponent(precision.weightsFilename).path
                let configPathLocal = cached.appendingPathComponent("config.json").path
                if FileManager.default.fileExists(atPath: modelPath) && FileManager.default.fileExists(atPath: configPathLocal) {
                    resolvedWeightsPath = modelPath
                    resolvedConfigPath = configPathLocal
                } else {
                    throw AudioToolError.modelNotFound("Precision \(precision.rawValue) weights or config not found for MossFormer2SR48K")
                }
            } else {
                // Auto-download from HuggingFace
                let modelDir = try await ModelDownloader.shared.downloadAndGetPath(
                    repo: Self.repo,
                    matching: [precision.weightsFilename, "config.json"]
                )
                resolvedWeightsPath = modelDir.appendingPathComponent(precision.weightsFilename).path
                resolvedConfigPath = modelDir.appendingPathComponent("config.json").path
            }
        }
        
        let configData = try Data(contentsOf: URL(fileURLWithPath: resolvedConfigPath))
        let modelConfig = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
        
        args = AttrDict(modelConfig)
        args["one_time_decode_length"] = 20.0
        args["decode_window"] = 4.0
        
        model = MossFormer2_SR_48K(args: args)
        
        let weights = try MLX.loadArrays(url: URL(fileURLWithPath: resolvedWeightsPath))
        let filteredWeights = weights.filter { !$0.key.contains("num_batches_tracked") }
        let parameters = ModuleParameters.unflattened(filteredWeights)
        model?.update(parameters: parameters)
        
        if let model = model {
            eval(model)
        }
    }
    
    /// Upsample audio to 48kHz
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormer2_SR_48K")
        }
        
        try validateSampleRate(input)
        
        // Interpolate up to the output rate. This is not a format conversion the
        // caller could have done instead - the model operates at 48 kHz and
        // reconstructs the band that 16 kHz input cannot carry, so the upsample is
        // the first step of inference rather than plumbing.
        let audio48k = try await resampleTo48k(input)
        
        let durationSeconds = Float(audio48k.count) / Float(outputSampleRate)
        
        // Use chunking for longer audio
        if durationSeconds > maxDirectDuration {
            return try await processWithChunking(audio48k, model: model)
        }
        
        return try await processChunk(audio48k, model: model)
    }
    
    /// Process with chunking using MLXOverlap
    /// Uses Hann weighted overlap-add for smooth blending
    private func processWithChunking(_ samples: [Float], model: MossFormer2_SR_48K) async throws -> AudioBuffer {
        let chunkingConfig = ChunkingConfig.mossformer2SR48K(sampleRate: outputSampleRate)
        let inputMLX = MLXArray(samples)
        
        // Split into chunks using MLXOverlap
        let chunks = MLXOverlap.split(
            audio: inputMLX,
            chunkSamples: chunkingConfig.chunkSamples,
            stride: chunkingConfig.strideSamples
        )
        
        // Process each chunk through model
        var processedChunks: [(chunk: MLXArray, startIdx: Int)] = []
        
        for (chunk, startIdx) in chunks {
            eval(chunk)
            let chunkSamples = chunk.asArray(Float.self)
            let processed = try await processChunk(chunkSamples, model: model)
            processedChunks.append((MLXArray(processed.samples), startIdx))
            
            // Clear GPU cache between chunks to reduce peak memory
            GPU.clearCache()
        }
        
        // Reassemble using Hann window for smooth blending
        let window = MLXOverlap.hannWindow(length: chunkingConfig.chunkSamples)
        let reassembled = MLXOverlap.reassembleOverlapAdd(
            processedChunks: processedChunks,
            chunkSamples: chunkingConfig.chunkSamples,
            stride: chunkingConfig.strideSamples,
            window: window,
            originalLength: samples.count
        )
        eval(reassembled)
        
        return AudioBuffer(samples: reassembled.asArray(Float.self), sampleRate: outputSampleRate, channels: 1)
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
    
    /// Resample input to 48kHz
    private func resampleTo48k(_ input: AudioBuffer) async throws -> [Float] {
        let tempDir = FileManager.default.temporaryDirectory
        let inputPath = tempDir.appendingPathComponent("sr_input_\(UUID().uuidString).wav").path
        
        defer {
            try? FileManager.default.removeItem(atPath: inputPath)
        }
        
        // Save input at its native sample rate
        let saverConfig = AudioSaver.Configuration(sampleRate: Double(input.sampleRate))
        let saver = AudioSaver(config: saverConfig)
        try saver.save(MLXArray(input.samples), to: inputPath)
        
        // Load and resample to 48kHz
        let loaderConfig = AudioLoader.Configuration(
            targetSampleRate: 48000,
            normalizationMode: .none
        )
        let loader = AudioLoader(config: loaderConfig)
        let audio48k = try loader.loadMono(from: URL(fileURLWithPath: inputPath))
        eval(audio48k)
        
        return audio48k.asArray(Float.self)
    }
}

// MARK: - StreamableOutput Conformance

extension MossFormer2SR48KProvider: StreamableOutput {
    /// Process audio and stream output chunks as they're ready.
    /// Uses Hann-window overlap-add with buffered blending.
    public nonisolated func processStream(_ input: AudioBuffer) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.processStreamImpl(input, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
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
        
        // Resample to 48kHz first
        let samples = try await resampleTo48k(input)
        let totalLength = samples.count
        let durationSeconds = Float(totalLength) / Float(outputSampleRate)
        
        // For short audio, just yield single result
        if durationSeconds <= maxDirectDuration {
            let result = try await processChunk(samples, model: model)
            continuation.yield(result)
            continuation.finish()
            return
        }
        
        // Streaming with chunking and overlap buffer
        let chunkingConfig = ChunkingConfig.mossformer2SR48K(sampleRate: outputSampleRate)
        let chunkSamples = chunkingConfig.chunkSamples
        let stride = chunkingConfig.strideSamples
        let window = MLXOverlap.hannWindow(length: chunkSamples).asArray(Float.self)
        
        let inputMLX = MLXArray(samples)
        let chunks = MLXOverlap.split(
            audio: inputMLX,
            chunkSamples: chunkSamples,
            stride: stride
        )
        
        // Streaming must produce the samples batch would. `IncrementalOverlapAdd`
        // is the same weighted overlap-add as `MLXOverlap.reassembleOverlapAdd`,
        // fed chunk by chunk.
        //
        // What was here before multiplied the first chunk by the rising half of the
        // Hann window and never divided by the accumulated weight, so every stream
        // opened with a `stride`-long fade-in from silence - a second and a half at
        // 48 kHz - and the middle chunks were weighted but unnormalized too.
        // Attaching a progress handler was enough to get that instead of the real
        // output.
        var assembler = IncrementalOverlapAdd(
            chunkSamples: chunkSamples,
            stride: stride,
            window: window,
            totalLength: totalLength
        )

        for (chunk, startIdx) in chunks {
            eval(chunk)
            let chunkSamplesArray = chunk.asArray(Float.self)
            let processed = try await processChunk(chunkSamplesArray, model: model)

            let ready = assembler.add(processed.samples, startIdx: startIdx)
            if !ready.isEmpty {
                continuation.yield(AudioBuffer(samples: ready, sampleRate: outputSampleRate, channels: 1))
            }

            GPU.clearCache()
        }

        let tail = assembler.finish()
        if !tail.isEmpty {
            continuation.yield(AudioBuffer(samples: tail, sampleRate: outputSampleRate, channels: 1))
        }

        continuation.finish()
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
