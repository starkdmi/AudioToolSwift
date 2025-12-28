//
//  MLXSuperResolutionProvider.swift
//  ClearVoiceMLX
//
//  MLX-based super resolution provider with chunking support
//

import Foundation
import ClearVoice
import ClearVoiceCore
import MLX
import MLXNN
import AudioUtils
import MossFormer2SR

// MARK: - MossFormer2 SR 48K Provider (Super Resolution)

/// MLX MossFormer2 Super Resolution - upsamples audio to 48kHz
/// Chunking: 4s chunks, 50% overlap, Hann window (from benchmarks)
public final class MossFormer2SR48KProvider: AudioUpscaler, @unchecked Sendable {
    
    // AudioUpscaler conformance
    public var inputSampleRate: Int { 16000 }
    public var outputSampleRate: Int { 48000 }
    
    // AudioProcessor conformance
    public var sampleRate: Int { outputSampleRate }
    public let inputChannels: Int = 1
    public let outputChannels: Int = 1
    
    private var model: MossFormer2_SR_48K?
    private var args: AttrDict = AttrDict()
    private let weightsPath: String
    private let configPath: String
    
    /// Max audio without chunking (seconds at 48kHz)
    private let maxDirectDuration: Float = 4.0
    
    public init(weightsPath: String, configPath: String) {
        self.weightsPath = weightsPath
        self.configPath = configPath
    }
    
    /// Load model with config and weights
    public func load() async throws {
        let configData = try Data(contentsOf: URL(fileURLWithPath: configPath))
        let modelConfig = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
        
        args = AttrDict(modelConfig)
        args["one_time_decode_length"] = 20.0
        args["decode_window"] = 4.0
        
        model = MossFormer2_SR_48K(args: args)
        
        let weights = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath))
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
            throw ClearVoiceError.modelNotLoaded("MossFormer2_SR_48K")
        }
        
        // Resample input to 48kHz first
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
