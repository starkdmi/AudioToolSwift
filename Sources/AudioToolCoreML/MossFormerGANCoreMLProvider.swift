//
//  MossFormerGANCoreMLProvider.swift
//  AudioToolCoreML
//
//  CoreML-based MossFormer GAN Speech Enhancement (16kHz)
//  Uses MLX for STFT/ISTFT, CoreML for model inference
//

import Foundation
@preconcurrency import CoreML
import Accelerate
import AudioTool
import AudioToolCore
import MLX
import MLXNN

// MARK: - MossFormer GAN CoreML Provider

/// CoreML MossFormer GAN Speech Enhancement (16kHz)
/// Uses Accelerate for STFT/ISTFT and CoreML for model inference
public actor MossFormerGANCoreMLProvider: SpeechEnhancer {
    
    public nonisolated let sampleRate: Int = 16000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1
    public nonisolated let minChunkSize: Int = 3200   // 0.2s at 16kHz
    public nonisolated let recommendedChunkSize: Int = 25500  // 1.594s at 16kHz (256 frames)
    
    // Audio parameters (must match model training)
    private let nFFT: Int = 400
    private let hopLength: Int = 100
    private let winLength: Int = 400
    private let powerCompress: Float = 0.3
    
    // Segment duration: 256 frames = 25500 samples = 1.594s
    private let segmentSamples: Int = 25500
    
    private var model: MLModel?
    private let modelPath: String
    private let computeUnits: MLComputeUnits
    
    // MLX STFT window
    private let mlxWindow: MLXArray
    
    public init(modelPath: String, computeUnits: MLComputeUnits = .cpuAndGPU) {
        self.modelPath = modelPath
        self.computeUnits = computeUnits
        
        // Create periodic Hann window for MLX STFT
        self.mlxWindow = createPeriodicHannWindow(length: winLength)
    }
    
    /// Load CoreML model (auto-compiles .mlpackage if needed)
    public func load() async throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        
        var modelURL = URL(fileURLWithPath: modelPath)
        
        // If it's an .mlpackage, compile it first
        if modelPath.hasSuffix(".mlpackage") {
            let compiledURL = try await MLModel.compileModel(at: modelURL)
            modelURL = compiledURL
        }
        
        model = try await MLModel.load(contentsOf: modelURL, configuration: config)
    }
    
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormerGAN_CoreML")
        }
        try validateSampleRate(input)
        
        // Process in segments for long audio
        if input.samples.count <= segmentSamples {
            return try await processSegment(input.samples, model: model)
        }
        
        // Process in segments without overlap (as per model recommendation)
        return try await processWithChunking(input.samples, model: model)
    }
    
    // MARK: - Background Extraction
    
    /// Result containing both enhanced speech and background/residual audio
    public struct EnhancedWithBackground: Sendable {
        public let enhanced: AudioBuffer
        public let background: AudioBuffer
    }
    
    /// Process audio and extract both enhanced speech and background track
    /// Uses mask-based spectral extraction for high-quality background separation
    public func processWithBackground(_ input: AudioBuffer) async throws -> EnhancedWithBackground {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormerGAN_CoreML")
        }
        
        // Process in segments for long audio
        if input.samples.count <= segmentSamples {
            return try await processSegmentWithBackground(input.samples, model: model)
        }
        
        // Process in segments without overlap
        return try await processWithChunkingAndBackground(input.samples, model: model)
    }
    
    /// Process a single segment with background extraction using MLX
    private func processSegmentWithBackground(_ samples: [Float], model: MLModel) async throws -> EnhancedWithBackground {
        // Pad to segment size if needed
        var paddedSamples = samples
        let trimStart: Int
        
        if samples.count < segmentSamples {
            let needPadding = segmentSamples - samples.count
            paddedSamples = [Float](repeating: 0, count: needPadding) + samples
            trimStart = needPadding
        } else {
            trimStart = 0
        }
        
        // Convert to MLXArray [1, samples]
        let inputMLX = MLXArray(paddedSamples).expandedDimensions(axis: 0).asType(.float32)
        
        // Normalize using MLX
        let inputLen = MLXArray(Float(paddedSamples.count))
        let sumSquares = sum(inputMLX * inputMLX, axis: 1, keepDims: true)
        let normFactor = sqrt(inputLen / (sumSquares + 1e-9))
        let normedAudio = inputMLX * normFactor
        
        // MLX STFT -> [batch, freq, time] (keep noisy for background extraction)
        let (noisyReal, noisyImag) = mlxSTFT(
            normedAudio,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: mlxWindow,
            center: true
        )
        
        // Power compress for model input
        let (realC, imagC) = mlxPowerCompress(real: noisyReal, imag: noisyImag)
        
        // Prepare for CoreML: [1, 2, T, F]
        let coremlInput = try prepareForCoreMLFromMLX(real: realC, imag: imagC)
        
        // CoreML inference
        let prediction = try await model.prediction(from: coremlInput)
        
        // Parse output -> [batch, freq, time]
        let (enhancedReal, enhancedImag) = parseModelOutputToMLX(prediction)
        
        // Power uncompress for enhanced
        let (realUC, imagUC) = mlxPowerUncompress(real: enhancedReal, imag: enhancedImag)
        
        // Compute magnitude mask and extract background using MLX
        let (backgroundReal, backgroundImag) = mlxComputeBackgroundSpectrogram(
            noisyReal: noisyReal,
            noisyImag: noisyImag,
            enhancedReal: realUC,
            enhancedImag: imagUC
        )
        
        // MLX ISTFT for enhanced
        let enhancedRecon = mlxISTFT(
            realPart: realUC,
            imagPart: imagUC,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: mlxWindow,
            center: true,
            audioLength: paddedSamples.count
        )
        
        // MLX ISTFT for background
        let backgroundRecon = mlxISTFT(
            realPart: backgroundReal,
            imagPart: backgroundImag,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: mlxWindow,
            center: true,
            audioLength: paddedSamples.count
        )
        
        // De-normalize both
        let enhancedOut = (enhancedRecon / normFactor).squeezed(axis: 0)
        let backgroundOut = (backgroundRecon / normFactor).squeezed(axis: 0)
        eval(enhancedOut, backgroundOut)
        
        // Convert to [Float] and trim
        let enhancedSamples = Array(enhancedOut.asArray(Float.self).suffix(from: trimStart))
        let backgroundSamples = Array(backgroundOut.asArray(Float.self).suffix(from: trimStart))
        
        return EnhancedWithBackground(
            enhanced: AudioBuffer(samples: enhancedSamples, sampleRate: sampleRate, channels: 1),
            background: AudioBuffer(samples: backgroundSamples, sampleRate: sampleRate, channels: 1)
        )
    }
    
    /// Compute background spectrogram using inverse magnitude mask (MLX version)
    private func mlxComputeBackgroundSpectrogram(
        noisyReal: MLXArray,
        noisyImag: MLXArray,
        enhancedReal: MLXArray,
        enhancedImag: MLXArray
    ) -> (MLXArray, MLXArray) {
        // Compute magnitudes
        let noisyMag = sqrt(noisyReal * noisyReal + noisyImag * noisyImag + 1e-9)
        let enhancedMag = sqrt(enhancedReal * enhancedReal + enhancedImag * enhancedImag + 1e-9)
        
        // Compute soft mask: enhanced / noisy (clipped to [0, 1])
        let mask = clip(enhancedMag / (noisyMag + 1e-9), min: MLXArray(0.0), max: MLXArray(1.0))
        
        // Background = noisy * (1 - mask)
        let inverseMask = 1.0 - mask
        let backgroundReal = noisyReal * inverseMask
        let backgroundImag = noisyImag * inverseMask
        
        return (backgroundReal, backgroundImag)
    }
    
    /// Process long audio in segments with background extraction
    private func processWithChunkingAndBackground(_ samples: [Float], model: MLModel) async throws -> EnhancedWithBackground {
        let numSegments = Int(ceil(Double(samples.count) / Double(segmentSamples)))
        var enhancedSegments: [Float] = []
        var backgroundSegments: [Float] = []
        
        for idx in 0..<numSegments {
            let start = idx * segmentSamples
            let end = min(start + segmentSamples, samples.count)
            let actualLen = end - start
            
            // For short final segment, use context from previous audio
            var segment: [Float]
            let trimAmount: Int
            
            if actualLen < segmentSamples {
                let needExtra = segmentSamples - actualLen
                let contextStart = max(0, start - needExtra)
                segment = Array(samples[contextStart..<end])
                
                if segment.count < segmentSamples {
                    let padAmt = segmentSamples - segment.count
                    segment = [Float](repeating: 0, count: padAmt) + segment
                    trimAmount = padAmt + (segment.count - actualLen - padAmt)
                } else {
                    trimAmount = segment.count - actualLen
                }
            } else {
                segment = Array(samples[start..<end])
                trimAmount = 0
            }
            
            let result = try await processSegmentWithBackground(segment, model: model)
            
            // Trim to actual portion
            if trimAmount > 0 && trimAmount < result.enhanced.samples.count {
                enhancedSegments.append(contentsOf: result.enhanced.samples.suffix(result.enhanced.samples.count - trimAmount).prefix(actualLen))
                backgroundSegments.append(contentsOf: result.background.samples.suffix(result.background.samples.count - trimAmount).prefix(actualLen))
            } else {
                enhancedSegments.append(contentsOf: result.enhanced.samples.prefix(actualLen))
                backgroundSegments.append(contentsOf: result.background.samples.prefix(actualLen))
            }
        }
        
        return EnhancedWithBackground(
            enhanced: AudioBuffer(samples: enhancedSegments, sampleRate: sampleRate, channels: 1),
            background: AudioBuffer(samples: backgroundSegments, sampleRate: sampleRate, channels: 1)
        )
    }
    
    /// Process a single segment (up to 1.594s) using MLX STFT/ISTFT
    private func processSegment(_ samples: [Float], model: MLModel) async throws -> AudioBuffer {
        // Pad to segment size if needed
        var paddedSamples = samples
        let trimStart: Int
        
        if samples.count < segmentSamples {
            let needPadding = segmentSamples - samples.count
            paddedSamples = [Float](repeating: 0, count: needPadding) + samples
            trimStart = needPadding
        } else {
            trimStart = 0
        }
        
        // Convert to MLXArray [1, samples]
        let inputMLX = MLXArray(paddedSamples).expandedDimensions(axis: 0).asType(.float32)
        
        // Normalize using MLX
        let inputLen = MLXArray(Float(paddedSamples.count))
        let sumSquares = sum(inputMLX * inputMLX, axis: 1, keepDims: true)
        let normFactor = sqrt(inputLen / (sumSquares + 1e-9))
        let normedAudio = inputMLX * normFactor
        
        // MLX STFT -> [batch, freq, time]
        let (stftReal, stftImag) = mlxSTFT(
            normedAudio,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: mlxWindow,
            center: true
        )
        
        // Power compress using MLX
        let (realC, imagC) = mlxPowerCompress(real: stftReal, imag: stftImag)
        
        // Prepare for CoreML: [1, 2, T, F] (transpose from [1, F, T])
        let coremlInput = try prepareForCoreMLFromMLX(real: realC, imag: imagC)
        
        // CoreML inference
        let prediction = try await model.prediction(from: coremlInput)
        
        // Parse output -> [batch, freq, time]
        let (enhancedReal, enhancedImag) = parseModelOutputToMLX(prediction)
        
        // Power uncompress using MLX
        let (realUC, imagUC) = mlxPowerUncompress(real: enhancedReal, imag: enhancedImag)
        
        // MLX ISTFT -> [batch, samples]
        let reconstructed = mlxISTFT(
            realPart: realUC,
            imagPart: imagUC,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: mlxWindow,
            center: true,
            audioLength: paddedSamples.count
        )
        
        // De-normalize
        let output = (reconstructed / normFactor).squeezed(axis: 0)
        eval(output)
        
        // Convert to [Float] and trim
        let outputSamples = output.asArray(Float.self)
        let finalSamples = Array(outputSamples.suffix(from: trimStart))
        
        return AudioBuffer(samples: finalSamples, sampleRate: sampleRate, channels: 1)
    }
    
    /// Process long audio in segments (no overlap per model recommendation)
    private func processWithChunking(_ samples: [Float], model: MLModel) async throws -> AudioBuffer {
        let numSegments = Int(ceil(Double(samples.count) / Double(segmentSamples)))
        var enhancedSegments: [Float] = []
        
        for idx in 0..<numSegments {
            let start = idx * segmentSamples
            let end = min(start + segmentSamples, samples.count)
            let actualLen = end - start
            
            // For short final segment, use context from previous audio
            var segment: [Float]
            let trimAmount: Int
            
            if actualLen < segmentSamples {
                let needExtra = segmentSamples - actualLen
                let contextStart = max(0, start - needExtra)
                segment = Array(samples[contextStart..<end])
                
                if segment.count < segmentSamples {
                    // Still short - pad with zeros at start
                    let padAmt = segmentSamples - segment.count
                    segment = [Float](repeating: 0, count: padAmt) + segment
                    trimAmount = padAmt + (segment.count - actualLen - padAmt)
                } else {
                    trimAmount = segment.count - actualLen
                }
            } else {
                segment = Array(samples[start..<end])
                trimAmount = 0
            }
            
            let result = try await processSegment(segment, model: model)
            
            // Trim to actual portion
            if trimAmount > 0 && trimAmount < result.samples.count {
                enhancedSegments.append(contentsOf: result.samples.suffix(result.samples.count - trimAmount).prefix(actualLen))
            } else {
                enhancedSegments.append(contentsOf: result.samples.prefix(actualLen))
            }
        }
        
        return AudioBuffer(samples: enhancedSegments, sampleRate: sampleRate, channels: 1)
    }
    
    // MARK: - MLX Power Compression
    
    /// Power compress spectrogram using MLX (matches Python exactly)
    private func mlxPowerCompress(real: MLXArray, imag: MLXArray) -> (MLXArray, MLXArray) {
        let mag = sqrt(real * real + imag * imag + 1e-8)
        let phase = atan2(imag, real)
        let magC = pow(mag, MLXArray(powerCompress))
        return (magC * cos(phase), magC * sin(phase))
    }
    
    /// Power uncompress spectrogram using MLX
    private func mlxPowerUncompress(real: MLXArray, imag: MLXArray) -> (MLXArray, MLXArray) {
        let mag = sqrt(real * real + imag * imag + 1e-8)
        let phase = atan2(imag, real)
        let magUC = pow(mag, MLXArray(1.0 / powerCompress))
        return (magUC * cos(phase), magUC * sin(phase))
    }
    
    // MARK: - MLX CoreML I/O
    
    /// Prepare MLXArray spectrogram for CoreML input [1, 2, T, F]
    private func prepareForCoreMLFromMLX(real: MLXArray, imag: MLXArray) throws -> MLFeatureProvider {
        // real/imag are [batch=1, F, T], need [1, 2, T, F]
        let realT = real.transposed(0, 2, 1)  // [1, T, F]
        let imagT = imag.transposed(0, 2, 1)  // [1, T, F]
        
        eval(realT, imagT)
        
        let T = realT.shape[1]
        let F = realT.shape[2]
        
        // Stack as [1, 2, T, F]
        let spec = MLX.stacked([realT, imagT], axis: 1)  // [1, 2, T, F]
        eval(spec)
        
        // Convert to MLMultiArray
        let shape: [NSNumber] = [1, 2, NSNumber(value: T), NSNumber(value: F)]
        let multiArray = try MLMultiArray(shape: shape, dataType: .float32)
        
        // Copy data using buffer pointer (much faster than element-by-element)
        let flatData = spec.flattened().asArray(Float.self)
        let dataPointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        flatData.withUnsafeBufferPointer { srcPtr in
            dataPointer.update(from: srcPtr.baseAddress!, count: flatData.count)
        }
        
        return try MLDictionaryFeatureProvider(dictionary: ["spectrogram": multiArray])
    }
    
    /// Parse CoreML output to MLXArray [batch, freq, time]
    private func parseModelOutputToMLX(_ output: MLFeatureProvider) -> (MLXArray, MLXArray) {
        guard let multiArray = output.featureValue(for: "enhanced_spectrogram")?.multiArrayValue ??
                              output.featureValue(for: output.featureNames.first ?? "")?.multiArrayValue else {
            return (MLXArray.zeros([1, nFFT / 2 + 1, 1]), MLXArray.zeros([1, nFFT / 2 + 1, 1]))
        }
        
        // Output is [1, 2, T, F]
        let T = multiArray.shape[2].intValue
        let F = multiArray.shape[3].intValue

        // Read through the array's own strides rather than assuming the backing
        // buffer is packed.
        //
        // It is not. CoreML returns this model's output as [1, 2, 256, 201] with
        // strides [106496, 53248, 208, 1] - the innermost dimension is 201 wide but
        // rows sit 208 floats apart, padded up to a 16-float boundary. A flat copy
        // of 2*T*F floats therefore reads every row after the first 7 floats
        // further out of place than the last, and 256 rows in the spectrogram is
        // sheared beyond recognition. That is what produced ganse_enhanced.wav.
        //
        // Python never hits this: coremltools unpacks MLMultiArray into a proper
        // C-contiguous ndarray before predict() returns, which is why run.py needs
        // no equivalent and its outputs were always correct. Swift's dataPointer is
        // the raw backing store, padding included.
        //
        // Measured on one segment, same prediction read both ways: flat copy gives
        // -1.1 dB against the MLX Python reference, this gives 129.3 dB. The
        // padding is present under every compute unit, so this path never worked.
        //
        // The fast path is kept for the case where CoreML does hand back packed
        // data, which is what the shape implies and what a smaller F would give.
        let strides = multiArray.strides.map(\.intValue)
        let totalCount = 2 * T * F
        var flatData = [Float](repeating: 0, count: totalCount)
        let srcPointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)

        if strides == [totalCount, T * F, F, 1] {
            flatData.withUnsafeMutableBufferPointer { dstPtr in
                dstPtr.baseAddress!.update(from: srcPointer, count: totalCount)
            }
        } else {
            flatData.withUnsafeMutableBufferPointer { dstPtr in
                guard let destination = dstPtr.baseAddress else { return }
                for part in 0..<2 {
                    for frame in 0..<T {
                        let row = part * strides[1] + frame * strides[2]
                        let offset = (part * T + frame) * F
                        if strides[3] == 1 {
                            // Rows are contiguous even though the array is not:
                            // copy each one whole rather than bin by bin.
                            destination.advanced(by: offset)
                                .update(from: srcPointer.advanced(by: row), count: F)
                        } else {
                            for bin in 0..<F {
                                destination[offset + bin] = srcPointer[row + bin * strides[3]]
                            }
                        }
                    }
                }
            }
        }

        let mlxData = MLXArray(flatData).reshaped([1, 2, T, F])
        
        // Extract real/imag and transpose to [1, F, T]
        let realT = mlxData[0..., 0, 0..., 0...]  // [1, T, F]
        let imagT = mlxData[0..., 1, 0..., 0...]  // [1, T, F]
        
        let real = realT.transposed(0, 2, 1)  // [1, F, T]
        let imag = imagT.transposed(0, 2, 1)  // [1, F, T]
        
        eval(real, imag)
        return (real, imag)
    }
    
    // MARK: - Streaming
    
    public nonisolated func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            Task {
                for await chunk in input {
                    do {
                        let processed = try await process(chunk)
                        continuation.yield(processed)
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
                continuation.finish()
            }
        }
    }
    
    public func reset() async {}
    
    // MARK: - Output Streaming
    
    /// Process audio and stream output chunks as they're ready.
    /// Uses no overlap - each segment is processed independently.
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
    
    /// Internal implementation for streaming - runs within actor context
    private func processStreamImpl(_ input: AudioBuffer, continuation: AsyncThrowingStream<AudioBuffer, Error>.Continuation) async throws {
        guard let model = model else {
            throw AudioToolError.modelNotLoaded("MossFormerGAN_CoreML")
        }
        
        let samples = input.samples
        let totalLength = samples.count
        
        // For short audio, just yield single result
        if totalLength <= segmentSamples {
            let result = try await processSegment(samples, model: model)
            continuation.yield(result)
            continuation.finish()
            return
        }
        
        // Process in segments (no overlap per model recommendation)
        let numSegments = Int(ceil(Double(totalLength) / Double(segmentSamples)))
        
        for idx in 0..<numSegments {
            let start = idx * segmentSamples
            let end = min(start + segmentSamples, totalLength)
            let actualLen = end - start
            
            // For short final segment, use context from previous audio
            var segment: [Float]
            let trimAmount: Int
            
            if actualLen < segmentSamples {
                let needExtra = segmentSamples - actualLen
                let contextStart = max(0, start - needExtra)
                segment = Array(samples[contextStart..<end])
                
                if segment.count < segmentSamples {
                    let padAmt = segmentSamples - segment.count
                    segment = [Float](repeating: 0, count: padAmt) + segment
                    trimAmount = padAmt + (segment.count - actualLen - padAmt)
                } else {
                    trimAmount = segment.count - actualLen
                }
            } else {
                segment = Array(samples[start..<end])
                trimAmount = 0
            }
            
            let result = try await processSegment(segment, model: model)
            
            // Trim to actual portion
            var outputSamples: [Float]
            if trimAmount > 0 && trimAmount < result.samples.count {
                outputSamples = Array(result.samples.suffix(result.samples.count - trimAmount).prefix(actualLen))
            } else {
                outputSamples = Array(result.samples.prefix(actualLen))
            }
            
            let chunkBuffer = AudioBuffer(
                samples: outputSamples,
                sampleRate: sampleRate,
                channels: 1
            )
            continuation.yield(chunkBuffer)
        }
        
        continuation.finish()
    }
}

// MARK: - StreamableOutput Conformance

extension MossFormerGANCoreMLProvider: StreamableOutput {}

// MARK: - ManagedModel

extension MossFormerGANCoreMLProvider: ManagedModel {
    public nonisolated var modelId: String { "mossformer_gan_se_16k" }

    /// ~30 MB: CoreML FP16, and the ANE holds much of it outside our accounting.
    public nonisolated var estimatedMemoryBytes: Int { 30_000_000 }

    public func checkIfLoaded() async -> Bool { model != nil }

    public func unload() async {
        model = nil
    }
}
