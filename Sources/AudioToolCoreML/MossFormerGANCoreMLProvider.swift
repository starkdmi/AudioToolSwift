//
//  MossFormerGANCoreMLProvider.swift
//  ClearVoiceCoreML
//
//  CoreML-based MossFormer GAN Speech Enhancement (16kHz)
//  Uses Accelerate for STFT/ISTFT
//

import Foundation
import CoreML
import Accelerate
import ClearVoice
import ClearVoiceCore

// MARK: - MossFormer GAN CoreML Provider

/// CoreML MossFormer GAN Speech Enhancement (16kHz)
/// Uses Accelerate for STFT/ISTFT and CoreML for model inference
public final class MossFormerGANCoreMLProvider: SpeechEnhancer, @unchecked Sendable {
    
    public let sampleRate: Int = 16000
    public let inputChannels: Int = 1
    public let outputChannels: Int = 1
    public let minChunkSize: Int = 3200   // 0.2s at 16kHz
    public let recommendedChunkSize: Int = 25500  // 1.594s at 16kHz (256 frames)
    
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
    
    // Accelerate FFT setup
    private var fftSetup: FFTSetup?
    private var log2n: UInt
    private var window: [Float]
    
    public init(modelPath: String, computeUnits: MLComputeUnits = .cpuAndGPU) {
        self.modelPath = modelPath
        self.computeUnits = computeUnits
        
        // FFT setup
        self.log2n = UInt(log2(Double(nFFT)))
        self.fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))
        
        // Create Hann window
        self.window = [Float](repeating: 0, count: winLength)
        vDSP_hann_window(&self.window, vDSP_Length(winLength), Int32(vDSP_HANN_NORM))
    }
    
    deinit {
        if let setup = fftSetup {
            vDSP_destroy_fftsetup(setup)
        }
    }
    
    /// Load CoreML model
    public func load() async throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        
        let modelURL = URL(fileURLWithPath: modelPath)
        model = try await MLModel.load(contentsOf: modelURL, configuration: config)
    }
    
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        guard let model = model else {
            throw ClearVoiceError.modelNotLoaded("MossFormerGAN_CoreML")
        }
        
        // Process in segments for long audio
        if input.samples.count <= segmentSamples {
            return try await processSegment(input.samples, model: model)
        }
        
        // Process in segments without overlap (as per model recommendation)
        return try await processWithChunking(input.samples, model: model)
    }
    
    /// Process a single segment (up to 1.594s)
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
        
        // Normalize
        let inputLen = Float(paddedSamples.count)
        var sumSquares: Float = 0
        vDSP_svesq(paddedSamples, 1, &sumSquares, vDSP_Length(paddedSamples.count))
        let normFactor = sqrt(inputLen / (sumSquares + 1e-9))
        var normedAudio = [Float](repeating: 0, count: paddedSamples.count)
        vDSP_vsmul(paddedSamples, 1, [normFactor], &normedAudio, 1, vDSP_Length(paddedSamples.count))
        
        // STFT using Accelerate
        let (stftReal, stftImag) = performSTFT(normedAudio)
        
        // Power compress
        let (realC, imagC) = powerCompress(real: stftReal, imag: stftImag)
        
        // Prepare for CoreML: [1, 2, T, F]
        let coremlInput = try prepareForCoreML(real: realC, imag: imagC)
        
        // CoreML inference
        let prediction = try await model.prediction(from: coremlInput)
        
        // Parse output
        let (enhancedReal, enhancedImag) = parseModelOutput(prediction)
        
        // Power uncompress
        let (realUC, imagUC) = powerUncompress(real: enhancedReal, imag: enhancedImag)
        
        // ISTFT using Accelerate
        var reconstructed = performISTFT(real: realUC, imag: imagUC, originalLength: paddedSamples.count)
        
        // De-normalize
        var invNorm = 1.0 / normFactor
        vDSP_vsmul(reconstructed, 1, &invNorm, &reconstructed, 1, vDSP_Length(reconstructed.count))
        
        // Trim to original length
        let outputSamples = Array(reconstructed[trimStart...])
        
        return AudioBuffer(samples: outputSamples, sampleRate: sampleRate, channels: 1)
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
    
    // MARK: - STFT/ISTFT using Accelerate
    
    private func performSTFT(_ samples: [Float]) -> ([[Float]], [[Float]]) {
        guard let fftSetup = fftSetup else { return ([], []) }
        
        // Pad for center mode
        let padAmount = nFFT / 2
        var paddedSamples = [Float](repeating: 0, count: padAmount) + samples + [Float](repeating: 0, count: padAmount)
        
        let numFrames = (paddedSamples.count - winLength) / hopLength + 1
        let freqBins = nFFT / 2 + 1
        
        var realFrames = [[Float]](repeating: [Float](repeating: 0, count: numFrames), count: freqBins)
        var imagFrames = [[Float]](repeating: [Float](repeating: 0, count: numFrames), count: freqBins)
        
        // FFT buffers
        var realBuffer = [Float](repeating: 0, count: nFFT)
        var imagBuffer = [Float](repeating: 0, count: nFFT)
        var splitComplex = DSPSplitComplex(realp: &realBuffer, imagp: &imagBuffer)
        
        for frame in 0..<numFrames {
            let start = frame * hopLength
            let end = start + winLength
            
            // Apply window
            var windowedFrame = [Float](repeating: 0, count: nFFT)
            for i in 0..<winLength {
                windowedFrame[i] = paddedSamples[start + i] * window[i]
            }
            
            // Forward FFT
            windowedFrame.withUnsafeMutableBufferPointer { framePtr in
                vDSP_ctoz(UnsafePointer<DSPComplex>(OpaquePointer(framePtr.baseAddress!)),
                          2, &splitComplex, 1, vDSP_Length(nFFT / 2))
            }
            
            vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
            
            // Scale
            var scale = Float(0.5)
            vDSP_vsmul(splitComplex.realp, 1, &scale, splitComplex.realp, 1, vDSP_Length(nFFT / 2))
            vDSP_vsmul(splitComplex.imagp, 1, &scale, splitComplex.imagp, 1, vDSP_Length(nFFT / 2))
            
            // Store result [F, T]
            for f in 0..<freqBins {
                if f < nFFT / 2 {
                    realFrames[f][frame] = realBuffer[f]
                    imagFrames[f][frame] = imagBuffer[f]
                } else {
                    realFrames[f][frame] = 0
                    imagFrames[f][frame] = 0
                }
            }
        }
        
        return (realFrames, imagFrames)
    }
    
    private func performISTFT(real: [[Float]], imag: [[Float]], originalLength: Int) -> [Float] {
        guard let fftSetup = fftSetup, !real.isEmpty, !real[0].isEmpty else { return [] }
        
        let freqBins = real.count
        let numFrames = real[0].count
        let padAmount = nFFT / 2
        let outputLength = (numFrames - 1) * hopLength + winLength + 2 * padAmount
        
        var output = [Float](repeating: 0, count: outputLength)
        var windowSum = [Float](repeating: 0, count: outputLength)
        
        // Buffers
        var realBuffer = [Float](repeating: 0, count: nFFT)
        var imagBuffer = [Float](repeating: 0, count: nFFT)
        var splitComplex = DSPSplitComplex(realp: &realBuffer, imagp: &imagBuffer)
        var frameOutput = [Float](repeating: 0, count: nFFT)
        
        for frame in 0..<numFrames {
            // Load frequency data
            for f in 0..<min(freqBins, nFFT / 2) {
                realBuffer[f] = real[f][frame]
                imagBuffer[f] = imag[f][frame]
            }
            
            // Inverse FFT
            vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_INVERSE))
            
            // Scale
            var scale = Float(1.0 / Float(nFFT))
            vDSP_vsmul(splitComplex.realp, 1, &scale, splitComplex.realp, 1, vDSP_Length(nFFT / 2))
            vDSP_vsmul(splitComplex.imagp, 1, &scale, splitComplex.imagp, 1, vDSP_Length(nFFT / 2))
            
            // Convert split complex to interleaved real signal
            // The inverse FFT produces interleaved results that we extract from realp
            for i in 0..<winLength {
                if i < nFFT / 2 {
                    frameOutput[i * 2] = realBuffer[i]
                    frameOutput[i * 2 + 1] = imagBuffer[i]
                }
            }
            // Simplified: just use the real part directly for reconstruction
            for i in 0..<winLength {
                frameOutput[i] = realBuffer[i % (nFFT / 2)] + imagBuffer[i % (nFFT / 2)]
            }
            
            // Apply window and overlap-add
            let start = frame * hopLength
            for i in 0..<winLength {
                let windowed = realBuffer[i % (nFFT / 2)] * window[i]
                output[start + i] += windowed
                windowSum[start + i] += window[i] * window[i]
            }
        }
        
        // Normalize by window sum
        for i in 0..<output.count {
            if windowSum[i] > 1e-8 {
                output[i] /= windowSum[i]
            }
        }
        
        // Remove padding
        let trimStart = padAmount
        let trimEnd = trimStart + originalLength
        return Array(output[trimStart..<min(trimEnd, output.count)])
    }
    
    // MARK: - Power Compression
    
    private func powerCompress(real: [[Float]], imag: [[Float]]) -> ([[Float]], [[Float]]) {
        var realC = real
        var imagC = imag
        
        for f in 0..<real.count {
            for t in 0..<real[f].count {
                let r = real[f][t]
                let im = imag[f][t]
                let mag = sqrt(r * r + im * im + 1e-9)
                let phase = atan2(im, r)
                let magC = pow(mag, powerCompress)
                realC[f][t] = magC * cos(phase)
                imagC[f][t] = magC * sin(phase)
            }
        }
        
        return (realC, imagC)
    }
    
    private func powerUncompress(real: [[Float]], imag: [[Float]]) -> ([[Float]], [[Float]]) {
        var realUC = real
        var imagUC = imag
        
        for f in 0..<real.count {
            for t in 0..<real[f].count {
                let r = real[f][t]
                let im = imag[f][t]
                let mag = sqrt(r * r + im * im + 1e-9)
                let phase = atan2(im, r)
                let magUC = pow(mag, 1.0 / powerCompress)
                realUC[f][t] = magUC * cos(phase)
                imagUC[f][t] = magUC * sin(phase)
            }
        }
        
        return (realUC, imagUC)
    }
    
    // MARK: - CoreML I/O
    
    private func prepareForCoreML(real: [[Float]], imag: [[Float]]) throws -> MLFeatureProvider {
        // real and imag are [F, T] shaped
        // CoreML expects [1, 2, T, F]
        let F = real.count
        let T = real.isEmpty ? 0 : real[0].count
        
        // Create [1, 2, T, F] MLMultiArray
        let shape: [NSNumber] = [1, 2, NSNumber(value: T), NSNumber(value: F)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        
        for t in 0..<T {
            for f in 0..<F {
                let idx0 = t * F + f  // channel 0 (real)
                let idx1 = T * F + t * F + f  // channel 1 (imag)
                array[idx0] = NSNumber(value: real[f][t])
                array[idx1] = NSNumber(value: imag[f][t])
            }
        }
        
        return try MLDictionaryFeatureProvider(dictionary: ["spectrogram": array])
    }
    
    private func parseModelOutput(_ output: MLFeatureProvider) -> ([[Float]], [[Float]]) {
        guard let multiArray = output.featureValue(for: "enhanced")?.multiArrayValue 
              ?? output.featureValue(for: output.featureNames.first!)?.multiArrayValue else {
            return ([], [])
        }
        
        // Output is [1, 2, T, F]
        let T = multiArray.shape[2].intValue
        let F = multiArray.shape[3].intValue
        
        var real = [[Float]](repeating: [Float](repeating: 0, count: T), count: F)
        var imag = [[Float]](repeating: [Float](repeating: 0, count: T), count: F)
        
        for t in 0..<T {
            for f in 0..<F {
                let idx0 = t * F + f  // channel 0 (real)
                let idx1 = T * F + t * F + f  // channel 1 (imag)
                real[f][t] = multiArray[idx0].floatValue
                imag[f][t] = multiArray[idx1].floatValue
            }
        }
        
        return (real, imag)
    }
    
    // MARK: - Streaming
    
    public func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
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
}
