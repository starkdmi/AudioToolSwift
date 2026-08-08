//
//  AudioBuffer+Resampling.swift
//  AudioTool
//
//  Native Swift resampling for AudioBuffer - follows SwiftAudio/AudioUtils algorithms
//  without MLXArray conversion overhead.
//

import Foundation
@preconcurrency import AVFoundation
import Accelerate
import AudioToolCore

/// Type alias to disambiguate from AVFoundation's AudioBuffer
public typealias CVAudioBuffer = AudioToolCore.AudioBuffer

// ResamplingQuality lives in AudioToolCore so providers can declare a preference.

extension CVAudioBuffer {
    
    /// Resample audio buffer to target sample rate
    /// - Parameters:
    ///   - targetSampleRate: Target sample rate in Hz
    ///   - quality: Resampling quality (default: .balanced)
    /// - Returns: New AudioBuffer at target sample rate
    public func resampled(to targetSampleRate: Int, quality: ResamplingQuality = .balanced) throws -> CVAudioBuffer {
        guard targetSampleRate > 0 else {
            throw ResamplingError.invalidParameters("Target sample rate must be positive")
        }
        guard sampleRate != targetSampleRate else { return self }
        guard samples.count.isMultiple(of: channels) else {
            throw ResamplingError.invalidParameters(
                "Interleaved sample count must be divisible by the channel count"
            )
        }
        guard !samples.isEmpty else {
            return CVAudioBuffer(samples: [], sampleRate: targetSampleRate, channels: channels)
        }

        let expectedFrames = expectedOutputFrames(
            inputFrames: frameCount,
            fromRate: sampleRate,
            toRate: targetSampleRate
        )
        var resampledChannels = [[Float]]()
        resampledChannels.reserveCapacity(channels)

        // `samples` is frame-interleaved. Resampling the flattened storage treats
        // every channel boundary as a point in one mono waveform, interpolating
        // L -> R -> L. Deinterleave first and give each channel its own converter.
        for channel in 0..<channels {
            var planar = [Float](repeating: 0, count: frameCount)
            for frame in 0..<frameCount {
                planar[frame] = samples[frame * channels + channel]
            }
            resampledChannels.append(try resampleChannel(
                planar,
                fromRate: sampleRate,
                toRate: targetSampleRate,
                quality: quality,
                expectedLength: expectedFrames
            ))
        }

        var interleaved = [Float](repeating: 0, count: expectedFrames * channels)
        for frame in 0..<expectedFrames {
            for channel in 0..<channels {
                interleaved[frame * channels + channel] = resampledChannels[channel][frame]
            }
        }

        return CVAudioBuffer(samples: interleaved, sampleRate: targetSampleRate, channels: channels)
    }
    
    /// Resample asynchronously (for large files)
    public func resampledAsync(to targetSampleRate: Int, quality: ResamplingQuality = .balanced) async throws -> CVAudioBuffer {
        try await Task.detached(priority: .userInitiated) {
            try self.resampled(to: targetSampleRate, quality: quality)
        }.value
    }
}

private func expectedOutputFrames(inputFrames: Int, fromRate: Int, toRate: Int) -> Int {
    Int((Double(inputFrames) * Double(toRate) / Double(fromRate)).rounded())
}

private func resampleChannel(
    _ samples: [Float],
    fromRate: Int,
    toRate: Int,
    quality: ResamplingQuality,
    expectedLength: Int
) throws -> [Float] {
    let output: [Float]
    switch quality {
    case .fast:
        output = resampleLinear(samples, fromRate: Float(fromRate),
                                toRate: Float(toRate), outputLength: expectedLength)
    case .balanced:
        output = resampleCubic(samples, fromRate: Float(fromRate),
                               toRate: Float(toRate), outputLength: expectedLength)
    case .high:
        output = try resampleAVAudioConverter(
            samples,
            fromRate: Float(fromRate),
            toRate: Float(toRate),
            algorithm: AVSampleRateConverterAlgorithm_Mastering,
            quality: AVAudioQuality.max,
            expectedLength: expectedLength
        )
    case .auto:
        if toRate > fromRate {
            output = resampleCubic(samples, fromRate: Float(fromRate),
                                   toRate: Float(toRate), outputLength: expectedLength)
        } else {
            output = try resampleAVAudioConverter(
                samples,
                fromRate: Float(fromRate),
                toRate: Float(toRate),
                algorithm: AVSampleRateConverterAlgorithm_Normal,
                quality: AVAudioQuality.high,
                expectedLength: expectedLength
            )
        }
    }
    return output
}

// MARK: - Linear Resampling

/// Simple linear interpolation resampling (fastest but lowest quality)
private func resampleLinear(
    _ samples: [Float],
    fromRate: Float,
    toRate: Float,
    outputLength: Int
) -> [Float] {
    let ratio = toRate / fromRate
    let inputLength = samples.count

    guard outputLength > 0 else { return [] }
    guard inputLength > 1 else {
        return [Float](repeating: samples.first ?? 0, count: outputLength)
    }
    
    var result = [Float](repeating: 0, count: outputLength)
    
    for i in 0..<outputLength {
        let inputPosition = Float(i) / ratio
        let index = min(max(Int(inputPosition), 0), inputLength - 1)
        let idx0 = index
        let idx1 = min(index + 1, inputLength - 1)
        // Once the sampling position reaches the last source frame there is no
        // right-hand neighbour to interpolate toward. Hold that endpoint instead
        // of clamping the index back by one while retaining the old fraction,
        // which made an upsampled signal jump backwards at its tail.
        let fraction = idx0 == idx1 ? 0 : inputPosition - Float(index)
        
        result[i] = samples[idx0] * (1 - fraction) + samples[idx1] * fraction
    }
    
    return result
}

// MARK: - Cubic Resampling (Catmull-Rom)

/// Catmull-Rom cubic interpolation resampling.
///
/// Pure interpolation with no anti-aliasing stage, so downsampling folds content
/// above the new Nyquist back into the band. That is a property to be aware of, not
/// necessarily a defect to fix - see the note on `ResamplingQuality`.
///
/// This is a second copy of the implementation in SwiftAudio's AudioUtils; the two
/// should be reconciled rather than left to drift.
///
/// - Note: The figure "~84.3 dB SNR with 0.01% THD" was carried over from AudioUtils
///   and has not been verified here; it cannot hold for downsampling without a
///   low-pass stage.
private func resampleCubic(
    _ samples: [Float],
    fromRate: Float,
    toRate: Float,
    outputLength: Int
) -> [Float] {
    let ratio = toRate / fromRate
    let inputLength = samples.count

    guard outputLength > 0 else { return [] }
    guard inputLength >= 4 else {
        // Fall back to linear for very short samples
        return resampleLinear(
            samples,
            fromRate: fromRate,
            toRate: toRate,
            outputLength: outputLength
        )
    }
    
    var result = [Float](repeating: 0, count: outputLength)
    
    // Catmull-Rom coefficients
    let c0: Float = -0.5
    let c1: Float = 1.5
    let c2: Float = 2.5
    let c3: Float = 2.0
    
    for i in 0..<outputLength {
        let inputPosition = Float(i) / ratio
        let index = Int(inputPosition)
        let f = inputPosition - Float(index)
        
        // Get 4 points for cubic interpolation with boundary clamping
        let idx0 = max(0, min(index - 1, inputLength - 1))
        let idx1 = max(0, min(index, inputLength - 1))
        let idx2 = max(0, min(index + 1, inputLength - 1))
        let idx3 = max(0, min(index + 2, inputLength - 1))
        
        let y0 = samples[idx0]
        let y1 = samples[idx1]
        let y2 = samples[idx2]
        let y3 = samples[idx3]
        
        // Catmull-Rom cubic interpolation
        let a0 = c0 * y0 + c1 * y1 - c1 * y2 + (-c0) * y3
        let a1 = y0 - c2 * y1 + c3 * y2 - (-c0) * y3
        let a2 = c0 * y0 + (-c0) * y2
        let a3 = y1
        
        // Polynomial evaluation: a0*f^3 + a1*f^2 + a2*f + a3
        let f2 = f * f
        let f3 = f2 * f
        
        result[i] = a0 * f3 + a1 * f2 + a2 * f + a3
    }
    
    return result
}

// MARK: - AVAudioConverter Resampling

/// High-quality resampling using AVAudioConverter
/// Provides professional-grade quality with hardware acceleration and anti-aliasing
private func resampleAVAudioConverter(
    _ samples: [Float],
    fromRate: Float,
    toRate: Float,
    algorithm: String,
    quality: AVAudioQuality,
    expectedLength: Int
) throws -> [Float] {
    guard let inputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(fromRate),
        channels: 1,
        interleaved: false
    ) else {
        throw ResamplingError.invalidFormat
    }
    
    guard let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(toRate),
        channels: 1,
        interleaved: false
    ) else {
        throw ResamplingError.invalidFormat
    }
    
    guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
        throw ResamplingError.conversionFailed
    }

    // Set explicitly rather than inherited. AVAudioConverter's own defaults are
    // Normal at quality 64, which is neither of the two settings this package needs
    // to be able to reproduce: Mastering/`.max` for the generators that ask for it,
    // and Normal/`.medium` for `AudioLoader`'s `.auto`. Leaving it implicit meant
    // `.high` silently delivered the former's promise with the latter's algorithm.
    converter.sampleRateConverterAlgorithm = algorithm
    converter.sampleRateConverterQuality = quality.rawValue
    
    // Create input buffer
    guard let inputBuffer = AVAudioPCMBuffer(
        pcmFormat: inputFormat,
        frameCapacity: AVAudioFrameCount(samples.count)
    ) else {
        throw ResamplingError.invalidFormat
    }
    inputBuffer.frameLength = AVAudioFrameCount(samples.count)
    
    // Copy input data using memcpy for efficiency
    if let channelData = inputBuffer.floatChannelData?[0] {
        samples.withUnsafeBufferPointer { ptr in
            channelData.update(from: ptr.baseAddress!, count: samples.count)
        }
    }
    
    // Use a class to wrap mutable state for the converter callback
    final class InputProvider: @unchecked Sendable {
        var consumed = false
        let buffer: AVAudioPCMBuffer
        init(buffer: AVAudioPCMBuffer) { self.buffer = buffer }
    }
    let provider = InputProvider(buffer: inputBuffer)

    var output = [Float]()
    output.reserveCapacity(expectedLength)
    var reachedEnd = false
    var emptyPasses = 0

    // AVAudioConverter is stateful and withholds its latency tail. Keep pulling,
    // and report end-of-stream after the single input buffer, until it is drained.
    while !reachedEnd && output.count < expectedLength {
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(min(16_384, max(1, expectedLength - output.count)))
        ) else {
            throw ResamplingError.invalidFormat
        }

        var error: NSError?
        let status = converter.convert(to: outputBuffer, error: &error) { _, outStatus in
        if provider.consumed {
            outStatus.pointee = .endOfStream
            return nil
        }
        outStatus.pointee = .haveData
        provider.consumed = true
        return provider.buffer
        }

        guard status != .error, error == nil else {
            throw ResamplingError.conversionFailed
        }

        let produced = min(Int(outputBuffer.frameLength), expectedLength - output.count)
        if produced > 0, let channelData = outputBuffer.floatChannelData?[0] {
            output.append(contentsOf: UnsafeBufferPointer(start: channelData, count: produced))
            emptyPasses = 0
        } else {
            emptyPasses += 1
        }
        reachedEnd = status == .endOfStream
        if emptyPasses > 2 { break }
    }

    if output.count > expectedLength {
        output.removeLast(output.count - expectedLength)
    } else if output.count < expectedLength {
        output.append(contentsOf: [Float](repeating: 0, count: expectedLength - output.count))
    }
    return output
}

// MARK: - vDSP Optimized Mixing (Bonus)

extension CVAudioBuffer {

    /// Convert interleaved channel layout explicitly.
    ///
    /// Model inputs in this package are mono or stereo. Mono conversion averages
    /// channels; mono-to-stereo duplicates the signal; inputs wider than stereo are
    /// downmixed before duplication so a two-channel model never receives arbitrary
    /// channel truncation.
    public func converted(toChannels targetChannels: Int) throws -> CVAudioBuffer {
        guard targetChannels > 0 else {
            throw AudioToolError.invalidAudioFormat(
                expected: "a positive channel count",
                found: "\(targetChannels) channels"
            )
        }
        guard samples.count.isMultiple(of: channels) else {
            throw AudioToolError.invalidAudioFormat(
                expected: "complete interleaved frames",
                found: "\(samples.count) samples for \(channels) channels"
            )
        }
        guard targetChannels != channels else { return self }
        if targetChannels == 1 { return mixedToMono() }

        let mono = channels == 1 ? samples : mixedToMono().samples
        var converted = [Float](repeating: 0, count: mono.count * targetChannels)
        for frame in mono.indices {
            for channel in 0..<targetChannels {
                converted[frame * targetChannels + channel] = mono[frame]
            }
        }
        return CVAudioBuffer(
            samples: converted,
            sampleRate: sampleRate,
            channels: targetChannels
        )
    }
    
    /// Mix down to mono using vDSP (optimized for multi-channel)
    public func mixedToMono() -> CVAudioBuffer {
        guard channels > 1 else { return self }
        
        let frameCount = self.frameCount
        var monoSamples = [Float](repeating: 0, count: frameCount)
        
        // Use vDSP for efficient mixing
        let scale = 1.0 / Float(channels)
        
        for ch in 0..<channels {
            // Extract channel samples (assuming interleaved)
            var channelSamples = [Float](repeating: 0, count: frameCount)
            for i in 0..<frameCount {
                channelSamples[i] = samples[i * channels + ch]
            }
            
            // Add to mono mix
            vDSP_vsma(channelSamples, 1, [scale], monoSamples, 1, &monoSamples, 1, vDSP_Length(frameCount))
        }
        
        return CVAudioBuffer(samples: monoSamples, sampleRate: sampleRate, channels: 1)
    }
}
