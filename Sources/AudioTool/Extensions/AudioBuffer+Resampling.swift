//
//  AudioBuffer+Resampling.swift
//  ClearVoice
//
//  Native Swift resampling for AudioBuffer - follows SwiftAudio/AudioUtils algorithms
//  without MLXArray conversion overhead.
//

import Foundation
import AVFoundation
import Accelerate
import ClearVoiceCore

/// Type alias to disambiguate from AVFoundation's AudioBuffer
public typealias CVAudioBuffer = ClearVoiceCore.AudioBuffer

// MARK: - Resampling Quality

/// Resampling quality options matching AudioUtils
public enum ResamplingQuality: Sendable {
    /// Linear interpolation (fastest, lowest quality)
    case fast
    /// Cubic interpolation - Catmull-Rom (excellent quality, ~84 dB SNR)
    case balanced
    /// AVAudioConverter (professional quality, anti-aliasing)
    case high
}

// MARK: - Resampling Errors

public enum ResamplingError: Error, Sendable {
    case invalidFormat
    case invalidParameters(String)
    case conversionFailed
}

// MARK: - AudioBuffer Resampling Extension

extension CVAudioBuffer {
    
    /// Resample audio buffer to target sample rate
    /// - Parameters:
    ///   - targetSampleRate: Target sample rate in Hz
    ///   - quality: Resampling quality (default: .balanced)
    /// - Returns: New AudioBuffer at target sample rate
    public func resampled(to targetSampleRate: Int, quality: ResamplingQuality = .balanced) throws -> CVAudioBuffer {
        guard sampleRate != targetSampleRate else { return self }
        guard !samples.isEmpty else { return self }
        guard targetSampleRate > 0 else {
            throw ResamplingError.invalidParameters("Target sample rate must be positive")
        }
        
        let resampled: [Float]
        switch quality {
        case .fast:
            resampled = resampleLinear(samples, fromRate: Float(sampleRate), toRate: Float(targetSampleRate))
        case .balanced:
            resampled = resampleCubic(samples, fromRate: Float(sampleRate), toRate: Float(targetSampleRate))
        case .high:
            resampled = try resampleAVAudioConverter(samples, fromRate: Float(sampleRate), toRate: Float(targetSampleRate))
        }
        
        return CVAudioBuffer(samples: resampled, sampleRate: targetSampleRate, channels: channels)
    }
    
    /// Resample asynchronously (for large files)
    public func resampledAsync(to targetSampleRate: Int, quality: ResamplingQuality = .balanced) async throws -> CVAudioBuffer {
        try await Task.detached(priority: .userInitiated) {
            try self.resampled(to: targetSampleRate, quality: quality)
        }.value
    }
}

// MARK: - Linear Resampling

/// Simple linear interpolation resampling (fastest but lowest quality)
private func resampleLinear(_ samples: [Float], fromRate: Float, toRate: Float) -> [Float] {
    let ratio = toRate / fromRate
    let inputLength = samples.count
    let outputLength = Int(Float(inputLength) * ratio)
    
    guard outputLength > 0 else { return [] }
    
    var result = [Float](repeating: 0, count: outputLength)
    
    for i in 0..<outputLength {
        let inputPosition = Float(i) / ratio
        let index = Int(inputPosition)
        let fraction = inputPosition - Float(index)
        
        let idx0 = min(max(index, 0), inputLength - 2)
        let idx1 = min(idx0 + 1, inputLength - 1)
        
        result[i] = samples[idx0] * (1 - fraction) + samples[idx1] * fraction
    }
    
    return result
}

// MARK: - Cubic Resampling (Catmull-Rom)

/// Catmull-Rom cubic interpolation resampling
/// Achieves ~84.3 dB SNR with 0.01% THD (matching AudioUtils quality)
private func resampleCubic(_ samples: [Float], fromRate: Float, toRate: Float) -> [Float] {
    let ratio = toRate / fromRate
    let inputLength = samples.count
    let outputLength = Int(Float(inputLength) * ratio)
    
    guard outputLength > 0 else { return [] }
    guard inputLength >= 4 else {
        // Fall back to linear for very short samples
        return resampleLinear(samples, fromRate: fromRate, toRate: toRate)
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
private func resampleAVAudioConverter(_ samples: [Float], fromRate: Float, toRate: Float) throws -> [Float] {
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
    
    // Calculate output size
    let outputFrames = AVAudioFrameCount(Float(samples.count) * toRate / fromRate)
    guard let outputBuffer = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity: outputFrames + 1024  // Extra capacity for rounding
    ) else {
        throw ResamplingError.invalidFormat
    }
    
    // Perform conversion
    var error: NSError?
    var inputConsumed = false
    
    let status = converter.convert(to: outputBuffer, error: &error) { inNumPackets, outStatus in
        if inputConsumed {
            outStatus.pointee = .noDataNow
            return nil
        }
        outStatus.pointee = .haveData
        inputConsumed = true
        return inputBuffer
    }
    
    guard status != .error, error == nil else {
        throw ResamplingError.conversionFailed
    }
    
    // Extract output data efficiently
    let outputCount = Int(outputBuffer.frameLength)
    var output = [Float](repeating: 0, count: outputCount)
    
    if let channelData = outputBuffer.floatChannelData?[0] {
        output.withUnsafeMutableBufferPointer { ptr in
            ptr.baseAddress!.update(from: channelData, count: outputCount)
        }
    }
    
    return output
}

// MARK: - vDSP Optimized Mixing (Bonus)

extension CVAudioBuffer {
    
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
