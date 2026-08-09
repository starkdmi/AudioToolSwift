//
//  AudioBuffer+Resampling.swift
//  AudioTool
//
//  AudioBuffer resampling, on AudioUtils' [Float] algorithms.
//

import Foundation
@preconcurrency import AVFoundation
import Accelerate
import AudioToolCore
import AudioUtils

/// Type alias to disambiguate from AVFoundation's AudioBuffer
public typealias CVAudioBuffer = AudioToolCore.AudioBuffer

// `ResamplingQuality` and `ResamplingError` are declared in both AudioToolCore and
// AudioUtils. AudioToolCore's are the ones this package's API is written in - the
// quality enum is what providers declare a preference with - so they are qualified
// throughout rather than left to whichever module wins a lookup.

extension CVAudioBuffer {
    
    /// Resample audio buffer to target sample rate
    /// - Parameters:
    ///   - targetSampleRate: Target sample rate in Hz
    ///   - quality: Resampling quality (default: .balanced)
    /// - Returns: New AudioBuffer at target sample rate
    public func resampled(to targetSampleRate: Int, quality: AudioToolCore.ResamplingQuality = .balanced) throws -> CVAudioBuffer {
        guard targetSampleRate > 0 else {
            throw AudioToolCore.ResamplingError.invalidParameters("Target sample rate must be positive")
        }
        guard sampleRate != targetSampleRate else { return self }
        guard samples.count.isMultiple(of: channels) else {
            throw AudioToolCore.ResamplingError.invalidParameters(
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
    public func resampledAsync(to targetSampleRate: Int, quality: AudioToolCore.ResamplingQuality = .balanced) async throws -> CVAudioBuffer {
        try await Task.detached(priority: .userInitiated) {
            try self.resampled(to: targetSampleRate, quality: quality)
        }.value
    }
}

/// Map `AudioToolCore`'s quality preference onto AudioUtils' resamplers.
///
/// The algorithms themselves live in `AudioUtils.FloatResampling`; this package used
/// to carry its own copies, which drifted ahead of the ones in the library and were
/// folded back in AudioUtils 1.2.0. What stays here is the policy - which resampler
/// each `AudioToolCore.ResamplingQuality` means - because `AudioToolCore.ResamplingQuality` is AudioToolCore's
/// type and the mapping is this package's decision, not the library's.
private func resampleChannel(
    _ samples: [Float],
    fromRate: Int,
    toRate: Int,
    quality: AudioToolCore.ResamplingQuality,
    expectedLength: Int
) throws -> [Float] {
    switch quality {
    case .fast:
        return resampleLinear(samples, fromRate: Float(fromRate),
                              toRate: Float(toRate), outputLength: expectedLength)
    case .balanced:
        return resampleCubic(samples, fromRate: Float(fromRate),
                             toRate: Float(toRate), outputLength: expectedLength)
    case .high:
        return try resampleAVAudioConverter(
            samples,
            fromRate: Float(fromRate),
            toRate: Float(toRate),
            algorithm: AVSampleRateConverterAlgorithm_Mastering,
            quality: AVAudioQuality.max,
            expectedLength: expectedLength
        )
    case .auto:
        // Upsampling has nothing to alias, so the cheap interpolator is honest there.
        // Downsampling does, so it goes through the converter.
        if toRate > fromRate {
            return resampleCubic(samples, fromRate: Float(fromRate),
                                 toRate: Float(toRate), outputLength: expectedLength)
        }
        return try resampleAVAudioConverter(
            samples,
            fromRate: Float(fromRate),
            toRate: Float(toRate),
            algorithm: AVSampleRateConverterAlgorithm_Normal,
            quality: AVAudioQuality.high,
            expectedLength: expectedLength
        )
    }
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
