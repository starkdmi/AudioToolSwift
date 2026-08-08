//
//  AudioBuffer.swift
//  AudioTool
//
//  Core immutable audio buffer type - thread-safe across isolation boundaries
//

import Foundation

/// Immutable audio buffer - Sendable for safe cross-isolation transfer
public struct AudioBuffer: Sendable, Hashable {
    
    // MARK: - Properties
    
    /// Raw audio samples (interleaved if multi-channel)
    public let samples: [Float]
    
    /// Sample rate in Hz
    public let sampleRate: Int
    
    /// Number of audio channels
    public let channels: Int
    
    /// Duration in seconds
    public var duration: Double {
        guard channels > 0, sampleRate > 0 else { return 0 }
        return Double(samples.count / channels) / Double(sampleRate)
    }
    
    /// Number of frames (samples per channel)
    public var frameCount: Int {
        guard channels > 0 else { return 0 }
        return samples.count / channels
    }
    
    /// Check if buffer is empty
    public var isEmpty: Bool {
        samples.isEmpty
    }
    
    // MARK: - Initialization
    
    /// Create buffer from samples
    public init(samples: [Float], sampleRate: Int, channels: Int = 1) {
        precondition(sampleRate > 0, "Sample rate must be positive")
        precondition(channels > 0, "Channels must be positive")
        precondition(
            samples.count.isMultiple(of: channels),
            "Interleaved samples must contain a whole number of frames"
        )
        self.samples = samples
        self.sampleRate = sampleRate
        self.channels = channels
    }
    
    /// Create empty buffer with capacity
    public init(capacity: Int, sampleRate: Int, channels: Int = 1) {
        precondition(capacity >= 0, "Capacity must not be negative")
        precondition(sampleRate > 0, "Sample rate must be positive")
        precondition(channels > 0, "Channels must be positive")
        let (sampleCapacity, overflow) = capacity.multipliedReportingOverflow(by: channels)
        precondition(!overflow, "Capacity and channel count are too large")
        self.samples = [Float](repeating: 0, count: sampleCapacity)
        self.sampleRate = sampleRate
        self.channels = channels
    }
    
    // MARK: - Slicing
    
    /// Extract slice by time range
    public func slice(_ range: Range<Double>) -> AudioBuffer {
        guard let startFrame = frameIndex(for: range.lowerBound),
              let endFrame = frameIndex(for: range.upperBound),
              startFrame < endFrame else {
            return AudioBuffer(samples: [], sampleRate: sampleRate, channels: channels)
        }
        let startSample = startFrame * channels
        let endSample = endFrame * channels
        
        return AudioBuffer(
            samples: Array(samples[startSample..<endSample]),
            sampleRate: sampleRate,
            channels: channels
        )
    }
    
    /// Extract slice by sample range
    public func slice(samples range: Range<Int>) -> AudioBuffer {
        let startFrame = min(frameCount, max(0, range.lowerBound))
        let endFrame = min(frameCount, max(0, range.upperBound))
        let startSample = startFrame * channels
        let endSample = endFrame * channels
        
        guard startSample < endSample else {
            return AudioBuffer(samples: [], sampleRate: sampleRate, channels: channels)
        }
        
        return AudioBuffer(
            samples: Array(samples[startSample..<endSample]),
            sampleRate: sampleRate,
            channels: channels
        )
    }
    
    // MARK: - Operations
    
    /// Replace segment at time range with new audio
    public func replacing(_ range: Range<Double>, with replacement: AudioBuffer) -> AudioBuffer {
        precondition(replacement.sampleRate == sampleRate, "Sample rates must match")
        precondition(replacement.channels == channels, "Channel counts must match")

        guard let startFrame = frameIndex(for: range.lowerBound),
              let endFrame = frameIndex(for: range.upperBound),
              startFrame < endFrame else {
            return self
        }
        let startSample = startFrame * channels
        let endSample = endFrame * channels
        
        var newSamples = samples
        newSamples.replaceSubrange(startSample..<endSample, with: replacement.samples)
        
        return AudioBuffer(
            samples: newSamples,
            sampleRate: sampleRate,
            channels: channels
        )
    }

    /// Convert a timeline position to a valid frame boundary without first
    /// converting an unbounded `Double` to `Int`, which can trap. Infinities clamp
    /// naturally; NaN has no meaningful position and is rejected by callers.
    private func frameIndex(for time: Double) -> Int? {
        guard !time.isNaN else { return nil }
        if time <= 0 { return 0 }
        if !time.isFinite || time >= duration { return frameCount }
        return min(frameCount, Int(time * Double(sampleRate)))
    }
    
    /// Subtract another buffer (for residual computation)
    public func subtracting(_ other: AudioBuffer) -> AudioBuffer {
        precondition(other.sampleRate == sampleRate, "Sample rates must match")
        precondition(other.channels == channels, "Channel counts must match")
        
        let minCount = min(samples.count, other.samples.count)
        var result = [Float](repeating: 0, count: minCount)
        
        for i in 0..<minCount {
            result[i] = samples[i] - other.samples[i]
        }
        
        return AudioBuffer(samples: result, sampleRate: sampleRate, channels: channels)
    }
    
    /// Mix with another buffer at optional offset
    public func mixing(with other: AudioBuffer, at offset: Double = 0) -> AudioBuffer {
        precondition(other.sampleRate == sampleRate, "Sample rates must match")
        precondition(other.channels == channels, "Channel counts must match")
        
        let offsetSamples = Int(offset * Double(sampleRate)) * channels

        // The returned buffer represents the union of both timelines. For a
        // negative offset, shift `self` to the right instead of indexing the result
        // with a negative value (which used to trap) or throwing away the leading
        // part of `other`.
        let timelineStart = min(0, offsetSamples)
        let timelineEnd = max(samples.count, offsetSamples + other.samples.count)
        let totalLength = max(0, timelineEnd - timelineStart)
        let selfStart = -timelineStart
        let otherStart = offsetSamples - timelineStart
        
        var result = [Float](repeating: 0, count: totalLength)
        
        // Copy original
        for i in 0..<samples.count {
            result[selfStart + i] = samples[i]
        }
        
        // Mix in other
        for i in 0..<other.samples.count {
            let idx = otherStart + i
            if idx >= 0 && idx < totalLength {
                result[idx] += other.samples[i]
            }
        }
        
        return AudioBuffer(samples: result, sampleRate: sampleRate, channels: channels)
    }
    
    /// Append another buffer
    public func appending(_ other: AudioBuffer) -> AudioBuffer {
        precondition(other.sampleRate == sampleRate, "Sample rates must match")
        precondition(other.channels == channels, "Channel counts must match")
        
        return AudioBuffer(
            samples: samples + other.samples,
            sampleRate: sampleRate,
            channels: channels
        )
    }
    
    /// Scale amplitude
    public func scaled(by factor: Float) -> AudioBuffer {
        AudioBuffer(
            samples: samples.map { $0 * factor },
            sampleRate: sampleRate,
            channels: channels
        )
    }
}

// MARK: - Test Utilities

extension AudioBuffer {
    
    /// Create silence buffer for testing
    public static func silence(duration: Double, sampleRate: Int, channels: Int = 1) -> AudioBuffer {
        let frameCount = Int(duration * Double(sampleRate))
        return AudioBuffer(
            samples: [Float](repeating: 0, count: frameCount * channels),
            sampleRate: sampleRate,
            channels: channels
        )
    }
    
    /// Create sine wave for testing
    public static func sine(frequency: Float, duration: Double, sampleRate: Int, amplitude: Float = 0.5) -> AudioBuffer {
        let frameCount = Int(duration * Double(sampleRate))
        let samples = (0..<frameCount).map { i in
            amplitude * sin(2 * .pi * frequency * Float(i) / Float(sampleRate))
        }
        return AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
    }
    
    /// Create white noise for testing
    public static func noise(duration: Double, sampleRate: Int, amplitude: Float = 0.5) -> AudioBuffer {
        let frameCount = Int(duration * Double(sampleRate))
        let samples = (0..<frameCount).map { _ in Float.random(in: -amplitude...amplitude) }
        return AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
    }
}
