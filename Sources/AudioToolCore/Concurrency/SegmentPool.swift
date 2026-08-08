//
//  SegmentPool.swift
//  AudioTool
//
//  Memory pool for efficient audio segment reuse
//

import Foundation

/// Pool statistics
public struct PoolStats: Sendable {
    public let totalAllocated: Int
    public let currentlyInUse: Int
    public let available: Int
    public let peakUsage: Int
}

/// Mutable storage leased by ``SegmentPool``.
///
/// `AudioBuffer` is deliberately immutable, so it cannot itself be a reusable
/// scratch allocation. A lease owns mutable Array storage while it is checked out;
/// convert it to an immutable `AudioBuffer` only when handing completed audio to a
/// consumer.
public struct PooledAudioSegment: Sendable {
    public var samples: [Float]
    public let sampleRate: Int
    public let channels: Int

    public var frameCount: Int { samples.count / channels }

    fileprivate init(frameCount: Int, sampleRate: Int, channels: Int) {
        self.samples = [Float](repeating: 0, count: frameCount * channels)
        self.sampleRate = sampleRate
        self.channels = channels
    }

    /// Immutable snapshot suitable for provider and pipeline APIs.
    public func audioBuffer() -> AudioBuffer {
        AudioBuffer(samples: samples, sampleRate: sampleRate, channels: channels)
    }

    fileprivate mutating func clear() {
        // These elements are already initialized. Assigning through the Array keeps
        // its capacity while respecting the initialized-memory contract.
        for index in samples.indices {
            samples[index] = 0
        }
    }
}

/// Segment storage pool for streaming.
public actor SegmentPool {
    
    private var available: [PooledAudioSegment] = []
    private let capacity: Int
    private let segmentSize: Int
    private let sampleRate: Int
    private let channels: Int
    
    private var totalAllocated = 0
    private var currentlyInUse = 0
    private var peakUsage = 0
    
    public init(
        capacity: Int = 32,
        segmentSize: Int = 16000,
        sampleRate: Int = 16000,
        channels: Int = 1
    ) {
        precondition(capacity >= 0, "Capacity cannot be negative")
        precondition(segmentSize > 0, "Segment size must be positive")
        precondition(sampleRate > 0, "Sample rate must be positive")
        precondition(channels > 0, "Channel count must be positive")
        self.capacity = capacity
        self.segmentSize = segmentSize
        self.sampleRate = sampleRate
        self.channels = channels
    }
    
    /// Acquire buffer from pool (or create new if empty)
    public func acquire() -> PooledAudioSegment {
        currentlyInUse += 1
        peakUsage = max(peakUsage, currentlyInUse)
        
        if var segment = available.popLast() {
            segment.clear()
            return segment
        }
        
        // Create new buffer
        totalAllocated += 1
        return PooledAudioSegment(
            frameCount: segmentSize,
            sampleRate: sampleRate,
            channels: channels
        )
    }
    
    /// Return a lease to the pool. The consuming parameter transfers its Array
    /// storage back instead of retaining a stale immutable audio value.
    public func release(_ segment: consuming PooledAudioSegment) {
        currentlyInUse = max(0, currentlyInUse - 1)

        guard available.count < capacity,
              segment.sampleRate == sampleRate,
              segment.channels == channels,
              segment.frameCount == segmentSize else {
            return
        }
        available.append(segment)
    }
    
    /// Clear pool
    public func clear() {
        available.removeAll()
    }
    
    /// Current pool statistics
    public var stats: PoolStats {
        PoolStats(
            totalAllocated: totalAllocated,
            currentlyInUse: currentlyInUse,
            available: available.count,
            peakUsage: peakUsage
        )
    }
}
