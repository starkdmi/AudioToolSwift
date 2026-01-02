//
//  SegmentPool.swift
//  ClearVoice
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

/// Segment buffer pool for streaming - reuses AudioBuffer allocations
public actor SegmentPool {
    
    private var available: [AudioBuffer] = []
    private let capacity: Int
    private let segmentSize: Int
    private let sampleRate: Int
    
    private var totalAllocated = 0
    private var currentlyInUse = 0
    private var peakUsage = 0
    
    public init(capacity: Int = 32, segmentSize: Int = 16000, sampleRate: Int = 16000) {
        self.capacity = capacity
        self.segmentSize = segmentSize
        self.sampleRate = sampleRate
    }
    
    /// Acquire buffer from pool (or create new if empty)
    public func acquire() -> AudioBuffer {
        currentlyInUse += 1
        peakUsage = max(peakUsage, currentlyInUse)
        
        if let buffer = available.popLast() {
            return buffer
        }
        
        // Create new buffer
        totalAllocated += 1
        return AudioBuffer(capacity: segmentSize, sampleRate: sampleRate)
    }
    
    /// Return buffer to pool for reuse
    ///
    /// Note: While `AudioBuffer` is immutable, the pool keeps preallocated
    /// buffers that can be reused. When you're done with a buffer, call
    /// release() so the pool can recycle it for future acquire() calls.
    ///
    /// The pool reuses the *allocation* by keeping the passed buffer
    /// (which has preallocated capacity) rather than creating new buffers.
    public func release(_ buffer: AudioBuffer) {
        currentlyInUse = max(0, currentlyInUse - 1)
        
        // Only keep up to capacity
        if available.count < capacity {
            // Reuse the actual buffer - it has preallocated capacity
            // Even though samples are immutable, we keep the allocation
            // and the next user will write new data via AudioBuffer(samples:...)
            available.append(buffer)
        }
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
