//
//  BoundedChannel.swift
//  ClearVoice
//
//  Bounded async channel with backpressure support
//

import Foundation

/// Bounded channel with backpressure for streaming pipelines
public actor BoundedChannel<T: Sendable> {
    
    private var buffer: [T] = []
    private let capacity: Int
    private var isClosed = false
    
    private var sendWaiters: [CheckedContinuation<Void, Never>] = []
    private var receiveWaiters: [CheckedContinuation<T?, Never>] = []
    
    public init(capacity: Int = 16) {
        precondition(capacity > 0, "Capacity must be positive")
        self.capacity = capacity
    }
    
    /// Send value (blocks if at capacity)
    public func send(_ value: T) async {
        guard !isClosed else { return }
        
        // Wait if buffer is full
        while buffer.count >= capacity && !isClosed {
            await withCheckedContinuation { continuation in
                sendWaiters.append(continuation)
            }
        }
        
        guard !isClosed else { return }
        
        buffer.append(value)
        
        // Wake up a waiting receiver
        if let waiter = receiveWaiters.first {
            receiveWaiters.removeFirst()
            waiter.resume(returning: buffer.removeFirst())
        }
    }
    
    /// Send value with timeout
    public func send(_ value: T, timeout: Duration) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                await self.send(value)
            }
            
            group.addTask {
                try await Task.sleep(for: timeout)
                throw ClearVoiceError.backpressureTimeout
            }
            
            // First to complete wins
            try await group.next()
            group.cancelAll()
        }
    }
    
    /// Receive next value (nil if closed and empty)
    public func receive() async -> T? {
        // Return buffered value if available
        if !buffer.isEmpty {
            let value = buffer.removeFirst()
            
            // Wake up a waiting sender
            if let waiter = sendWaiters.first {
                sendWaiters.removeFirst()
                waiter.resume()
            }
            
            return value
        }
        
        // Return nil if closed
        if isClosed {
            return nil
        }
        
        // Wait for value
        return await withCheckedContinuation { continuation in
            receiveWaiters.append(continuation)
        }
    }
    
    /// Close the channel
    public func close() {
        isClosed = true
        
        // Resume all waiting senders
        for waiter in sendWaiters {
            waiter.resume()
        }
        sendWaiters.removeAll()
        
        // Resume all waiting receivers with nil
        for waiter in receiveWaiters {
            waiter.resume(returning: nil)
        }
        receiveWaiters.removeAll()
    }
    
    /// Async sequence of values
    public var values: AsyncStream<T> {
        AsyncStream { continuation in
            Task {
                while let value = await self.receive() {
                    continuation.yield(value)
                }
                continuation.finish()
            }
        }
    }
    
    /// Current buffer count
    public var count: Int {
        buffer.count
    }
    
    /// Check if channel is closed
    public var closed: Bool {
        isClosed
    }
}
