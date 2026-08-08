//
//  BoundedChannel.swift
//  AudioTool
//
//  Bounded async channel with backpressure support
//

import Foundation

/// Bounded channel with backpressure for streaming pipelines
public actor BoundedChannel<T: Sendable> {
    
    private var buffer: [T] = []
    private let capacity: Int
    private var isClosed = false
    
    private struct SendWaiter {
        let id: UUID
        let continuation: CheckedContinuation<Bool, Never>
    }

    private struct ReceiveWaiter {
        let id: UUID
        let continuation: CheckedContinuation<T?, Never>
    }

    private var sendWaiters: [SendWaiter] = []
    private var receiveWaiters: [ReceiveWaiter] = []

    /// Sender continuations that have been resumed for a specific free slot but
    /// have not yet re-entered the actor to publish their value. Counting these
    /// reservations prevents a newly arriving sender from stealing a granted
    /// slot and lets cancellation transfer the grant to the next waiter.
    private var reservedSenderIDs: Set<UUID> = []

    private var occupiedSendSlots: Int {
        buffer.count + reservedSenderIDs.count
    }
    
    public init(capacity: Int = 16) {
        precondition(capacity > 0, "Capacity must be positive")
        self.capacity = capacity
    }
    
    /// Send value (blocks if at capacity)
    public func send(_ value: T) async {
        guard !isClosed, !Task.isCancelled else { return }

        var ownsReservedSlot = false

        // Wait if buffer is full - re-check after each wake
        while !ownsReservedSlot && occupiedSendSlots >= capacity && !isClosed {
            let waiterID = UUID()
            let mayContinue = await withTaskCancellationHandler {
                await withCheckedContinuation { continuation in
                    if Task.isCancelled || isClosed {
                        continuation.resume(returning: false)
                    } else if occupiedSendSlots < capacity {
                        continuation.resume(returning: true)
                    } else {
                        sendWaiters.append(SendWaiter(
                            id: waiterID,
                            continuation: continuation
                        ))
                    }
                }
            } onCancel: {
                Task { await self.cancelSendWaiter(waiterID) }
            }

            ownsReservedSlot = reservedSenderIDs.remove(waiterID) != nil
            guard mayContinue else { return }
            guard !Task.isCancelled else {
                // A receive granted this sender a slot, but cancellation won the
                // race before the sender could use it. Pass that capacity to the
                // next queued sender instead of leaving it suspended forever.
                wakeWaitingSendersForAvailableCapacity()
                return
            }
        }

        guard !isClosed else { return }
        guard !Task.isCancelled else {
            wakeWaitingSendersForAvailableCapacity()
            return
        }

        // Hand off directly when a receiver is already suspended. This avoids an
        // unnecessary buffer insertion/removal and keeps capacity available.
        if !receiveWaiters.isEmpty {
            let waiter = receiveWaiters.removeFirst()
            waiter.continuation.resume(returning: value)
            wakeWaitingSendersForAvailableCapacity()
        } else {
            buffer.append(value)
        }
    }
    
    /// Send value with timeout
    public func send(_ value: T, timeout: Duration) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try Task.checkCancellation()
                await self.send(value)
                try Task.checkCancellation()
            }
            
            group.addTask {
                try await Task.sleep(for: timeout)
                throw AudioToolError.backpressureTimeout
            }
            
            // First to complete wins
            _ = try await group.next()
            group.cancelAll()
        }
    }
    
    /// Receive next value (nil if closed and empty)
    public func receive() async -> T? {
        // Return buffered value if available
        if !buffer.isEmpty {
            let value = buffer.removeFirst()

            wakeWaitingSendersForAvailableCapacity()
            
            return value
        }
        
        // Return nil if closed
        if isClosed {
            return nil
        }
        
        // Wait for value
        let waiterID = UUID()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                if Task.isCancelled || isClosed {
                    continuation.resume(returning: nil)
                } else if !buffer.isEmpty {
                    let value = buffer.removeFirst()
                    wakeWaitingSendersForAvailableCapacity()
                    continuation.resume(returning: value)
                } else {
                    receiveWaiters.append(ReceiveWaiter(
                        id: waiterID,
                        continuation: continuation
                    ))
                }
            }
        } onCancel: {
            Task { await self.cancelReceiveWaiter(waiterID) }
        }
    }
    
    /// Close the channel
    public func close() {
        isClosed = true
        
        // Resume all waiting senders
        for waiter in sendWaiters {
            waiter.continuation.resume(returning: false)
        }
        sendWaiters.removeAll()
        reservedSenderIDs.removeAll()
        
        // Resume all waiting receivers with nil
        for waiter in receiveWaiters {
            waiter.continuation.resume(returning: nil)
        }
        receiveWaiters.removeAll()
    }
    
    /// Async sequence of values
    public var values: AsyncStream<T> {
        AsyncStream { continuation in
            let producer = Task {
                while let value = await self.receive() {
                    guard case .terminated = continuation.yield(value) else {
                        continue
                    }
                    return
                }
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in
                producer.cancel()
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

    private func cancelSendWaiter(_ id: UUID) {
        if let index = sendWaiters.firstIndex(where: { $0.id == id }) {
            let waiter = sendWaiters.remove(at: index)
            waiter.continuation.resume(returning: false)
            return
        }

        // Cancellation can arrive after receive() has removed and resumed the
        // waiter but before send() re-enters the actor. Release and transfer the
        // reserved slot in that case.
        if reservedSenderIDs.remove(id) != nil {
            wakeWaitingSendersForAvailableCapacity()
        }
    }

    private func cancelReceiveWaiter(_ id: UUID) {
        guard let index = receiveWaiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = receiveWaiters.remove(at: index)
        waiter.continuation.resume(returning: nil)
    }

    private func wakeWaitingSendersForAvailableCapacity() {
        guard !isClosed else { return }

        while occupiedSendSlots < capacity, !sendWaiters.isEmpty {
            let waiter = sendWaiters.removeFirst()
            reservedSenderIDs.insert(waiter.id)
            waiter.continuation.resume(returning: true)
        }
    }
}
