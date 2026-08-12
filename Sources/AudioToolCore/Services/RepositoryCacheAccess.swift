//
//  RepositoryCacheAccess.swift
//  AudioToolCore
//
//  Per-repository serialization for HuggingFace cache mutations
//

import Foundation

/// Serializes cache-mutating operations for the same repository while allowing
/// unrelated repositories to proceed concurrently.
///
/// ``withGlobalExclusiveAccess(operation:)`` is the whole-cache counterpart, for
/// operations that are not scoped to one repository - clearing the cache. It waits
/// for every in-flight repository operation to finish and holds off new ones for its
/// duration. Without it, `clearCache()` deleted files from under an active download
/// or an integrity verification: the downloader actor is reentrant while awaiting a
/// snapshot, so "the actor is busy downloading" never stopped anything.
actor RepositoryCacheAccess {
    static let shared = RepositoryCacheAccess()

    private final class Waiter: @unchecked Sendable {
        let id: UUID

        private let lock = NSLock()
        private var result: Bool?
        private var continuation: CheckedContinuation<Bool, Never>?

        init(id: UUID) {
            self.id = id
        }

        func wait() async -> Bool {
            await withCheckedContinuation { continuation in
                lock.lock()
                if let result {
                    lock.unlock()
                    continuation.resume(returning: result)
                } else {
                    self.continuation = continuation
                    lock.unlock()
                }
            }
        }

        /// Resolve exactly once. Returns true when this call won the race.
        @discardableResult
        func resolve(acquired: Bool) -> Bool {
            lock.lock()
            guard result == nil else {
                lock.unlock()
                return false
            }
            result = acquired
            let continuation = continuation
            self.continuation = nil
            lock.unlock()

            continuation?.resume(returning: acquired)
            return true
        }
    }

    /// One waiter's claim, either on a single repository or on the whole cache.
    private struct Blocked {
        let token: UUID
        /// `nil` for a whole-cache claim.
        let repository: String?
        let waiter: Waiter
    }

    /// Repositories with an operation in progress.
    private var activeRepositories: Set<String> = []

    /// Whether a whole-cache operation holds the barrier.
    private var globalActive = false

    /// Queued claims, oldest first. Whole-cache claims take priority over repository
    /// claims queued behind them, so a `clearCache()` cannot be starved by a steady
    /// stream of downloads.
    private var blocked: [Blocked] = []

    func withExclusiveAccess<Value: Sendable>(
        to repository: String,
        operation: @Sendable () async throws -> Value
    ) async throws -> Value {
        let token = UUID()
        try await acquire(repository: repository, token: token)
        defer { release(repository: repository, token: token) }

        try Task.checkCancellation()
        return try await operation()
    }

    /// Run `operation` with no repository operation in flight, and none admitted
    /// until it returns.
    func withGlobalExclusiveAccess<Value: Sendable>(
        operation: @Sendable () async throws -> Value
    ) async throws -> Value {
        let token = UUID()
        try await acquire(repository: nil, token: token)
        defer { release(repository: nil, token: token) }

        try Task.checkCancellation()
        return try await operation()
    }

    func waitingCount(for repository: String) -> Int {
        blocked.filter { $0.repository == repository }.count
    }

    /// Waiters for the whole-cache barrier.
    func globalWaitingCount() -> Int {
        blocked.filter { $0.repository == nil }.count
    }

    /// `repository == nil` claims the whole cache.
    private func acquire(repository: String?, token: UUID) async throws {
        try Task.checkCancellation()

        if canAdmit(repository: repository) {
            take(repository: repository)
            return
        }

        let waiter = Waiter(id: token)
        blocked.append(Blocked(token: token, repository: repository, waiter: waiter))

        let acquired = await withTaskCancellationHandler {
            await waiter.wait()
        } onCancel: {
            waiter.resolve(acquired: false)
        }

        guard acquired else {
            removeWaiter(token: token)
            // A cancelled waiter may have been the one holding up the queue.
            admitWaiting()
            throw CancellationError()
        }
        if Task.isCancelled {
            release(repository: repository, token: token)
            throw CancellationError()
        }
    }

    /// Whether a claim can start right now, ignoring the queue's own ordering.
    private func canAdmit(repository: String?) -> Bool {
        guard !globalActive else { return false }
        guard let repository else {
            // A whole-cache claim needs the cache quiet.
            return activeRepositories.isEmpty
        }
        // A repository claim waits behind any queued whole-cache claim, so that
        // claim is not starved by repositories that keep arriving.
        guard !blocked.contains(where: { $0.repository == nil }) else { return false }
        return !activeRepositories.contains(repository)
    }

    private func take(repository: String?) {
        if let repository {
            activeRepositories.insert(repository)
        } else {
            globalActive = true
        }
    }

    private func removeWaiter(token: UUID) {
        blocked.removeAll { $0.token == token }
    }

    private func release(repository: String?, token: UUID) {
        if let repository {
            activeRepositories.remove(repository)
        } else {
            globalActive = false
        }
        admitWaiting()
    }

    /// Hand the lock to whoever can take it now, in queue order.
    private func admitWaiting() {
        var index = 0
        while index < blocked.count {
            let candidate = blocked[index]
            guard canAdmit(repository: candidate.repository) else {
                // A queued whole-cache claim blocks everything behind it.
                if candidate.repository == nil { return }
                index += 1
                continue
            }
            blocked.remove(at: index)
            if candidate.waiter.resolve(acquired: true) {
                take(repository: candidate.repository)
            }
            // The removed entry may have been the whole-cache claim that was
            // holding the rest of the queue; rescan from the top either way.
            index = 0
        }
    }
}
