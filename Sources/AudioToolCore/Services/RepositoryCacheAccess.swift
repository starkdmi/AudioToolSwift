//
//  RepositoryCacheAccess.swift
//  AudioToolCore
//
//  Per-repository serialization for HuggingFace cache mutations
//

import Foundation

/// Serializes cache-mutating operations for the same repository while allowing
/// unrelated repositories to proceed concurrently.
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

    private struct RepositoryState {
        var owner: UUID?
        var waiters: [Waiter] = []
    }

    private var repositories: [String: RepositoryState] = [:]

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

    func waitingCount(for repository: String) -> Int {
        repositories[repository]?.waiters.count ?? 0
    }

    private func acquire(repository: String, token: UUID) async throws {
        try Task.checkCancellation()

        var state = repositories[repository] ?? RepositoryState()
        guard state.owner != nil else {
            state.owner = token
            repositories[repository] = state
            return
        }

        let waiter = Waiter(id: token)
        state.waiters.append(waiter)
        repositories[repository] = state

        let acquired = await withTaskCancellationHandler {
            await waiter.wait()
        } onCancel: {
            waiter.resolve(acquired: false)
        }

        guard acquired else {
            removeWaiter(repository: repository, token: token)
            throw CancellationError()
        }
        if Task.isCancelled {
            release(repository: repository, token: token)
            throw CancellationError()
        }
    }

    private func removeWaiter(repository: String, token: UUID) {
        guard var state = repositories[repository] else { return }
        state.waiters.removeAll { $0.id == token }
        repositories[repository] = state
    }

    private func release(repository: String, token: UUID) {
        guard var state = repositories[repository], state.owner == token else {
            return
        }

        state.owner = nil
        while !state.waiters.isEmpty {
            let waiter = state.waiters.removeFirst()
            if waiter.resolve(acquired: true) {
                state.owner = waiter.id
                break
            }
        }

        if state.owner == nil && state.waiters.isEmpty {
            repositories.removeValue(forKey: repository)
        } else {
            repositories[repository] = state
        }
    }
}
