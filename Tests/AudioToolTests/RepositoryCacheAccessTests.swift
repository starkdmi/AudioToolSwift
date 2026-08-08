//
//  RepositoryCacheAccessTests.swift
//  AudioToolTests
//
//  Tests for per-repository cache mutation serialization
//

import Testing
@testable import AudioToolCore

private actor RepositoryTestGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        guard !isOpen else { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func open() {
        isOpen = true
        let waiting = waiters
        waiters.removeAll()
        for waiter in waiting { waiter.resume() }
    }
}

private actor RepositoryEventLog {
    private var events: [String] = []

    func record(_ event: String) {
        events.append(event)
    }

    func contains(_ event: String) -> Bool {
        events.contains(event)
    }

    func index(of event: String) -> Int? {
        events.firstIndex(of: event)
    }
}

@Suite("Repository cache access")
struct RepositoryCacheAccessTests {

    @Test("Same-repository operations serialize and cancelled waiters are removed")
    func repositoryOperationsAreExclusive() async throws {
        let access = RepositoryCacheAccess()
        let gate = RepositoryTestGate()
        let events = RepositoryEventLog()
        let repository = "test/shared-repository"

        let first = Task {
            try await access.withExclusiveAccess(to: repository) {
                await events.record("first-start")
                await gate.wait()
                await events.record("first-end")
            }
        }

        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(2))
        while !(await events.contains("first-start")), clock.now < deadline {
            await Task.yield()
        }
        #expect(await events.contains("first-start"))

        let cancelled = Task {
            try await access.withExclusiveAccess(to: repository) {
                await events.record("cancelled-entered")
            }
        }
        while await access.waitingCount(for: repository) == 0,
              clock.now < deadline {
            await Task.yield()
        }
        #expect(await access.waitingCount(for: repository) == 1)
        cancelled.cancel()
        do {
            try await cancelled.value
            Issue.record("cancelled repository waiter unexpectedly acquired the lock")
        } catch is CancellationError {
            // Expected.
        } catch {
            Issue.record("unexpected cancellation error: \(error)")
        }
        #expect(!(await events.contains("cancelled-entered")))
        #expect(await access.waitingCount(for: repository) == 0)

        let third = Task {
            try await access.withExclusiveAccess(to: repository) {
                await events.record("third-start")
            }
        }
        let unrelated = Task {
            try await access.withExclusiveAccess(to: "test/other-repository") {
                await events.record("other-start")
            }
        }
        try await unrelated.value
        #expect(await events.contains("other-start"))
        #expect(!(await events.contains("third-start")))

        await gate.open()
        try await first.value
        try await third.value

        let firstEnd = await events.index(of: "first-end")
        let thirdStart = await events.index(of: "third-start")
        #expect(firstEnd != nil)
        #expect(thirdStart != nil)
        if let firstEnd, let thirdStart {
            #expect(firstEnd < thirdStart)
        }
    }
}
