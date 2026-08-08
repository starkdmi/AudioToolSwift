//
//  ModelResidencyTests.swift
//  AudioToolTests
//
//  Tests for ModelResidency memory tracking and LRU eviction
//

import Foundation
import Testing
@testable import AudioToolCore

// MARK: - Mock ManagedModel

/// Mock model for testing
actor MockManagedModel: ManagedModel {
    let modelId: String
    let estimatedMemoryBytes: Int
    private(set) var _isLoaded: Bool = false
    private(set) var loadCallCount: Int = 0
    private(set) var unloadCallCount: Int = 0
    private(set) var loadFinished: Bool = false
    private(set) var unloadFinished: Bool = true
    private let loadDelay: Duration
    private let unloadDelay: Duration
    private let failLoading: Bool
    private let ignoresLoadCancellation: Bool

    init(
        modelId: String,
        memoryBytes: Int,
        loadDelay: Duration = .zero,
        unloadDelay: Duration = .zero,
        failLoading: Bool = false,
        ignoresLoadCancellation: Bool = false
    ) {
        self.modelId = modelId
        self.estimatedMemoryBytes = memoryBytes
        self.loadDelay = loadDelay
        self.unloadDelay = unloadDelay
        self.failLoading = failLoading
        self.ignoresLoadCancellation = ignoresLoadCancellation
    }

    func checkIfLoaded() async -> Bool {
        _isLoaded
    }

    func load() async throws {
        loadCallCount += 1
        loadFinished = false
        _isLoaded = true
        if loadDelay > .zero {
            if ignoresLoadCancellation {
                let delay = loadDelay
                await Task.detached {
                    try? await Task.sleep(for: delay)
                }.value
            } else {
                try await Task.sleep(for: loadDelay)
            }
        }
        loadFinished = true
        if failLoading {
            throw AudioToolError.modelLoadFailed(
                modelId,
                underlying: CocoaError(.fileReadCorruptFile)
            )
        }
        _isLoaded = true
    }

    func unload() async {
        unloadCallCount += 1
        unloadFinished = false
        if unloadDelay > .zero {
            let delay = unloadDelay
            await Task.detached {
                try? await Task.sleep(for: delay)
            }.value
        }
        _isLoaded = false
        unloadFinished = true
    }

    func getLoadCallCount() async -> Int { loadCallCount }
    func getUnloadCallCount() async -> Int { unloadCallCount }
    func getIsLoaded() async -> Bool { _isLoaded }
    func getLoadFinished() async -> Bool { loadFinished }
    func getUnloadFinished() async -> Bool { unloadFinished }
}

// MARK: - Tests

@Suite("ModelResidency Tests")
struct ModelResidencyTests {

    /// The normal shape of a call: claim, use, release.
    private func use(_ manager: ModelResidency, _ model: MockManagedModel) async throws {
        let lease = try await manager.beginUse(model)
        await manager.endUse(lease)
    }

    private func waitForLoadToStart(_ model: MockManagedModel) async throws {
        for _ in 0..<1_000 {
            if await model.getLoadCallCount() > 0 { return }
            try await Task.sleep(for: .milliseconds(1))
        }
        Issue.record("model load did not start within one second")
    }

    private func waitForUnloadToStart(_ model: MockManagedModel) async throws {
        for _ in 0..<1_000 {
            if await model.getUnloadCallCount() > 0 { return }
            try await Task.sleep(for: .milliseconds(1))
        }
        Issue.record("model unload did not start within one second")
    }

    private func waitForPendingWaiters(
        _ expectedCount: Int,
        modelId: String,
        manager: ModelResidency
    ) async throws {
        for _ in 0..<1_000 {
            if await manager.pendingWaiterCount(for: modelId) == expectedCount {
                return
            }
            try await Task.sleep(for: .milliseconds(1))
        }
        Issue.record("pending waiter count did not become \(expectedCount) within one second")
    }

    private func beginUseWhenAvailable(
        _ manager: ModelResidency,
        _ model: MockManagedModel
    ) async throws -> ModelResidency.Lease {
        for _ in 0..<1_000 {
            do {
                return try await manager.beginUse(model)
            } catch let error as AudioToolError {
                guard case .resourceUnavailable = error else { throw error }
                try await Task.sleep(for: .milliseconds(1))
            }
        }
        throw AudioToolError.resourceUnavailable(
            "Model teardown did not finish within one second"
        )
    }

    @Test("Manager tracks loaded models")
    func testManagerTracksModels() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let mock = MockManagedModel(modelId: "test_model", memoryBytes: 100_000_000)

        try await use(manager, mock)

        let loadedIds = await manager.loadedModelIds
        let memoryUsage = await manager.totalMemoryUsage

        #expect(loadedIds.contains("test_model"))
        #expect(memoryUsage == 100_000_000)
    }

    @Test("Manager reports available memory")
    func testAvailableMemory() async throws {
        let manager = ModelResidency(memoryLimitBytes: 500_000_000)
        let mock = MockManagedModel(modelId: "test", memoryBytes: 200_000_000)

        let initialAvailable = await manager.availableMemory
        #expect(initialAvailable == 500_000_000)

        try await use(manager, mock)

        let afterAvailable = await manager.availableMemory
        #expect(afterAvailable == 300_000_000)
    }

    @Test("LRU eviction when limit exceeded")
    func testLRUEviction() async throws {
        let manager = ModelResidency(memoryLimitBytes: 150_000_000)

        let model1 = MockManagedModel(modelId: "model1", memoryBytes: 100_000_000)
        let model2 = MockManagedModel(modelId: "model2", memoryBytes: 100_000_000)

        try await use(manager, model1)

        // Wait a bit to ensure different timestamps
        try await Task.sleep(for: .milliseconds(10))

        try await use(manager, model2)  // Should evict model1

        let loadedIds = await manager.loadedModelIds
        let unloadCount1 = await model1.getUnloadCallCount()

        #expect(!loadedIds.contains("model1"), "model1 should be evicted")
        #expect(loadedIds.contains("model2"), "model2 should be loaded")
        #expect(unloadCount1 == 1, "model1 unload should be called once")
    }

    @Test("A model cannot reload while eviction is tearing it down")
    func testEvictionBlocksSameModelReacquisition() async throws {
        let manager = ModelResidency(memoryLimitBytes: 150_000_000)
        let first = MockManagedModel(
            modelId: "eviction_barrier",
            memoryBytes: 100_000_000,
            unloadDelay: .milliseconds(500)
        )
        let second = MockManagedModel(modelId: "replacement", memoryBytes: 100_000_000)
        try await use(manager, first)

        let secondAdmission = Task {
            let lease = try await manager.beginUse(second)
            await manager.endUse(lease)
        }
        try await waitForUnloadToStart(first)
        #expect(await first.getUnloadFinished() == false)

        await #expect(throws: AudioToolError.self) {
            _ = try await manager.beginUse(first)
        }

        try await secondAdmission.value
        #expect(await first.getUnloadFinished())
        #expect(await first.getUnloadCallCount() == 1)
        #expect(await first.getIsLoaded() == false)
    }

    @Test("Explicit unload clears memory")
    func testExplicitUnload() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let mock = MockManagedModel(modelId: "test", memoryBytes: 100_000_000)

        try await use(manager, mock)

        await manager.unload(modelId: "test")

        let loadedIds = await manager.loadedModelIds
        let memoryUsage = await manager.totalMemoryUsage
        let unloadCount = await mock.getUnloadCallCount()

        #expect(loadedIds.isEmpty)
        #expect(memoryUsage == 0)
        #expect(unloadCount == 1)
    }

    @Test("Duplicate unload waits for the running teardown")
    func testDuplicateUnloadJoinsRunningTeardown() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let model = MockManagedModel(
            modelId: "duplicate_unload",
            memoryBytes: 100_000_000,
            unloadDelay: .milliseconds(500)
        )
        try await use(manager, model)

        let firstUnload = Task { await manager.unload(modelId: model.modelId) }
        try await waitForUnloadToStart(model)
        #expect(await model.getUnloadFinished() == false)

        // This is a completion barrier, not an idempotent fire-and-forget call.
        await manager.unload(modelId: model.modelId)

        #expect(await model.getUnloadFinished())
        #expect(await model.getUnloadCallCount() == 1)
        await firstUnload.value
    }

    @Test("Unload all joins a running per-model teardown")
    func testUnloadAllJoinsRunningModelTeardown() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let model = MockManagedModel(
            modelId: "unload_all_barrier",
            memoryBytes: 100_000_000,
            unloadDelay: .milliseconds(500)
        )
        try await use(manager, model)

        let modelUnload = Task { await manager.unload(modelId: model.modelId) }
        try await waitForUnloadToStart(model)
        #expect(await model.getUnloadFinished() == false)

        await manager.unloadAll()

        #expect(await model.getUnloadFinished())
        #expect(await model.getUnloadCallCount() == 1)
        await modelUnload.value
    }

    @Test("Unload all models")
    func testUnloadAll() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)

        let model1 = MockManagedModel(modelId: "model1", memoryBytes: 100_000_000)
        let model2 = MockManagedModel(modelId: "model2", memoryBytes: 100_000_000)

        try await use(manager, model1)
        try await use(manager, model2)

        await manager.unloadAll()

        let loadedIds = await manager.loadedModelIds
        let memoryUsage = await manager.totalMemoryUsage

        #expect(loadedIds.isEmpty)
        #expect(memoryUsage == 0)
    }

    @Test("Re-using a model updates its LRU timestamp")
    func testReuseUpdatesLRU() async throws {
        let manager = ModelResidency(memoryLimitBytes: 250_000_000)

        let model1 = MockManagedModel(modelId: "model1", memoryBytes: 100_000_000)
        let model2 = MockManagedModel(modelId: "model2", memoryBytes: 100_000_000)
        let model3 = MockManagedModel(modelId: "model3", memoryBytes: 100_000_000)

        try await use(manager, model1)
        try await Task.sleep(for: .milliseconds(10))
        try await use(manager, model2)
        try await Task.sleep(for: .milliseconds(10))

        // Use model1 again to update its LRU timestamp
        try await use(manager, model1)
        try await Task.sleep(for: .milliseconds(10))

        // Now use model3 - should evict model2 (older LRU) not model1
        try await use(manager, model3)

        let loadedIds = await manager.loadedModelIds

        #expect(loadedIds.contains("model1"), "model1 should still be loaded (recently accessed)")
        #expect(!loadedIds.contains("model2"), "model2 should be evicted (older LRU)")
        #expect(loadedIds.contains("model3"), "model3 should be loaded")
    }

    @Test("Stats reports correct values")
    func testStatsReporting() async throws {
        let manager = ModelResidency(memoryLimitBytes: 500_000_000)

        let model1 = MockManagedModel(modelId: "model1", memoryBytes: 100_000_000)
        let model2 = MockManagedModel(modelId: "model2", memoryBytes: 150_000_000)

        try await use(manager, model1)
        try await use(manager, model2)

        let stats = await manager.stats

        #expect(stats.loadedModelCount == 2)
        #expect(stats.totalMemoryBytes == 250_000_000)
        #expect(stats.memoryLimitBytes == 500_000_000)
        #expect(stats.availableBytes == 250_000_000)
    }

    // MARK: - Eviction safety

    /// The regression that made eviction unsafe rather than merely invisible.
    ///
    /// Callers hold their provider directly and call `process` on it. Providers here
    /// have no lazy-load path - an unloaded one throws `modelNotLoaded`. So a model
    /// that was evicted has to come back before the next call, and the only place
    /// that can happen is the next `beginUse`.
    @Test("An evicted model is reloaded on next use")
    func testEvictedModelReloads() async throws {
        let manager = ModelResidency(memoryLimitBytes: 150_000_000)

        let model1 = MockManagedModel(modelId: "model1", memoryBytes: 100_000_000)
        let model2 = MockManagedModel(modelId: "model2", memoryBytes: 100_000_000)

        try await use(manager, model1)
        try await Task.sleep(for: .milliseconds(10))
        try await use(manager, model2)   // evicts model1

        #expect(await model1.getIsLoaded() == false, "model1 should have been unloaded by eviction")

        // Using model1 again must make it usable, not hand back a dead provider.
        let lease = try await manager.beginUse(model1)
        #expect(await model1.getIsLoaded(), "model1 should be reloaded by beginUse")
        #expect(await model1.getLoadCallCount() == 2, "load should have been called a second time")
        await manager.endUse(lease)
    }

    /// Eviction must not pull the weights out from under a call that is running.
    @Test("A model in use is not evicted")
    func testInUseModelSurvivesEviction() async throws {
        let manager = ModelResidency(memoryLimitBytes: 150_000_000)

        let inFlight = MockManagedModel(modelId: "in_flight", memoryBytes: 100_000_000)
        let other = MockManagedModel(modelId: "other", memoryBytes: 100_000_000)

        // Claim and do not release - this stands in for an inference in progress.
        let inFlightLease = try await manager.beginUse(inFlight)
        try await Task.sleep(for: .milliseconds(10))

        // Over budget, but the only candidate is in use, so nothing is evicted.
        let otherLease = try await manager.beginUse(other)

        #expect(await inFlight.getIsLoaded(), "a model mid-call must not be unloaded")
        #expect(await inFlight.getUnloadCallCount() == 0)

        let loadedIds = await manager.loadedModelIds
        #expect(loadedIds.contains("in_flight"))
        #expect(loadedIds.contains("other"))

        // Once released it becomes a candidate again.
        await manager.endUse(inFlightLease)
        await manager.endUse(otherLease)

        let third = MockManagedModel(modelId: "third", memoryBytes: 100_000_000)
        try await use(manager, third)
        #expect(await inFlight.getUnloadCallCount() == 1, "released model is evictable again")
    }

    /// A failed claim must not leave the model pinned for the process's lifetime.
    @Test("A model too large to fit does not hold a claim")
    func testFailedClaimIsReleased() async throws {
        let manager = ModelResidency(
            memoryLimitBytes: 100_000_000,
            allowsOversizedSingleton: false
        )
        let tooBig = MockManagedModel(modelId: "too_big", memoryBytes: 500_000_000)
        let small = MockManagedModel(modelId: "small", memoryBytes: 100_000_000)

        await #expect(throws: AudioToolError.self) {
            _ = try await manager.beginUse(tooBig)
        }

        // The failure left no claim behind, so ordinary eviction still works.
        try await use(manager, small)
        try await Task.sleep(for: .milliseconds(10))
        let second = MockManagedModel(modelId: "second", memoryBytes: 100_000_000)
        try await use(manager, second)

        #expect(await small.getUnloadCallCount() == 1)
    }

    @Test("An oversized model can be the sole resident")
    func testOversizedSingletonAdmission() async throws {
        let manager = ModelResidency(memoryLimitBytes: 100_000_000)
        let oversized = MockManagedModel(modelId: "oversized", memoryBytes: 500_000_000)

        try await use(manager, oversized)

        #expect(await oversized.getIsLoaded())
        #expect(await manager.totalMemoryUsage == 500_000_000)

        // A normal model still enforces the configured budget and evicts the
        // idle oversized singleton before loading.
        let ordinary = MockManagedModel(modelId: "ordinary", memoryBytes: 100_000_000)
        try await use(manager, ordinary)
        #expect(await oversized.getUnloadCallCount() == 1)
        #expect(await ordinary.getIsLoaded())
        #expect(await manager.totalMemoryUsage == 100_000_000)
    }

    /// A provider unloaded behind the manager's back must be reloaded, not assumed.
    @Test("A stale entry is reloaded rather than trusted")
    func testStaleEntryIsReloaded() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let model = MockManagedModel(modelId: "model", memoryBytes: 100_000_000)

        try await use(manager, model)
        #expect(await model.getLoadCallCount() == 1)

        // The caller holds the provider too, and unloads it themselves.
        await model.unload()

        let lease = try await manager.beginUse(model)
        #expect(await model.getIsLoaded(), "beginUse must verify with the model, not the table")
        #expect(await model.getLoadCallCount() == 2)
        await manager.endUse(lease)
    }

    /// Going over budget to protect a running call is fine. Staying over budget
    /// once the calls finish is not - the limit would then hold only when models
    /// happened not to overlap, which is exactly when it does not matter.
    @Test("The budget is reconciled once claims are released")
    func testBudgetReconcilesAfterRelease() async throws {
        let manager = ModelResidency(memoryLimitBytes: 150_000_000)

        let first = MockManagedModel(modelId: "first", memoryBytes: 100_000_000)
        let second = MockManagedModel(modelId: "second", memoryBytes: 100_000_000)

        // Overlapping calls: neither can be evicted, so both become resident.
        let firstLease = try await manager.beginUse(first)
        try await Task.sleep(for: .milliseconds(10))
        let secondLease = try await manager.beginUse(second)
        #expect(await manager.totalMemoryUsage == 200_000_000, "deliberately over budget while both are in use")

        // Releasing the first is the moment the budget can be honoured again.
        await manager.endUse(firstLease)
        await manager.endUse(secondLease)

        let usage = await manager.totalMemoryUsage
        #expect(usage <= 150_000_000, "still \(usage / 1_000_000)MB resident under a 150MB limit")
        #expect(await first.getUnloadCallCount() == 1, "the LRU model should have been evicted on release")
        #expect(await second.getIsLoaded(), "the most recently used model should survive")
    }

    /// Two providers can share a `modelId` - the engine's register methods key on
    /// the model enum, and nothing stops a caller swapping the instance behind one.
    /// Dropping the old entry without unloading it leaves its weights resident,
    /// untracked, and beyond the manager's reach.
    @Test("A replaced instance is unloaded, not orphaned")
    func testReplacedInstanceIsUnloaded() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)

        let original = MockManagedModel(modelId: "shared_id", memoryBytes: 100_000_000)
        let replacement = MockManagedModel(modelId: "shared_id", memoryBytes: 100_000_000)

        try await use(manager, original)
        #expect(await original.getIsLoaded())

        let replacementLease = try await manager.beginUse(replacement)
        await manager.endUse(replacementLease)

        #expect(await original.getUnloadCallCount() == 1,
                "the displaced instance must be unloaded or its weights leak")
        #expect(await original.getIsLoaded() == false)
        #expect(await replacement.getIsLoaded())
        #expect(await manager.totalMemoryUsage == 100_000_000, "one model resident, not two")
    }

    /// Replacing an active instance cannot be made safe: unloading breaks its call,
    /// while accepting the replacement orphans the original outside accounting.
    @Test("A conflicting active instance is rejected")
    func testReplacedInstanceInUseIsRejected() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)

        let original = MockManagedModel(modelId: "shared_id", memoryBytes: 100_000_000)
        let replacement = MockManagedModel(modelId: "shared_id", memoryBytes: 100_000_000)

        let originalLease = try await manager.beginUse(original)   // still running
        await #expect(throws: AudioToolError.self) {
            _ = try await manager.beginUse(replacement)
        }

        #expect(await original.getIsLoaded(), "a model mid-call must not be unloaded even when displaced")
        #expect(await original.getUnloadCallCount() == 0)
        #expect(await replacement.getLoadCallCount() == 0)
        #expect(await manager.totalMemoryUsage == 100_000_000)

        await manager.endUse(originalLease)
    }

    @Test("Concurrent claims share one model load")
    func testConcurrentClaimsUseSingleFlightLoad() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let model = MockManagedModel(
            modelId: "single_flight",
            memoryBytes: 100_000_000,
            loadDelay: .milliseconds(50)
        )

        async let first = manager.beginUse(model)
        async let second = manager.beginUse(model)
        let (firstLease, secondLease) = try await (first, second)

        #expect(await model.getLoadCallCount() == 1)
        #expect(await manager.totalMemoryUsage == 100_000_000)
        await manager.endUse(firstLease)
        await manager.endUse(secondLease)
    }

    @Test("Canceling one waiter keeps a shared load alive")
    func testCancelOneOfMultiplePendingClaims() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let model = MockManagedModel(
            modelId: "shared_cancellation",
            memoryBytes: 100_000_000,
            loadDelay: .milliseconds(300)
        )

        let cancelledClaim = Task { try await manager.beginUse(model) }
        try await waitForLoadToStart(model)
        let survivingClaim = Task { try await manager.beginUse(model) }
        try await waitForPendingWaiters(2, modelId: model.modelId, manager: manager)

        cancelledClaim.cancel()
        await #expect(throws: CancellationError.self) {
            _ = try await cancelledClaim.value
        }
        #expect(await model.getLoadFinished() == false)
        #expect(await manager.pendingWaiterCount(for: model.modelId) == 1)

        let survivingLease = try await survivingClaim.value
        #expect(await model.getLoadCallCount() == 1)
        #expect(await model.getUnloadCallCount() == 0)
        #expect(await manager.loadedModelIds == [model.modelId])
        await manager.endUse(survivingLease)
    }

    @Test("Canceling the final waiter returns before load rollback finishes")
    func testCancelFinalPendingClaimIsImmediate() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let model = MockManagedModel(
            modelId: "abandoned_load",
            memoryBytes: 100_000_000,
            loadDelay: .milliseconds(300),
            ignoresLoadCancellation: true
        )
        let claim = Task { try await manager.beginUse(model) }
        try await waitForLoadToStart(model)
        try await waitForPendingWaiters(1, modelId: model.modelId, manager: manager)

        claim.cancel()
        await #expect(throws: CancellationError.self) {
            _ = try await claim.value
        }

        // The provider deliberately ignores cancellation while loading. Returning
        // before this flips proves the waiter no longer blocks on `Task.value`.
        #expect(await model.getLoadFinished() == false)
        #expect(await manager.loadedModelIds.isEmpty)
        #expect(await manager.totalMemoryUsage == 0)

        // The abandoned generation remains unavailable while its provider ignores
        // cancellation. This prevents its eventual unload from tearing down a new
        // provider that happens to reuse the same identifier.
        let replacement = MockManagedModel(
            modelId: model.modelId,
            memoryBytes: 100_000_000
        )
        await #expect(throws: AudioToolError.self) {
            _ = try await manager.beginUse(replacement)
        }

        try await waitForUnloadToStart(model)
        #expect(await model.getLoadFinished())
        #expect(await model.getUnloadCallCount() == 1)
        #expect(await model.getIsLoaded() == false)

        let replacementLease = try await beginUseWhenAvailable(manager, replacement)
        #expect(await replacement.getIsLoaded())
        #expect(await replacement.getLoadCallCount() == 1)
        await manager.endUse(replacementLease)
    }

    @Test("Unload all joins abandoned load cleanup")
    func testUnloadAllJoinsAbandonedLoadCleanup() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let model = MockManagedModel(
            modelId: "abandoned_unload_all",
            memoryBytes: 100_000_000,
            loadDelay: .milliseconds(300),
            ignoresLoadCancellation: true
        )
        let claim = Task { try await manager.beginUse(model) }
        try await waitForLoadToStart(model)

        claim.cancel()
        await #expect(throws: CancellationError.self) {
            _ = try await claim.value
        }
        #expect(await model.getLoadFinished() == false)

        await manager.unloadAll()

        #expect(await model.getLoadFinished())
        #expect(await model.getUnloadCallCount() == 1)
        #expect(await model.getIsLoaded() == false)
        #expect(await manager.loadedModelIds.isEmpty)
    }

    @Test("A conflicting instance cannot join an in-flight load")
    func testPendingLoadRejectsConflictingInstance() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let original = MockManagedModel(
            modelId: "pending_conflict",
            memoryBytes: 100_000_000,
            loadDelay: .milliseconds(50)
        )
        let replacement = MockManagedModel(
            modelId: "pending_conflict",
            memoryBytes: 100_000_000
        )
        let originalClaim = Task { try await manager.beginUse(original) }
        try await waitForLoadToStart(original)

        await #expect(throws: AudioToolError.self) {
            _ = try await manager.beginUse(replacement)
        }

        let lease = try await originalClaim.value
        #expect(await original.getLoadCallCount() == 1)
        #expect(await replacement.getLoadCallCount() == 0)
        await manager.endUse(lease)
    }

    @Test("Explicit unload cancels an in-flight load without publishing it")
    func testUnloadCancelsPendingLoad() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let model = MockManagedModel(
            modelId: "pending_unload",
            memoryBytes: 100_000_000,
            loadDelay: .seconds(5)
        )
        let claim = Task { try await manager.beginUse(model) }
        try await waitForLoadToStart(model)

        await manager.unload(modelId: model.modelId)

        await #expect(throws: CancellationError.self) {
            _ = try await claim.value
        }
        #expect(await model.getUnloadCallCount() == 1)
        #expect(await model.getIsLoaded() == false)
        #expect(await manager.loadedModelIds.isEmpty)
        #expect(await manager.totalMemoryUsage == 0)
    }

    @Test("A stale lease cannot release a replacement instance")
    func testLeaseIsInstanceGenerationScoped() async throws {
        let manager = ModelResidency(memoryLimitBytes: 150_000_000)
        let original = MockManagedModel(modelId: "shared", memoryBytes: 100_000_000)
        let replacement = MockManagedModel(modelId: "shared", memoryBytes: 100_000_000)
        let other = MockManagedModel(modelId: "other", memoryBytes: 100_000_000)

        let staleLease = try await manager.beginUse(original)
        await manager.unload(modelId: original.modelId)
        let replacementLease = try await manager.beginUse(replacement)

        // This release belongs to the explicitly unloaded generation and must not
        // decrement the replacement's active claim.
        await manager.endUse(staleLease)
        let otherLease = try await manager.beginUse(other)

        #expect(await replacement.getUnloadCallCount() == 0)
        #expect(await replacement.getIsLoaded())

        await manager.endUse(replacementLease)
        await manager.endUse(otherLease)
    }

    @Test("A failed load is rolled back")
    func testFailedLoadRollsBackProviderState() async {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let model = MockManagedModel(
            modelId: "failing",
            memoryBytes: 100_000_000,
            failLoading: true
        )

        await #expect(throws: AudioToolError.self) {
            _ = try await manager.beginUse(model)
        }

        #expect(await model.getIsLoaded() == false)
        #expect(await model.getUnloadCallCount() == 1)
        #expect(await manager.loadedModelIds.isEmpty)
        #expect(await manager.totalMemoryUsage == 0)
    }

    @Test("Nested use of the same model is reference counted")
    func testNestedUseIsCounted() async throws {
        let manager = ModelResidency(memoryLimitBytes: 150_000_000)
        let model = MockManagedModel(modelId: "model", memoryBytes: 100_000_000)

        let firstLease = try await manager.beginUse(model)
        let secondLease = try await manager.beginUse(model)
        await manager.endUse(firstLease)

        // One claim outstanding, so it is still protected.
        let other = MockManagedModel(modelId: "other", memoryBytes: 100_000_000)
        let otherLease = try await manager.beginUse(other)
        #expect(await model.getUnloadCallCount() == 0, "one outstanding claim still protects the model")

        await manager.endUse(secondLease)
        await manager.endUse(otherLease)
    }
}
