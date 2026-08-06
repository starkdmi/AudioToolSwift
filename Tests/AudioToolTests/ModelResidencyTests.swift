//
//  ModelResidencyTests.swift
//  AudioToolTests
//
//  Tests for ModelResidency memory tracking and LRU eviction
//

import Testing
import Foundation
@testable import AudioToolCore

// MARK: - Mock ManagedModel

/// Mock model for testing
actor MockManagedModel: ManagedModel {
    let modelId: String
    let estimatedMemoryBytes: Int
    private(set) var _isLoaded: Bool = false
    private(set) var loadCallCount: Int = 0
    private(set) var unloadCallCount: Int = 0

    init(modelId: String, memoryBytes: Int) {
        self.modelId = modelId
        self.estimatedMemoryBytes = memoryBytes
    }

    func checkIfLoaded() async -> Bool {
        _isLoaded
    }

    func load() async throws {
        loadCallCount += 1
        _isLoaded = true
    }

    func unload() async {
        unloadCallCount += 1
        _isLoaded = false
    }

    func getLoadCallCount() async -> Int { loadCallCount }
    func getUnloadCallCount() async -> Int { unloadCallCount }
    func getIsLoaded() async -> Bool { _isLoaded }
}

// MARK: - Tests

@Suite("ModelResidency Tests")
struct ModelResidencyTests {

    /// The normal shape of a call: claim, use, release.
    private func use(_ manager: ModelResidency, _ model: MockManagedModel) async throws {
        try await manager.beginUse(model)
        await manager.endUse(model.modelId)
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
        try await manager.beginUse(model1)
        #expect(await model1.getIsLoaded(), "model1 should be reloaded by beginUse")
        #expect(await model1.getLoadCallCount() == 2, "load should have been called a second time")
        await manager.endUse(model1.modelId)
    }

    /// Eviction must not pull the weights out from under a call that is running.
    @Test("A model in use is not evicted")
    func testInUseModelSurvivesEviction() async throws {
        let manager = ModelResidency(memoryLimitBytes: 150_000_000)

        let inFlight = MockManagedModel(modelId: "in_flight", memoryBytes: 100_000_000)
        let other = MockManagedModel(modelId: "other", memoryBytes: 100_000_000)

        // Claim and do not release - this stands in for an inference in progress.
        try await manager.beginUse(inFlight)
        try await Task.sleep(for: .milliseconds(10))

        // Over budget, but the only candidate is in use, so nothing is evicted.
        try await manager.beginUse(other)

        #expect(await inFlight.getIsLoaded(), "a model mid-call must not be unloaded")
        #expect(await inFlight.getUnloadCallCount() == 0)

        let loadedIds = await manager.loadedModelIds
        #expect(loadedIds.contains("in_flight"))
        #expect(loadedIds.contains("other"))

        // Once released it becomes a candidate again.
        await manager.endUse(inFlight.modelId)
        await manager.endUse(other.modelId)

        let third = MockManagedModel(modelId: "third", memoryBytes: 100_000_000)
        try await use(manager, third)
        #expect(await inFlight.getUnloadCallCount() == 1, "released model is evictable again")
    }

    /// A failed claim must not leave the model pinned for the process's lifetime.
    @Test("A model too large to fit does not hold a claim")
    func testFailedClaimIsReleased() async throws {
        let manager = ModelResidency(memoryLimitBytes: 100_000_000)
        let tooBig = MockManagedModel(modelId: "too_big", memoryBytes: 500_000_000)
        let small = MockManagedModel(modelId: "small", memoryBytes: 100_000_000)

        await #expect(throws: AudioToolError.self) {
            try await manager.beginUse(tooBig)
        }

        // The failure left no claim behind, so ordinary eviction still works.
        try await use(manager, small)
        try await Task.sleep(for: .milliseconds(10))
        let second = MockManagedModel(modelId: "second", memoryBytes: 100_000_000)
        try await use(manager, second)

        #expect(await small.getUnloadCallCount() == 1)
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

        try await manager.beginUse(model)
        #expect(await model.getIsLoaded(), "beginUse must verify with the model, not the table")
        #expect(await model.getLoadCallCount() == 2)
        await manager.endUse(model.modelId)
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
        try await manager.beginUse(first)
        try await Task.sleep(for: .milliseconds(10))
        try await manager.beginUse(second)
        #expect(await manager.totalMemoryUsage == 200_000_000, "deliberately over budget while both are in use")

        // Releasing the first is the moment the budget can be honoured again.
        await manager.endUse(first.modelId)
        await manager.endUse(second.modelId)

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

        try await manager.beginUse(replacement)
        await manager.endUse(replacement.modelId)

        #expect(await original.getUnloadCallCount() == 1,
                "the displaced instance must be unloaded or its weights leak")
        #expect(await original.getIsLoaded() == false)
        #expect(await replacement.getIsLoaded())
        #expect(await manager.totalMemoryUsage == 100_000_000, "one model resident, not two")
    }

    /// The exception: replacing an instance that is mid-call. Unloading it there
    /// would break the running inference, so it is released to its caller instead.
    @Test("A replaced instance that is in use is left alone")
    func testReplacedInstanceInUseIsNotUnloaded() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)

        let original = MockManagedModel(modelId: "shared_id", memoryBytes: 100_000_000)
        let replacement = MockManagedModel(modelId: "shared_id", memoryBytes: 100_000_000)

        try await manager.beginUse(original)   // still running
        try await manager.beginUse(replacement)

        #expect(await original.getIsLoaded(), "a model mid-call must not be unloaded even when displaced")
        #expect(await original.getUnloadCallCount() == 0)

        await manager.endUse(replacement.modelId)
        await manager.endUse(original.modelId)
    }

    @Test("Nested use of the same model is reference counted")
    func testNestedUseIsCounted() async throws {
        let manager = ModelResidency(memoryLimitBytes: 150_000_000)
        let model = MockManagedModel(modelId: "model", memoryBytes: 100_000_000)

        try await manager.beginUse(model)
        try await manager.beginUse(model)
        await manager.endUse(model.modelId)

        // One claim outstanding, so it is still protected.
        let other = MockManagedModel(modelId: "other", memoryBytes: 100_000_000)
        try await manager.beginUse(other)
        #expect(await model.getUnloadCallCount() == 0, "one outstanding claim still protects the model")

        await manager.endUse(model.modelId)
        await manager.endUse(other.modelId)
    }
}
