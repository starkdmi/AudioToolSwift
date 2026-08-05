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
}

// MARK: - Tests

@Suite("ModelResidency Tests")
struct ModelResidencyTests {
    
    @Test("Manager tracks loaded models")
    func testManagerTracksModels() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let mock = MockManagedModel(modelId: "test_model", memoryBytes: 100_000_000)
        
        try await manager.register(mock)
        
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
        
        try await manager.register(mock)
        
        let afterAvailable = await manager.availableMemory
        #expect(afterAvailable == 300_000_000)
    }
    
    @Test("LRU eviction when limit exceeded")
    func testLRUEviction() async throws {
        let manager = ModelResidency(memoryLimitBytes: 150_000_000)
        
        let model1 = MockManagedModel(modelId: "model1", memoryBytes: 100_000_000)
        let model2 = MockManagedModel(modelId: "model2", memoryBytes: 100_000_000)
        
        try await manager.register(model1)
        
        // Wait a bit to ensure different timestamps
        try await Task.sleep(for: .milliseconds(10))
        
        try await manager.register(model2)  // Should evict model1
        
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
        
        try await manager.register(mock)
        
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
        
        try await manager.register(model1)
        try await manager.register(model2)
        
        await manager.unloadAll()
        
        let loadedIds = await manager.loadedModelIds
        let memoryUsage = await manager.totalMemoryUsage
        
        #expect(loadedIds.isEmpty)
        #expect(memoryUsage == 0)
    }
    
    @Test("Re-registering updates LRU timestamp")
    func testReregisterUpdatesLRU() async throws {
        let manager = ModelResidency(memoryLimitBytes: 250_000_000)
        
        let model1 = MockManagedModel(modelId: "model1", memoryBytes: 100_000_000)
        let model2 = MockManagedModel(modelId: "model2", memoryBytes: 100_000_000)
        let model3 = MockManagedModel(modelId: "model3", memoryBytes: 100_000_000)
        
        try await manager.register(model1)
        try await Task.sleep(for: .milliseconds(10))
        try await manager.register(model2)
        try await Task.sleep(for: .milliseconds(10))
        
        // Re-register model1 to update its LRU timestamp
        try await manager.register(model1)
        try await Task.sleep(for: .milliseconds(10))
        
        // Now register model3 - should evict model2 (older LRU) not model1
        try await manager.register(model3)
        
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
        
        try await manager.register(model1)
        try await manager.register(model2)
        
        let stats = await manager.stats
        
        #expect(stats.loadedModelCount == 2)
        #expect(stats.totalMemoryBytes == 250_000_000)
        #expect(stats.memoryLimitBytes == 500_000_000)
        #expect(stats.availableBytes == 250_000_000)
    }
}
