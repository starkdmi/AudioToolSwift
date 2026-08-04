//
//  ModelLifecycleManager.swift
//  AudioToolCore
//
//  Centralized model lifecycle management with memory tracking and LRU eviction
//

import Foundation

// MARK: - Model Lifecycle Manager

/// Centralized model lifecycle management with memory tracking and LRU eviction.
///
/// The manager tracks loaded models, their memory footprint, and access patterns.
/// When memory limit is exceeded, least-recently-used models are automatically evicted.
///
/// Usage:
/// ```swift
/// let manager = ModelLifecycleManager(memoryLimitBytes: 4_000_000_000)  // 4GB limit
///
/// // Register a model (loads if needed, evicts LRU if over limit)
/// try await manager.register(enhancerProvider)
///
/// // Access updates LRU timestamp
/// if let model = await manager.access(modelId: "mossformer2_se_48k") {
///     let enhanced = try await model.process(audio)
/// }
///
/// // Explicit unload
/// await manager.unload(modelId: "mossformer2_se_48k")
///
/// // Query status
/// print("Loaded: \(await manager.loadedModelIds)")
/// print("Memory: \(await manager.totalMemoryUsage / 1_000_000)MB")
/// ```
public actor ModelLifecycleManager {
    
    // MARK: - Types
    
    /// Entry for a loaded model with metadata
    private struct LoadedModelEntry: Sendable {
        let model: any ManagedModel
        let loadedAt: Date
        var lastAccessedAt: Date
        let memoryBytes: Int
        
        var modelId: String { model.modelId }
    }
    
    /// Manager statistics
    public struct Stats: Sendable {
        public let loadedModelCount: Int
        public let totalMemoryBytes: Int
        public let memoryLimitBytes: Int
        public let availableBytes: Int
        public let evictionCount: Int
    }
    
    // MARK: - Properties
    
    /// Memory limit in bytes
    public let memoryLimitBytes: Int
    
    /// Loaded models keyed by modelId
    private var loadedModels: [String: LoadedModelEntry] = [:]
    
    /// Total evictions performed
    private var evictionCount: Int = 0
    
    // MARK: - Initialization
    
    /// Create a manager with specified memory limit
    /// - Parameter memoryLimitBytes: Maximum combined memory for all loaded models (default: 2GB)
    public init(memoryLimitBytes: Int = 2_000_000_000) {
        self.memoryLimitBytes = memoryLimitBytes
    }
    
    // MARK: - Core Operations
    
    /// Register and load a model.
    ///
    /// If the model is already loaded, this updates its LRU timestamp.
    /// If loading would exceed memory limit, LRU models are evicted first.
    ///
    /// - Parameter model: The model to register
    /// - Throws: If model fails to load or exceeds memory limit
    public func register(_ model: any ManagedModel) async throws {
        let modelId = model.modelId
        
        // Already loaded? Just update access time
        if var existing = loadedModels[modelId] {
            existing.lastAccessedAt = Date()
            loadedModels[modelId] = existing
            return
        }
        
        // Preflight check: reject models that exceed memory limit entirely
        let requiredMemory = model.estimatedMemoryBytes
        if requiredMemory > memoryLimitBytes {
            throw AudioToolError.memoryExhausted(
                required: requiredMemory,
                available: memoryLimitBytes
            )
        }
        
        // Check if we need to evict before loading
        try await evictIfNeeded(forNewMemory: requiredMemory)
        
        // Load the model
        if !(await model.checkIfLoaded()) {
            try await model.load()
        }
        
        // Store entry
        let now = Date()
        let entry = LoadedModelEntry(
            model: model,
            loadedAt: now,
            lastAccessedAt: now,
            memoryBytes: requiredMemory
        )
        loadedModels[modelId] = entry
    }
    
    /// Unload a specific model by ID.
    ///
    /// - Parameter modelId: The model identifier to unload
    public func unload(modelId: String) async {
        guard let entry = loadedModels.removeValue(forKey: modelId) else {
            return
        }
        await entry.model.unload()
    }
    
    /// Unload all models.
    public func unloadAll() async {
        let entries = loadedModels.values
        loadedModels.removeAll()
        
        for entry in entries {
            await entry.model.unload()
        }
    }
    
    /// Access a loaded model by ID.
    ///
    /// Updates the model's LRU timestamp to prevent eviction.
    ///
    /// - Parameter modelId: The model identifier
    /// - Returns: The model if loaded, nil otherwise
    public func access(modelId: String) -> (any ManagedModel)? {
        guard var entry = loadedModels[modelId] else {
            return nil
        }
        
        // Update LRU timestamp
        entry.lastAccessedAt = Date()
        loadedModels[modelId] = entry
        
        return entry.model
    }
    
    // MARK: - Query
    
    /// IDs of all currently loaded models
    public var loadedModelIds: [String] {
        Array(loadedModels.keys)
    }
    
    /// Total memory used by loaded models in bytes
    public var totalMemoryUsage: Int {
        loadedModels.values.reduce(0) { $0 + $1.memoryBytes }
    }
    
    /// Available memory before eviction is triggered
    public var availableMemory: Int {
        max(0, memoryLimitBytes - totalMemoryUsage)
    }
    
    /// Check if a model is currently loaded
    public func isLoaded(modelId: String) -> Bool {
        loadedModels[modelId] != nil
    }
    
    /// Get manager statistics
    public var stats: Stats {
        Stats(
            loadedModelCount: loadedModels.count,
            totalMemoryBytes: totalMemoryUsage,
            memoryLimitBytes: memoryLimitBytes,
            availableBytes: availableMemory,
            evictionCount: evictionCount
        )
    }
    
    // MARK: - LRU Eviction
    
    /// Evict least-recently-used models until there's room for new memory
    private func evictIfNeeded(forNewMemory requiredBytes: Int) async throws {
        let targetUsage = memoryLimitBytes - requiredBytes
        
        // Keep evicting until we have enough room
        while totalMemoryUsage > targetUsage && !loadedModels.isEmpty {
            // Find LRU model
            guard let lruEntry = loadedModels.values.min(by: { $0.lastAccessedAt < $1.lastAccessedAt }) else {
                break
            }
            
            // Evict it
            loadedModels.removeValue(forKey: lruEntry.modelId)
            await lruEntry.model.unload()
            evictionCount += 1
            
            #if DEBUG
            print("ModelLifecycleManager: Evicted '\(lruEntry.modelId)' to free \(lruEntry.memoryBytes / 1_000_000)MB")
            #endif
        }
    }
}

// MARK: - CustomStringConvertible

extension ModelLifecycleManager: CustomStringConvertible {
    public nonisolated var description: String {
        "ModelLifecycleManager(limit: \(memoryLimitBytes / 1_000_000)MB)"
    }
}

extension ModelLifecycleManager.Stats: CustomStringConvertible {
    public var description: String {
        """
        ModelLifecycleManager.Stats:
          Loaded models: \(loadedModelCount)
          Memory usage: \(totalMemoryBytes / 1_000_000)MB / \(memoryLimitBytes / 1_000_000)MB
          Available: \(availableBytes / 1_000_000)MB
          Total evictions: \(evictionCount)
        """
    }
}
