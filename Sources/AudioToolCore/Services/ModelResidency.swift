//
//  ModelResidency.swift
//  AudioToolCore
//
//  Centralized model lifecycle management with memory tracking and LRU eviction
//

import Foundation
import os

// MARK: - Model Lifecycle Manager

/// Centralized model lifecycle management with memory tracking and LRU eviction.
///
/// The manager tracks loaded models, their memory footprint, and access patterns.
/// When memory limit is exceeded, least-recently-used models are automatically evicted.
///
/// Residency is a bracket around each inference, not a one-time registration:
///
/// ```swift
/// let manager = ModelResidency(memoryLimitBytes: 4_000_000_000)  // 4GB limit
///
/// // Make the model resident, mark it most-recently-used, and protect it from
/// // eviction for the duration of the call.
/// try await manager.beginUse(enhancer)
/// let enhanced = try await enhancer.process(audio)
/// await manager.endUse(enhancer.modelId)
///
/// // Explicit unload
/// await manager.unload(modelId: "mossformer2_se_48k")
///
/// // Query status
/// print("Loaded: \(await manager.loadedModelIds)")
/// print("Memory: \(await manager.totalMemoryUsage / 1_000_000)MB")
/// ```
///
/// The bracket is the point. An earlier version registered each provider once, at
/// registration time, and never consulted the manager again. Two consequences, both
/// bad: LRU order reflected registration order rather than use, so the model used
/// most was the first evicted; and eviction unloaded a provider the caller still
/// held, whose next `process` call threw ``AudioToolError/modelNotLoaded`` instead
/// of reloading - providers here have no lazy-load path. ``beginUse(_:)`` fixes
/// both. It reloads an evicted model, and it will not evict one that is mid-call.
public actor ModelResidency {

    /// Eviction and residency events. A library has no business writing to stdout -
    /// a host app decides what surfaces where, and os.Logger lets it.
    private static let logger = Logger(subsystem: "AudioToolSwift", category: "ModelResidency")

    
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

    /// Whether one model may exceed the normal combined residency budget. Some
    /// providers (notably translation LLMs) are individually larger than the
    /// default budget; admitting one as the sole idle resident keeps the budget
    /// useful for eviction without making those providers impossible to use.
    public let allowsOversizedSingleton: Bool
    
    /// Loaded models keyed by modelId
    private var loadedModels: [String: LoadedModelEntry] = [:]

    /// How many in-flight calls are using each model.
    ///
    /// Non-zero means "do not unload": a provider evicted while its `process` was
    /// running would have its weights torn out from under a live inference.
    /// Balanced by ``endUse(_:)``, and counted rather than flagged because
    /// concurrent calls to the same model are legal.
    private var activeUseCounts: [String: Int] = [:]

    /// Total evictions performed
    private var evictionCount: Int = 0
    
    // MARK: - Initialization
    
    /// Create a manager with a specified normal residency budget.
    /// - Parameters:
    ///   - memoryLimitBytes: Maximum combined memory for ordinary loaded models (default: 2GB).
    ///   - allowsOversizedSingleton: Permit a single model larger than the budget,
    ///     evicting other idle models to make it the sole resident (default: true).
    public init(
        memoryLimitBytes: Int = 2_000_000_000,
        allowsOversizedSingleton: Bool = true
    ) {
        self.memoryLimitBytes = memoryLimitBytes
        self.allowsOversizedSingleton = allowsOversizedSingleton
    }
    
    // MARK: - Core Operations
    
    /// Make a model resident and claim it for the duration of one call.
    ///
    /// Loads the model if it is not loaded - including after an eviction, which is
    /// what makes eviction safe rather than merely silent. Marks it
    /// most-recently-used, and increments its in-use count so that a concurrent
    /// ``beginUse(_:)`` for some other model cannot evict it mid-inference.
    ///
    /// Every successful call must be balanced by ``endUse(_:)``, including on the
    /// error path, or the model is pinned in memory for the process's lifetime.
    ///
    /// - Parameter model: The model about to be used
    /// - Throws: If the model fails to load, or cannot fit in the memory limit at all
    public func beginUse(_ model: any ManagedModel) async throws {
        let modelId = model.modelId

        if let existing = loadedModels[modelId] {
            if Self.isSameInstance(existing.model, model) {
                // Trust the model, not the table. An entry can be stale - callers
                // hold the provider directly and may unload it themselves.
                if await model.checkIfLoaded() {
                    touch(modelId)
                    activeUseCounts[modelId, default: 0] += 1
                    return
                }
                // Same instance, unloaded behind our back: reload it below.
            } else if activeUseCounts[modelId] == nil {
                // A different instance is registering under this id. The one in the
                // table is about to become unreachable through the manager, so it
                // has to be released here or its weights stay resident, untracked
                // and impossible to evict - memory the accounting cannot see.
                await existing.model.unload()
            } else {
                // Displaced while someone is mid-call on it. Unloading would tear
                // weights out of a live inference, so the old instance is left to
                // its caller; it simply stops being the manager's business.
                Self.logger.error(
                    "'\(modelId, privacy: .public)' was replaced by a different instance while in use - the previous one is no longer tracked")
            }
            loadedModels.removeValue(forKey: modelId)
        }

        // Strict managers reject a model that can never fit. The default policy
        // instead admits it as an oversized singleton; TranslateGemma is larger
        // than AudioEngine's normal 2 GB residency budget.
        let requiredMemory = model.estimatedMemoryBytes
        if requiredMemory > memoryLimitBytes, !allowsOversizedSingleton {
            throw AudioToolError.memoryExhausted(
                required: requiredMemory,
                available: memoryLimitBytes
            )
        }

        // Claim before loading, so a reentrant beginUse for another model cannot
        // pick this one as its eviction victim while its weights are being read.
        activeUseCounts[modelId, default: 0] += 1
        do {
            await evictIfNeeded(forNewMemory: requiredMemory)
            if !(await model.checkIfLoaded()) {
                try await model.load()
            }
        } catch {
            releaseUse(modelId)
            throw error
        }

        let now = Date()
        loadedModels[modelId] = LoadedModelEntry(
            model: model,
            loadedAt: now,
            lastAccessedAt: now,
            memoryBytes: requiredMemory
        )
    }

    /// Release the claim taken by ``beginUse(_:)``, and settle the budget.
    ///
    /// Releasing is also the moment the budget can be honoured again. ``beginUse(_:)``
    /// will go over the limit rather than evict a model that is mid-inference, so
    /// two overlapping 100 MB calls under a 150 MB limit leave 200 MB resident. If
    /// nothing reconciled afterwards that overage would persist for the lifetime of
    /// the process - the limit would be respected only when models happened not to
    /// overlap, which is precisely when it does not matter.
    ///
    /// - Parameter modelId: The model identifier passed to ``beginUse(_:)``
    public func endUse(_ modelId: String) async {
        releaseUse(modelId)
        guard activeUseCounts[modelId] == nil else { return }  // still claimed elsewhere
        await evict(downTo: idleResidencyLimit)
    }

    private func releaseUse(_ modelId: String) {
        guard let count = activeUseCounts[modelId] else { return }
        if count <= 1 {
            activeUseCounts.removeValue(forKey: modelId)
        } else {
            activeUseCounts[modelId] = count - 1
        }
    }

    private func touch(_ modelId: String) {
        guard var entry = loadedModels[modelId] else { return }
        entry.lastAccessedAt = Date()
        loadedModels[modelId] = entry
    }

    /// Identity, not equality. `ManagedModel` carries no `Equatable` requirement,
    /// and two providers sharing a `modelId` are two different things holding two
    /// different sets of weights.
    private static func isSameInstance(_ lhs: any ManagedModel, _ rhs: any ManagedModel) -> Bool {
        // Value-type models have no instance identity to compare, and `as AnyObject`
        // would box each one into a fresh object. Treating them as distinct is the
        // conservative answer: it costs a reload, where the opposite leaks weights.
        guard type(of: lhs) is AnyClass, type(of: lhs) == type(of: rhs) else {
            return false
        }
        return (lhs as AnyObject) === (rhs as AnyObject)
    }

    /// Unload a specific model by ID.
    ///
    /// Unlike eviction this is unconditional - an explicit unload is the caller
    /// saying they know what they are doing, even if an inference is in flight.
    ///
    /// - Parameter modelId: The model identifier to unload
    public func unload(modelId: String) async {
        activeUseCounts.removeValue(forKey: modelId)
        guard let entry = loadedModels.removeValue(forKey: modelId) else {
            return
        }
        await entry.model.unload()
    }

    /// Unload all models.
    public func unloadAll() async {
        let entries = loadedModels.values
        loadedModels.removeAll()
        activeUseCounts.removeAll()

        for entry in entries {
            await entry.model.unload()
        }
    }
    
    /// Access a loaded model by ID.
    ///
    /// Updates the model's LRU timestamp. Prefer ``beginUse(_:)``, which also
    /// reloads an evicted model and holds it for the call; this is for callers that
    /// only want to look up what is already resident.
    ///
    /// - Parameter modelId: The model identifier
    /// - Returns: The model if loaded, nil otherwise
    public func access(modelId: String) -> (any ManagedModel)? {
        guard let entry = loadedModels[modelId] else {
            return nil
        }
        touch(modelId)
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
    
    /// Evict least-recently-used models to make room for a model about to load.
    private func evictIfNeeded(forNewMemory requiredBytes: Int) async {
        let admissionLimit = allowsOversizedSingleton
            ? max(memoryLimitBytes, requiredBytes)
            : memoryLimitBytes
        await evict(downTo: max(0, admissionLimit - requiredBytes))
    }

    /// Preserve at most one over-budget idle model. If multiple models are
    /// resident, LRU eviction brings their combined footprint down to the larger
    /// of the configured budget and the largest single model.
    private var idleResidencyLimit: Int {
        guard allowsOversizedSingleton else { return memoryLimitBytes }
        let largestModel = loadedModels.values.map(\.memoryBytes).max() ?? 0
        return max(memoryLimitBytes, largestModel)
    }

    /// Evict least-recently-used models until usage is at or below `targetUsage`.
    ///
    /// Models with an outstanding ``beginUse(_:)`` are not candidates. If the only
    /// things left to evict are in use, this stops rather than freeing them: going
    /// over the limit is recoverable, tearing weights out of a running inference is
    /// not. The limit is a budget, not an invariant - but ``endUse(_:)`` calls back
    /// here once a claim clears, so an overage lasts only as long as the calls that
    /// forced it.
    private func evict(downTo targetUsage: Int) async {
        while totalMemoryUsage > targetUsage {
            let candidates = loadedModels.values.filter { activeUseCounts[$0.modelId] == nil }
            guard let lruEntry = candidates.min(by: { $0.lastAccessedAt < $1.lastAccessedAt }) else {
                #if DEBUG
                Self.logger.info(
                    "Over budget by \((self.totalMemoryUsage - targetUsage) / 1_000_000, privacy: .public)MB but every resident model is in use - not evicting")
                #endif
                return
            }

            loadedModels.removeValue(forKey: lruEntry.modelId)
            await lruEntry.model.unload()
            evictionCount += 1

            #if DEBUG
            Self.logger.info(
                "Evicted '\(lruEntry.modelId, privacy: .public)' to free \(lruEntry.memoryBytes / 1_000_000, privacy: .public)MB")
            #endif
        }
    }
}

// MARK: - CustomStringConvertible

extension ModelResidency: CustomStringConvertible {
    public nonisolated var description: String {
        "ModelResidency(limit: \(memoryLimitBytes / 1_000_000)MB)"
    }
}

extension ModelResidency.Stats: CustomStringConvertible {
    public var description: String {
        """
        ModelResidency.Stats:
          Loaded models: \(loadedModelCount)
          Memory usage: \(totalMemoryBytes / 1_000_000)MB / \(memoryLimitBytes / 1_000_000)MB
          Available: \(availableBytes / 1_000_000)MB
          Total evictions: \(evictionCount)
        """
    }
}
