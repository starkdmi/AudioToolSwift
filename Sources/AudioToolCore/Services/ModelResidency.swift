//
//  ModelResidency.swift
//  AudioToolCore
//
//  Centralized model lifecycle management with memory tracking and LRU eviction
//

import Foundation
import os

/// Broadcasts one shared load result to cancellation-aware waiters. Completion and
/// each caller's cancellation race under one lock, so every checked continuation
/// is resumed exactly once without creating a blocked observer task per caller.
private final class PendingLoadCompletion: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<Void, Error>?
    private var continuations: [UUID: CheckedContinuation<Void, Error>] = [:]
    private var cancelledWaiters: Set<UUID> = []

    func wait() async throws {
        let waiterId = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                install(continuation, for: waiterId)
            }
        } onCancel: {
            self.cancel(waiterId)
        }
    }

    func complete(with result: Result<Void, Error>) {
        lock.lock()
        guard self.result == nil else {
            lock.unlock()
            return
        }
        self.result = result
        let waitingContinuations = Array(continuations.values)
        self.continuations.removeAll()
        cancelledWaiters.removeAll()
        lock.unlock()

        for continuation in waitingContinuations {
            continuation.resume(with: result)
        }
    }

    private func install(
        _ continuation: CheckedContinuation<Void, Error>,
        for waiterId: UUID
    ) {
        lock.lock()
        if let result {
            lock.unlock()
            continuation.resume(with: result)
            return
        }
        if cancelledWaiters.remove(waiterId) != nil {
            lock.unlock()
            continuation.resume(throwing: CancellationError())
            return
        }
        continuations[waiterId] = continuation
        lock.unlock()
    }

    private func cancel(_ waiterId: UUID) {
        lock.lock()
        guard result == nil else {
            lock.unlock()
            return
        }
        if let continuation = continuations.removeValue(forKey: waiterId) {
            lock.unlock()
            continuation.resume(throwing: CancellationError())
        } else {
            cancelledWaiters.insert(waiterId)
            lock.unlock()
        }
    }
}

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
/// let lease = try await manager.beginUse(enhancer)
/// let enhanced = try await enhancer.process(audio)
/// await manager.endUse(lease)
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

    /// An instance-scoped claim on one resident model.
    ///
    /// Releasing by token prevents a late `endUse` from an explicitly unloaded old
    /// provider from decrementing the claim count of a newer provider with the same
    /// model identifier.
    public struct Lease: Sendable, Hashable {
        public let modelId: String
        fileprivate let id: UUID
    }
    
    /// Entry for a loaded model with metadata
    private struct LoadedModelEntry: Sendable {
        let model: any ManagedModel
        let loadedAt: Date
        var lastAccessedAt: Date
        let memoryBytes: Int
        
        var modelId: String { model.modelId }
    }

    /// One shared load operation for every caller waiting on the same provider.
    /// The token prevents a cancelled/replaced operation from publishing itself
    /// after a newer operation has taken its place.
    private struct PendingLoad: Sendable {
        let token: UUID
        let model: any ManagedModel
        let replacedModel: (any ManagedModel)?
        let memoryBytes: Int
        let startedAt: Date
        let task: Task<Void, Error>
        let completion: PendingLoadCompletion
        var waiterCount: Int
    }

    /// One provider teardown, shared by explicit unloads, abandoned loads, and
    /// eviction. The token prevents a late waiter from clearing a newer operation.
    private struct ModelTeardown: Sendable {
        let token: UUID
        let replacedModel: (any ManagedModel)?
        let task: Task<Void, Never>
    }

    /// One `unloadAll()` barrier. Keeping the model teardowns it joined lets the
    /// final waiter remove their completed records before reopening admission.
    private struct GlobalTeardown: Sendable {
        let token: UUID
        let joinedModelTeardowns: [String: ModelTeardown]
        let task: Task<Void, Never>
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

    /// Loads are single-flight per model identifier. Actor isolation alone is not
    /// sufficient because calls re-enter this actor while a provider awaits I/O.
    private var pendingLoads: [String: PendingLoad] = [:]

    /// Provider teardown permits actor reentrancy. Retaining the task makes every
    /// duplicate unload a completion barrier and blocks admission until teardown
    /// has actually finished.
    private var modelTeardowns: [String: ModelTeardown] = [:]
    private var globalTeardown: GlobalTeardown?

    /// How many in-flight calls are using each model.
    ///
    /// Non-zero means "do not unload": a provider evicted while its `process` was
    /// running would have its weights torn out from under a live inference.
    /// Balanced by ``endUse(_:)``, and counted rather than flagged because
    /// concurrent calls to the same model are legal.
    private var activeUseCounts: [String: Int] = [:]
    private var activeLeases: [UUID: String] = [:]

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
    /// Cancelling a caller while it waits for a shared load releases its claim
    /// immediately. The load itself is cancelled and rolled back once its final
    /// waiter leaves.
    ///
    /// - Parameter model: The model about to be used
    /// - Throws: If the model fails to load, or cannot fit in the memory limit at all
    /// - Returns: A lease that must be passed to ``endUse(_:)``.
    public func beginUse(_ model: any ManagedModel) async throws -> Lease {
        let modelId = model.modelId
        let requiredMemory = model.estimatedMemoryBytes

        guard globalTeardown == nil, modelTeardowns[modelId] == nil else {
            throw AudioToolError.resourceUnavailable(
                "Model '\(modelId)' is currently being unloaded"
            )
        }

        // Strict managers reject a model that can never fit. The default policy
        // instead admits it as an oversized singleton; TranslateGemma is larger
        // than AudioEngine's normal 2 GB residency budget.
        if requiredMemory > memoryLimitBytes, !allowsOversizedSingleton {
            throw AudioToolError.memoryExhausted(
                required: requiredMemory,
                available: memoryLimitBytes
            )
        }

        if let pending = pendingLoads[modelId] {
            try ensureSameInstance(pending.model, model, modelId: modelId)
            let lease = claim(modelId)
            try await awaitPendingLoad(pending, lease: lease)
            return lease
        }

        if let existing = loadedModels[modelId] {
            if Self.isSameInstance(existing.model, model) {
                // Claim before the await. A concurrent admission may otherwise
                // evict this entry while its readiness is being checked.
                let lease = claim(modelId)
                // Trust the model, not the table. An entry can be stale - callers
                // hold the provider directly and may unload it themselves.
                if await model.checkIfLoaded() {
                    if let current = loadedModels[modelId],
                       Self.isSameInstance(current.model, model),
                        pendingLoads[modelId] == nil {
                        touch(modelId)
                        return lease
                    }
                    // State changed during the await. Retry without leaking the
                    // claim taken above.
                    release(lease)
                    return try await beginUse(model)
                }

                // Same instance, unloaded behind our back. Another reentrant call
                // may already have started the reload while this check awaited.
                if let pending = pendingLoads[modelId] {
                    do {
                        try ensureSameInstance(pending.model, model, modelId: modelId)
                        try await awaitPendingLoad(pending, lease: lease)
                    } catch {
                        // `awaitPendingLoad` releases load errors itself. A validation
                        // error occurs before it is called and still owns this lease.
                        if activeLeases[lease.id] != nil { release(lease) }
                        throw error
                    }
                    return lease
                }
                if let current = loadedModels[modelId],
                   Self.isSameInstance(current.model, model) {
                    loadedModels.removeValue(forKey: modelId)
                    return try await startPendingLoad(
                        model,
                        memoryBytes: requiredMemory,
                        replacing: nil,
                        existingLease: lease
                    )
                }

                release(lease)
                return try await beginUse(model)
            } else if activeUseCounts[modelId] == nil {
                return try await startPendingLoad(
                    model,
                    memoryBytes: requiredMemory,
                    replacing: existing,
                    existingLease: nil
                )
            } else {
                // A modelId is the manager's ownership key. Replacing an active
                // instance would either unload live weights or orphan them outside
                // memory accounting, so fail explicitly.
                throw AudioToolError.pipelineConfigurationInvalid(
                    "A different model instance with id '\(modelId)' is already in use"
                )
            }
        }

        return try await startPendingLoad(
            model,
            memoryBytes: requiredMemory,
            replacing: nil,
            existingLease: nil
        )
    }

    private func startPendingLoad(
        _ model: any ManagedModel,
        memoryBytes: Int,
        replacing existing: LoadedModelEntry?,
        existingLease: Lease?
    ) async throws -> Lease {
        let modelId = model.modelId
        let lease = existingLease ?? claim(modelId)
        let token = UUID()
        let startedAt = Date()
        let completion = PendingLoadCompletion()
        let task = Task {
            do {
                try await self.prepareAndLoad(
                    model,
                    memoryBytes: memoryBytes,
                    replacing: existing
                )
                completion.complete(with: .success(()))
            } catch {
                completion.complete(with: .failure(error))
                throw error
            }
        }
        let pending = PendingLoad(
            token: token,
            model: model,
            replacedModel: existing?.model,
            memoryBytes: memoryBytes,
            startedAt: startedAt,
            task: task,
            completion: completion,
            waiterCount: 0
        )
        pendingLoads[modelId] = pending

        try await awaitPendingLoad(pending, lease: lease)
        return lease
    }

    private func awaitPendingLoad(
        _ pending: PendingLoad,
        lease: Lease
    ) async throws {
        let modelId = lease.modelId
        guard registerWaiter(for: pending, modelId: modelId) else {
            release(lease)
            throw CancellationError()
        }

        do {
            try Task.checkCancellation()
            try await pending.completion.wait()
            try Task.checkCancellation()
            if let current = pendingLoads[modelId], current.token == pending.token {
                pendingLoads.removeValue(forKey: modelId)
                loadedModels[modelId] = LoadedModelEntry(
                    model: pending.model,
                    loadedAt: pending.startedAt,
                    lastAccessedAt: Date(),
                    memoryBytes: pending.memoryBytes
                )
            } else if let loaded = loadedModels[modelId],
                      Self.isSameInstance(loaded.model, pending.model) {
                // Another waiter won the race to publish this shared load.
                touch(modelId)
            } else {
                throw CancellationError()
            }
        } catch {
            let waiterWasCancelled = Task.isCancelled
            let wasLastWaiter = unregisterWaiter(for: pending, modelId: modelId)
            release(lease)

            if waiterWasCancelled,
               wasLastWaiter {
                abandonPendingLoad(pending, modelId: modelId)
            } else if !waiterWasCancelled,
                      pendingLoads[modelId]?.token == pending.token {
                pendingLoads.removeValue(forKey: modelId)
                // This load displaced a resident of the same identifier and then
                // failed. Put the old one back now that its slot is free again.
                if let replaced = pending.replacedModel {
                    await restore(replaced)
                }
            }
            throw error
        }
    }

    private func registerWaiter(
        for pending: PendingLoad,
        modelId: String
    ) -> Bool {
        guard var current = pendingLoads[modelId],
              current.token == pending.token else { return false }
        current.waiterCount += 1
        pendingLoads[modelId] = current
        return true
    }

    /// Returns whether this caller was the final waiter for the current pending
    /// generation. Active inference leases are intentionally not part of this
    /// count: they are residency claims, not callers awaiting this load.
    private func unregisterWaiter(
        for pending: PendingLoad,
        modelId: String
    ) -> Bool {
        guard var current = pendingLoads[modelId],
              current.token == pending.token,
              current.waiterCount > 0 else { return false }
        current.waiterCount -= 1
        pendingLoads[modelId] = current
        return current.waiterCount == 0
    }

    /// Cancel a shared load only after its final waiter has released its lease.
    /// Keep the abandoned generation behind a tracked teardown until provider
    /// rollback finishes; otherwise a new generation could start and then be torn
    /// down by the old generation's delayed `unload()`.
    private func abandonPendingLoad(
        _ pending: PendingLoad,
        modelId: String
    ) {
        guard let current = pendingLoads[modelId],
              current.token == pending.token,
              current.waiterCount == 0 else { return }

        pendingLoads.removeValue(forKey: modelId)
        pending.task.cancel()

        let cleanupTask = Task {
            let outcome = await pending.task.result
            // `prepareAndLoad` rolls back every throwing path. A task that
            // completed just before cancellation still needs explicit teardown
            // because it was never published as a resident entry.
            if case .success = outcome {
                await pending.model.unload()
            }
        }
        recordModelTeardown(
            modelId: modelId,
            replacedModel: pending.replacedModel,
            task: cleanupTask
        )
    }

    @discardableResult
    private func recordModelTeardown(
        modelId: String,
        replacedModel: (any ManagedModel)?,
        task: Task<Void, Never>
    ) -> ModelTeardown {
        let teardown = ModelTeardown(
            token: UUID(),
            replacedModel: replacedModel,
            task: task
        )
        modelTeardowns[modelId] = teardown
        Task { await settleModelTeardown(teardown, modelId: modelId) }
        return teardown
    }

    private func settleModelTeardown(
        _ teardown: ModelTeardown,
        modelId: String
    ) async {
        await teardown.task.value
        finishModelTeardown(teardown, modelId: modelId)
    }

    private func finishModelTeardown(
        _ teardown: ModelTeardown,
        modelId: String
    ) {
        guard modelTeardowns[modelId]?.token == teardown.token else { return }
        modelTeardowns.removeValue(forKey: modelId)
    }

    private func finishGlobalTeardown(_ teardown: GlobalTeardown) {
        guard globalTeardown?.token == teardown.token else { return }
        for (modelId, modelTeardown) in teardown.joinedModelTeardowns {
            finishModelTeardown(modelTeardown, modelId: modelId)
        }
        globalTeardown = nil
    }

    private func prepareAndLoad(
        _ model: any ManagedModel,
        memoryBytes: Int,
        replacing existing: LoadedModelEntry?
    ) async throws {
        if let existing {
            if let current = loadedModels[existing.modelId],
               Self.isSameInstance(current.model, existing.model) {
                loadedModels.removeValue(forKey: existing.modelId)
            }
            await existing.model.unload()
        }

        do {
            try Task.checkCancellation()
            await evictIfNeeded(forNewMemory: memoryBytes)
            if !(await model.checkIfLoaded()) {
                try await model.load()
            }
            try Task.checkCancellation()
        } catch {
            // A failed or cancelled load must never publish partially initialized
            // provider state through the residency manager.
            await model.unload()
            throw error
        }
    }

    /// Put back a resident that was displaced for a replacement that then failed.
    ///
    /// Replacement unloads first by design - two instances of the same model are two
    /// copies of the same several gigabytes, and the budget exists precisely to stop
    /// that. The cost of unloading first is that a failed replacement takes the
    /// working provider with it: the caller still holds and has registered the old
    /// instance, and would find it unloaded through no action of its own. Reloading
    /// an already-downloaded model is the cheap half of a load, and it is the state
    /// the caller was in before it asked for something this manager could not give.
    ///
    /// Goes through ``beginUse(_:)`` rather than loading and publishing by hand. A
    /// hand-rolled restore is invisible to this actor while it awaits: `unloadAll()`
    /// could return and *then* have this publish behind it, a concurrent admission
    /// would size its eviction against a budget that did not know about these bytes,
    /// and losing a race to a same-identifier winner left the restored model loaded
    /// but untracked - resident memory the manager could never reclaim. `beginUse` is
    /// the one path that reserves the identifier before suspending, accounts for the
    /// memory, and refuses while a teardown is in flight.
    ///
    /// Best effort by construction. If the restore fails, the original error is what
    /// the caller needs to see, and `beginUse` reloads the model on its next use
    /// regardless. Skipped entirely on cancellation, where nobody is waiting for the
    /// old model either.
    private func restore(_ model: any ManagedModel) async {
        guard let lease = try? await beginUse(model) else {
            Self.logger.error(
                """
                Could not restore '\(model.modelId, privacy: .public)' after a failed \
                replacement; it will reload on next use
                """
            )
            return
        }
        // Immediately: this is a restoration, not a use. The claim exists only to
        // keep the entry from being evicted between admission and here.
        release(lease)
    }

    private func ensureSameInstance(
        _ existing: any ManagedModel,
        _ proposed: any ManagedModel,
        modelId: String
    ) throws {
        guard Self.isSameInstance(existing, proposed) else {
            throw AudioToolError.pipelineConfigurationInvalid(
                "A different model instance with id '\(modelId)' is already loading"
            )
        }
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
    /// - Parameter lease: The exact lease returned by ``beginUse(_:)``.
    public func endUse(_ lease: Lease) async {
        release(lease)
        let modelId = lease.modelId
        guard activeUseCounts[modelId] == nil else { return }  // still claimed elsewhere
        await evict(downTo: idleResidencyLimit)
    }

    private func claim(_ modelId: String) -> Lease {
        let lease = Lease(modelId: modelId, id: UUID())
        activeLeases[lease.id] = modelId
        activeUseCounts[modelId, default: 0] += 1
        return lease
    }

    private func release(_ lease: Lease) {
        guard activeLeases.removeValue(forKey: lease.id) == lease.modelId else {
            return
        }
        releaseUseCount(lease.modelId)
    }

    private func releaseUseCount(_ modelId: String) {
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
        if let globalTeardown {
            await globalTeardown.task.value
            finishGlobalTeardown(globalTeardown)
            return
        }

        // Explicit unload invalidates every lease for this identifier, even when
        // it is joining an abandonment or eviction teardown started elsewhere.
        activeUseCounts.removeValue(forKey: modelId)
        activeLeases = activeLeases.filter { $0.value != modelId }
        if let teardown = modelTeardowns[modelId] {
            await teardown.task.value
            finishModelTeardown(teardown, modelId: modelId)
            return
        }

        let pending = pendingLoads.removeValue(forKey: modelId)
        pending?.task.cancel()
        let entry = loadedModels.removeValue(forKey: modelId)

        guard pending != nil || entry != nil else { return }
        let shouldUnloadEntry: Bool
        if let entry, let replaced = pending?.replacedModel {
            shouldUnloadEntry = !Self.isSameInstance(replaced, entry.model)
        } else {
            shouldUnloadEntry = entry != nil
        }
        let task = Task {
            if let pending {
                let outcome = await pending.task.result
                if case .success = outcome {
                    await pending.model.unload()
                }
            }
            if shouldUnloadEntry, let entry {
                await entry.model.unload()
            }
        }
        let teardown = recordModelTeardown(
            modelId: modelId,
            replacedModel: pending?.replacedModel,
            task: task
        )
        await teardown.task.value
        finishModelTeardown(teardown, modelId: modelId)
    }

    /// Unload all models.
    public func unloadAll() async {
        if let globalTeardown {
            await globalTeardown.task.value
            finishGlobalTeardown(globalTeardown)
            return
        }

        let joinedModelTeardowns = modelTeardowns
        let entries = Array(loadedModels.values)
        let pending = Array(pendingLoads.values)
        loadedModels.removeAll()
        pendingLoads.removeAll()
        activeUseCounts.removeAll()
        activeLeases.removeAll()

        for load in pending {
            load.task.cancel()
        }
        let entriesToUnload = entries.filter { entry in
            let replacedByPendingLoad = pending.contains(where: { load in
                guard let replaced = load.replacedModel else { return false }
                return Self.isSameInstance(replaced, entry.model)
            })
            let replacedByExistingTeardown = joinedModelTeardowns.values.contains(
                where: { teardown in
                    guard let replaced = teardown.replacedModel else { return false }
                    return Self.isSameInstance(replaced, entry.model)
                }
            )
            return !replacedByPendingLoad && !replacedByExistingTeardown
        }
        let task = Task {
            for teardown in joinedModelTeardowns.values {
                await teardown.task.value
            }
            for load in pending {
                let outcome = await load.task.result
                if case .success = outcome {
                    await load.model.unload()
                }
            }
            for entry in entriesToUnload {
                await entry.model.unload()
            }
        }
        let teardown = GlobalTeardown(
            token: UUID(),
            joinedModelTeardowns: joinedModelTeardowns,
            task: task
        )
        globalTeardown = teardown
        await teardown.task.value
        finishGlobalTeardown(teardown)
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

    /// Number of callers currently waiting for a model's shared load. Internal so
    /// concurrency tests can synchronize on actor state instead of timing sleeps.
    func pendingWaiterCount(for modelId: String) -> Int {
        pendingLoads[modelId]?.waiterCount ?? 0
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
    /// Models with an outstanding ``beginUse(_:)`` or teardown are not candidates.
    /// If nothing safe remains, this stops rather than freeing a live model: going
    /// over the limit is recoverable, tearing weights out of a running inference is
    /// not. The limit is a budget, not an invariant - but ``endUse(_:)`` calls back
    /// here once a claim clears, so an overage lasts only as long as the calls that
    /// forced it.
    private func evict(downTo targetUsage: Int) async {
        while totalMemoryUsage > targetUsage {
            let candidates = loadedModels.values.filter {
                activeUseCounts[$0.modelId] == nil && modelTeardowns[$0.modelId] == nil
            }
            guard let lruEntry = candidates.min(by: { $0.lastAccessedAt < $1.lastAccessedAt }) else {
                #if DEBUG
                Self.logger.info(
                    "Over budget by \((self.totalMemoryUsage - targetUsage) / 1_000_000, privacy: .public)MB but every resident model is in use or unloading - not evicting")
                #endif
                return
            }

            loadedModels.removeValue(forKey: lruEntry.modelId)
            let task = Task { await lruEntry.model.unload() }
            let teardown = recordModelTeardown(
                modelId: lruEntry.modelId,
                replacedModel: nil,
                task: task
            )
            await teardown.task.value
            finishModelTeardown(teardown, modelId: lruEntry.modelId)
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
