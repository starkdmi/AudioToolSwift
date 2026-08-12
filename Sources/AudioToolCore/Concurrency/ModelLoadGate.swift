//
//  ModelLoadGate.swift
//  AudioToolCore
//
//  One in-flight load per provider
//

import Foundation

/// Coalesces concurrent `load()` calls on one provider into a single load.
///
/// Actor isolation does not give this for free. A provider actor's `load()`
/// characteristically reads its state, finds it empty, and then *awaits* - a
/// download, a `MLModel.load`, a `loadContainer` - before assigning what it built.
/// The actor is reentrant across that await, so two callers can both pass the
/// emptiness check and both allocate. For the models here that is a second copy of
/// something between several hundred megabytes and four gigabytes, and the loser's
/// copy is not freed so much as overwritten.
///
/// ```swift
/// public func load() async throws {
///     try await loadGate.run { [self] in try await performLoad() }
/// }
///
/// public func unload() async {
///     let teardown = await loadGate.beginTeardown()
///     defer { loadGate.endTeardown(teardown) }
///     model = nil
/// }
/// ```
///
/// A successful load is remembered, so a second `load()` returns immediately rather
/// than repeating the work; tearing down forgets it, which is what makes
/// unload-then-load reload rather than no-op. A failed load is forgotten too, so the
/// next caller retries - a download that failed while the network was down should not
/// poison the provider for the process's lifetime.
///
/// ## Cancellation
///
/// Each caller waits on its own continuation rather than on the shared task, so a
/// cancelled caller returns immediately and the callers still waiting keep their
/// load. When the *last* waiter leaves, the load has no one left to deliver to and is
/// cancelled.
///
/// This matters more than it looks. The load task is unstructured, and an earlier
/// version had every caller `await task.value`: awaiting an unstructured task ignores
/// the awaiting task's cancellation, so cancelling the final residency waiter left a
/// multi-gigabyte download running with nothing to receive it, a subsequent
/// `unloadAll()` had to wait for that abandoned work before it could tear anything
/// down, and the cancelled caller was returned a success it had stopped waiting for.
///
/// ## Teardown
///
/// Cancellation is cooperative, so a cancelled load is *draining*, not gone, and the
/// gate stays shut until it has actually stopped - otherwise unload-then-load could
/// have two `MLModel.compileModel` calls or two LLM container loads in flight at once.
///
/// The gate then stays shut for as long as the provider needs to clear its own state.
/// ``beginTeardown()`` and ``endTeardown(_:)`` bracket that, and the bracket is not
/// ceremony: a provider's `unload()` awaits the drain and only then assigns `nil` to
/// its model, and between those two steps the actor is free. A load admitted in that
/// window would publish a model that the `nil` immediately wipes - leaving the gate
/// saying "loaded", the provider holding nothing, every later `load()` a no-op and
/// every `process()` a `modelNotLoaded`.
/// A one-shot gate a task waits on before starting work.
///
/// `Task { }` begins running immediately, which is a problem when the handle it
/// returns is what everything else uses to cancel and drain it: between creating the
/// task and recording the handle, a teardown sees nothing to cancel while the body is
/// already downloading. The body waits here instead, and is released - or told it was
/// cancelled - only once the handle is recorded.
private final class StartLatch: @unchecked Sendable {
    private let lock = NSLock()
    private var opened = false
    private var abandoned = false
    private var waiter: CheckedContinuation<Bool, Never>?

    /// Returns whether the body should proceed.
    func wait() async -> Bool {
        await withCheckedContinuation { continuation in
            let decided = lock.withLock { () -> Bool? in
                if opened { return true }
                if abandoned { return false }
                waiter = continuation
                return nil
            }
            if let decided { continuation.resume(returning: decided) }
        }
    }

    func open() { settle(proceed: true) }
    func abandon() { settle(proceed: false) }

    private func settle(proceed: Bool) {
        let waiter = lock.withLock { () -> CheckedContinuation<Bool, Never>? in
            if proceed { opened = true } else { abandoned = true }
            let waiter = self.waiter
            self.waiter = nil
            return waiter
        }
        waiter?.resume(returning: proceed)
    }
}

public final class ModelLoadGate: @unchecked Sendable {

    /// A teardown in progress. Hand it back to ``endTeardown(_:)`` to reopen the gate.
    public struct Teardown: Sendable, Hashable {
        fileprivate let generation: UInt64
    }

    /// What the gate is doing, as one value rather than several booleans that can
    /// contradict each other. The `loaded`/`loading` distinction is what stops a
    /// cancellation arriving just after success from undoing it.
    private enum Phase {
        /// Nothing loaded, nothing running.
        case idle
        /// A load is running and callers may be waiting on it.
        ///
        /// The task is `nil` for the instant between reserving the slot and
        /// publishing the task that fills it - see ``admit(_:)``, which must not
        /// create that task while holding the lock.
        case loading(Task<Void, Error>?)
        /// A load finished successfully. Further `run` calls return at once.
        case loaded
        /// A cancelled load has not yet stopped. Nothing may start until it has.
        case draining(Task<Void, Error>)
        /// The provider is resetting its own state. Nothing may start until it has.
        case tearingDown
    }

    /// What `run` should do next, decided under the lock and acted on outside it.
    private enum Admission {
        case alreadyLoaded
        case wait(UInt64)
        case waitForGate
    }

    private let lock = NSLock()

    private var phase: Phase = .idle

    /// Bumped whenever a load is abandoned, so a completion or a cancellation from a
    /// superseded load cannot disturb its replacement.
    private var generation: UInt64 = 0

    /// How many callers still want the current load.
    private var waiterCount = 0

    /// Waiters parked on the current load.
    private var waiters: [UUID: CheckedContinuation<Void, Error>] = [:]

    /// Callers cancelled between joining and installing their continuation.
    private var cancelledWaiters: Set<UUID> = []

    /// Callers parked because the gate is shut - draining or tearing down - who will
    /// try again once it opens.
    private var gateWaiters: [UUID: CheckedContinuation<Void, Error>] = [:]

    /// Gate waiters cancelled between entering the wait and installing their
    /// continuation.
    ///
    /// `onCancel` can run before the continuation body does, and then it has nothing
    /// to remove and nobody to resume - so the body would park a continuation that no
    /// drain and no teardown will ever come back for. A leaked continuation is not an
    /// inert leak: destroying one whose task has gone takes the process with it.
    private var cancelledGateWaiters: Set<UUID> = []

    /// Results of recently completed loads, by generation.
    ///
    /// A load can fail before the caller that started it has installed its
    /// continuation - an integrity mismatch on an already-downloaded file takes
    /// microseconds. Without this, that caller arrived to find the phase back at
    /// `idle` and was told `CancellationError`, discarding the real error it needed.
    ///
    /// Keyed by generation and kept for several generations rather than just the
    /// last, because the next caller's load starts a new generation: a single slot
    /// was cleared by that caller before the first one had read its own failure.
    private var terminals: [UInt64: Result<Void, Error>] = [:]

    /// Callers admitted to a generation who have not yet looked at its outcome.
    ///
    /// Every `.wait` admission installs a continuation exactly once, so this reaches
    /// zero for every generation. Retaining a fixed number of recent results instead
    /// was a guess - eight - about how long a caller might take to install, and a
    /// caller descheduled across nine fast retries would have found its own failure
    /// evicted and been handed `CancellationError`.
    private var unobserved: [UInt64: Int] = [:]

    /// Test seam: called once a completed load has published its phase and taken its
    /// waiters, but before those waiters resume.
    ///
    /// That gap is where a late cancellation used to undo a successful load, and it
    /// is microseconds wide - a stress test racing against it passed just as happily
    /// with the bug present as without, which is worse than no test. This makes the
    /// interleaving something a test can ask for. Never set outside tests.
    internal var didPublishCompletion: (@Sendable () -> Void)?

    public init() {}


    /// Run `body` once on behalf of every concurrent caller, and deliver its result
    /// to all of them.
    public func run(_ body: @escaping @Sendable () async throws -> Void) async throws {
        while true {
            try Task.checkCancellation()
            switch admit(body) {
            case .alreadyLoaded:
                return
            case .wait(let token):
                return try await wait(for: token)
            case .waitForGate:
                // Shut for a drain or a teardown. Parked on a continuation rather
                // than on the blocking task, so cancelling this caller releases it
                // now instead of after someone else's download finishes unwinding.
                try await waitForGate()
            }
        }
    }

    /// Cancel any in-flight load, wait for it to stop, and hold the gate shut until
    /// the matching ``endTeardown(_:)``.
    ///
    /// Waiters on an in-flight load are failed: the provider they were waiting for is
    /// being torn down, and handing them a load that will not be published is worse
    /// than telling them so.
    ///
    /// Call this before clearing provider state, and `endTeardown` after:
    ///
    /// ```swift
    /// let teardown = await loadGate.beginTeardown()
    /// defer { loadGate.endTeardown(teardown) }
    /// model = nil
    /// ```
    public func beginTeardown() async -> Teardown {
        // Claim the phase *first*, in the same critical section that takes the
        // in-flight task. Draining first and claiming afterwards left a window: the
        // drain watcher could reach `.idle`, wake a parked `run()`, and let it start
        // a replacement load - which this method would then overwrite the phase of,
        // orphaning a load that nothing cancels, nothing drains, and that is free to
        // publish into provider state the caller is about to clear.
        let (inFlight, pending, teardown) = lock.withLock {
            () -> (Task<Void, Error>?, [CheckedContinuation<Void, Error>], Teardown) in
            let pending = Array(waiters.values)
            waiters.removeAll()
            cancelledWaiters.removeAll()
            waiterCount = 0
            generation &+= 1
            terminals.removeAll()
            unobserved.removeAll()

            let inFlight: Task<Void, Error>?
            switch phase {
            case .loading(let task):
                inFlight = task
            case .draining(let task):
                inFlight = task
            case .loaded, .idle, .tearingDown:
                inFlight = nil
            }
            phase = .tearingDown
            return (inFlight, pending, Teardown(generation: generation))
        }

        inFlight?.cancel()
        for continuation in pending {
            continuation.resume(throwing: CancellationError())
        }
        // Wait for it here rather than through a watcher: the phase is already
        // `.tearingDown`, so nothing can be admitted while this unwinds.
        if let inFlight {
            _ = try? await inFlight.value
        }

        return teardown
    }

    /// Reopen the gate after the provider has finished resetting.
    public func endTeardown(_ teardown: Teardown) {
        let waiting = lock.withLock { () -> [CheckedContinuation<Void, Error>] in
            // A newer teardown has taken over; it owns the reopening.
            guard generation == teardown.generation, case .tearingDown = phase else {
                return []
            }
            phase = .idle
            let waiting = Array(gateWaiters.values)
            gateWaiters.removeAll()
            cancelledGateWaiters.removeAll()
            return waiting
        }
        for continuation in waiting { continuation.resume() }
    }

    /// Cancel any in-flight load, wait for it to stop, and reopen the gate.
    ///
    /// For callers with no state of their own to reset. A provider clearing a model
    /// wants ``beginTeardown()`` and ``endTeardown(_:)`` instead.
    public func cancelAndDrain() async {
        endTeardown(await beginTeardown())
    }

    /// Cancel without waiting, returning the task still unwinding, if any.
    ///
    /// For callers with nowhere to await - `deinit`, or a non-async teardown path.
    @discardableResult
    public func cancelWithoutDraining() -> Task<Void, Error>? {
        let (inFlight, pending) = lock.withLock {
            () -> (Task<Void, Error>?, [CheckedContinuation<Void, Error>]) in
            let pending = Array(waiters.values)
            waiters.removeAll()
            cancelledWaiters.removeAll()
            waiterCount = 0
            generation &+= 1
            terminals.removeAll()
            unobserved.removeAll()

            switch phase {
            case .loading(let task):
                // No task yet means it has not been published; `admit` cancels it
                // when it finds the generation moved on.
                guard let task else {
                    phase = .idle
                    return (nil, pending)
                }
                phase = .draining(task)
                return (task, pending)
            case .draining(let task):
                phase = .draining(task)
                return (task, pending)
            case .loaded, .idle, .tearingDown:
                phase = .idle
                return (nil, pending)
            }
        }

        inFlight?.cancel()
        for continuation in pending {
            continuation.resume(throwing: CancellationError())
        }
        if let inFlight { watchDrain(of: inFlight) }
        return inFlight
    }

    /// Whether a load has completed successfully and not since been torn down.
    public var hasLoaded: Bool {
        lock.withLock {
            if case .loaded = phase { return true }
            return false
        }
    }

    // MARK: - Admission

    private func admit(_ body: @escaping @Sendable () async throws -> Void) -> Admission {
        // Reserve first, and only then build the task.
        //
        // Creating it inside `withLock` is a use-after-free waiting to happen: the
        // closure passed to `withLock` is non-escaping, so its context can live on
        // the stack, and the `Task` that captures it outlives the call. It ran
        // against freed stack memory as soon as the scheduler started the body
        // before the lock was released - intermittently, which is the worst way for
        // this class of bug to present.
        let reserved: UInt64? = lock.withLock {
            switch phase {
            case .loaded:
                return nil

            case .draining, .tearingDown:
                return nil

            case .loading:
                return nil

            case .idle:
                generation &+= 1
                waiterCount = 1
                unobserved[generation, default: 0] += 1
                // Reserved, not yet filled: concurrent callers see a load in flight
                // and join it rather than starting a second one.
                phase = .loading(nil)
                return generation
            }
        }

        guard let loadGeneration = reserved else {
            // Not ours to start; decide what this caller does instead.
            return lock.withLock {
                switch phase {
                case .loaded: return .alreadyLoaded
                case .draining, .tearingDown: return .waitForGate
                case .loading:
                    waiterCount += 1
                    unobserved[generation, default: 0] += 1
                    return .wait(generation)
                case .idle:
                    // Raced with a load that has just finished or been abandoned.
                    return .waitForGate
                }
            }
        }

        // The body does nothing until the handle below is recorded. Otherwise a
        // teardown landing in between finds `.loading(nil)` - a load in flight with
        // no handle to cancel - and returns believing nothing is running, while the
        // body allocates or downloads behind it. `task.cancel()` afterwards is only
        // cooperative, and these bodies do real work before their first cancellation
        // check.
        let latch = StartLatch()
        let task = Task { [weak self] in
            guard await latch.wait() else { throw CancellationError() }
            // Opened, but possibly cancelled since: a teardown that arrived after
            // publication cancels the task, and the latch has no opinion about that.
            // Without this the body would start anyway, and `beginTeardown` would sit
            // waiting for the multi-gigabyte load it had just cancelled.
            try Task.checkCancellation()
            do {
                try await body()
                self?.complete(.success(()), generation: loadGeneration)
            } catch {
                self?.complete(.failure(error), generation: loadGeneration)
                throw error
            }
        }

        let published = lock.withLock { () -> Bool in
            guard generation == loadGeneration, case .loading(nil) = phase else { return false }
            phase = .loading(task)
            return true
        }

        if published {
            latch.open()
        } else {
            // Abandoned between reserving and publishing - an `unload()` landing in
            // that window. Released only to be told to stop, having done nothing.
            task.cancel()
            latch.abandon()
        }

        return .wait(loadGeneration)
    }

    /// One caller of `generation` has read its outcome. The result is kept only
    /// while someone may still need it.
    ///
    /// Must be called with the lock held.
    private func observed(_ generation: UInt64) {
        guard let remaining = unobserved[generation] else { return }
        if remaining <= 1 {
            unobserved.removeValue(forKey: generation)
            terminals.removeValue(forKey: generation)
        } else {
            unobserved[generation] = remaining - 1
        }
    }

    /// Reopen the gate once a cancelled load has actually stopped.
    ///
    /// One watcher per abandoned task, so callers wait on a continuation instead of
    /// on the task itself and can be cancelled out of that wait.
    private func watchDrain(of drained: Task<Void, Error>) {
        Task { [weak self] in
            _ = try? await drained.value
            self?.drainFinished(drained)
        }
    }

    private func drainFinished(_ drained: Task<Void, Error>) {
        let waiting = lock.withLock { () -> [CheckedContinuation<Void, Error>] in
            guard case .draining(let current) = phase, current == drained else { return [] }
            phase = .idle
            let waiting = Array(gateWaiters.values)
            gateWaiters.removeAll()
            cancelledGateWaiters.removeAll()
            return waiting
        }
        for continuation in waiting { continuation.resume() }
    }

    // MARK: - Waiting

    /// Park until the gate is admissible again, or until this caller is cancelled.
    private func waitForGate() async throws {
        enum Outcome { case parked, admissible, cancelled }

        let waiterId = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let outcome = lock.withLock { () -> Outcome in
                    // Cancelled before getting here; `onCancel` had nothing to resume.
                    if cancelledGateWaiters.remove(waiterId) != nil { return .cancelled }
                    switch phase {
                    case .draining, .tearingDown:
                        gateWaiters[waiterId] = continuation
                        return .parked
                    case .idle, .loading, .loaded:
                        return .admissible
                    }
                }
                switch outcome {
                case .parked: break
                case .admissible: continuation.resume()
                case .cancelled: continuation.resume(throwing: CancellationError())
                }
            }
        } onCancel: {
            let continuation = self.lock.withLock { () -> CheckedContinuation<Void, Error>? in
                if let installed = self.gateWaiters.removeValue(forKey: waiterId) {
                    return installed
                }
                self.cancelledGateWaiters.insert(waiterId)
                return nil
            }
            continuation?.resume(throwing: CancellationError())
        }
    }

    private func wait(for token: UInt64) async throws {
        let waiterId = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                install(continuation, id: waiterId, token: token)
            }
        } onCancel: {
            self.leave(waiterId, token: token)
        }
    }

    private func install(
        _ continuation: CheckedContinuation<Void, Error>,
        id waiterId: UUID,
        token: UInt64
    ) {
        enum Outcome {
            case parked
            case superseded
            case cancelled
            case settled(Result<Void, Error>)
        }

        let outcome = lock.withLock { () -> Outcome in
            // Read the outcome before marking it observed - `observed` drops the
            // result once the last caller of that generation has taken it, and
            // dropping it first would hand this caller `CancellationError` in place
            // of the very error it came for.
            // Read the outcome before marking it observed - `observed` drops the
            // result once the last caller of that generation has taken it, and
            // dropping it first would hand this caller `CancellationError` in place
            // of the very error it came for.
            //
            // This path is defensive: with the start latch, a load cannot finish
            // before its caller parks unless that caller is descheduled in between.
            // It is the difference between a rare wrong error and a rare right one.
            let settled = terminals[token]
            observed(token)
            if let settled { return .settled(settled) }
            if case .loaded = phase, generation == token { return .settled(.success(())) }
            // The load this caller joined has been abandoned or replaced.
            guard generation == token, case .loading = phase else { return .superseded }
            // Cancelled after joining but before getting here.
            if cancelledWaiters.remove(waiterId) != nil { return .cancelled }
            waiters[waiterId] = continuation
            return .parked
        }

        switch outcome {
        case .parked: break
        case .settled(let result): continuation.resume(with: result)
        case .superseded, .cancelled: continuation.resume(throwing: CancellationError())
        }
    }

    /// One caller stops waiting. The load goes with it if it was the last.
    private func leave(_ waiterId: UUID, token: UInt64) {
        let (abandoned, continuation) = lock.withLock {
            () -> (Task<Void, Error>?, CheckedContinuation<Void, Error>?) in
            // Only a load still running can be abandoned. A cancellation that lands
            // after `complete` has published success must not undo it: completion
            // has already taken the waiters and set `waiterCount` to zero, so the
            // arithmetic below would otherwise read "last waiter left" and throw away
            // a provider that is loaded and in use.
            guard generation == token, case .loading(let inFlight) = phase else {
                return (nil, nil)
            }

            let continuation = waiters.removeValue(forKey: waiterId)
            if continuation == nil {
                // Ran before `install`; it will see this and resume itself.
                cancelledWaiters.insert(waiterId)
            }
            waiterCount = max(0, waiterCount - 1)

            guard waiterCount == 0 else { return (nil, continuation) }

            // Nobody is left to receive this load.
            waiters.removeAll()
            cancelledWaiters.removeAll()
            generation &+= 1
            guard let inFlight else {
                // Not published yet; `admit` cancels it on the moved generation.
                phase = .idle
                return (nil, continuation)
            }
            phase = .draining(inFlight)
            return (inFlight, continuation)
        }

        abandoned?.cancel()
        if let abandoned { watchDrain(of: abandoned) }
        continuation?.resume(throwing: CancellationError())
    }

    private func complete(_ result: Result<Void, Error>, generation loadGeneration: UInt64) {
        let pending = lock.withLock { () -> [CheckedContinuation<Void, Error>] in
            guard generation == loadGeneration, case .loading = phase else { return [] }

            let pending = Array(waiters.values)
            waiters.removeAll()
            cancelledWaiters.removeAll()
            waiterCount = 0
            terminals[loadGeneration] = result

            switch result {
            case .success:
                phase = .loaded
            case .failure:
                // Forgotten, so the next caller retries rather than inheriting an
                // error from an attempt it never made.
                phase = .idle
            }
            return pending
        }

        didPublishCompletion?()

        for continuation in pending {
            continuation.resume(with: result)
        }
    }
}
