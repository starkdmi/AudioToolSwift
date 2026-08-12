//
//  MLXCachePolicy.swift
//  AudioToolMLX
//
//  Process-global MLX memory policy: what is capped, and what is trimmed.
//

import MLX

/// The two process-global MLX limits this package sets, and the periodic trimming
/// that runs underneath them.
///
/// MLX sizes both of its defaults from the machine rather than the work. The
/// memory limit defaults to 1.5x the device's recommended working set - about
/// 19 GB on a 16 GB Mac - and the *cache* limit defaults to the memory limit. So
/// out of the box a long chunked job's resident size climbs until the OS pushes
/// back, and "how much memory does this model need" answers with a description of
/// the host.
///
/// Both limits are applied together and once, from ``applyProcessLimits()``, which
/// every chunked path calls so the policy does not depend on which provider
/// happened to run first.
enum MLXCachePolicy {

    // MARK: - Cache limit

    /// Ceiling on MLX's allocator cache: buffers the work has freed but which are
    /// retained for reuse.
    ///
    /// 512 MB, measured rather than assumed. This was 3 GB, inherited from
    /// `USSInference` and never checked. Sweeping the cap on an M1 Pro over a 24x
    /// range changed throughput by nothing outside noise:
    ///
    /// | cap     | SR direct 19 s | SS 2spk chunked 30 s |
    /// | ------- | -------------- | -------------------- |
    /// | 3072 MB | 2.56x          | 1.7x                 |
    /// |  512 MB | 2.54x          | 1.7x                 |
    /// |  128 MB | -              | 1.7x                 |
    ///
    /// So a smaller cache is close to free, and 3 GB of retained buffers was
    /// 3 GB the host could not use. What it does *not* do is lower a model's peak:
    /// in the same sweep `GPU.peakMemory` was identical to three decimals at every
    /// cap, because these peaks are live activations inside a single forward pass
    /// and no cache policy can release a live tensor. The cap bounds sustained
    /// residency, which is what it is for.
    ///
    /// Process-global, so its value is mirrored by `USSInference.mlxCacheLimitBytes`
    /// - `USSMLXSwift` sits below `AudioToolMLX` and the two cannot share a
    /// constant. Change both together.
    static let cacheLimitBytes = 512 * 1024 * 1024

    // MARK: - Memory limit

    /// Ceiling on total MLX allocation.
    ///
    /// Nothing set this before, so MLX kept its own default of 1.5x the recommended
    /// working set and nothing applied backpressure to a runaway allocation. The
    /// benchmark harness set a limit for its own runs, which meant the measuring
    /// tool was better behaved than the library it measured.
    ///
    /// 60% of physical memory, capped by what the device recommends for its GPU and
    /// floored at 2 GB. On a 16 GB Mac that is ~9.6 GB, comfortably above the
    /// largest peak in this package (MossFormer2 SS 2-speaker, 5.1 GB on a 30 s
    /// file) while leaving the rest of the machine usable.
    ///
    /// This is backpressure, not a wall. `GPU.set(memoryLimit:)` defaults to
    /// `relaxed: true`, so past the limit MLX waits on scheduled work and then
    /// allocates anyway rather than failing. A job that exceeds it gets slower, not
    /// broken - which is the right failure mode for a library that cannot know what
    /// else the host is doing.
    static let memoryLimitBytes: Int = {
        let info = GPU.deviceInfo()
        var limit = info.memorySize / 5 * 3
        let recommended = Int(info.maxRecommendedWorkingSetSize)
        if recommended > 0 {
            limit = min(limit, recommended)
        }
        // A floor, so an unreadable device or a very small machine cannot produce a
        // limit under which nothing loads.
        return max(limit, 2 * 1024 * 1024 * 1024)
    }()

    // MARK: - Trimming

    /// Cache size above which a chunk boundary may trim.
    ///
    /// A quarter of ``cacheLimitBytes``. Scaled with the cap rather than left at
    /// its old 512 MB: at the new cap that would have meant "trim only when the
    /// cache is completely full", which is a different policy wearing the same name.
    static let defaultThresholdBytes = 128 * 1024 * 1024

    static let defaultChunkInterval = 8

    /// Above this, trim on the next chunk boundary rather than waiting for the
    /// interval. The interval exists to stop a global purge after every chunk; it
    /// is not a bound, and on its own it lets a run of chunks retain whatever the
    /// allocator hands back.
    static let hardCeilingBytes = 4 * defaultThresholdBytes

    private static let limitsApplied: Void = {
        GPU.set(cacheLimit: cacheLimitBytes)
        GPU.set(memoryLimit: memoryLimitBytes)
    }()

    /// Idempotent. Called from every chunked path so the policy does not depend on
    /// which provider happened to run first.
    static func applyProcessLimits() {
        _ = limitsApplied
    }

    static func shouldTrim(
        afterChunk completedChunks: Int,
        cacheMemory: Int,
        thresholdBytes: Int = defaultThresholdBytes,
        chunkInterval: Int = defaultChunkInterval,
        hardCeilingBytes: Int = hardCeilingBytes
    ) -> Bool {
        guard completedChunks > 0, chunkInterval > 0 else { return false }
        if cacheMemory >= hardCeilingBytes { return true }
        return completedChunks.isMultiple(of: chunkInterval)
            && cacheMemory >= thresholdBytes
    }

    /// `GPU.clearCache()` affects every MLX user in the process. Only perform it
    /// periodically and after the allocator cache has grown materially, instead of
    /// forcing a global purge after every model chunk - but never let it grow past
    /// `hardCeilingBytes` waiting for the interval to come round.
    static func trimIfNeeded(afterChunk completedChunks: Int) {
        applyProcessLimits()
        guard shouldTrim(
            afterChunk: completedChunks,
            cacheMemory: GPU.cacheMemory
        ) else { return }
        GPU.clearCache()
    }

    /// The same policy for work that is not a chunk sequence: one direct inference,
    /// one synthesis.
    ///
    /// There is no interval to count against here, so the threshold does all the
    /// deciding. The paths that call this used to purge unconditionally on every
    /// call, which is exactly what the threshold exists to avoid - `GPU.clearCache()`
    /// is process-global, so a provider clearing after each short inference throws
    /// away buffers belonging to whatever else is using MLX, and hands back the
    /// allocator reuse that makes the next call fast.
    static func trimIfCacheGrew(thresholdBytes: Int = defaultThresholdBytes) {
        applyProcessLimits()
        guard GPU.cacheMemory >= thresholdBytes else { return }
        GPU.clearCache()
    }
}
