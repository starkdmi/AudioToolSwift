//
//  MLXCachePolicy.swift
//  AudioToolMLX
//
//  Coarse, memory-pressure-driven cache trimming for chunked inference
//

import MLX

enum MLXCachePolicy {
    static let defaultThresholdBytes = 512 * 1024 * 1024
    static let defaultChunkInterval = 8

    /// Above this, trim on the next chunk boundary rather than waiting for the
    /// interval. The interval exists to stop a global purge after every chunk; it
    /// is not a bound, and on its own it lets a run of chunks retain whatever the
    /// allocator hands back.
    static let hardCeilingBytes = 4 * defaultThresholdBytes

    /// MLX's default cache ceiling is the device's recommended working set - about
    /// 10.6 GB on a 16 GB Mac. Chunked inference frees a segment's buffers every
    /// chunk, and with no explicit cap MLX retains them all the way to that ceiling,
    /// so resident size tracks the machine rather than the work.
    ///
    /// 3 GB is what this package ran under until the call was dropped from
    /// `USSInference.prewarm`. The setting is process-global, so it is applied once
    /// and its value is mirrored by `USSInference.mlxCacheLimitBytes` - the two
    /// targets cannot share a constant, since `USSMLXSwift` sits below `AudioToolMLX`.
    static let cacheLimitBytes = 3 * 1024 * 1024 * 1024

    private static let cacheLimitApplied: Void = {
        GPU.set(cacheLimit: cacheLimitBytes)
    }()

    /// Idempotent. Called from every chunked path so the cap does not depend on
    /// which provider happened to run first.
    static func applyProcessCacheLimit() {
        _ = cacheLimitApplied
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
        applyProcessCacheLimit()
        guard shouldTrim(
            afterChunk: completedChunks,
            cacheMemory: GPU.cacheMemory
        ) else { return }
        GPU.clearCache()
    }
}
