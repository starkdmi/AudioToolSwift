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

    static func shouldTrim(
        afterChunk completedChunks: Int,
        cacheMemory: Int,
        thresholdBytes: Int = defaultThresholdBytes,
        chunkInterval: Int = defaultChunkInterval
    ) -> Bool {
        guard completedChunks > 0, chunkInterval > 0 else { return false }
        return completedChunks.isMultiple(of: chunkInterval)
            && cacheMemory >= thresholdBytes
    }

    /// `GPU.clearCache()` affects every MLX user in the process. Only perform it
    /// periodically and after the allocator cache has grown materially, instead of
    /// forcing a global purge after every model chunk.
    static func trimIfNeeded(afterChunk completedChunks: Int) {
        guard shouldTrim(
            afterChunk: completedChunks,
            cacheMemory: GPU.cacheMemory
        ) else { return }
        GPU.clearCache()
    }
}
