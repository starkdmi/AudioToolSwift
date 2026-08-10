//
//  MemoryBudget.swift
//  AudioToolBenchmark
//
//  The explicit caps a benchmark run enforces, and why they are not MLX's defaults.
//

import Foundation
import Metal
@preconcurrency import MLX

/// Process-global MLX caps applied before any model loads.
///
/// MLX's own defaults scale with the machine, not the work: the memory limit is
/// 1.5x the device's recommended working set, and the cache limit *defaults to the
/// memory limit*. On a 16 GB Mac that is roughly a 10.6 GB working set, a 16 GB
/// memory limit and a 16 GB cache ceiling - so a long chunked job's resident size
/// climbs until the machine pushes back, and the peak recorded for a model
/// describes the host rather than the model.
///
/// This package caps the cache in production, in two places that cannot share a
/// constant: ``MLXCachePolicy.cacheLimitBytes`` in `AudioToolMLX` and
/// `USSInference.mlxCacheLimitBytes` in `USSMLXSwift`. The benchmark defaults to
/// the same value so its numbers describe shipped behaviour rather than a
/// configuration nobody runs - a third copy of the number, but the alternative is
/// measuring a fourth policy.
///
/// Note that for any *chunked* path the library wins regardless: `MLXCachePolicy`
/// applies its own limits on the first trim, overriding whatever was requested
/// here. `CaseResult.effectiveGpuCacheLimitBytes` records what was actually in
/// force, read back from MLX after the case.
///
/// The memory limit is the one that matters on a 16 GB machine. MLX's `relaxed`
/// mode - the default - responds to the limit by allocating anyway, into swap. A
/// benchmark that swaps has stopped measuring the model, so this sets the limit
/// low enough that a single model cannot get there, and every case runs alone in
/// its own process so nothing else is competing for it.
public struct MemoryBudget: Sendable {

    /// Ceiling on MLX's allocator cache: buffers freed by the work but retained
    /// for reuse.
    public var cacheLimitBytes: Int

    /// Ceiling on total MLX allocation. Past it, MLX waits on scheduled work and
    /// then allocates anyway (see `relaxed:` in `GPU.set(memoryLimit:)`), so this
    /// is backpressure rather than a hard wall.
    public var memoryLimitBytes: Int

    public init(cacheLimitBytes: Int, memoryLimitBytes: Int) {
        self.cacheLimitBytes = cacheLimitBytes
        self.memoryLimitBytes = memoryLimitBytes
    }

    /// What this package caps its cache at in production.
    ///
    /// Mirrors `MLXCachePolicy.cacheLimitBytes` (512 MB) so the harness measures
    /// shipped behaviour rather than a configuration nobody runs. A third copy of
    /// the number, for the same reason the other two exist: this target sits above
    /// `AudioToolMLX`, `USSMLXSwift` sits below it, and none of the three can share
    /// a constant. `MLXProcessLimitsTests` pins the other pair together.
    public static let productionCacheLimitBytes = 512 * 1024 * 1024

    /// A default sized to the host rather than assumed.
    ///
    /// Memory limit: 60% of physical RAM, capped by the Metal device's recommended
    /// working set - the same formula as `MLXCachePolicy.memoryLimitBytes`, so the
    /// harness does not advertise a ceiling different from the one the library will
    /// install over it at load. On a 16 GB machine that is ~9.6 GB, above every
    /// peak in this catalog while leaving the rest of the machine usable.
    public static func forHost() -> MemoryBudget {
        let physical = Sysctl.int("hw.memsize") ?? Int(ProcessInfo.processInfo.physicalMemory)
        var limit = physical / 5 * 3
        if let recommended = MemoryBudget.recommendedWorkingSet {
            limit = min(limit, recommended)
        }
        // A floor, so a small machine or an unreadable sysctl cannot produce a
        // limit under which nothing can load.
        limit = max(limit, 2 * 1024 * 1024 * 1024)
        return MemoryBudget(
            cacheLimitBytes: productionCacheLimitBytes,
            memoryLimitBytes: limit
        )
    }

    private static var recommendedWorkingSet: Int? {
        MTLCreateSystemDefaultDevice().map { Int($0.recommendedMaxWorkingSetSize) }
    }

    /// Apply to the process. Both settings are global and idempotent.
    ///
    /// Called once per child process, before the first model touches MLX -
    /// deliberately not per case, because in `--in-process` mode changing the cap
    /// between cases would mean each case ran under a different one.
    public func apply() {
        GPU.set(cacheLimit: cacheLimitBytes)
        GPU.set(memoryLimit: memoryLimitBytes)
    }

    /// A copy with per-case overrides folded in.
    public func overridden(cache: Int?, memory: Int?) -> MemoryBudget {
        MemoryBudget(
            cacheLimitBytes: cache ?? cacheLimitBytes,
            memoryLimitBytes: memory ?? memoryLimitBytes
        )
    }
}

// MARK: - Headroom

/// Whether this machine can be expected to run a case.
///
/// Checked before starting rather than discovered by being killed. The estimate
/// is the catalog's declared working set, which is a stated expectation and not a
/// measurement - so this refuses only the clearly impossible and says so, rather
/// than second-guessing a machine it cannot see.
public enum Headroom {

    public enum Verdict: Sendable {
        case fits
        case tight(String)
        case insufficient(String)
    }

    public static func check(estimatedBytes: Int) -> Verdict {
        let physical = Sysctl.int("hw.memsize") ?? Int(ProcessInfo.processInfo.physicalMemory)
        let estimateGB = Double(estimatedBytes) / 1_073_741_824
        let physicalGB = Double(physical) / 1_073_741_824

        if estimatedBytes > physical {
            return .insufficient(String(
                format: "case expects ~%.1f GB, machine has %.1f GB",
                estimateGB, physicalGB
            ))
        }
        if Double(estimatedBytes) > Double(physical) * 0.5 {
            return .tight(String(
                format: "case expects ~%.1f GB of %.1f GB - close other applications",
                estimateGB, physicalGB
            ))
        }
        return .fits
    }
}
