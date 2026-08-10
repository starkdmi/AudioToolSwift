//
//  ResourceProbe.swift
//  AudioToolBenchmark
//
//  Reading what the process is costing, without a profiler attached.
//

import Darwin
import Foundation

// MARK: - Task memory

/// One reading of the process's memory, in bytes.
public struct MemorySample: Sendable {
    /// `phys_footprint`: what Activity Monitor shows as Memory and what the OS
    /// charges against a memory limit. Includes compressed and IOKit-mapped pages,
    /// which is why it is the number that matters on a 16 GB machine holding
    /// unified-memory model weights.
    public var footprintBytes: Int
    /// `resident_size`: physical pages currently mapped. Lower than the footprint
    /// once anything is compressed, so it is recorded but not headlined.
    public var residentBytes: Int
}

public enum TaskMemory {

    /// Current footprint and resident size, or nil if the kernel refused.
    public static func current() -> MemorySample? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        // `phys_footprint` sits partway into the struct and older kernels fill
        // fewer fields, so a short reply would hand back whatever was already in
        // that memory. The SDK's `TASK_VM_INFO_REV1_COUNT` macro is not importable
        // - it expands through a structure Swift declines to bridge - so the same
        // bound is computed from the layout instead.
        guard count >= minimumCountForFootprint else { return nil }
        return MemorySample(
            footprintBytes: Int(info.phys_footprint),
            residentBytes: Int(info.resident_size)
        )
    }

    /// How many `natural_t` words the kernel must have written for
    /// `phys_footprint` to be real.
    private static let minimumCountForFootprint: mach_msg_type_number_t = {
        guard let offset = MemoryLayout<task_vm_info_data_t>.offset(of: \.phys_footprint) else {
            // No layout answer available; accept whatever the kernel returned
            // rather than refusing to measure anything.
            return 0
        }
        let bytes = offset + MemoryLayout<mach_vm_size_t>.size
        return mach_msg_type_number_t(bytes / MemoryLayout<natural_t>.size)
    }()
}

// MARK: - Peak sampling

/// Polls the process footprint on a background queue and keeps the maximum.
///
/// There is no kernel counter for peak footprint - `rusage.ru_maxrss` is resident
/// size, not footprint, and is never reset - so a peak has to be sampled. 25 ms is
/// short against a model load or a multi-second inference and costs a `task_info`
/// call, which is a syscall and nothing more.
///
/// The consequence of sampling is that a spike shorter than the interval can be
/// missed. That is acceptable here and stated rather than hidden: these are
/// hundreds-of-megabytes allocations held for whole chunks, not transient peaks.
public final class FootprintSampler: @unchecked Sendable {

    private let lock = NSLock()
    private var peakFootprint = 0
    private var peakResident = 0
    private var timer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "audiotool.benchmark.sampler", qos: .userInitiated)

    public init() {}

    /// Begin sampling, seeding the peak with the current reading so a case that
    /// finishes between two ticks still reports something real.
    public func start(interval: TimeInterval = 0.025) {
        record()
        let source = DispatchSource.makeTimerSource(queue: queue)
        source.schedule(deadline: .now() + interval, repeating: interval, leeway: .milliseconds(5))
        source.setEventHandler { [weak self] in self?.record() }
        lock.lock()
        timer = source
        lock.unlock()
        source.resume()
    }

    /// Stop sampling and return the peak, including one final reading.
    @discardableResult
    public func stop() -> MemorySample {
        lock.lock()
        let source = timer
        timer = nil
        lock.unlock()
        source?.cancel()
        record()
        lock.lock()
        defer { lock.unlock() }
        return MemorySample(footprintBytes: peakFootprint, residentBytes: peakResident)
    }

    /// The peak so far, without stopping.
    public var peak: MemorySample {
        lock.lock()
        defer { lock.unlock() }
        return MemorySample(footprintBytes: peakFootprint, residentBytes: peakResident)
    }

    private func record() {
        guard let sample = TaskMemory.current() else { return }
        lock.lock()
        peakFootprint = max(peakFootprint, sample.footprintBytes)
        peakResident = max(peakResident, sample.residentBytes)
        lock.unlock()
    }
}

// MARK: - CPU time

/// Cumulative CPU time for this process, in seconds.
public struct CPUTime: Sendable {
    public var user: Double
    public var system: Double

    public static func current() -> CPUTime {
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else {
            return CPUTime(user: .nan, system: .nan)
        }
        return CPUTime(
            user: seconds(usage.ru_utime),
            system: seconds(usage.ru_stime)
        )
    }

    /// Difference from an earlier reading.
    public func since(_ earlier: CPUTime) -> CPUTime {
        CPUTime(user: user - earlier.user, system: system - earlier.system)
    }

    private static func seconds(_ value: timeval) -> Double {
        Double(value.tv_sec) + Double(value.tv_usec) / 1_000_000
    }
}

// MARK: - Formatting

public enum ByteFormat {

    /// Binary units, because that is what every memory tool on this platform
    /// reports and mixing conventions inside one report is worse than either.
    public static func bytes(_ value: Int) -> String {
        let units = ["B", "KiB", "MiB", "GiB"]
        var amount = Double(value)
        var index = 0
        while amount >= 1024, index < units.count - 1 {
            amount /= 1024
            index += 1
        }
        return index == 0
            ? "\(value) B"
            : String(format: "%.1f %@", amount, units[index])
    }

    /// Megabytes as a bare number, for table columns.
    public static func megabytes(_ value: Int) -> String {
        String(format: "%.0f", Double(value) / 1_048_576)
    }
}
