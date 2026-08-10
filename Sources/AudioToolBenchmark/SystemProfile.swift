//
//  SystemProfile.swift
//  AudioToolBenchmark
//
//  What machine produced the numbers.
//

import Foundation
import Metal

#if canImport(IOKit)
import IOKit.ps
#endif

/// The machine, captured once per run.
///
/// A benchmark that travels between machines is only worth anything if the
/// machine travels with it. Everything here is read from the host rather than
/// configured, so a report cannot claim to come from hardware it did not.
///
/// Deliberately not captured: user name, serial number, the machine's network
/// name. ``hostName`` is included but is the only identifying field, and
/// ``redactingHost()`` removes it for reports that get published.
public struct SystemProfile: Codable, Sendable {

    // MARK: Hardware

    /// `hw.model`, e.g. `MacBookPro18,3`. The one field that identifies the
    /// machine class unambiguously; marketing names are not available to a process.
    public var hardwareModel: String

    /// `machdep.cpu.brand_string`, e.g. `Apple M1 Pro`.
    public var chip: String

    public var logicalCores: Int
    /// Performance cores, from `hw.perflevel0.logicalcpu`. Nil on Intel, which has
    /// no perflevel split.
    public var performanceCores: Int?
    /// Efficiency cores, from `hw.perflevel1.logicalcpu`.
    public var efficiencyCores: Int?

    public var physicalMemoryBytes: Int

    /// Metal device name, e.g. `Apple M1 Pro`.
    public var gpuName: String?
    /// `recommendedMaxWorkingSetSize`. MLX's default memory limit is 1.5x this,
    /// and its default cache limit follows the memory limit - which is exactly
    /// why this package caps both explicitly. See ``MemoryBudget``.
    public var gpuRecommendedWorkingSetBytes: Int?
    public var gpuHasUnifiedMemory: Bool?

    // MARK: Software

    public var osVersion: String
    public var kernelVersion: String
    /// The Swift language mode this binary was compiled under, as a coarse floor.
    public var swiftCompiler: String
    public var architecture: String

    /// Revisions of the dependencies that decide the numbers, read from
    /// `Package.resolved` when the binary is still next to its source checkout.
    /// Nil for a binary copied elsewhere - honest absence rather than a guess.
    public var dependencyRevisions: [String: String]?

    // MARK: Identity

    public var hostName: String?

    // MARK: - Capture

    public static func capture() -> SystemProfile {
        let device = MTLCreateSystemDefaultDevice()
        let info = ProcessInfo.processInfo

        return SystemProfile(
            hardwareModel: Sysctl.string("hw.model") ?? "unknown",
            chip: Sysctl.string("machdep.cpu.brand_string") ?? "unknown",
            logicalCores: Sysctl.int("hw.logicalcpu_max") ?? info.processorCount,
            performanceCores: Sysctl.int("hw.perflevel0.logicalcpu"),
            efficiencyCores: Sysctl.int("hw.perflevel1.logicalcpu"),
            physicalMemoryBytes: Sysctl.int("hw.memsize") ?? Int(info.physicalMemory),
            gpuName: device?.name,
            gpuRecommendedWorkingSetBytes: device.map { Int($0.recommendedMaxWorkingSetSize) },
            gpuHasUnifiedMemory: device?.hasUnifiedMemory,
            osVersion: info.operatingSystemVersionString,
            kernelVersion: Sysctl.string("kern.osrelease") ?? "unknown",
            swiftCompiler: Self.swiftCompilerDescription,
            architecture: Self.architecture,
            dependencyRevisions: PackageResolved.revisions(),
            hostName: info.hostName
        )
    }

    /// A copy with the only identifying field removed, for a report that will be
    /// shared. Everything else describes a hardware configuration, not a person.
    public func redactingHost() -> SystemProfile {
        var copy = self
        copy.hostName = nil
        return copy
    }

    /// A short line for a report header, e.g. `Apple M1 Pro, 10 cores, 16 GB`.
    public var summaryLine: String {
        let memoryGB = Double(physicalMemoryBytes) / 1_073_741_824
        var parts = [chip, "\(logicalCores) cores"]
        if let performance = performanceCores, let efficiency = efficiencyCores {
            parts[1] = "\(logicalCores) cores (\(performance)P/\(efficiency)E)"
        }
        parts.append(String(format: "%.0f GB", memoryGB))
        parts.append(hardwareModel)
        return parts.joined(separator: ", ")
    }

    // MARK: - Static facts about this binary

    /// A floor, not an exact toolchain version: a binary cannot read the compiler
    /// that produced it, only answer which language versions it was allowed to use.
    private static var swiftCompilerDescription: String {
        #if compiler(>=6.3)
        return "Swift >= 6.3"
        #elseif compiler(>=6.2)
        return "Swift 6.2"
        #elseif compiler(>=6.1)
        return "Swift 6.1"
        #else
        return "Swift 6.0"
        #endif
    }

    private static var architecture: String {
        #if arch(arm64)
        return "arm64"
        #elseif arch(x86_64)
        return "x86_64"
        #else
        return "unknown"
        #endif
    }

    /// `"release"` or `"debug"`, so a report cannot quietly be a debug run.
    public static var buildConfiguration: String {
        #if DEBUG
        return "debug"
        #else
        return "release"
        #endif
    }
}

// MARK: - Live conditions

/// The conditions that move while a benchmark runs.
///
/// Separate from ``SystemProfile`` because these are sampled at the start and
/// end of every case, not once per run: a laptop that starts on mains and drops
/// to battery, or crosses into `serious` thermal pressure, has changed the thing
/// being measured partway through.
public enum HostConditions {

    public static var thermalState: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    public static var lowPowerModeEnabled: Bool {
        ProcessInfo.processInfo.isLowPowerModeEnabled
    }

    /// `"ac"`, `"battery"` or `"unknown"`.
    ///
    /// Worth recording rather than assuming: an Apple laptop on battery runs the
    /// GPU at a lower sustained power budget, and a set of numbers that silently
    /// mixes the two states explains a 20% spread that looks like a regression.
    public static var powerSource: String {
        #if canImport(IOKit) && os(macOS)
        guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(snapshot)?.takeUnretainedValue()
        else { return "unknown" }
        switch type as String {
        case kIOPMACPowerKey: return "ac"
        case kIOPMBatteryPowerKey: return "battery"
        case kIOPMUPSPowerKey: return "ups"
        default: return "unknown"
        }
        #else
        return "unknown"
        #endif
    }

    public static func snapshot(start: String) -> EnvironmentSnapshot {
        EnvironmentSnapshot(
            thermalStateAtStart: start,
            thermalStateAtEnd: thermalState,
            lowPowerModeEnabled: lowPowerModeEnabled,
            powerSource: powerSource
        )
    }
}

// MARK: - sysctl

enum Sysctl {

    static func string(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        // sysctl strings are NUL-terminated and the reported size includes the
        // terminator; decoding the whole buffer would append a stray U+0000 that
        // then travels all the way into the JSON.
        let bytes = buffer.prefix { $0 != 0 }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Reads at whatever width the kernel reports.
    ///
    /// `hw.memsize` is 64-bit and `hw.logicalcpu` is 32-bit; a single fixed-width
    /// read gets one of them wrong - and gets it wrong quietly, by returning a
    /// plausible number with three extra bytes of adjacent stack in it.
    static func int(_ name: String) -> Int? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0 else { return nil }
        switch size {
        case MemoryLayout<Int32>.size:
            var value: Int32 = 0
            guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
            return Int(value)
        case MemoryLayout<Int64>.size:
            var value: Int64 = 0
            guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
            return Int(value)
        default:
            return nil
        }
    }
}

// MARK: - Package.resolved

/// Dependency revisions, when the binary is still beside the checkout it came from.
///
/// mlx-swift's version decides most of what this benchmark measures, so a report
/// that does not name it can be compared with the wrong thing. Read at runtime
/// rather than baked in at compile time because SwiftPM offers no way to inject
/// the resolved graph into a target, and a hand-maintained copy would drift.
enum PackageResolved {

    /// The handful that move the numbers. A full dump of forty transitive pins
    /// would bury them.
    private static let interesting: Set<String> = [
        "mlx-swift", "mlx-swift-lm", "SwiftAudio", "swift-transformers",
        "FluidAudio", "MisakiSwift",
    ]

    static func revisions() -> [String: String]? {
        guard let url = locate() else { return nil }
        guard let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        // Version 2 and 3 of the file differ in where the array lives.
        let pins = (root["pins"] as? [[String: Any]])
            ?? ((root["object"] as? [String: Any])?["pins"] as? [[String: Any]])
        guard let pins else { return nil }

        var found: [String: String] = [:]
        for pin in pins {
            guard let identity = (pin["identity"] as? String) ?? (pin["package"] as? String)
            else { continue }
            guard interesting.contains(where: { $0.lowercased() == identity.lowercased() })
            else { continue }
            let state = (pin["state"] as? [String: Any]) ?? [:]
            if let version = state["version"] as? String {
                found[identity] = version
            } else if let revision = state["revision"] as? String {
                found[identity] = String(revision.prefix(12))
            }
        }
        return found.isEmpty ? nil : found
    }

    private static func locate() -> URL? {
        // SystemProfile.swift -> AudioToolBenchmark -> Sources -> AudioToolSwift
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<3 { url.deleteLastPathComponent() }
        let candidate = url.appendingPathComponent("Package.resolved")
        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
    }
}
