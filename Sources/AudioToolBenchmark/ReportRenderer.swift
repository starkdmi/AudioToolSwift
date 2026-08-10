//
//  ReportRenderer.swift
//  AudioToolBenchmark
//
//  Turning a report into something a person reads without opening the JSON.
//

import Foundation

/// Markdown and terminal rendering of a ``BenchmarkReport``.
///
/// The JSON is the record; this is the part that gets pasted into a pull request.
/// Both come out of the same run, so a summary cannot disagree with the data it
/// summarises.
public enum ReportRenderer {

    // MARK: - Markdown

    public static func markdown(_ report: BenchmarkReport) -> String {
        var lines: [String] = []

        lines.append("# AudioTool benchmark")
        lines.append("")
        if let tag = report.tag, !tag.isEmpty {
            lines.append("**\(tag)**")
            lines.append("")
        }
        lines.append(contentsOf: systemSection(report))
        lines.append("")
        lines.append(contentsOf: configurationSection(report))
        lines.append("")
        lines.append(contentsOf: timingTable(report))
        lines.append("")
        lines.append(contentsOf: memoryTable(report))

        let notes = noteSection(report)
        if !notes.isEmpty {
            lines.append("")
            lines.append(contentsOf: notes)
        }

        lines.append("")
        return lines.joined(separator: "\n")
    }

    private static func systemSection(_ report: BenchmarkReport) -> [String] {
        let system = report.system
        var lines = ["## Machine", ""]
        lines.append("| | |")
        lines.append("| --- | --- |")
        lines.append("| Chip | \(system.chip) |")
        lines.append("| Model | \(system.hardwareModel) |")
        var cores = "\(system.logicalCores)"
        if let performance = system.performanceCores, let efficiency = system.efficiencyCores {
            cores += " (\(performance)P / \(efficiency)E)"
        }
        lines.append("| Cores | \(cores) |")
        lines.append("| Memory | \(ByteFormat.bytes(system.physicalMemoryBytes)) |")
        if let gpu = system.gpuName {
            let workingSet = system.gpuRecommendedWorkingSetBytes.map(ByteFormat.bytes) ?? "unknown"
            lines.append("| GPU | \(gpu), recommended working set \(workingSet) |")
        }
        lines.append("| OS | \(system.osVersion) |")
        lines.append("| Architecture | \(system.architecture) |")
        if let revisions = system.dependencyRevisions, !revisions.isEmpty {
            let rendered = revisions.sorted { $0.key < $1.key }
                .map { "\($0.key) \($0.value)" }
                .joined(separator: ", ")
            lines.append("| Dependencies | \(rendered) |")
        }
        return lines
    }

    private static func configurationSection(_ report: BenchmarkReport) -> [String] {
        let config = report.configuration
        var lines = ["## Run", ""]
        lines.append("| | |")
        lines.append("| --- | --- |")
        lines.append("| Started | \(iso(report.startedAt)) |")
        lines.append(String(
            format: "| Duration | %.0f s |",
            report.finishedAt.timeIntervalSince(report.startedAt)
        ))
        lines.append("| Build | \(config.buildConfiguration) |")
        lines.append("| Isolation | \(config.isolation) |")
        lines.append(String(format: "| Input | %@, %.0f s |", config.inputSource, config.inputSeconds))
        lines.append("| Iterations | \(config.iterations) timed, \(config.warmupIterations) warm-up |")
        lines.append("| MLX cache limit | \(ByteFormat.bytes(config.gpuCacheLimitBytes)) |")
        lines.append("| MLX memory limit | \(ByteFormat.bytes(config.gpuMemoryLimitBytes)) |")
        lines.append(String(format: "| Cooldown | %.0f s between cases |", config.cooldownSeconds))
        return lines
    }

    private static func timingTable(_ report: BenchmarkReport) -> [String] {
        var lines = ["## Speed", ""]
        lines.append("RTF is audio seconds per wall second, from the median iteration.")
        lines.append("Higher is faster. `load` and `first` are reported separately because")
        lines.append("the first inference carries MLX graph compilation.")
        lines.append("")
        lines.append("| case | rate | load (s) | first (s) | median (s) | ±sd | RTF | best RTF |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        for result in report.results {
            guard let timing = result.timing else {
                lines.append("| `\(result.id)` | | | | | | \(statusNote(result)) | |")
                continue
            }
            let rate = result.inputSampleRate.map { "\($0 / 1000)k" } ?? ""
            lines.append(String(
                format: "| `%@` | %@ | %.2f | %.2f | %.3f | %.3f | **%.1fx** | %.1fx |",
                result.id, rate,
                timing.loadSeconds, timing.firstRunSeconds,
                timing.medianSeconds, timing.stdDevSeconds,
                timing.realTimeFactor, timing.realTimeFactorBest
            ))
        }
        return lines
    }

    private static func memoryTable(_ report: BenchmarkReport) -> [String] {
        var lines = ["## Memory", ""]
        lines.append("`peak` is the process `phys_footprint` - what Activity Monitor calls Memory.")
        lines.append("`weights` is the footprint growth across `load()`. The `mlx` columns are")
        lines.append("MLX's own accounting: `mlx peak` is total MLX memory at its highest during")
        lines.append("the timed runs, **weights included** - resetting the peak counter zeroes the")
        lines.append("high-water mark but frees nothing, so the next allocation restores it.")
        lines.append("`act.` is that minus the resident weights: an estimate of the activations,")
        lines.append("not a measurement, and `-` where the provider had not materialised its")
        lines.append("weights by the end of `load()` (see Notes).")
        lines.append("")
        lines.append("| case | peak | weights | mlx weights | mlx peak | mlx act. | mlx cache | cpu util |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        for result in report.results {
            guard let memory = result.memory else {
                lines.append("| `\(result.id)` | \(statusNote(result)) | | | | | | |")
                continue
            }
            let utilization = result.cpu.map { String(format: "%.2f", $0.utilization) } ?? ""
            let mlxWeights = memory.weightsMaterializedAtLoad
                ? ByteFormat.bytes(memory.mlxActiveAfterLoadBytes)
                : "lazy"
            let activations = memory.mlxRunActivationsBytes.map(ByteFormat.bytes) ?? "-"
            lines.append(String(
                format: "| `%@` | %@ | %@ | %@ | %@ | %@ | %@ | %@ |",
                result.id,
                ByteFormat.bytes(memory.peakFootprintBytes),
                ByteFormat.bytes(memory.loadDeltaBytes),
                mlxWeights,
                ByteFormat.bytes(memory.mlxPeakDuringRunBytes),
                activations,
                ByteFormat.bytes(memory.mlxCacheAtEndBytes),
                utilization
            ))
        }
        return lines
    }

    /// Everything that would make a reader distrust a row, collected rather than
    /// buried per-line: a cold download folded into a load time, a thermal state
    /// that moved mid-case, a machine on battery, a debug build.
    private static func noteSection(_ report: BenchmarkReport) -> [String] {
        var notes: [String] = []

        if report.configuration.buildConfiguration != "release" {
            notes.append(
                "- **Debug build.** These numbers measure the optimiser, not the models. "
                    + "Rebuild with `-configuration Release`."
            )
        }
        if report.configuration.isolation != "process" {
            notes.append(
                "- Cases shared one process (`--in-process`), so peak-memory columns "
                    + "include whatever earlier cases left resident."
            )
        }

        let onBattery = report.results.compactMap(\.environment)
            .contains { $0.powerSource == "battery" }
        if onBattery {
            notes.append("- At least one case ran on battery, where the GPU power budget is lower.")
        }
        let lowPower = report.results.compactMap(\.environment)
            .contains { $0.lowPowerModeEnabled }
        if lowPower {
            notes.append("- Low Power Mode was enabled for at least one case.")
        }

        let lazy = report.results.filter {
            $0.status == .completed && $0.memory?.weightsMaterializedAtLoad == false
        }
        if !lazy.isEmpty {
            let names = lazy.map { "`\($0.id)`" }.joined(separator: ", ")
            notes.append(
                "- \(names) had not materialised weights when `load()` returned - MLX "
                    + "arrays are lazy and these providers do not `eval` at load. Their "
                    + "`load` column excludes weight materialisation and their `first` "
                    + "column includes it, so those two columns are not the same split "
                    + "of work as the other rows'. Total and RTF are unaffected."
            )
        }

        let throttled = report.results.filter { $0.environment?.thermalStateChanged == true }
        for result in throttled {
            guard let environment = result.environment else { continue }
            notes.append(
                "- `\(result.id)`: thermal state moved "
                    + "\(environment.thermalStateAtStart) -> \(environment.thermalStateAtEnd) "
                    + "during the case; its later iterations were throttled."
            )
        }

        let cold = report.results.filter {
            $0.status == .completed && $0.weightsSource == .network
        }
        for result in cold {
            notes.append(
                "- `\(result.id)`: weights were not on this machine when the case "
                    + "started, so its load time includes a network transfer. Re-run "
                    + "for a comparable number."
            )
        }

        let localSourced = report.results.filter { $0.weightsSource == .local }
        if !localSourced.isEmpty {
            let names = localSourced.map { "`\($0.id)`" }.joined(separator: ", ")
            notes.append(
                "- \(names) loaded from explicit local weight files rather than the "
                    + "HuggingFace cache. Inference is identical; only the load path "
                    + "differs."
            )
        }

        let skipped = report.results.filter { $0.status == .skipped }
        for result in skipped {
            notes.append("- `\(result.id)` skipped: \(result.message ?? "no reason recorded")")
        }
        let failed = report.results.filter { $0.status == .failed }
        for result in failed {
            notes.append("- `\(result.id)` **failed**: \(result.message ?? "no reason recorded")")
        }

        return notes.isEmpty ? [] : ["## Notes", ""] + notes
    }

    private static func statusNote(_ result: CaseResult) -> String {
        switch result.status {
        case .completed: "-"
        case .skipped: "skipped"
        case .failed: "failed"
        }
    }

    private static func iso(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }

    // MARK: - Terminal

    /// A compact table for stdout while the run is still going.
    ///
    /// Deliberately narrower than the markdown: this is read in an 80-column
    /// terminal alongside the log lines, and a table that wraps is unreadable.
    public static func consoleLine(_ result: CaseResult) -> String {
        let name = result.id.padding(toLength: 34, withPad: " ", startingAt: 0)
        switch result.status {
        case .skipped:
            return "\(name)  skipped  \(result.message ?? "")"
        case .failed:
            return "\(name)  FAILED   \(result.message ?? "")"
        case .completed:
            guard let timing = result.timing, let memory = result.memory else {
                return "\(name)  ok"
            }
            let peak = ByteFormat.bytes(memory.peakFootprintBytes)
            return String(
                format: "%@  %6.1fx RTF  %7.3f s  peak %@",
                name,
                timing.realTimeFactor,
                timing.medianSeconds,
                peak
            )
        }
    }

    public static func consoleHeader() -> String {
        "case                                   speed        median      peak memory"
    }
}
