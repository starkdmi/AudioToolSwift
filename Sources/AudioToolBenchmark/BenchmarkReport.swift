//
//  BenchmarkReport.swift
//  AudioToolBenchmark
//
//  The on-disk shape of a benchmark run. This is the deliverable; everything
//  else in this target exists to fill it in.
//

import Foundation

// MARK: - Report

/// One benchmark run: what machine, what settings, what each model cost.
///
/// Written as JSON so runs from different machines can be diffed and merged. The
/// numbers here are only comparable across machines because ``RunConfiguration``
/// and ``SystemProfile`` travel with them - an RTF without the input length, the
/// build configuration and the thermal state it was measured under is a rumour.
public struct BenchmarkReport: Codable, Sendable {

    /// Bumped whenever a field changes meaning. Readers should refuse a schema
    /// they do not know rather than silently misinterpret a column.
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var runID: String
    /// Free-form label from `--tag`, e.g. a branch name or "before-chunk-rewrite".
    public var tag: String?
    public var startedAt: Date
    public var finishedAt: Date
    public var system: SystemProfile
    public var configuration: RunConfiguration
    public var results: [CaseResult]

    public init(
        schemaVersion: Int = BenchmarkReport.currentSchemaVersion,
        runID: String,
        tag: String?,
        startedAt: Date,
        finishedAt: Date,
        system: SystemProfile,
        configuration: RunConfiguration,
        results: [CaseResult]
    ) {
        self.schemaVersion = schemaVersion
        self.runID = runID
        self.tag = tag
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.system = system
        self.configuration = configuration
        self.results = results
    }
}

// MARK: - Configuration

/// Everything a reader needs in order to reproduce the run.
public struct RunConfiguration: Codable, Sendable {

    /// How many timed iterations per case, after the warm-ups.
    public var iterations: Int

    /// Untimed runs between the first inference and the measured ones.
    ///
    /// The first inference pays MLX graph compilation and Metal pipeline
    /// construction, which is a real cost but a different one - it is reported
    /// separately as ``TimingMetrics/firstRunSeconds`` rather than folded into a
    /// steady-state RTF.
    public var warmupIterations: Int

    /// Length of the input handed to every case, in seconds of audio.
    public var inputSeconds: Double

    /// `"synthetic"` or the path the input was read from.
    public var inputSource: String

    /// `"process"` when each case ran in its own child process, `"in-process"`
    /// when they shared one. Peak-memory numbers from the two are not comparable.
    public var isolation: String

    /// Idle seconds between cases, so one model's heat is not the next one's
    /// starting condition.
    public var cooldownSeconds: Double

    /// The process-global MLX caps in force. See ``MemoryBudget``.
    public var gpuCacheLimitBytes: Int
    public var gpuMemoryLimitBytes: Int

    /// `"release"` or `"debug"`. A debug number measures the optimiser.
    public var buildConfiguration: String

    public init(
        iterations: Int,
        warmupIterations: Int,
        inputSeconds: Double,
        inputSource: String,
        isolation: String,
        cooldownSeconds: Double,
        gpuCacheLimitBytes: Int,
        gpuMemoryLimitBytes: Int,
        buildConfiguration: String
    ) {
        self.iterations = iterations
        self.warmupIterations = warmupIterations
        self.inputSeconds = inputSeconds
        self.inputSource = inputSource
        self.isolation = isolation
        self.cooldownSeconds = cooldownSeconds
        self.gpuCacheLimitBytes = gpuCacheLimitBytes
        self.gpuMemoryLimitBytes = gpuMemoryLimitBytes
        self.buildConfiguration = buildConfiguration
    }
}

// MARK: - Case result

/// How one case ended.
public enum CaseStatus: String, Codable, Sendable {
    /// Ran to completion; ``CaseResult/timing`` and friends are populated.
    case completed
    /// Deliberately not run - weights absent, model file not supplied, and so on.
    /// Not a failure: a fresh machine has none of the optional inputs.
    case skipped
    /// Started and did not finish. ``CaseResult/message`` says why.
    case failed
}

/// One model, measured.
public struct CaseResult: Codable, Sendable {
    public var id: String
    public var label: String
    public var category: String
    public var backend: String
    public var status: CaseStatus

    /// Skip reason or failure description. Nil when ``status`` is `.completed`.
    public var message: String?

    /// Sample rate the model consumed, and how much audio it was given. These
    /// differ per case - SS 3-speaker runs at 8 kHz, SE 48K at 48 kHz - so the
    /// sample count is not derivable from ``RunConfiguration/inputSeconds`` alone.
    public var inputSampleRate: Int?
    public var inputFrames: Int?

    /// What came back, so a case that silently produced nothing is visible.
    public var outputStreams: Int?
    public var outputFrames: Int?
    public var outputSampleRate: Int?

    /// True when the weights were already in the HuggingFace cache before the
    /// case started. When false, ``TimingMetrics/loadSeconds`` includes a
    /// download and is not comparable to anything.
    public var weightsPreCached: Bool?

    /// Which route the weights took: an explicit local file, the HuggingFace
    /// cache, or a transfer during this run. Three different measurements, and
    /// only the first two belong in a comparison.
    public var weightsSource: WeightsSource?

    public var timing: TimingMetrics?
    public var memory: MemoryMetrics?
    public var cpu: CPUMetrics?
    public var environment: EnvironmentSnapshot?

    /// The caps this case *requested*, which may differ from the run-wide defaults
    /// when the catalog overrides them.
    public var gpuCacheLimitBytes: Int?
    public var gpuMemoryLimitBytes: Int?

    /// The caps actually in force when the case finished, read back from MLX.
    ///
    /// Not the same question as the two above, and the difference is real:
    /// `MLXCachePolicy` applies the package's own process-global limits the first
    /// time any chunked path runs, overriding whatever the harness asked for. A
    /// report that printed only the requested value would describe a configuration
    /// that was replaced before the first chunk finished.
    public var effectiveGpuCacheLimitBytes: Int?
    public var effectiveGpuMemoryLimitBytes: Int?

    public init(
        id: String,
        label: String,
        category: String,
        backend: String,
        status: CaseStatus,
        message: String? = nil,
        inputSampleRate: Int? = nil,
        inputFrames: Int? = nil,
        outputStreams: Int? = nil,
        outputFrames: Int? = nil,
        outputSampleRate: Int? = nil,
        weightsPreCached: Bool? = nil,
        timing: TimingMetrics? = nil,
        memory: MemoryMetrics? = nil,
        cpu: CPUMetrics? = nil,
        environment: EnvironmentSnapshot? = nil,
        gpuCacheLimitBytes: Int? = nil,
        gpuMemoryLimitBytes: Int? = nil,
        effectiveGpuCacheLimitBytes: Int? = nil,
        effectiveGpuMemoryLimitBytes: Int? = nil
    ) {
        self.id = id
        self.label = label
        self.category = category
        self.backend = backend
        self.status = status
        self.message = message
        self.inputSampleRate = inputSampleRate
        self.inputFrames = inputFrames
        self.outputStreams = outputStreams
        self.outputFrames = outputFrames
        self.outputSampleRate = outputSampleRate
        self.weightsPreCached = weightsPreCached
        self.timing = timing
        self.memory = memory
        self.cpu = cpu
        self.environment = environment
        self.gpuCacheLimitBytes = gpuCacheLimitBytes
        self.gpuMemoryLimitBytes = gpuMemoryLimitBytes
        self.effectiveGpuCacheLimitBytes = effectiveGpuCacheLimitBytes
        self.effectiveGpuMemoryLimitBytes = effectiveGpuMemoryLimitBytes
    }
}

// MARK: - Metrics

/// Wall-clock costs, in seconds.
public struct TimingMetrics: Codable, Sendable {

    /// `load()`: reading weights, materialising them, and whatever prewarm the
    /// provider does inside its own load.
    public var loadSeconds: Double

    /// The first `process()` after load. Carries MLX graph compilation and Metal
    /// pipeline construction, so it is routinely several times the steady state
    /// and is deliberately excluded from ``medianSeconds``.
    public var firstRunSeconds: Double

    /// Every timed iteration, in order, so a reader can see drift rather than
    /// take the summary on trust.
    public var iterationSeconds: [Double]

    public var minSeconds: Double
    public var medianSeconds: Double
    public var meanSeconds: Double
    public var stdDevSeconds: Double

    /// Audio seconds processed per wall second, from ``medianSeconds``.
    ///
    /// Median rather than min: unlike a pure numeric kernel, these runs include
    /// cache trimming and allocator behaviour that a real caller also pays, so
    /// the fastest observed run is not the honest one. The min is recorded too.
    public var realTimeFactor: Double

    /// From ``minSeconds``, for comparison with suites that headline a best-of-N.
    public var realTimeFactorBest: Double

    public init(
        loadSeconds: Double,
        firstRunSeconds: Double,
        iterationSeconds: [Double],
        audioSeconds: Double
    ) {
        self.loadSeconds = loadSeconds
        self.firstRunSeconds = firstRunSeconds
        self.iterationSeconds = iterationSeconds

        let sorted = iterationSeconds.sorted()
        let count = sorted.count
        let minimum = sorted.first ?? .nan
        let mean = count > 0 ? sorted.reduce(0, +) / Double(count) : .nan
        let median: Double
        if count == 0 {
            median = .nan
        } else if count.isMultiple(of: 2) {
            median = (sorted[count / 2 - 1] + sorted[count / 2]) / 2
        } else {
            median = sorted[count / 2]
        }
        // Sample standard deviation. With one iteration there is no spread to
        // report, and dividing by zero would put a NaN in the JSON.
        let variance = count > 1
            ? sorted.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(count - 1)
            : 0

        self.minSeconds = minimum
        self.medianSeconds = median
        self.meanSeconds = mean
        self.stdDevSeconds = variance.squareRoot()
        self.realTimeFactor = median > 0 ? audioSeconds / median : .nan
        self.realTimeFactorBest = minimum > 0 ? audioSeconds / minimum : .nan
    }
}

/// What the process and MLX held, in bytes.
///
/// Two independent views, because neither alone is the answer. `phys_footprint`
/// is what Activity Monitor calls Memory and what the OS charges against a
/// jetsam/pressure limit; MLX's own counters say how much of that is the model
/// rather than the runtime, the audio buffers and the Swift heap.
public struct MemoryMetrics: Codable, Sendable {

    /// Process footprint before `load()`.
    public var baselineFootprintBytes: Int
    /// Process footprint immediately after `load()`.
    public var afterLoadFootprintBytes: Int
    /// Highest footprint seen by the sampler across load and every run.
    public var peakFootprintBytes: Int
    /// Highest resident size seen by the sampler.
    public var peakResidentBytes: Int

    /// ``afterLoadFootprintBytes`` minus ``baselineFootprintBytes``: roughly the
    /// weights, once. Clamped at zero, since an allocator can hand memory back
    /// during load and make the difference negative.
    public var loadDeltaBytes: Int

    /// MLX active bytes after load - the weights as MLX accounts for them.
    public var mlxActiveAfterLoadBytes: Int
    /// MLX peak during load.
    public var mlxPeakDuringLoadBytes: Int

    /// MLX peak across the timed runs: **total** MLX memory, weights included.
    ///
    /// The counter is reset after load, but resetting it sets the high-water mark
    /// to zero and does not free anything - the weights are still active, so the
    /// very next allocation pushes the peak back to at least their size. What the
    /// reset does buy is dropping any transient spike from *loading* (converting,
    /// unflattening, `verify: .all`), which is real but is not what an inference
    /// costs.
    ///
    /// So this is the ceiling a caller has to have room for while the model runs,
    /// which is the number that matters on a memory-constrained machine.
    /// ``mlxRunActivationsBytes`` is the activations-only estimate.
    public var mlxPeakDuringRunBytes: Int

    /// Whether MLX had the weights materialised by the time `load()` returned.
    ///
    /// Not every provider does, and the difference is not cosmetic. MLX arrays are
    /// lazy: `MossFormer2SE48KProvider.load()` calls `loadWeights` and returns
    /// without an `eval`, so its active memory after load is *8 bytes* and the
    /// weights are really materialised inside the first inference. `FRCRNSE16K`
    /// runs a dummy forward pass and evals, `MossFormer2SS` evals the module, and
    /// both report their real size here.
    ///
    /// When this is false, `loadSeconds` excludes weight materialisation and
    /// `firstRunSeconds` includes it - so the two columns are not measuring the
    /// same split of work across providers, and ``mlxRunActivationsBytes`` cannot
    /// be computed at all.
    public var weightsMaterializedAtLoad: Bool

    /// Approximate inference working set: ``mlxPeakDuringRunBytes`` minus the
    /// weights that were resident before the first run.
    ///
    /// An estimate, not a measurement, and labelled as one - it assumes the weights
    /// stay resident and unchanged across the run, which is true of every provider
    /// here but is not something the counters attest.
    ///
    /// Nil when ``weightsMaterializedAtLoad`` is false. Subtracting a baseline that
    /// does not yet contain the weights would return the whole peak and call it
    /// activations, which is exactly the wrong answer stated confidently.
    public var mlxRunActivationsBytes: Int?

    /// MLX cache at the end, against the cap that was in force.
    public var mlxCacheAtEndBytes: Int

    public init(
        baselineFootprintBytes: Int,
        afterLoadFootprintBytes: Int,
        peakFootprintBytes: Int,
        peakResidentBytes: Int,
        mlxActiveAfterLoadBytes: Int,
        mlxPeakDuringLoadBytes: Int,
        mlxPeakDuringRunBytes: Int,
        mlxCacheAtEndBytes: Int
    ) {
        self.baselineFootprintBytes = baselineFootprintBytes
        self.afterLoadFootprintBytes = afterLoadFootprintBytes
        self.peakFootprintBytes = peakFootprintBytes
        self.peakResidentBytes = peakResidentBytes
        self.loadDeltaBytes = max(0, afterLoadFootprintBytes - baselineFootprintBytes)
        self.mlxActiveAfterLoadBytes = mlxActiveAfterLoadBytes
        self.mlxPeakDuringLoadBytes = mlxPeakDuringLoadBytes
        self.mlxPeakDuringRunBytes = mlxPeakDuringRunBytes
        // A megabyte is three orders of magnitude below the smallest set of weights
        // in this catalog (USS FP16, ~53 MB), and the lazy case reports single-digit
        // bytes - so anything under it means "nothing was materialised", with no
        // real configuration anywhere near the boundary.
        let materialized = mlxActiveAfterLoadBytes >= 1024 * 1024
        self.weightsMaterializedAtLoad = materialized
        self.mlxRunActivationsBytes = materialized
            ? max(0, mlxPeakDuringRunBytes - mlxActiveAfterLoadBytes)
            : nil
        self.mlxCacheAtEndBytes = mlxCacheAtEndBytes
    }
}

/// CPU time consumed by the whole case, from `getrusage`.
///
/// ``utilization`` above 1 means real parallelism; well below 1 on a GPU model
/// means the CPU was waiting on Metal, which is the expected shape here and the
/// reason a wall-clock RTF alone cannot tell a GPU win from a CPU one.
public struct CPUMetrics: Codable, Sendable {
    public var userSeconds: Double
    public var systemSeconds: Double
    public var wallSeconds: Double
    public var utilization: Double

    public init(userSeconds: Double, systemSeconds: Double, wallSeconds: Double) {
        self.userSeconds = userSeconds
        self.systemSeconds = systemSeconds
        self.wallSeconds = wallSeconds
        self.utilization = wallSeconds > 0 ? (userSeconds + systemSeconds) / wallSeconds : .nan
    }
}

/// Conditions that invalidate a measurement if they moved during it.
public struct EnvironmentSnapshot: Codable, Sendable {
    public var thermalStateAtStart: String
    public var thermalStateAtEnd: String
    public var lowPowerModeEnabled: Bool
    public var powerSource: String

    /// True when the thermal state changed mid-case. A run that starts `nominal`
    /// and ends `serious` was throttled partway through and its later iterations
    /// are not the same measurement as its earlier ones.
    public var thermalStateChanged: Bool { thermalStateAtStart != thermalStateAtEnd }

    public init(
        thermalStateAtStart: String,
        thermalStateAtEnd: String,
        lowPowerModeEnabled: Bool,
        powerSource: String
    ) {
        self.thermalStateAtStart = thermalStateAtStart
        self.thermalStateAtEnd = thermalStateAtEnd
        self.lowPowerModeEnabled = lowPowerModeEnabled
        self.powerSource = powerSource
    }

    private enum CodingKeys: String, CodingKey {
        case thermalStateAtStart, thermalStateAtEnd, lowPowerModeEnabled, powerSource
    }
}

// MARK: - Redaction

/// Removing what identifies a person from a report that will be published.
///
/// The host name is not the only thing that leaks. `--input /Users/someone/...`
/// lands in ``RunConfiguration/inputSource``, and a skip or failure message can
/// carry the path of a `.mlpackage` or a weights file - all of which contain a
/// home directory, and so a user name. Redacting only the host name and calling
/// the report safe to publish would be worse than not offering redaction at all,
/// because it invites trust the output has not earned.
///
/// Best-effort and described as such: it removes the host name, reduces a file
/// input to its last path component, and rewrites the home directory as `~`
/// wherever it appears. A path deliberately placed outside the home directory
/// survives, so read a redacted report before publishing it rather than assuming.
public enum Redaction {

    public static func apply(to report: BenchmarkReport) -> BenchmarkReport {
        var copy = report
        copy.system = report.system.redactingHost()
        copy.configuration.inputSource = redactPath(report.configuration.inputSource)
        copy.tag = report.tag.map(scrubHome)
        copy.results = report.results.map { result in
            var redacted = result
            redacted.message = result.message.map(scrubHome)
            return redacted
        }
        return copy
    }

    /// A file input becomes its basename; `"synthetic"` is left alone.
    ///
    /// The directory is what identifies someone; the file name is what makes the
    /// report meaningful, so it stays.
    static func redactPath(_ source: String) -> String {
        guard source != "synthetic" else { return source }
        return URL(fileURLWithPath: source).lastPathComponent
    }

    /// `/Users/someone/x` becomes `~/x`.
    static func scrubHome(_ text: String) -> String {
        let home = NSHomeDirectory()
        guard !home.isEmpty, home != "/" else { return text }
        return text.replacingOccurrences(of: home, with: "~")
    }
}

// MARK: - Coding

public enum BenchmarkCoding {

    /// One encoder configuration for the child, the parent and any tooling, so a
    /// merged report is byte-comparable with a fresh one.
    public static func encoder(pretty: Bool = true) -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = pretty
            ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            : [.sortedKeys, .withoutEscapingSlashes]
        // Infinity and NaN are legitimate outcomes here - a case with a single
        // iteration has no standard deviation, and a zero-length run has no RTF -
        // and JSON cannot express either. Encoding them as strings keeps the file
        // writable instead of throwing at the very end of a long run.
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan"
        )
        return encoder
    }

    public static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(
            positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan"
        )
        return decoder
    }
}
