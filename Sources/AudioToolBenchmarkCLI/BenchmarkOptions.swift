//
//  BenchmarkOptions.swift
//  AudioToolBenchmarkCLI
//
//  Command line surface for `audio-tool-bench`.
//

import Foundation
import AudioToolBenchmark

/// Parsed invocation.
///
/// Hand-rolled rather than swift-argument-parser, to match `audio-tool` and to
/// keep the benchmark from adding a dependency that would then appear in the
/// dependency list it reports.
struct BenchmarkOptions: Sendable {

    // Selection
    var patterns: [String] = []
    var listOnly = false

    // Measurement
    var iterations = 5
    var warmupIterations = 1
    var inputSeconds = 30.0
    var inputPath: String?

    // Isolation and pacing
    var inProcess = false
    var cooldownSeconds = 5.0
    var caseTimeoutSeconds = 900.0
    var stopOnFailure = false

    // Limits
    var gpuCacheLimitBytes: Int?
    var gpuMemoryLimitBytes: Int?

    // Inputs the catalog cannot discover
    var coreMLGANModelPath: String?
    /// Root of a checkout holding weights already on this machine. Autodetected.
    var weightsRoot: String?

    // Output
    var outputDirectory = "BenchmarkResults"
    var tag: String?
    var redactHost = false
    var quiet = false

    // Modes
    var prefetchOnly = false
    /// Set by the parent when it re-executes this binary for a single case.
    var childCaseID: String?
    /// Where the child writes its single ``CaseResult``.
    var childResultPath: String?

    var isChild: Bool { childCaseID != nil }

    var inputSource: BenchmarkInput.Source {
        inputPath.map { .file(URL(fileURLWithPath: $0)) } ?? .synthetic
    }

    var catalogOptions: BenchmarkCatalog.CatalogOptions {
        let local = weightsRoot.map { LocalWeights(root: URL(fileURLWithPath: $0)) }
            ?? LocalWeights(root: LocalWeights.autodetect())
        return BenchmarkCatalog.CatalogOptions(
            coreMLGANModelPath: coreMLGANModelPath,
            localWeights: local
        )
    }

    // MARK: - Parsing

    static func parse(_ arguments: [String]) throws -> BenchmarkOptions {
        var options = BenchmarkOptions()

        // Environment first, so an explicit flag always wins. The CoreML model is
        // the only input with no download path, so a machine that has one is
        // better off exporting it once than remembering a flag every run.
        let environment = ProcessInfo.processInfo.environment
        options.coreMLGANModelPath = environment["AUDIOTOOL_BENCH_COREML_GAN"]
        options.weightsRoot = environment["AUDIOTOOL_BENCH_WEIGHTS_ROOT"]
        if let directory = environment["AUDIOTOOL_BENCH_OUTPUT_DIR"] {
            options.outputDirectory = directory
        }

        var index = 1
        func next(_ flag: String) throws -> String {
            index += 1
            guard index < arguments.count else { throw OptionError.missingValue(flag) }
            return arguments[index]
        }
        func nextInt(_ flag: String) throws -> Int {
            let raw = try next(flag)
            guard let value = Int(raw) else { throw OptionError.notAnInteger(flag, raw) }
            return value
        }
        func nextDouble(_ flag: String) throws -> Double {
            let raw = try next(flag)
            guard let value = Double(raw) else { throw OptionError.notANumber(flag, raw) }
            return value
        }

        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--help", "-h":
                print(usage)
                exit(0)
            case "--list", "-l":
                options.listOnly = true
            case "--filter", "-f":
                options.patterns.append(contentsOf:
                    try next(argument).split(separator: ",").map(String.init))
            case "--case", "-c":
                options.patterns.append(try next(argument))
            case "--iterations", "-n":
                options.iterations = try nextInt(argument)
            case "--warmup":
                options.warmupIterations = try nextInt(argument)
            case "--seconds", "-s":
                options.inputSeconds = try nextDouble(argument)
            case "--input", "-i":
                options.inputPath = try next(argument)
            case "--in-process":
                options.inProcess = true
            case "--cooldown":
                options.cooldownSeconds = try nextDouble(argument)
            case "--timeout":
                options.caseTimeoutSeconds = try nextDouble(argument)
            case "--stop-on-failure":
                options.stopOnFailure = true
            case "--gpu-cache-limit":
                options.gpuCacheLimitBytes = try nextInt(argument) * 1024 * 1024
            case "--gpu-memory-limit":
                options.gpuMemoryLimitBytes = try nextInt(argument) * 1024 * 1024
            case "--coreml-gan":
                options.coreMLGANModelPath = try next(argument)
            case "--weights-root":
                options.weightsRoot = try next(argument)
            case "--out", "-o":
                options.outputDirectory = try next(argument)
            case "--tag", "-t":
                options.tag = try next(argument)
            case "--redact-host":
                options.redactHost = true
            case "--quiet", "-q":
                options.quiet = true
            case "--prefetch":
                options.prefetchOnly = true
            case "--run-case":
                options.childCaseID = try next(argument)
            case "--result-out":
                options.childResultPath = try next(argument)
            default:
                throw OptionError.unknown(argument)
            }
            index += 1
        }

        guard options.iterations >= 1 else { throw OptionError.range("--iterations must be >= 1") }
        guard options.warmupIterations >= 0 else { throw OptionError.range("--warmup must be >= 0") }
        guard options.inputSeconds > 0 else { throw OptionError.range("--seconds must be > 0") }
        guard options.cooldownSeconds >= 0 else { throw OptionError.range("--cooldown must be >= 0") }
        guard options.caseTimeoutSeconds > 0 else { throw OptionError.range("--timeout must be > 0") }
        if let path = options.inputPath, !FileManager.default.fileExists(atPath: path) {
            throw OptionError.range("--input file not found: \(path)")
        }

        return options
    }

    /// Arguments for a child that runs exactly one case.
    ///
    /// Rebuilt from the parsed options rather than forwarded verbatim, so a child
    /// cannot inherit `--list`, another `--run-case`, or an output directory it
    /// would write a second report into.
    func childArguments(caseID: String, resultPath: String) -> [String] {
        var arguments = [
            "--run-case", caseID,
            "--result-out", resultPath,
            "--iterations", String(iterations),
            "--warmup", String(warmupIterations),
            "--seconds", String(inputSeconds),
        ]
        if let path = inputPath { arguments += ["--input", path] }
        if let bytes = gpuCacheLimitBytes {
            arguments += ["--gpu-cache-limit", String(bytes / 1024 / 1024)]
        }
        if let bytes = gpuMemoryLimitBytes {
            arguments += ["--gpu-memory-limit", String(bytes / 1024 / 1024)]
        }
        if let path = coreMLGANModelPath { arguments += ["--coreml-gan", path] }
        if let path = weightsRoot { arguments += ["--weights-root", path] }
        if prefetchOnly { arguments.append("--prefetch") }
        if quiet { arguments.append("--quiet") }
        return arguments
    }

    /// The caps this run enforces: host defaults unless overridden.
    var budget: MemoryBudget {
        MemoryBudget.forHost().overridden(
            cache: gpuCacheLimitBytes,
            memory: gpuMemoryLimitBytes
        )
    }

    func runConfiguration(budget: MemoryBudget) -> RunConfiguration {
        RunConfiguration(
            iterations: iterations,
            warmupIterations: warmupIterations,
            inputSeconds: inputSeconds,
            inputSource: inputSource.description,
            isolation: inProcess ? "in-process" : "process",
            cooldownSeconds: cooldownSeconds,
            gpuCacheLimitBytes: budget.cacheLimitBytes,
            gpuMemoryLimitBytes: budget.memoryLimitBytes,
            buildConfiguration: SystemProfile.buildConfiguration
        )
    }
}

// MARK: - Errors

enum OptionError: Error, CustomStringConvertible {
    case unknown(String)
    case missingValue(String)
    case notAnInteger(String, String)
    case notANumber(String, String)
    case range(String)

    var description: String {
        switch self {
        case .unknown(let flag): "unknown option: \(flag)"
        case .missingValue(let flag): "\(flag) needs a value"
        case .notAnInteger(let flag, let value): "\(flag) expects an integer, got '\(value)'"
        case .notANumber(let flag, let value): "\(flag) expects a number, got '\(value)'"
        case .range(let message): message
        }
    }
}

// MARK: - Usage

let usage = """
audio-tool-bench - measure AudioTool's models on this machine

Usage: audio-tool-bench [options]

By default every case runs in its own child process, one at a time. That is what
makes the memory numbers mean anything: each model starts from a clean allocator,
nothing from the previous one is still resident, and a model that runs the machine
out of memory takes down one case instead of the whole run.

Selection
  -l, --list              List the cases and exit
  -f, --filter <p>[,<p>]  Run cases whose id, category or backend contains p
  -c, --case <id>         Same, repeatable

Measurement
  -n, --iterations <n>    Timed runs per case (default 5)
      --warmup <n>        Untimed runs before them (default 1)
  -s, --seconds <d>       Seconds of audio per run (default 30)
  -i, --input <path>      Use a real file instead of the synthetic signal.
                          Resampled per case; looped if shorter than --seconds.

Pacing and isolation
      --cooldown <s>      Idle seconds between cases (default 5), so one model's
                          heat is not the next one's starting condition
      --timeout <s>       Kill a case after this long (default 900)
      --in-process        Run every case in this process. Faster, and the peak
                          memory columns stop meaning anything.
      --stop-on-failure   Abort the run on the first failed case

Limits
      --gpu-cache-limit <MB>   MLX allocator cache ceiling (default 3072, which is
                               what this package applies in production)
      --gpu-memory-limit <MB>  MLX total allocation ceiling (default: 55% of RAM,
                               capped by the GPU's recommended working set)

Inputs
      --coreml-gan <path>  MossFormerGAN .mlpackage to measure instead of the
                           published FP32 one. Optional - both CoreML precisions
                           download without it.
                           Also read from AUDIOTOOL_BENCH_COREML_GAN.
      --weights-root <dir> Use weights already on this machine instead of fetching
                           them. Defaults to the sibling research checkout when one
                           is present, so a development machine needs no downloads.
                           Also read from AUDIOTOOL_BENCH_WEIGHTS_ROOT.

Output
  -o, --out <dir>         Where the report goes (default ./BenchmarkResults,
                          or AUDIOTOOL_BENCH_OUTPUT_DIR)
  -t, --tag <label>       Free-form label recorded in the report
      --redact-host       Prepare the report for publication: drop the host name,
                          reduce --input to a basename, rewrite the home directory
                          as ~ (including in failure messages). Best-effort.
  -q, --quiet             Only print the summary

Modes
      --prefetch          Download and load every selected model, then exit.
                          Run this once on a new machine: a load time with a
                          download in it is not a load time.

Examples
  audio-tool-bench --list
  audio-tool-bench --prefetch
  audio-tool-bench -t "main@a1b2c3"
  audio-tool-bench -f enhancement -n 10 -s 60
  audio-tool-bench -c mlx.demucs.all_stems --cooldown 30

Build it the way `audio-tool` is built, and run it from the products directory -
that is where MLX's metallib lives:

  xcodebuild build -scheme audio-tool-bench -configuration Release \\
    -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
  .build/DerivedData/Build/Products/Release/audio-tool-bench
"""
