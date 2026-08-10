//
//  BenchmarkDriver.swift
//  AudioToolBenchmarkCLI
//
//  Sequencing the run: one model at a time, each in its own process.
//

import Foundation
import AudioToolBenchmark

/// The parent side of a run.
///
/// The design decision worth stating: every case is a child process, and the
/// parent does nothing but sequence them, collect their JSON and write a report.
/// That is more machinery than calling the runner in a loop, and it buys three
/// things a loop cannot:
///
/// - **Honest peak memory.** MLX's allocator is process-global and its cache
///   survives `unload()`. Two models measured in one process share a high-water
///   mark, and the second one inherits the first one's.
/// - **Containment.** A model that exhausts memory, traps, or wedges Metal kills
///   one case. The run continues and the report says which one died.
/// - **A real cold start.** Graph compilation, Metal pipeline construction and
///   the weights cache are per-process. Measuring the second model's "first
///   inference" in a warmed process measures something no user ever pays.
///
/// The cost is a process launch per case, which is milliseconds against loads and
/// inferences measured in seconds.
enum BenchmarkDriver {

    // MARK: - Listing

    static func list(_ options: BenchmarkOptions) {
        let cases = BenchmarkCatalog.select(options.patterns, options: options.catalogOptions)
        print("\(cases.count) case(s)")
        print("")
        for benchmarkCase in cases {
            let name = benchmarkCase.id.padding(toLength: 36, withPad: " ", startingAt: 0)
            let category = benchmarkCase.category.padding(toLength: 16, withPad: " ", startingAt: 0)
            let rate = "\(benchmarkCase.sampleRate / 1000) kHz"
                .padding(toLength: 8, withPad: " ", startingAt: 0)
            print("\(name)\(category)\(rate)\(benchmarkCase.label)")
            if let reason = benchmarkCase.unavailableReason() {
                print("\(String(repeating: " ", count: 36))unavailable: \(reason)")
            } else if benchmarkCase.weightsPreCached() == false {
                print("\(String(repeating: " ", count: 36))weights not yet downloaded")
            }
        }
    }

    // MARK: - Child entry point

    /// Run one case in this process and write its ``CaseResult`` where the parent
    /// expects it.
    ///
    /// Exit status is deliberately not the channel for the outcome: a case that
    /// fails still writes a result describing the failure and exits 0, having done
    /// its job. A non-zero exit means this process could not produce a result at
    /// all - unknown case id, unwritable path, or a crash - which the parent
    /// reports differently, because the two mean different things to whoever
    /// reads the report.
    static func runSingleCase(_ caseID: String, options: BenchmarkOptions) async -> Int32 {
        let cases = BenchmarkCatalog.allCases(options: options.catalogOptions)
        guard let benchmarkCase = cases.first(where: { $0.id == caseID }) else {
            FileHandle.standardError.write(Data("unknown case: \(caseID)\n".utf8))
            return 1
        }
        guard let resultPath = options.childResultPath else {
            FileHandle.standardError.write(Data("--run-case requires --result-out\n".utf8))
            return 1
        }

        let budget = options.budget
        let quiet = options.quiet
        let log: @Sendable (String) -> Void = { message in
            if !quiet { print(message) }
        }

        let result: CaseResult
        if options.prefetchOnly {
            result = await CaseRunner.prefetch(benchmarkCase, budget: budget, log: log)
        } else {
            result = await CaseRunner.run(
                benchmarkCase,
                configuration: options.runConfiguration(budget: budget),
                inputSource: options.inputSource,
                budget: budget,
                log: log
            )
        }

        do {
            let data = try BenchmarkCoding.encoder().encode(result)
            try data.write(to: URL(fileURLWithPath: resultPath))
        } catch {
            FileHandle.standardError.write(Data("could not write result: \(error)\n".utf8))
            return 1
        }
        return 0
    }

    // MARK: - Parent

    static func runAll(_ options: BenchmarkOptions) async -> Int32 {
        let cases = BenchmarkCatalog.select(options.patterns, options: options.catalogOptions)
        guard !cases.isEmpty else {
            FileHandle.standardError.write(Data(
                "no cases match \(options.patterns.joined(separator: ", "))\n".utf8
            ))
            return 1
        }

        let budget = options.budget
        let configuration = options.runConfiguration(budget: budget)
        let system = SystemProfile.capture()

        let startedAt = Date()
        if !options.quiet {
            print("audio-tool-bench - \(system.summaryLine)")
            print("\(cases.count) case(s), \(configuration.isolation) isolation, "
                + "\(configuration.buildConfiguration) build")
            if configuration.buildConfiguration != "release" {
                print("WARNING: debug build - these numbers measure the optimiser")
            }
            print("MLX caps: cache \(ByteFormat.bytes(budget.cacheLimitBytes)), "
                + "memory \(ByteFormat.bytes(budget.memoryLimitBytes))")
            print("")
            print(ReportRenderer.consoleHeader())
        }

        var results: [CaseResult] = []
        for (index, benchmarkCase) in cases.enumerated() {
            let result: CaseResult
            if options.inProcess {
                result = await runInProcess(benchmarkCase, options: options, budget: budget)
            } else {
                result = await runIsolated(benchmarkCase, options: options)
            }
            results.append(result)
            if !options.quiet {
                print(ReportRenderer.consoleLine(result))
            }

            if options.stopOnFailure, result.status == .failed {
                if !options.quiet { print("stopping: --stop-on-failure and \(result.id) failed") }
                break
            }

            // Cooldown after every case but the last. A benchmark that runs eleven
            // models back to back on a laptop is measuring the eleventh under the
            // thermal load of the first ten.
            let isLast = index == cases.count - 1
            if !isLast, options.cooldownSeconds > 0 {
                try? await Task.sleep(for: .seconds(options.cooldownSeconds))
            }
        }

        var report = BenchmarkReport(
            runID: UUID().uuidString,
            tag: options.tag,
            startedAt: startedAt,
            finishedAt: Date(),
            system: system,
            configuration: configuration,
            results: results
        )
        // Applied to the finished report rather than to the profile up front, so
        // it also reaches the input path and the failure messages - both of which
        // carry a home directory and so a user name.
        if options.redactHost {
            report = Redaction.apply(to: report)
        }

        do {
            let written = try write(report, to: options.outputDirectory)
            print("")
            print("report: \(written.json.path)")
            print("summary: \(written.markdown.path)")
        } catch {
            FileHandle.standardError.write(Data("could not write report: \(error)\n".utf8))
            return 1
        }

        return results.contains { $0.status == .failed } ? 2 : 0
    }

    /// Download and load every selected model, so the measured run is not a
    /// download benchmark. Reports rather than writes a report file.
    static func prefetchAll(_ options: BenchmarkOptions) async -> Int32 {
        let cases = BenchmarkCatalog.select(options.patterns, options: options.catalogOptions)
        let budget = options.budget
        print("prefetching \(cases.count) case(s)")
        var failures = 0
        for benchmarkCase in cases {
            let result = options.inProcess
                ? await CaseRunner.prefetch(benchmarkCase, budget: budget)
                : await runIsolated(benchmarkCase, options: options)
            if result.status == .failed { failures += 1 }
            let status = result.status.rawValue.padding(toLength: 10, withPad: " ", startingAt: 0)
            let name = result.id.padding(toLength: 36, withPad: " ", startingAt: 0)
            print("\(status)\(name)\(result.message ?? "")")
        }
        return failures == 0 ? 0 : 2
    }

    // MARK: - In-process

    private static func runInProcess(
        _ benchmarkCase: BenchmarkCase,
        options: BenchmarkOptions,
        budget: MemoryBudget
    ) async -> CaseResult {
        let configuration = options.runConfiguration(budget: budget)
        if options.prefetchOnly {
            return await CaseRunner.prefetch(benchmarkCase, budget: budget)
        }
        return await CaseRunner.run(
            benchmarkCase,
            configuration: configuration,
            inputSource: options.inputSource,
            budget: budget
        )
    }

    // MARK: - Child process

    private static func runIsolated(
        _ benchmarkCase: BenchmarkCase,
        options: BenchmarkOptions
    ) async -> CaseResult {

        func failure(_ message: String) -> CaseResult {
            CaseResult(
                id: benchmarkCase.id,
                label: benchmarkCase.label,
                category: benchmarkCase.category,
                backend: benchmarkCase.backend,
                status: .failed,
                message: message
            )
        }

        guard let executable = Bundle.main.executableURL else {
            return failure("cannot locate this executable to re-run it per case")
        }

        let resultURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("audio-tool-bench-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: resultURL) }

        let process = Process()
        process.executableURL = executable
        process.arguments = options.childArguments(
            caseID: benchmarkCase.id,
            resultPath: resultURL.path
        )
        // Same working directory, so a relative --input resolves the same way it
        // would have in the parent.
        process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        var environment = ProcessInfo.processInfo.environment
        environment["AUDIOTOOL_BENCH_CHILD"] = "1"
        process.environment = environment

        // Both streams are drained continuously. A child that writes more than a
        // pipe buffer's worth and is never read blocks forever in `write`, and the
        // symptom is a benchmark that hangs on one model and looks like the model.
        let output = StreamCollector()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            output.append(handle.availableData)
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            output.append(handle.availableData)
        }

        let watchdog = TimeoutFlag()
        let timeoutTask = Task {
            try await Task.sleep(for: .seconds(options.caseTimeoutSeconds))
            guard process.isRunning else { return }
            watchdog.trip()
            process.terminate()
            try await Task.sleep(for: .seconds(5))
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
            }
        }

        let exitCode: Int32 = await withCheckedContinuation { continuation in
            let resumed = OnceFlag()
            process.terminationHandler = { finished in
                guard resumed.claim() else { return }
                continuation.resume(returning: finished.terminationStatus)
            }
            do {
                try process.run()
            } catch {
                process.terminationHandler = nil
                if resumed.claim() {
                    output.append(Data("could not launch child: \(error)\n".utf8))
                    continuation.resume(returning: -1)
                }
            }
        }
        timeoutTask.cancel()
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        stderrPipe.fileHandleForReading.readabilityHandler = nil
        output.append(stdoutPipe.fileHandleForReading.availableData)
        output.append(stderrPipe.fileHandleForReading.availableData)

        // The child's own JSON is the authority when it exists: it was written
        // before exit and carries the measurements. Only when it is missing does
        // the exit code have to stand in for a result.
        if let data = try? Data(contentsOf: resultURL),
           let decoded = try? BenchmarkCoding.decoder().decode(CaseResult.self, from: data) {
            return decoded
        }

        if watchdog.tripped {
            return failure(String(
                format: "timed out after %.0f s and was killed", options.caseTimeoutSeconds
            ))
        }
        let tail = output.tail(lines: 12)
        return failure(
            "child exited \(exitCode) without writing a result"
                + (tail.isEmpty ? "" : "\n\(tail)")
        )
    }

    // MARK: - Writing

    private static func write(
        _ report: BenchmarkReport,
        to directory: String
    ) throws -> (json: URL, markdown: URL) {
        let base = URL(fileURLWithPath: directory)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        formatter.timeZone = TimeZone(identifier: "UTC")
        // The machine is in the filename because the whole point is collecting
        // these from several machines into one directory.
        let machine = report.system.hardwareModel
            .replacingOccurrences(of: ",", with: "-")
            .replacingOccurrences(of: " ", with: "-")
        // The run id is in the name because the timestamp is only second-accurate:
        // two short runs, or two started in parallel, would otherwise land on the
        // same name and the second would overwrite the first without a word. It
        // also ties the file back to `runID` inside it.
        let shortID = report.runID.prefix(8).lowercased()
        let stem = "bench-\(machine)-\(formatter.string(from: report.startedAt))-\(shortID)"

        let jsonURL = base.appendingPathComponent("\(stem).json")
        let markdownURL = base.appendingPathComponent("\(stem).md")
        try BenchmarkCoding.encoder().encode(report).write(to: jsonURL)
        try Data(ReportRenderer.markdown(report).utf8).write(to: markdownURL)
        return (jsonURL, markdownURL)
    }
}

// MARK: - Small concurrency helpers

/// Accumulates a child's stdout and stderr from the pipe reader threads.
///
/// `readabilityHandler` fires on a Dispatch-owned queue, so the buffer needs a
/// lock; `@unchecked Sendable` is the accurate label for that rather than a
/// concession.
final class StreamCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = Data()

    func append(_ data: Data) {
        guard !data.isEmpty else { return }
        lock.lock()
        buffer.append(data)
        lock.unlock()
    }

    /// The last `lines` lines, for a failure message. A child that died on a stack
    /// trace produces megabytes; the useful part is at the end.
    func tail(lines: Int) -> String {
        lock.lock()
        let text = String(decoding: buffer, as: UTF8.self)
        lock.unlock()
        let all = text.split(separator: "\n", omittingEmptySubsequences: false)
        return all.suffix(lines).joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// One-shot latch, so a continuation cannot be resumed twice when a launch fails
/// after the termination handler is installed.
final class OnceFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var used = false

    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if used { return false }
        used = true
        return true
    }
}

final class TimeoutFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    func trip() {
        lock.lock()
        value = true
        lock.unlock()
    }

    var tripped: Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}
