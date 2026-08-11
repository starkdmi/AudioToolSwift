//
//  CaseRunner.swift
//  AudioToolBenchmark
//
//  Running exactly one case, and everything that has to be true around it.
//

import Foundation
import AudioToolCore
@preconcurrency import MLX

/// Runs a single case start to finish and returns its ``CaseResult``.
///
/// Never throws. A model that fails to load, times out, or produces nothing is a
/// result with `status == .failed`, because a run of eleven models that aborts on
/// the third has told you less than one that finishes and reports the third as
/// broken. The only thing that stops a run is the process dying, and the parent
/// handles that separately.
public enum CaseRunner {

    /// Phase order, and why it is this order:
    ///
    /// 1. Apply the caps *before* touching MLX, so no allocation predates them.
    /// 2. Clear and reset the MLX counters, so the numbers start from this case.
    /// 3. Read the baseline footprint after that, so the runtime's own memory is
    ///    baseline rather than being attributed to the model.
    /// 4. Load, timed.
    /// 5. One first inference, timed separately - it carries graph compilation.
    /// 6. Warm-ups, untimed.
    /// 7. `iterations` timed runs.
    ///
    /// Steps 5-7 exist as three phases rather than one because collapsing them
    /// produces an RTF that is mostly compilation on short inputs and mostly not on
    /// long ones, which is how the same model comes to have two published numbers.
    public static func run(
        _ benchmarkCase: BenchmarkCase,
        configuration: RunConfiguration,
        inputSource: BenchmarkInput.Source,
        budget: MemoryBudget,
        log: @Sendable (String) -> Void = { _ in }
    ) async -> CaseResult {

        func skeleton(status: CaseStatus, message: String?) -> CaseResult {
            CaseResult(
                id: benchmarkCase.id,
                label: benchmarkCase.label,
                category: benchmarkCase.category,
                backend: benchmarkCase.backend,
                status: status,
                message: message
            )
        }

        if let reason = benchmarkCase.unavailableReason() {
            log("skip \(benchmarkCase.id): \(reason)")
            return skeleton(status: .skipped, message: reason)
        }

        switch Headroom.check(estimatedBytes: benchmarkCase.estimatedMemoryBytes) {
        case .fits:
            break
        case .tight(let note):
            log("warning \(benchmarkCase.id): \(note)")
        case .insufficient(let note):
            log("skip \(benchmarkCase.id): \(note)")
            return skeleton(status: .skipped, message: note)
        }

        let caseBudget = budget.overridden(
            cache: benchmarkCase.gpuCacheLimitBytes,
            memory: benchmarkCase.gpuMemoryLimitBytes
        )
        caseBudget.apply()

        let input: AudioBuffer
        do {
            input = try BenchmarkInput.make(
                source: inputSource,
                seconds: configuration.inputSeconds,
                sampleRate: benchmarkCase.sampleRate,
                channels: benchmarkCase.inputChannels
            )
        } catch {
            return skeleton(status: .failed, message: "input: \(error)")
        }

        let thermalAtStart = HostConditions.thermalState
        let weightsPreCached = benchmarkCase.weightsPreCached()
        let weightsSource = benchmarkCase.weightsSource()
        let workload = benchmarkCase.makeWorkload()

        // A clean allocator and a zeroed peak, so nothing from a previous case in
        // the same process is charged to this one. Redundant under process
        // isolation and load-bearing under `--in-process`.
        GPU.clearCache()
        GPU.resetPeakMemory()

        let sampler = FootprintSampler()
        let baseline = TaskMemory.current()?.footprintBytes ?? 0
        let cpuStart = CPUTime.current()
        let clock = ContinuousClock()
        let wallStart = clock.now
        sampler.start()

        func abandon(_ stage: String, _ error: any Error) async -> CaseResult {
            sampler.stop()
            await workload.unload()
            GPU.clearCache()
            log("fail \(benchmarkCase.id): \(stage): \(error)")
            var result = skeleton(status: .failed, message: "\(stage): \(error)")
            result.weightsPreCached = weightsPreCached
            result.weightsSource = weightsSource
            result.inputSampleRate = benchmarkCase.sampleRate
            result.inputFrames = input.frameCount
            result.gpuCacheLimitBytes = caseBudget.cacheLimitBytes
            result.gpuMemoryLimitBytes = caseBudget.memoryLimitBytes
            result.environment = HostConditions.snapshot(start: thermalAtStart)
            return result
        }

        // MARK: Load

        let loadStart = clock.now
        do {
            try await workload.load()
        } catch {
            return await abandon("load", error)
        }
        let loadSeconds = seconds(from: loadStart, to: clock.now)
        let afterLoadFootprint = TaskMemory.current()?.footprintBytes ?? 0
        let mlxAfterLoad = GPU.snapshot()

        // Drops the load-time spike - conversion, unflattening, `verify: .all` -
        // so the run peak is not dominated by something no inference pays.
        //
        // It does not isolate the activations. `resetPeakMemory` zeroes the
        // high-water mark and frees nothing, so the weights are still active and
        // the next allocation puts the peak straight back to at least their size.
        // ``MemoryMetrics/mlxRunActivationsBytes`` subtracts them, and says it is
        // an estimate.
        GPU.resetPeakMemory()

        // MARK: First inference

        let firstStart = clock.now
        let firstOutput: WorkloadOutput
        do {
            firstOutput = try await workload.run(input)
        } catch {
            return await abandon("first inference", error)
        }
        let firstRunSeconds = seconds(from: firstStart, to: clock.now)

        // MARK: Warm-up

        for index in 0..<configuration.warmupIterations {
            do {
                _ = try await workload.run(input)
            } catch {
                return await abandon("warm-up \(index + 1)", error)
            }
        }

        // MARK: Timed iterations

        var iterationSeconds: [Double] = []
        iterationSeconds.reserveCapacity(configuration.iterations)
        var lastOutput = firstOutput
        for index in 0..<configuration.iterations {
            let start = clock.now
            do {
                lastOutput = try await workload.run(input)
            } catch {
                return await abandon("iteration \(index + 1)", error)
            }
            iterationSeconds.append(seconds(from: start, to: clock.now))
        }

        // MARK: Collect

        let mlxAfterRun = GPU.snapshot()
        let peak = sampler.stop()
        let wallSeconds = seconds(from: wallStart, to: clock.now)
        let cpu = CPUTime.current().since(cpuStart)
        let environment = HostConditions.snapshot(start: thermalAtStart)

        await workload.unload()
        GPU.clearCache()

        var result = skeleton(status: .completed, message: nil)
        result.inputSampleRate = input.sampleRate
        result.inputFrames = input.frameCount
        result.outputStreams = lastOutput.streams
        result.outputFrames = lastOutput.frames
        result.outputSampleRate = lastOutput.sampleRate
        result.weightsPreCached = weightsPreCached
        result.weightsSource = weightsSource
        result.gpuCacheLimitBytes = caseBudget.cacheLimitBytes
        result.gpuMemoryLimitBytes = caseBudget.memoryLimitBytes
        // Read back rather than assumed: a chunked path will have applied
        // MLXCachePolicy's own limits over the ones requested above.
        result.effectiveGpuCacheLimitBytes = GPU.cacheLimit
        result.effectiveGpuMemoryLimitBytes = GPU.memoryLimit
        // Synthesis ignores the buffer it is handed, so dividing by that buffer's
        // duration would report the length of a signal nothing looked at. See
        // ``RateBasis``.
        let ratedSeconds: Double
        switch benchmarkCase.rateBasis {
        case .input:
            ratedSeconds = input.duration
        case .output:
            ratedSeconds = lastOutput.sampleRate > 0
                ? Double(lastOutput.frames) / Double(lastOutput.sampleRate)
                : 0
        }
        result.timing = TimingMetrics(
            loadSeconds: loadSeconds,
            firstRunSeconds: firstRunSeconds,
            iterationSeconds: iterationSeconds,
            audioSeconds: ratedSeconds
        )
        result.memory = MemoryMetrics(
            baselineFootprintBytes: baseline,
            afterLoadFootprintBytes: afterLoadFootprint,
            peakFootprintBytes: peak.footprintBytes,
            peakResidentBytes: peak.residentBytes,
            mlxActiveAfterLoadBytes: mlxAfterLoad.activeMemory,
            mlxPeakDuringLoadBytes: mlxAfterLoad.peakMemory,
            mlxPeakDuringRunBytes: mlxAfterRun.peakMemory,
            mlxCacheAtEndBytes: mlxAfterRun.cacheMemory
        )
        result.cpu = CPUMetrics(
            userSeconds: cpu.user,
            systemSeconds: cpu.system,
            wallSeconds: wallSeconds
        )
        result.environment = environment

        // Output that is empty, or that arrives at an unexpected rate, is a
        // successful run of something that did not work. Reported as completed
        // with a note rather than silently tabulated as a fast model.
        if lastOutput.frames == 0 {
            result.status = .failed
            result.message = "model returned zero frames"
        }

        return result
    }

    /// Weights only: load, discard, report. Used by `--prefetch` so the measured
    /// run is not a download benchmark, and so the first run on a new machine
    /// fails on the network at a point where that is the obvious explanation.
    public static func prefetch(
        _ benchmarkCase: BenchmarkCase,
        budget: MemoryBudget,
        log: @Sendable (String) -> Void = { _ in }
    ) async -> CaseResult {
        var result = CaseResult(
            id: benchmarkCase.id,
            label: benchmarkCase.label,
            category: benchmarkCase.category,
            backend: benchmarkCase.backend,
            status: .completed
        )

        if let reason = benchmarkCase.unavailableReason() {
            result.status = .skipped
            result.message = reason
            log("skip \(benchmarkCase.id): \(reason)")
            return result
        }

        budget.overridden(
            cache: benchmarkCase.gpuCacheLimitBytes,
            memory: benchmarkCase.gpuMemoryLimitBytes
        ).apply()

        let workload = benchmarkCase.makeWorkload()
        let clock = ContinuousClock()
        let start = clock.now
        do {
            try await workload.load()
        } catch {
            result.status = .failed
            result.message = "load: \(error)"
            log("fail \(benchmarkCase.id): \(error)")
            return result
        }
        let elapsed = seconds(from: start, to: clock.now)
        await workload.unload()
        GPU.clearCache()

        result.weightsPreCached = benchmarkCase.weightsPreCached()
        result.weightsSource = benchmarkCase.weightsSource()
        result.message = String(format: "fetched and loaded in %.1f s", elapsed)
        log(String(format: "ok   %@ (%.1f s)", benchmarkCase.id, elapsed))
        return result
    }

    private static func seconds(
        from start: ContinuousClock.Instant,
        to end: ContinuousClock.Instant
    ) -> Double {
        let duration = end - start
        let components = duration.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
