//
//  ReviewFixTests.swift
//  AudioTool
//
//  Regression tests for defects found in review
//

import AVFoundation
import Foundation
import Testing
@testable import AudioTool
@testable import AudioToolCore

// MARK: - Saving

@Suite("Audio file writing")
struct AudioFileWritingTests {

    private func temporaryURL(_ ext: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("audiotool-write-\(UUID().uuidString)")
            .appendingPathExtension(ext)
    }

    @Test("Stereo is written as two channels, not twice the duration")
    func stereoRoundTrip() async throws {
        // Interleaved L/R: left is +0.5 throughout, right is -0.5.
        let frames = 1000
        var samples: [Float] = []
        for _ in 0..<frames {
            samples.append(0.5)
            samples.append(-0.5)
        }
        let buffer = AudioBuffer(samples: samples, sampleRate: 48000, channels: 2)

        let url = temporaryURL("wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try await AudioEngine().saveAudio(buffer, to: url, format: .wav)

        let file = try AVAudioFile(forReading: url)
        #expect(file.fileFormat.channelCount == 2)
        #expect(file.length == AVAudioFramePosition(frames))
        // The bug this covers: a mono file of 2000 frames, alternating +0.5/-0.5.
        #expect(file.fileFormat.sampleRate == 48000)
    }

    @Test("Mono round-trips its samples")
    func monoRoundTrip() async throws {
        let samples = (0..<500).map { Float($0) / 500.0 - 0.5 }
        let buffer = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)

        let url = temporaryURL("wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try await AudioEngine().saveAudio(buffer, to: url, format: .wav)

        let file = try AVAudioFile(forReading: url)
        #expect(file.fileFormat.channelCount == 1)
        #expect(file.length == AVAudioFramePosition(samples.count))

        let readBuffer = try #require(
            AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
        )
        try file.read(into: readBuffer)
        let channel = try #require(readBuffer.floatChannelData?[0])
        for index in stride(from: 0, to: samples.count, by: 50) {
            #expect(abs(channel[index] - samples[index]) < 1e-5)
        }
    }

    @Test("FLAC produces a FLAC file rather than WAV bytes with a .flac name")
    func flacIsFLAC() async throws {
        let buffer = AudioBuffer.sine(frequency: 440, duration: 0.1, sampleRate: 48000)

        let url = temporaryURL("flac")
        defer { try? FileManager.default.removeItem(at: url) }
        try await AudioEngine().saveAudio(buffer, to: url, format: .flac)

        let file = try AVAudioFile(forReading: url)
        #expect(file.fileFormat.streamDescription.pointee.mFormatID == kAudioFormatFLAC)
    }

    @Test("M4A produces an AAC file")
    func m4aIsAAC() async throws {
        let buffer = AudioBuffer.sine(frequency: 440, duration: 0.1, sampleRate: 48000)

        let url = temporaryURL("m4a")
        defer { try? FileManager.default.removeItem(at: url) }
        try await AudioEngine().saveAudio(buffer, to: url, format: .m4a)

        let file = try AVAudioFile(forReading: url)
        #expect(file.fileFormat.streamDescription.pointee.mFormatID == kAudioFormatMPEG4AAC)
    }

    @Test("MP3 is refused instead of silently written as WAV")
    func mp3Throws() async throws {
        let buffer = AudioBuffer.sine(frequency: 440, duration: 0.1, sampleRate: 48000)
        let url = temporaryURL("mp3")
        defer { try? FileManager.default.removeItem(at: url) }

        await #expect(throws: AudioToolError.self) {
            try await AudioEngine().saveAudio(buffer, to: url, format: .mp3)
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }
}

// MARK: - Numeric guards

@Suite("AudioBuffer numeric guards")
struct AudioBufferNumericGuardTests {

    @Test("Non-finite durations produce an empty buffer instead of trapping")
    func nonFiniteDurations() {
        #expect(AudioBuffer.silence(duration: .nan, sampleRate: 16000).samples.isEmpty)
        #expect(AudioBuffer.silence(duration: .infinity, sampleRate: 16000).samples.isEmpty)
        #expect(AudioBuffer.silence(duration: -1, sampleRate: 16000).samples.isEmpty)
        #expect(AudioBuffer.sine(frequency: 440, duration: .nan, sampleRate: 16000).samples.isEmpty)
        #expect(AudioBuffer.noise(duration: -.infinity, sampleRate: 16000).samples.isEmpty)
    }

    @Test("A finite duration is unaffected")
    func finiteDuration() {
        let buffer = AudioBuffer.silence(duration: 0.5, sampleRate: 16000, channels: 2)
        #expect(buffer.frameCount == 8000)
        #expect(buffer.samples.count == 16000)
    }

    @Test("A non-finite mix offset is treated as zero")
    func nonFiniteMixOffset() {
        let a = AudioBuffer(samples: [1, 1, 1, 1], sampleRate: 16000)
        let b = AudioBuffer(samples: [1, 1, 1, 1], sampleRate: 16000)

        let mixed = a.mixing(with: b, at: .nan)
        #expect(mixed.samples == [2, 2, 2, 2])
    }

    @Test("A finite mix offset still shifts")
    func finiteMixOffset() {
        let a = AudioBuffer(samples: [1, 1], sampleRate: 4)
        let b = AudioBuffer(samples: [1, 1], sampleRate: 4)

        let mixed = a.mixing(with: b, at: 0.5)  // two samples at 4 Hz
        #expect(mixed.samples == [1, 1, 1, 1])
    }
}

// MARK: - Pipeline validation

@Suite("Pipeline stage validation")
struct PipelineStageValidationTests {

    @Test("An unsupported speaker count fails instead of quietly separating two")
    func unsupportedSpeakerCount() async throws {
        let separator = MockSeparator(outputCount: 2)
        let engine = AudioEngine(
            configuration: .default,
            separator: (.mossformerWhamr, separator)
        )

        for count in [0, 1, 4, -2] {
            await #expect(throws: AudioToolError.self) {
                _ = try await engine.pipeline()
                    .separate(speakers: count)
                    .process(audio: AudioBuffer.silence(duration: 0.1, sampleRate: 16000))
            }
        }
        #expect(separator.separateCallCount == 0)
    }
}

// MARK: - Nested pipelines

@Suite("Nested pipeline context")
struct NestedPipelineContextTests {

    @Test("useOriginal inside a conditional branch gets the original audio")
    func useOriginalInsideBranch() async throws {
        let enhancer = MockEnhancer()
        enhancer.scaleFactor = 0.5
        let separator = MockSeparator(outputCount: 2)

        let engine = AudioEngine(
            configuration: .default,
            enhancer: (.mossformerSE16k, enhancer),
            separator: (.mossformerWhamr, separator)
        )

        let input = AudioBuffer(
            samples: [Float](repeating: 1.0, count: 16000),
            sampleRate: 16000
        )

        _ = try await engine.pipeline()
            .enhance(.mossformerSE16k)
            .conditionally({ _ in true }) {
                $0.separate(speakers: 2, useOriginal: true)
            }
            .process(audio: input)

        let separatorInput = try #require(separator.receivedInputs.first)
        // 1.0 is the original; 0.5 would be the enhanced audio that `useOriginal`
        // exists to bypass. The nested run used to reset `originalAudio` to whatever
        // it was handed, which is the enhanced buffer.
        #expect(abs((separatorInput.samples.first ?? 0) - 1.0) < 1e-6)
    }

    @Test("An upscale inside a branch survives the parent's edge conversion")
    func branchUpscaleSetsOutputRate() async throws {
        let engine = AudioEngine(configuration: .default, upscaler: MockUpscaler())

        let result = try await engine.pipeline()
            .conditionally({ _ in true }) { $0.upscale() }
            .process(audio: .silence(duration: 0.5, sampleRate: 16000))

        // Without the rate travelling up, the parent's target is still 16 kHz and the
        // extra bandwidth is resampled straight back out.
        #expect(result.audio?.sampleRate == 48000)
    }

    @Test("An explicit output rate outranks a branch upscale")
    func explicitOutputRateWinsOverBranchUpscale() async throws {
        let engine = AudioEngine(configuration: .default, upscaler: MockUpscaler())

        var pipeline = engine.pipeline()
        pipeline = pipeline.conditionally({ _ in true }) { $0.upscale() }

        let result = try await engine.executePipeline(
            pipeline,
            audio: .silence(duration: 0.5, sampleRate: 16000),
            outputSampleRate: 24000
        )

        // The caller named a rate; a nested upscale does not get to overrule it, the
        // same way a direct `.upscale` stage does not.
        #expect(result.audio?.sampleRate == 24000)
    }

    @Test("The output rate follows the parallel branch whose audio wins")
    func parallelOutputRateFollowsWinningBranch() async throws {
        let enhancer = MockEnhancer()
        let engine = AudioEngine(
            configuration: .default,
            enhancer: (.mossformerSE16k, enhancer),
            upscaler: MockUpscaler()
        )

        let result = try await engine.pipeline()
            .parallel {
                [
                    PipelineBuilder(voice: engine).upscale(),
                    PipelineBuilder(voice: engine).enhance(.mossformerSE16k),
                ]
            }
            .process(audio: .silence(duration: 0.5, sampleRate: 16000))

        // Merge is last-branch-wins, so the enhanced 16 kHz audio is what comes back.
        // Taking the rate from the discarded upscale branch would resample it up to
        // 48 kHz - a rate its content does not justify.
        #expect(result.audio?.sampleRate == 16000)
    }

    @Test("A parallel upscale that does win the audio sets the rate")
    func parallelUpscaleWinsWhenItProducesTheAudio() async throws {
        let engine = AudioEngine(
            configuration: .default,
            enhancer: (.mossformerSE16k, MockEnhancer()),
            upscaler: MockUpscaler()
        )

        let result = try await engine.pipeline()
            .parallel {
                [
                    PipelineBuilder(voice: engine).enhance(.mossformerSE16k),
                    PipelineBuilder(voice: engine).upscale(),
                ]
            }
            .process(audio: .silence(duration: 0.5, sampleRate: 16000))

        #expect(result.audio?.sampleRate == 48000)
    }

    @Test("forEach tracks come back at the pipeline's output rate")
    func forEachTracksAreConvertedAtTheEdge() async throws {
        let engine = AudioEngine(
            configuration: .default,
            enhancer: (.mossformerSE16k, MockEnhancer()),
            separator: (.mossformerWhamr, MockSeparator(outputCount: 2))
        )

        // 48 kHz in, separated and enhanced by 16 kHz providers.
        let result = try await engine.pipeline()
            .separate(speakers: 2)
            .forEach { $0.enhance(.mossformerSE16k) }
            .process(audio: .silence(duration: 0.5, sampleRate: 48000))

        // Each track pipeline is a nested run and so skips its own edge conversion;
        // the single conversion at the real edge is what returns them to the caller's
        // rate. Converting inside each track run as well is the compounding this
        // pipeline is built to avoid - the tracks would go 48 -> 16 -> 48 -> 48.
        let tracks = try #require(result.separatedTracks)
        #expect(tracks.count == 2)
        for track in tracks {
            #expect(track.sampleRate == 48000)
        }
    }

    @Test("A branch inherits the analysis produced before it")
    func analysisReachesNestedBranch() async throws {
        let diarization = MockDiarization()
        diarization.mockTimeline = SpeakerTimeline(segments: [
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 3), speakerID: SpeakerID(0), confidence: 0.9),
            DiarizedSegment(timeRange: TimeRange(start: 2, end: 5), speakerID: SpeakerID(1), confidence: 0.85),
        ])
        let separator = MockSeparator(outputCount: 2)

        let engine = AudioEngine(
            configuration: .default,
            vad: MockVAD(),
            diarization: diarization,
            separator: (.mossformerWhamr, separator)
        )

        // The inner condition runs inside the outer branch's own pipeline. That
        // pipeline started with `analysis: nil`, so this used to be false however
        // many speakers were diarized - which is exactly the shape of
        // `separateOverlappingSpeakers`, whose three-speaker branch is nested.
        _ = try await engine.pipeline()
            .analyze()
            .conditionally({ _ in true }) { branch in
                branch.conditionally({ context in
                    (context.analysis?.speakers.maxOverlappingSpeakers ?? 0) >= 2
                }) {
                    $0.separate(speakers: 2)
                }
            }
            .process(audio: .silence(duration: 5.0, sampleRate: 16000))

        #expect(separator.separateCallCount == 1)
    }
}

// MARK: - Residency replacement

@Suite("Same-identifier replacement")
struct ResidencyReplacementTests {

    @Test("A failed replacement gives the previous provider back")
    func failedReplacementRestoresPrevious() async throws {
        let manager = ModelResidency(memoryLimitBytes: 1_000_000_000)
        let original = MockManagedModel(modelId: "shared_id", memoryBytes: 100_000_000)
        let replacement = MockManagedModel(
            modelId: "shared_id",
            memoryBytes: 100_000_000,
            failLoading: true
        )

        let lease = try await manager.beginUse(original)
        await manager.endUse(lease)
        #expect(await original.getIsLoaded())

        // Replacement unloads the resident first - two copies of the same model is
        // what the budget exists to prevent - so a failure here used to leave the
        // caller's still-registered provider unloaded.
        await #expect(throws: (any Error).self) {
            _ = try await manager.beginUse(replacement)
        }

        #expect(await original.getIsLoaded())
        #expect(await manager.loadedModelIds.contains("shared_id"))

        // And it is usable without a reload.
        let loadsBefore = await original.getLoadCallCount()
        let second = try await manager.beginUse(original)
        await manager.endUse(second)
        #expect(await original.getLoadCallCount() == loadsBefore)
    }
}

// MARK: - Load coalescing

@Suite("Model load gate")
struct ModelLoadGateTests {

    private actor Counter {
        private(set) var starts = 0
        private(set) var concurrent = 0
        private(set) var peakConcurrent = 0

        func record() { starts += 1 }

        func enter() {
            starts += 1
            concurrent += 1
            peakConcurrent = max(peakConcurrent, concurrent)
        }

        func leave() { concurrent -= 1 }
    }

    @Test("Concurrent callers share one load")
    func concurrentLoadsShareOne() async throws {
        let gate = ModelLoadGate()
        let counter = Counter()

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<8 {
                group.addTask {
                    try? await gate.run {
                        await counter.record()
                        try await Task.sleep(for: .milliseconds(20))
                    }
                }
            }
        }

        #expect(await counter.starts == 1)
    }

    @Test("A failed load is retried by the next caller")
    func failedLoadIsForgotten() async throws {
        struct Failure: Error {}
        let gate = ModelLoadGate()
        let counter = Counter()

        for _ in 0..<3 {
            _ = try? await gate.run {
                await counter.record()
                throw Failure()
            }
        }

        #expect(await counter.starts == 3)
    }

    @Test("Cancelling forgets a completed load, so the next call reloads")
    func cancelForgetsCompletedLoad() async throws {
        let gate = ModelLoadGate()
        let counter = Counter()

        try await gate.run { await counter.record() }
        try await gate.run { await counter.record() }
        #expect(await counter.starts == 1)

        await gate.cancelAndDrain()
        try await gate.run { await counter.record() }
        #expect(await counter.starts == 2)
    }

    @Test("Cancelling the last waiter cancels the load it was waiting for")
    func lastWaiterLeavingCancelsTheLoad() async throws {
        let gate = ModelLoadGate()
        let started = AsyncStream.makeStream(of: Void.self)
        let observed = AsyncStream.makeStream(of: Bool.self)

        let caller = Task {
            try await gate.run {
                started.continuation.yield()
                started.continuation.finish()
                // A real load's download or allocation, which must not outlive the
                // last caller that wanted it.
                do {
                    try await Task.sleep(for: .seconds(10))
                    observed.continuation.yield(false)
                } catch {
                    observed.continuation.yield(true)
                }
                observed.continuation.finish()
            }
        }

        var startIterator = started.stream.makeAsyncIterator()
        _ = await startIterator.next()
        caller.cancel()

        // The caller returns rather than waiting out the abandoned load.
        await #expect(throws: CancellationError.self) { try await caller.value }

        var observedIterator = observed.stream.makeAsyncIterator()
        #expect(await observedIterator.next() == true)
        #expect(gate.hasLoaded == false)
    }

    @Test("A cancelled caller does not cancel a load others are still waiting for")
    func remainingWaitersKeepTheLoad() async throws {
        let gate = ModelLoadGate()
        let counter = Counter()
        let started = AsyncStream.makeStream(of: Void.self)
        let release = AsyncStream.makeStream(of: Void.self)

        let keeper = Task {
            try await gate.run {
                await counter.record()
                started.continuation.yield()
                started.continuation.finish()
                var iterator = release.stream.makeAsyncIterator()
                _ = await iterator.next()
            }
        }

        var startIterator = started.stream.makeAsyncIterator()
        _ = await startIterator.next()

        let leaver = Task { try await gate.run { await counter.record() } }
        // Give the second caller time to join the in-flight load before leaving.
        try await Task.sleep(for: .milliseconds(50))
        leaver.cancel()
        await #expect(throws: CancellationError.self) { try await leaver.value }

        release.continuation.yield()
        release.continuation.finish()
        try await keeper.value

        #expect(await counter.starts == 1)
        #expect(gate.hasLoaded)
    }

    /// Holds the task under test so the completion hook can cancel it.
    private final class TaskBox: @unchecked Sendable {
        private let lock = NSLock()
        private var task: Task<Void, Error>?

        func set(_ task: Task<Void, Error>) { lock.withLock { self.task = task } }
        func cancel() { lock.withLock { task }?.cancel() }
    }

    @Test("Cancellation landing on a completed load does not undo it")
    func cancellationAfterSuccessDoesNotUnload() async throws {
        // The window is between `complete` publishing success - which takes the
        // waiters and zeroes the waiter count - and those waiters resuming. A
        // cancellation arriving there used to read "last waiter left", abandon the
        // task and clear the loaded flag, so a provider that had just loaded
        // successfully was reported unloaded and reloaded on next use.
        //
        // Requested rather than raced for: `didPublishCompletion` is exactly that
        // gap. Racing for it is not a test - a stress loop passed identically with
        // the bug reintroduced.
        let gate = ModelLoadGate()
        let box = TaskBox()
        gate.didPublishCompletion = { box.cancel() }

        let caller = Task { try await gate.run { try await Task.sleep(for: .milliseconds(50)) } }
        box.set(caller)

        try await caller.value
        #expect(gate.hasLoaded)

        // And the load is still there to be used, rather than started again.
        let reloads = Counter()
        try await gate.run { await reloads.record() }
        #expect(await reloads.starts == 0)
    }

    @Test("A load cannot slip in while the provider is resetting its state")
    func teardownBracketHoldsTheGateShut() async throws {
        let gate = ModelLoadGate()
        let counter = Counter()

        try await gate.run { await counter.record() }
        #expect(gate.hasLoaded)

        // What a provider's `unload()` does: drain, then clear its own state. The
        // gap between those two is the window a load must not be admitted into -
        // anything it published would be wiped by the state reset that follows.
        let teardown = await gate.beginTeardown()

        let racer = Task { try await gate.run { await counter.record() } }
        try await Task.sleep(for: .milliseconds(50))
        #expect(await counter.starts == 1, "a load started while the gate was shut")

        gate.endTeardown(teardown)
        try await racer.value

        // Reopened, and the load that was waiting ran once.
        #expect(await counter.starts == 2)
        #expect(gate.hasLoaded)
    }

    @Test("A caller waiting on a teardown can be cancelled out of the wait")
    func teardownWaiterIsCancellable() async throws {
        let gate = ModelLoadGate()
        let counter = Counter()
        let teardown = await gate.beginTeardown()

        let waiter = Task { try await gate.run { await counter.record() } }
        try await Task.sleep(for: .milliseconds(20))
        waiter.cancel()

        // Released now, rather than whenever the teardown happens to end.
        await #expect(throws: CancellationError.self) { try await waiter.value }
        #expect(await counter.starts == 0)

        gate.endTeardown(teardown)
    }

    @Test("A load that fails immediately reports its own error")
    func fastFailureKeepsItsError() async throws {
        struct WeightsCorrupt: Error, Equatable {}
        let gate = ModelLoadGate()

        // Fails before the caller can install its continuation, which used to leave
        // the caller holding `CancellationError` instead of this.
        await #expect(throws: WeightsCorrupt.self) {
            try await gate.run { throw WeightsCorrupt() }
        }
    }

    @Test("Every caller of a failed load is given the real error")
    func failureReachesEveryCaller() async throws {
        struct WeightsCorrupt: Error {}
        let gate = ModelLoadGate()
        let attempts = Counter()

        // Eight callers on one load, so the result has to survive being read
        // repeatedly - and each of them wants the load's own error, not the
        // `CancellationError` a discarded result gets replaced by.
        let errors = await withTaskGroup(of: Bool.self) { group in
            for _ in 0..<8 {
                group.addTask {
                    do {
                        try await gate.run {
                            await attempts.record()
                            try await Task.sleep(for: .milliseconds(20))
                            throw WeightsCorrupt()
                        }
                        return false
                    } catch {
                        return error is WeightsCorrupt
                    }
                }
            }
            var real = 0
            for await wasReal in group where wasReal { real += 1 }
            return real
        }

        #expect(errors == 8)
        #expect(await attempts.starts == 1)

        // And the failure is not sticky: the gate is reusable.
        try await gate.run { await attempts.record() }
        #expect(await attempts.starts == 2)
    }

    @Test("A cancelled load is drained before another may start")
    func cancelledLoadIsDrainedBeforeRestart() async throws {
        let gate = ModelLoadGate()
        let running = Counter()
        let started = AsyncStream.makeStream(of: Void.self)

        // A load that ignores cancellation, as `MLModel.compileModel` and an LLM
        // container load effectively do: it keeps its allocations until it returns.
        let first = Task {
            try await gate.run {
                await running.enter()
                started.continuation.yield()
                started.continuation.finish()
                await Task.detached { try? await Task.sleep(for: .milliseconds(300)) }.value
                await running.leave()
            }
        }

        var iterator = started.stream.makeAsyncIterator()
        _ = await iterator.next()
        await gate.cancelAndDrain()
        _ = try? await first.value

        // The second load must not begin while the first is still winding down -
        // that is two of the same multi-gigabyte allocation at once.
        try await gate.run {
            #expect(await running.concurrent == 0)
            await running.enter()
            await running.leave()
        }

        #expect(await running.peakConcurrent == 1)
    }

    @Test("An in-flight load sees cancellation, so it can decline to publish")
    func cancelSignalsInFlightLoad() async throws {
        let gate = ModelLoadGate()
        let started = AsyncStream.makeStream(of: Void.self)
        let runner = Task {
            try await gate.run {
                started.continuation.yield()
                started.continuation.finish()
                try await Task.sleep(for: .seconds(5))
            }
        }

        var iterator = started.stream.makeAsyncIterator()
        _ = await iterator.next()
        await gate.cancelAndDrain()

        await #expect(throws: (any Error).self) {
            try await runner.value
        }
    }
}
