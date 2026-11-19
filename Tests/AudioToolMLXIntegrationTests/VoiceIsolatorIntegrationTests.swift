//
//  VoiceIsolatorIntegrationTests.swift
//  AudioToolMLXIntegrationTests
//
//  The two Voice Isolator providers: declared format, pins, and - given weights -
//  the streaming identities the realtime one promises.
//

import XCTest
import AudioToolCore
@testable import AudioToolMLX
import AudioToolTestSupport
import AudioUtils
import MLX

// MARK: - Hermetic

/// What the providers claim about themselves. No weights, no network, no GPU:
/// constructing a provider loads nothing.
final class VoiceIsolatorDeclarationTests: XCTestCase {

    func testBothProvidersDeclare48kHzMono() {
        let offline = VoiceIsolator48KProvider()
        let realtime = VoiceIsolatorRealtime48KProvider()

        for provider in [offline as any AudioProcessor, realtime as any AudioProcessor] {
            XCTAssertEqual(provider.sampleRate, 48000)
            XCTAssertEqual(provider.inputChannels, 1)
            XCTAssertEqual(provider.outputChannels, 1)
        }
    }

    /// The reason this is pinned rather than left at the default is in the doc
    /// comment on the property: an interpolating upsample leaves images above the
    /// source band at -25 to -32 dB, and this model scales energy that is present
    /// rather than deciding some of it is an artifact. Dropping to `.balanced` or
    /// `.auto` would feed it those images on every input that is not already
    /// 48 kHz, and nothing else in the package would notice.
    func testBothProvidersAskForABandLimitedResampler() {
        XCTAssertEqual(VoiceIsolator48KProvider().preferredResamplingQuality, .high)
        XCTAssertEqual(VoiceIsolatorRealtime48KProvider().preferredResamplingQuality, .high)
    }

    /// Every precision the providers advertise must be pinned by file hash.
    ///
    /// `ModelPinTests` sweeps the *catalog*, which is a different list: a provider
    /// can offer a precision the catalog does not list, and that one would then
    /// download from a mutable default branch. Kokoro has the same gap and the same
    /// kind of test; these two are the other models reachable by precision.
    func testEveryAdvertisedPrecisionIsPinned() {
        let repositories = [
            (VoiceIsolator48KProvider.repo, VoiceIsolator48KProvider.supportedPrecisions),
            (
                VoiceIsolatorRealtime48KProvider.repo,
                VoiceIsolatorRealtime48KProvider.supportedPrecisions
            ),
        ]

        for (repo, precisions) in repositories {
            let pin = ModelPins.pin(for: repo)
            XCTAssertNotNil(pin, "\(repo) is reachable from a provider but has no pin")
            for precision in precisions {
                XCTAssertNotNil(
                    pin?.fileHashes[precision.weightsFilename],
                    "\(repo)/\(precision.weightsFilename) is advertised but has no pinned hash"
                )
            }
        }
    }

    /// fp16 is the default here, against fp32 everywhere else in the package, and
    /// that is deliberate - it is what the port computes in upstream. A silent
    /// flip back to fp32 would double the download and change the output.
    func testDefaultPrecisionIsFP16() {
        XCTAssertEqual(ModelCatalog.shared.model(id: "voice_isolator_48k")?.defaultVariant?.quantization, .fp16)
        XCTAssertEqual(
            ModelCatalog.shared.model(id: "voice_isolator_realtime_48k")?.defaultVariant?.quantization,
            .fp16
        )
    }

    /// The realtime look-ahead is a trained position, not a dial.
    func testLookaheadChoicesAreTheTrainedOnes() {
        XCTAssertEqual(VoiceIsolatorRealtime48KProvider.supportedLookaheadFrames, [0, 2, 4])
    }

    /// A block size below one frame is refused at load, before any weights are
    /// read - otherwise `process` would work, since it clamps the block size up,
    /// and the error would wait for the first live stream.
    func testRealtimeRefusesABlockSizeBelowOneFrame() async {
        let provider = VoiceIsolatorRealtime48KProvider(
            weightsPath: "/nonexistent/model_fp16.safetensors",
            blockFrames: 0
        )
        do {
            try await provider.load()
            XCTFail("expected pipelineConfigurationInvalid")
        } catch let error as AudioToolError {
            guard case .pipelineConfigurationInvalid = error else {
                return XCTFail("expected pipelineConfigurationInvalid, got \(error)")
            }
        } catch {
            XCTFail("expected AudioToolError, got \(error)")
        }
    }
}

// MARK: - Model-backed

/// Needs real checkpoints. Point these at local files:
///
/// ```bash
/// AUDIOTOOL_VOICE_ISOLATOR_WEIGHTS=/path/VoiceIsolator_v1_48K_MLX/model_fp16.safetensors \
/// AUDIOTOOL_VOICE_ISOLATOR_REALTIME_WEIGHTS=/path/VoiceIsolator_v1_Realtime_48K_MLX/model_fp16.safetensors \
///   RUN_INTEGRATION_TESTS=1 swift test --filter VoiceIsolatorModelTests
/// ```
///
/// Both gates matter: the suite is gated on `RUN_INTEGRATION_TESTS` before it looks
/// at either path, so passing only the paths gives a green run in which everything
/// skipped.
final class VoiceIsolatorModelTests: IntegrationTestCase {

    private func offlineWeights() throws -> String {
        try weights("AUDIOTOOL_VOICE_ISOLATOR_WEIGHTS")
    }

    private func realtimeWeights() throws -> String {
        try weights("AUDIOTOOL_VOICE_ISOLATOR_REALTIME_WEIGHTS")
    }

    private func weights(_ variable: String) throws -> String {
        guard let path = ProcessInfo.processInfo.environment[variable],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(variable) to a VoiceIsolator .safetensors to run this suite")
        }
        return path
    }

    /// Uniform noise plus two tones, for tests that check formats, lengths,
    /// progress and session bookkeeping rather than sample values.
    ///
    /// Not for comparing outputs: tones and noise are what this model removes, so
    /// its output on this signal is close to silence. ``speech(from:seconds:)``
    /// is for that.
    private func testSignal(seconds: Double = 6.0, sampleRate: Int = 48000) -> AudioBuffer {
        let count = Int(seconds * Double(sampleRate))
        var generator = SystemRandomNumberGenerator()
        var samples = [Float](repeating: 0, count: count)
        for index in 0..<count {
            let t = Float(index) / Float(sampleRate)
            let voiceish = 0.3 * sinf(2 * .pi * 220 * t) + 0.15 * sinf(2 * .pi * 440 * t)
            let noise = Float.random(in: -0.05...0.05, using: &generator)
            samples[index] = voiceish + noise
        }
        return AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
    }

    /// Long enough for a realtime session to change attention lanes.
    ///
    /// A session's attention never looks back across a 600-frame (6 s) boundary:
    /// a second lane opens 200 frames before it, the two crossfade, and the new one
    /// takes over at the boundary. That swap is the most stateful thing a session
    /// does, and an input shorter than 6 s never reaches it.
    private let realtimeSeconds = 10.0

    /// Speech at 48 kHz: a slice of the tracked CC0 fixture `speech_long.wav`,
    /// three speakers in turn over 25.5 s at 22.05 kHz, resampled on load.
    ///
    /// For every test whose assertions are about sample values. The model's
    /// output on ``testSignal(seconds:sampleRate:)`` peaks around 3e-4, so an
    /// absolute tolerance of 1e-3 on it is looser than the whole output - an
    /// all-zero result would pass - and even a relative comparison there measures
    /// the model's residue rather than its work. On this slice it peaks around 0.8.
    ///
    /// Read in place from the FluidAudio suite's fixtures rather than copied into
    /// this target: it is tracked there, with its provenance in that directory's
    /// README.
    private func speech(from start: Double, seconds: Double) throws -> AudioBuffer {
        var url = URL(fileURLWithPath: #filePath)
        url.deleteLastPathComponent()
        url.deleteLastPathComponent()
        url.appendPathComponent("AudioToolFluidAudioTests/Fixtures/speech_long.wav")
        let loader = AudioLoader(config: AudioLoader.Configuration(targetSampleRate: 48000))
        let audio = try loader.loadMono(from: url)
        eval(audio)
        let samples = audio.asArray(Float.self)
        let first = Int(start * 48000)
        let last = min(samples.count, first + Int(seconds * 48000))
        XCTAssertEqual(last - first, Int(seconds * 48000), "speech_long.wav is shorter than the slice")
        return AudioBuffer(samples: Array(samples[first..<last]), sampleRate: 48000, channels: 1)
    }

    /// RMS of the difference over RMS of the reference: 1e-3 is an error 60 dB
    /// below the signal.
    ///
    /// The bound the streaming identities use, set from measurement. On this
    /// fixture a live session (4-frame blocks) differs from `process` (25-frame
    /// blocks) by 7-9e-5 at fp16 - rounding in a different reduction order - while
    /// two streams sharing one session, the bug the concurrency test pins,
    /// differ by 0.3 to 1.4. 1e-3 sits an order of magnitude from each.
    private let streamTolerance: Float = 1e-3

    private func relativeError(_ actual: [Float], _ reference: [Float]) -> Float {
        let count = min(actual.count, reference.count)
        var error: Float = 0
        var energy: Float = 0
        for index in 0..<count {
            let difference = actual[index] - reference[index]
            error += difference * difference
            energy += reference[index] * reference[index]
        }
        return energy > 0 ? (error / energy).squareRoot() : .infinity
    }

    /// A NaN on either side, or infinity on both, makes the result NaN rather than
    /// being skipped. `max` alone would skip it - `max(worst, .nan)` is `worst` -
    /// so non-finite output would read as agreement and pass a `== 0` check.
    private func maxAbsoluteDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
        var worst: Float = 0
        for index in 0..<min(lhs.count, rhs.count) {
            let difference = abs(lhs[index] - rhs[index])
            if difference.isNaN { return .nan }
            worst = max(worst, difference)
        }
        return worst
    }

    // MARK: Offline

    func testOfflinePreservesLengthAndProducesFiniteAudio() async throws {
        let provider = VoiceIsolator48KProvider(weightsPath: try offlineWeights())
        try await provider.load()

        let input = try speech(from: 0, seconds: 6)
        let output = try await provider.process(input)

        XCTAssertEqual(output.sampleRate, 48000)
        XCTAssertEqual(output.channels, 1)
        XCTAssertEqual(output.samples.count, input.samples.count)
        XCTAssertTrue(output.samples.allSatisfy { $0.isFinite })
        // Speech in, so the voice comes out at speech level: about 0.8 here.
        XCTAssertGreaterThan(output.samples.map { abs($0) }.max() ?? 0, 0.05,
                             "the model returned (near) silence for speech")

        await provider.unload()
    }

    /// The pair must reconstruct the input exactly, because the background is
    /// defined as the subtraction rather than produced by a second forward pass.
    func testOfflineBackgroundAndVoiceSumToTheInput() async throws {
        let provider = VoiceIsolator48KProvider(weightsPath: try offlineWeights())
        try await provider.load()

        let input = testSignal(seconds: 3.0)
        let result = try await provider.processWithBackground(input)

        XCTAssertEqual(result.enhanced.samples.count, input.samples.count)
        XCTAssertEqual(result.background.samples.count, input.samples.count)

        var reconstructed = [Float](repeating: 0, count: input.samples.count)
        for index in 0..<input.samples.count {
            reconstructed[index] = result.enhanced.samples[index] + result.background.samples[index]
        }
        XCTAssertLessThan(maxAbsoluteDifference(reconstructed, input.samples), 1e-6)

        await provider.unload()
    }

    /// `process` validates rather than resampling, like every other provider.
    func testOfflineRejectsAnUnexpectedSampleRate() async throws {
        let provider = VoiceIsolator48KProvider(weightsPath: try offlineWeights())
        try await provider.load()

        let input = testSignal(seconds: 1.0, sampleRate: 16000)
        do {
            _ = try await provider.process(input)
            XCTFail("expected sampleRateMismatch")
        } catch let error as AudioToolError {
            guard case .sampleRateMismatch(let expected, let found) = error else {
                return XCTFail("expected sampleRateMismatch, got \(error)")
            }
            XCTAssertEqual(expected, 48000)
            XCTAssertEqual(found, 16000)
        }

        await provider.unload()
    }

    /// `isolate` is the opt-in path that does convert, with the model's own filter.
    func testOfflineIsolateAcceptsAnotherRateAndReturnsIt() async throws {
        let provider = VoiceIsolator48KProvider(weightsPath: try offlineWeights())
        try await provider.load()

        let input = testSignal(seconds: 2.0, sampleRate: 16000)
        let output = try await provider.isolate(input)

        XCTAssertEqual(output.sampleRate, 16000)
        XCTAssertEqual(output.samples.count, input.samples.count)
        XCTAssertTrue(output.samples.allSatisfy { $0.isFinite })

        await provider.unload()
    }

    func testOfflineReportsProgressEndingAtOneHundred() async throws {
        let provider = VoiceIsolator48KProvider(weightsPath: try offlineWeights())
        try await provider.load()

        let recorder = ProgressRecorder()
        _ = try await provider.process(testSignal(seconds: 20.0)) { percent in
            await recorder.record(percent)
        }

        let reported = await recorder.values
        XCTAssertFalse(reported.isEmpty, "no progress was reported")
        XCTAssertEqual(reported.last ?? 0, 100.0, accuracy: 0.001,
                       "the final fraction must survive the bridge's buffering")
        XCTAssertTrue(reported.allSatisfy { $0 >= 0 && $0 <= 100.0001 })

        await provider.unload()
    }

    // MARK: Realtime

    func testRealtimeReportsNonZeroLatency() async throws {
        let provider = VoiceIsolatorRealtime48KProvider(weightsPath: try realtimeWeights())
        try await provider.load()

        let samples = try await provider.latencySamples()
        XCTAssertGreaterThan(samples, 0)
        // One analysis window is nFFT = 2048 at minimum, plus hop * lookahead.
        XCTAssertGreaterThanOrEqual(samples, 2048)

        await provider.unload()
    }

    /// `processStream` must agree with `process`: both drive one session over the
    /// whole buffer with the same block and piece sizes, so they are the same
    /// computation and must agree bit for bit, length included.
    ///
    /// This is the property that makes the realtime model testable at all. If the
    /// two diverged, one of them would be restarting the session or changing the
    /// block size behind the caller's back.
    func testRealtimeStreamAgreesWithWholeBufferProcessing() async throws {
        let provider = VoiceIsolatorRealtime48KProvider(weightsPath: try realtimeWeights())
        try await provider.load()

        let input = try speech(from: 0, seconds: realtimeSeconds)
        let whole = try await provider.process(input)

        var streamed: [Float] = []
        for try await piece in provider.processStream(input) {
            streamed.append(contentsOf: piece.samples)
        }

        // Exact, not within a hop: a session releases precisely as many samples as
        // it was given, and the engine's streaming path splices this output back
        // into the input's timeline by length.
        XCTAssertEqual(streamed.count, input.samples.count)
        let difference = maxAbsoluteDifference(streamed, whole.samples)
        XCTAssertEqual(difference, 0, "processStream and process differ by up to \(difference)")

        await provider.unload()
    }

    /// The live path: one session across several input buffers, which is the whole
    /// reason this provider exists. Pushing in uneven pieces must not change the
    /// result, because the session buffers what it cannot yet finalize.
    func testRealtimeLiveSessionCarriesStateAcrossBuffers() async throws {
        let provider = VoiceIsolatorRealtime48KProvider(weightsPath: try realtimeWeights())
        try await provider.load()

        let input = try speech(from: 0, seconds: realtimeSeconds)
        let whole = try await provider.process(input)

        // Deliberately uneven and not a multiple of the hop.
        let pieceSizes = [7001, 23_117, 48_000, 1_024, 60_000]
        let (buffers, continuation) = AsyncStream<AudioBuffer>.makeStream()
        var index = 0
        var sizeIndex = 0
        while index < input.samples.count {
            let size = pieceSizes[sizeIndex % pieceSizes.count]
            let end = min(input.samples.count, index + size)
            continuation.yield(
                AudioBuffer(
                    samples: Array(input.samples[index..<end]),
                    sampleRate: 48000,
                    channels: 1
                )
            )
            index = end
            sizeIndex += 1
        }
        continuation.finish()

        var live: [Float] = []
        for try await piece in provider.stream(buffers) {
            live.append(contentsOf: piece.samples)
        }

        XCTAssertEqual(live.count, input.samples.count)
        // Not bit for bit, unlike the `processStream` comparison: a live session
        // runs 4-frame blocks where `process` runs 25, and the library's claim
        // that the block size does not change the output holds at fp16 to within
        // rounding. See ``streamTolerance`` for what that measures.
        let error = relativeError(live, whole.samples)
        XCTAssertLessThan(error, streamTolerance,
                          "a live session must match whole-buffer processing; relative error was \(error)")

        await provider.unload()
    }

    /// Two live streams on one provider, pushing in alternation, must each get
    /// exactly what they would have got alone.
    ///
    /// The regression this pins: with one session slot per provider, the second
    /// `stream(_:)` replaced the first's session, both then pushed into the same
    /// one, and each caller got the two inputs mixed into its output. Two live
    /// sources on one provider is the natural way to use it - a second provider
    /// is a second copy of the weights - so this is not an exotic case.
    ///
    /// Alternation is forced rather than left to scheduling: each piece is fed and
    /// its output awaited before the other stream's next piece goes in, so both
    /// sessions are open and interleaved for the whole run. One-second pieces
    /// always finalize output - the session holds back only its latency, a few
    /// tens of milliseconds - so every await returns.
    func testRealtimeConcurrentStreamsKeepTheirOwnSessions() async throws {
        let provider = VoiceIsolatorRealtime48KProvider(weightsPath: try realtimeWeights())
        try await provider.load()

        let first = try speech(from: 0, seconds: realtimeSeconds)
        let second = try speech(from: 12, seconds: realtimeSeconds)
        let expectedFirst = try await provider.process(first).samples
        let expectedSecond = try await provider.process(second).samples

        let (firstInput, firstFeed) = AsyncStream<AudioBuffer>.makeStream()
        let (secondInput, secondFeed) = AsyncStream<AudioBuffer>.makeStream()
        var firstOutput = provider.stream(firstInput).makeAsyncIterator()
        var secondOutput = provider.stream(secondInput).makeAsyncIterator()

        func piece(_ signal: AudioBuffer, _ range: Range<Int>) -> AudioBuffer {
            AudioBuffer(samples: Array(signal.samples[range]), sampleRate: 48000, channels: 1)
        }

        var firstSamples: [Float] = []
        var secondSamples: [Float] = []
        let size = 48000
        var index = 0
        while index < first.samples.count {
            let range = index..<min(first.samples.count, index + size)
            firstFeed.yield(piece(first, range))
            firstSamples += try await firstOutput.next()?.samples ?? []
            secondFeed.yield(piece(second, range))
            secondSamples += try await secondOutput.next()?.samples ?? []
            index = range.upperBound
        }
        firstFeed.finish()
        secondFeed.finish()
        while let rest = try await firstOutput.next() { firstSamples += rest.samples }
        while let rest = try await secondOutput.next() { secondSamples += rest.samples }

        for (name, actual, expected) in [
            ("first", firstSamples, expectedFirst),
            ("second", secondSamples, expectedSecond),
        ] {
            XCTAssertEqual(actual.count, expected.count, name)
            let error = relativeError(actual, expected)
            XCTAssertLessThan(error, streamTolerance,
                              "the \(name) stream's output depends on the other's input; relative error was \(error)")
        }
        let open = try await waitForSessionsToClose(provider)
        XCTAssertEqual(open, 0, "a finished stream left its session open")

        await provider.unload()
    }

    /// Every way a live stream can end must close the session it opened.
    ///
    /// A session is a whole stream's model state - attention caches included -
    /// so one left behind is a leak per stream, and nothing else would ever free
    /// it short of unloading the provider.
    func testRealtimeStreamClosesItsSessionOnEveryExit() async throws {
        let provider = VoiceIsolatorRealtime48KProvider(weightsPath: try realtimeWeights())
        try await provider.load()

        let input = testSignal(seconds: 2.0)

        // Input ends normally.
        let (finished, finishedFeed) = AsyncStream<AudioBuffer>.makeStream()
        finishedFeed.yield(input)
        finishedFeed.finish()
        for try await _ in provider.stream(finished) {}
        var open = await provider.openSessionCount
        XCTAssertEqual(open, 0, "a stream that ran to the end left its session open")

        // A push fails validation.
        let (rejected, rejectedFeed) = AsyncStream<AudioBuffer>.makeStream()
        rejectedFeed.yield(input)
        rejectedFeed.yield(testSignal(seconds: 0.5, sampleRate: 16000))
        rejectedFeed.finish()
        do {
            for try await _ in provider.stream(rejected) {}
            XCTFail("expected sampleRateMismatch")
        } catch let error as AudioToolError {
            guard case .sampleRateMismatch = error else {
                return XCTFail("expected sampleRateMismatch, got \(error)")
            }
        }
        open = await provider.openSessionCount
        XCTAssertEqual(open, 0, "a stream that failed left its session open")

        // The consumer walks away while the input is still open.
        let (abandoned, abandonedFeed) = AsyncStream<AudioBuffer>.makeStream()
        abandonedFeed.yield(input)
        for try await _ in provider.stream(abandoned) { break }
        open = try await waitForSessionsToClose(provider)
        XCTAssertEqual(open, 0, "an abandoned stream left its session open")
        abandonedFeed.finish()

        await provider.unload()
    }

    /// NaN or infinity must be refused on every realtime entry point.
    ///
    /// Most of all on the live one: `IsolationStream.push` does not check, and the
    /// session's Mamba state is recurrent, so a NaN that got in would make every
    /// later output of that stream NaN. The stream ends with an error and closes
    /// its session instead.
    func testRealtimeRejectsNonFiniteInputOnEveryPath() async throws {
        let provider = VoiceIsolatorRealtime48KProvider(weightsPath: try realtimeWeights())
        try await provider.load()

        var poisoned = testSignal(seconds: 1.0).samples
        poisoned[poisoned.count / 2] = .nan
        let bad = AudioBuffer(samples: poisoned, sampleRate: 48000, channels: 1)

        func expectRejected(_ path: String, _ body: () async throws -> Void) async {
            do {
                try await body()
                XCTFail("\(path) accepted a NaN sample")
            } catch let error as AudioToolError {
                guard case .invalidAudioFormat = error else {
                    return XCTFail("\(path): expected invalidAudioFormat, got \(error)")
                }
            } catch {
                XCTFail("\(path): expected AudioToolError, got \(error)")
            }
        }

        await expectRejected("process") { _ = try await provider.process(bad) }
        await expectRejected("processStream") {
            for try await _ in provider.processStream(bad) {}
        }
        await expectRejected("stream") {
            let (input, feed) = AsyncStream<AudioBuffer>.makeStream()
            feed.yield(self.testSignal(seconds: 1.0))
            feed.yield(bad)
            feed.finish()
            for try await _ in provider.stream(input) {}
        }
        let open = await provider.openSessionCount
        XCTAssertEqual(open, 0, "the rejected stream left its session open")

        await provider.unload()
    }

    /// Unloading must end open streams with an error rather than let them finish
    /// as if a truncated signal were the whole one.
    func testRealtimeUnloadEndsOpenStreamsWithAnError() async throws {
        let provider = VoiceIsolatorRealtime48KProvider(weightsPath: try realtimeWeights())
        try await provider.load()

        let (input, feed) = AsyncStream<AudioBuffer>.makeStream()
        var output = provider.stream(input).makeAsyncIterator()
        feed.yield(testSignal(seconds: 1.0))
        _ = try await output.next()

        await provider.unload()
        feed.finish()

        do {
            while try await output.next() != nil {}
            XCTFail("expected modelNotLoaded after unload")
        } catch let error as AudioToolError {
            guard case .modelNotLoaded = error else {
                return XCTFail("expected modelNotLoaded, got \(error)")
            }
        }
    }

    /// Polls, because an abandoned stream closes its session on its own task after
    /// the consumer has already moved on. Returns the count it gave up at.
    private func waitForSessionsToClose(
        _ provider: VoiceIsolatorRealtime48KProvider
    ) async throws -> Int {
        for _ in 0..<200 {
            if await provider.openSessionCount == 0 { return 0 }
            try await Task.sleep(for: .milliseconds(25))
        }
        return await provider.openSessionCount
    }
}

/// Collects progress callbacks from whatever task the bridge reports on.
private actor ProgressRecorder {
    private(set) var values: [Double] = []

    func record(_ percent: Double) {
        values.append(percent)
    }
}
