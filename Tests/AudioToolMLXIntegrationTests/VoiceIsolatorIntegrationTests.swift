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

    /// Six seconds of band-limited noise plus two tones.
    ///
    /// Synthetic rather than a fixture because none of these tests is about
    /// enhancement quality - they are about lengths, identities between the three
    /// entry points, and format validation, all of which hold for any input. Long
    /// enough to cross the model's 8-second window boundary logic in the streaming
    /// paths while staying under it offline.
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

    private func maxAbsoluteDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
        var worst: Float = 0
        for index in 0..<min(lhs.count, rhs.count) {
            worst = max(worst, abs(lhs[index] - rhs[index]))
        }
        return worst
    }

    // MARK: Offline

    func testOfflinePreservesLengthAndProducesFiniteAudio() async throws {
        let provider = VoiceIsolator48KProvider(weightsPath: try offlineWeights())
        try await provider.load()

        let input = testSignal()
        let output = try await provider.process(input)

        XCTAssertEqual(output.sampleRate, 48000)
        XCTAssertEqual(output.channels, 1)
        XCTAssertEqual(output.samples.count, input.samples.count)
        XCTAssertTrue(output.samples.allSatisfy { $0.isFinite })
        XCTAssertGreaterThan(output.samples.map { abs($0) }.max() ?? 0, 1e-4,
                             "the model returned silence")

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
    /// whole buffer with the same block and piece sizes, so the only difference is
    /// that `process` fits the result to the input length.
    ///
    /// This is the property that makes the realtime model testable at all. If the
    /// two diverged, one of them would be restarting the session or changing the
    /// block size behind the caller's back.
    func testRealtimeStreamAgreesWithWholeBufferProcessing() async throws {
        let provider = VoiceIsolatorRealtime48KProvider(weightsPath: try realtimeWeights())
        try await provider.load()

        let input = testSignal(seconds: 4.0)
        let whole = try await provider.process(input)

        var streamed: [Float] = []
        for try await piece in provider.processStream(input) {
            streamed.append(contentsOf: piece.samples)
        }

        XCTAssertFalse(streamed.isEmpty)
        // `process` trims to the input length; the raw session emits whatever the
        // window alignment produced, which is within one hop either side.
        XCTAssertLessThanOrEqual(abs(streamed.count - whole.samples.count), 480)
        let difference = maxAbsoluteDifference(streamed, whole.samples)
        XCTAssertLessThan(difference, 1e-4, "max |difference| was \(difference)")

        await provider.unload()
    }

    /// The live path: one session across several input buffers, which is the whole
    /// reason this provider exists. Pushing in uneven pieces must not change the
    /// result, because the session buffers what it cannot yet finalize.
    func testRealtimeLiveSessionCarriesStateAcrossBuffers() async throws {
        let provider = VoiceIsolatorRealtime48KProvider(weightsPath: try realtimeWeights())
        try await provider.load()

        let input = testSignal(seconds: 4.0)
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

        XCTAssertFalse(live.isEmpty)
        XCTAssertLessThanOrEqual(abs(live.count - whole.samples.count), 480)
        // Looser than the `processStream` comparison on purpose: that one uses the
        // same block size as `process`, while a live session uses the smaller
        // `blockFrames` a latency-sensitive caller wants. The library's claim is
        // that the block size does not change the output, which at fp16 means to
        // within rounding rather than bit for bit.
        let difference = maxAbsoluteDifference(live, whole.samples)
        XCTAssertLessThan(difference, 1e-3,
                          "a live session must match whole-buffer processing; max |difference| was \(difference)")

        await provider.unload()
    }

    /// Ending a stream must leave the provider reusable rather than holding a
    /// half-drained session that the next stream would continue from.
    func testRealtimeSessionIsNotCarriedBetweenStreams() async throws {
        let provider = VoiceIsolatorRealtime48KProvider(weightsPath: try realtimeWeights())
        try await provider.load()

        let input = testSignal(seconds: 2.0)

        func runStream() async throws -> [Float] {
            var collected: [Float] = []
            for try await piece in provider.processStream(input) {
                collected.append(contentsOf: piece.samples)
            }
            return collected
        }

        let first = try await runStream()
        let second = try await runStream()

        XCTAssertEqual(first.count, second.count)
        XCTAssertLessThan(maxAbsoluteDifference(first, second), 1e-6,
                          "a second stream must start from a fresh session")

        await provider.unload()
    }
}

/// Collects progress callbacks from whatever task the bridge reports on.
private actor ProgressRecorder {
    private(set) var values: [Double] = []

    func record(_ percent: Double) {
        values.append(percent)
    }
}
