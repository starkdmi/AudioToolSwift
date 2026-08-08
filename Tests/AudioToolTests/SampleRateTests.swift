//
//  SampleRateTests.swift
//  AudioToolTests
//
//  Providers validate their rate; conversion happens once, at the edge
//

import Testing
import Foundation
@testable import AudioTool
@testable import AudioToolCore

@Suite("Sample rate handling")
struct SampleRateTests {

    private func tone(sampleRate: Int, seconds: Double = 1.0) -> AudioBuffer {
        let count = Int(Double(sampleRate) * seconds)
        let samples = (0..<count).map { i in
            sin(2 * Float.pi * 220 * Float(i) / Float(sampleRate)) * 0.5
        }
        return AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
    }

    // MARK: - Providers validate

    @Test("A provider rejects audio at the wrong rate")
    func providerRejectsWrongRate() async throws {
        let enhancer = MockEnhancer()  // 16 kHz
        await #expect(throws: AudioToolError.self) {
            _ = try await enhancer.process(tone(sampleRate: 48000))
        }
    }

    @Test("The mismatch error names both rates")
    func mismatchErrorIsSpecific() async throws {
        let enhancer = MockEnhancer()
        do {
            _ = try await enhancer.process(tone(sampleRate: 44100))
            Issue.record("expected a sampleRateMismatch")
        } catch let error as AudioToolError {
            guard case .sampleRateMismatch(let expected, let found) = error else {
                Issue.record("wrong error: \(error)")
                return
            }
            #expect(expected == 16000)
            #expect(found == 44100)
        }
    }

    @Test("A provider accepts audio at its own rate")
    func providerAcceptsMatchingRate() async throws {
        let enhancer = MockEnhancer()
        let output = try await enhancer.process(tone(sampleRate: 16000))
        #expect(output.sampleRate == 16000)
    }

    @Test("A mono provider rejects matching-rate stereo input")
    func providerRejectsWrongChannelCount() async throws {
        let enhancer = MockEnhancer()
        let stereo = AudioBuffer(
            samples: [0.25, -0.25, 0.5, -0.5],
            sampleRate: 16000,
            channels: 2
        )

        do {
            _ = try await enhancer.process(stereo)
            Issue.record("expected a channelCountMismatch")
        } catch let error as AudioToolError {
            guard case .channelCountMismatch(let expected, let found) = error else {
                Issue.record("wrong error: \(error)")
                return
            }
            #expect(expected == 1)
            #expect(found == 2)
        }
    }

    // MARK: - The facade converts, once

    @Test("Facade adapts input to the provider and restores the caller's rate")
    func facadeRoundTripsByDefault() async throws {
        let engine = AudioEngine()
        let enhancer = MockEnhancer()
        await engine.register(enhancer: enhancer, for: .mossformerSE16k)

        let output = try await engine.enhance(tone(sampleRate: 48000), model: .mossformerSE16k)

        // Exactly one buffer reached the provider, already at its rate.
        #expect(enhancer.receivedSampleRates == [16000])
        // And the caller got their rate back.
        #expect(output.sampleRate == 48000)
    }

    @Test("preservingSampleRate false leaves output at the model's rate")
    func facadeCanSkipTheReturnTrip() async throws {
        let engine = AudioEngine()
        let enhancer = MockEnhancer()
        await engine.register(enhancer: enhancer, for: .mossformerSE16k)

        let output = try await engine.enhance(
            tone(sampleRate: 48000),
            model: .mossformerSE16k,
            preservingSampleRate: false
        )

        #expect(enhancer.receivedSampleRates == [16000])
        // No upsample back to 48 kHz - which is the point when composing by hand.
        #expect(output.sampleRate == 16000)
    }

    @Test("Facade downmixes channels even when the sample rate already matches")
    func facadeAdaptsChannelsWithoutRateConversion() async throws {
        let engine = AudioEngine()
        let enhancer = MockEnhancer()
        enhancer.processDelay = .zero
        enhancer.scaleFactor = 1
        await engine.register(enhancer: enhancer, for: .mossformerSE16k)

        let input = AudioBuffer(
            samples: [1, 0, 0.5, -0.5, -1, 1],
            sampleRate: 16000,
            channels: 2
        )
        let output = try await engine.enhance(input, model: .mossformerSE16k)

        #expect(output.channels == 1)
        #expect(output.samples == [0.5, 0, 0])
    }

    // MARK: - The pipeline converts once, not once per stage

    /// The regression this fix exists for. A 16 kHz enhancement stage on 48 kHz input
    /// used to run 48 -> 16 -> 48; a following stage then pulled it back down again.
    /// The provider must see 16 kHz exactly once, and the result must come back at
    /// 48 kHz having been converted a single time on the way out.
    @Test("Pipeline enhancement converts down once and back once")
    func pipelineConvertsOnce() async throws {
        let engine = AudioEngine()
        let enhancer = MockEnhancer()
        await engine.register(enhancer: enhancer, for: .mossformerSE16k)

        let result = try await engine.pipeline()
            .enhance(.mossformerSE16k)
            .process(audio: tone(sampleRate: 48000))

        #expect(enhancer.receivedSampleRates == [16000],
                "provider should be handed 16 kHz exactly once, got \(enhancer.receivedSampleRates)")
        #expect(result.audio?.sampleRate == 48000,
                "pipeline should return the caller's rate")
    }

    @Test("Pipeline honours an explicit output rate")
    func pipelineRespectsRequestedOutputRate() async throws {
        let engine = AudioEngine()
        let enhancer = MockEnhancer()
        await engine.register(enhancer: enhancer, for: .mossformerSE16k)

        let result = try await engine.executePipeline(
            engine.pipeline().enhance(.mossformerSE16k),
            audio: tone(sampleRate: 48000),
            outputSampleRate: 24000
        )

        #expect(enhancer.receivedSampleRates == [16000])
        #expect(result.audio?.sampleRate == 24000)
    }

    @Test("Audio already at the model's rate is not converted at all")
    func noConversionWhenRatesAlreadyMatch() async throws {
        let engine = AudioEngine()
        let enhancer = MockEnhancer()
        await engine.register(enhancer: enhancer, for: .mossformerSE16k)

        let input = tone(sampleRate: 16000)
        let result = try await engine.pipeline()
            .enhance(.mossformerSE16k)
            .process(audio: input)

        #expect(enhancer.receivedSampleRates == [16000])
        #expect(result.audio?.sampleRate == 16000)
        #expect(result.audio?.samples.count == input.samples.count,
                "a no-op conversion should not change the sample count")
    }
}

@Suite("Interleaved resampling")
struct InterleavedResamplingTests {
    @Test("Stereo channels are resampled independently")
    func stereoChannelsRemainIndependent() throws {
        let frames = 64
        let interleaved = (0..<frames).flatMap { frame -> [Float] in
            [Float(frame) / Float(frames), -1]
        }
        let input = AudioBuffer(samples: interleaved, sampleRate: 16000, channels: 2)
        let output = try input.resampled(to: 24000, quality: .fast)

        #expect(output.channels == 2)
        #expect(output.frameCount == 96)
        for frame in 0..<output.frameCount {
            #expect(output.samples[frame * 2 + 1] == -1,
                    "right channel was contaminated at frame \(frame)")
        }
    }

    @Test("Every quality returns the exact mathematical frame count")
    func exactFrameCount() throws {
        let input = AudioBuffer(
            samples: [Float](repeating: 0.25, count: 48_001),
            sampleRate: 48000,
            channels: 1
        )
        let expected = Int((Double(input.frameCount) * 16000 / 48000).rounded())

        for quality in [ResamplingQuality.fast, .balanced, .high, .auto] {
            let output = try input.resampled(to: 16000, quality: quality)
            #expect(output.frameCount == expected,
                    "\(quality) returned \(output.frameCount), expected \(expected)")
        }
    }

    @Test("Empty input adopts the requested rate")
    func emptyInputChangesRate() throws {
        let input = AudioBuffer(samples: [], sampleRate: 16000, channels: 2)
        let output = try input.resampled(to: 48000)
        #expect(output.samples.isEmpty)
        #expect(output.sampleRate == 48000)
        #expect(output.channels == 2)
    }

    @Test("A one-frame interpolation is well-defined")
    func oneFrameInput() throws {
        let input = AudioBuffer(samples: [0.75], sampleRate: 16000, channels: 1)
        let output = try input.resampled(to: 48000, quality: .fast)
        #expect(output.samples == [0.75, 0.75, 0.75])
    }

    @Test("Linear interpolation holds the final source frame")
    func linearInterpolationEndpoint() throws {
        let input = AudioBuffer(samples: [0, 1], sampleRate: 2, channels: 1)
        let output = try input.resampled(to: 4, quality: .fast)
        #expect(output.samples == [0, 0.5, 1, 1])
    }
}

// MARK: - The regression this fix exists for

@Suite("Chained resampling")
struct ChainedResamplingTests {

    private func tone(sampleRate: Int, seconds: Double = 0.5) -> AudioBuffer {
        let count = Int(Double(sampleRate) * seconds)
        let samples = (0..<count).map { i in
            sin(2 * Float.pi * 220 * Float(i) / Float(sampleRate)) * 0.5
        }
        return AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
    }

    /// `.enhance(.mossformerSE16k).upscale()` used to run
    /// source -> 16 -> source -> 16 -> source: the enhancement stage converted back up,
    /// and the upscale stage immediately pulled it down to 16 kHz again.
    ///
    /// Measured at 44.1 kHz deliberately, not 48 kHz. The resampler is Catmull-Rom
    /// interpolation, so at an exact 3:1 ratio the samples land back on the original
    /// grid and 48 -> 16 -> 48 -> 16 is bit-identical - the old code wasted work there
    /// but lost nothing. At 44.1 kHz the ratio is 2.756 and the round trip carries a
    /// measurable RMS error of ~0.004, which is what this asserts is gone.
    @Test("Enhance then upscale converts down exactly once")
    func enhanceThenUpscaleDoesNotRoundTrip() async throws {
        let engine = AudioEngine()
        let enhancer = MockEnhancer()
        let upscaler = MockUpscaler()
        await engine.register(enhancer: enhancer, for: .mossformerSE16k)
        await engine.register(upscaler: upscaler)

        let input = tone(sampleRate: 44100)
        let result = try await engine.pipeline()
            .enhance(.mossformerSE16k)
            .upscale()
            .process(audio: input)

        // Each model saw 16 kHz, once.
        #expect(enhancer.receivedSampleRates == [16000])
        #expect(upscaler.receivedSampleRates == [16000])

        // The decisive check: the upscaler received precisely what the enhancer
        // produced. A 16 -> 48 -> 16 round trip between the stages would perturb
        // every sample, so equality here is what proves it did not happen.
        let expected = try input.resampled(to: 16000).samples.map { $0 * enhancer.scaleFactor }
        #expect(upscaler.lastReceivedSamples.count == expected.count)
        #expect(upscaler.lastReceivedSamples == expected,
                "upscaler input was altered between stages - audio was round-tripped")

        // Upscaling raises the rate, so the pipeline returns the upscaler's rate.
        #expect(result.audio?.sampleRate == 48000)

        // And returns the upscaler's actual samples. Asserting only the rate is not
        // enough: the pipeline used to carry the previous stage's buffer forward and
        // then resample it up, which produced a 48 kHz result that had never been
        // through the upscaler at all. MockUpscaler triples its input, so the count
        // is what distinguishes real output from a resampled impostor.
        let expectedCount = upscaler.lastReceivedSamples.count * 3
        #expect(result.audio?.samples.count == expectedCount,
                "pipeline returned audio the upscaler never produced")
        #expect(result.audio?.samples == upscaler.lastReceivedSamples.flatMap { [$0, $0, $0] })
    }

    @Test("Upscaler output rate is preserved, not forced back to the input rate")
    func upscaleKeepsItsOutputRate() async throws {
        let engine = AudioEngine()
        let upscaler = MockUpscaler()
        await engine.register(upscaler: upscaler)

        // Upscaling is a rate change by definition, so a 16 kHz caller still gets
        // 48 kHz back rather than having the gain thrown away.
        let output = try await engine.upscale(tone(sampleRate: 16000))
        #expect(output.sampleRate == 48000)
    }
}

// MARK: - Per-model resampling preference

@Suite("Resampling preference")
struct ResamplingPreferenceTests {

    /// A provider that reproduces a Python pipeline needs the facade to convert audio
    /// the way that pipeline did. Without this the edge conversion was always cubic,
    /// so a provider validated against a different resampler silently got another one.
    struct PickyProcessor: AudioProcessor {
        let sampleRate = 16000
        let inputChannels = 1
        let outputChannels = 1
        var preferredResamplingQuality: ResamplingQuality { .high }
    }

    struct DefaultProcessor: AudioProcessor {
        let sampleRate = 16000
        let inputChannels = 1
        let outputChannels = 1
    }

    @Test("Providers default to balanced")
    func defaultsToBalanced() {
        #expect(DefaultProcessor().preferredResamplingQuality == .balanced)
    }

    @Test("A provider can declare a different quality")
    func canDeclarePreference() {
        #expect(PickyProcessor().preferredResamplingQuality == .high)
    }

    /// The qualities must actually differ, otherwise honouring the preference would be
    /// a no-op and this whole seam would be decorative.
    @Test("Quality choice changes the converted samples")
    func qualityIsObservable() throws {
        let rate = 44100
        let samples = (0..<rate).map { i in
            sin(2 * Float.pi * 300 * Float(i) / Float(rate)) * 0.4
                + sin(2 * Float.pi * 9000 * Float(i) / Float(rate)) * 0.3
        }
        let input = AudioBuffer(samples: samples, sampleRate: rate, channels: 1)

        let balanced = try input.resampled(to: 16000, quality: .balanced)
        let high = try input.resampled(to: 16000, quality: .high)

        #expect(balanced.samples != high.samples,
                "balanced and high produced identical output - the preference would be meaningless")
    }
}

// MARK: - Residency registration

@Suite("Model residency wiring")
struct ResidencyWiringTests {

    /// A provider that participates in residency accounting.
    final class ManagedEnhancer: SpeechEnhancer, ManagedModel, @unchecked Sendable {
        let sampleRate = 16000
        let inputChannels = 1
        let outputChannels = 1
        let minChunkSize = 512
        let recommendedChunkSize = 16000
        nonisolated let modelId: String
        nonisolated let estimatedMemoryBytes = 1_000_000
        var loaded = true

        init(modelId: String = "managed_mock") { self.modelId = modelId }
        func load() async throws { loaded = true }
        func unload() async { loaded = false }
        func checkIfLoaded() async -> Bool { loaded }
        func process(_ input: AudioBuffer) async throws -> AudioBuffer { input }
        func stream(_ i: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
            AsyncThrowingStream { $0.finish() }
        }
        func reset() async {}
    }

    /// Registration is bookkeeping. It used to spawn a detached task that called
    /// `register` on the manager - which loads, and for a real provider downloads -
    /// from a call that looks synchronous and returns no error.
    @Test("Registering a provider does not load it")
    func registrationDoesNotLoad() async throws {
        let engine = AudioEngine()
        let enhancer = ManagedEnhancer()
        enhancer.loaded = false
        await engine.register(enhancer: enhancer, for: .mossformerSE16k)

        try await Task.sleep(for: .milliseconds(100))
        let stats = await engine.modelStats
        #expect(stats.loadedModelCount == 0, "registration should not make anything resident")
        #expect(enhancer.loaded == false, "registration should not trigger a load")
    }

    /// The defect this closes: nothing ever entered the residency manager outside of
    /// tests, so its LRU eviction could not fire no matter how many models were held.
    @Test("Using a ManagedModel provider makes it resident")
    func managedProviderIsTrackedOnUse() async throws {
        let engine = AudioEngine()
        await engine.register(enhancer: ManagedEnhancer(), for: .mossformerSE16k)

        _ = try await engine.enhance(
            AudioBuffer(samples: [Float](repeating: 0.1, count: 16000), sampleRate: 16000, channels: 1),
            model: .mossformerSE16k)

        let stats = await engine.modelStats
        #expect(stats.loadedModelCount == 1, "provider should be resident after use, got \(stats.loadedModelCount)")
    }

    /// End to end: evict, then use again. Before this, the second call reached an
    /// unloaded provider and a real one would have thrown `modelNotLoaded`.
    @Test("A model evicted under pressure is reloaded by the next call")
    func evictedModelIsUsableAgain() async throws {
        // Room for one model at a time.
        let engine = AudioEngine(configuration: AudioToolConfiguration(modelMemoryLimit: 1_500_000))
        let first = ManagedEnhancer()
        let second = ManagedEnhancer(modelId: "managed_mock_2")
        await engine.register(enhancer: first, for: .mossformerSE16k)
        await engine.register(enhancer: second, for: .mossformerSE48k)

        let audio = AudioBuffer(samples: [Float](repeating: 0.1, count: 16000), sampleRate: 16000, channels: 1)

        _ = try await engine.enhance(audio, model: .mossformerSE16k)
        _ = try await engine.enhance(audio, model: .mossformerSE48k)
        #expect(first.loaded == false, "the first model should have been evicted to make room")

        // The call that used to fail.
        _ = try await engine.enhance(audio, model: .mossformerSE16k)
        #expect(first.loaded, "using an evicted model must reload it")
    }

    @Test("A provider that is not ManagedModel still registers and simply is not tracked")
    func unmanagedProviderStillWorks() async throws {
        let engine = AudioEngine()
        await engine.register(enhancer: MockEnhancer(), for: .mossformerSE16k)

        let out = try await engine.enhance(
            AudioBuffer(samples: [Float](repeating: 0.1, count: 16000), sampleRate: 16000, channels: 1),
            model: .mossformerSE16k)
        #expect(out.samples.count == 16000)

        let stats = await engine.modelStats
        #expect(stats.loadedModelCount == 0)
    }
}

// MARK: - Streaming stages and sample rate

/// Attaching a progress handler must not change what a stage does.
///
/// The pipeline picks a streaming implementation whenever a provider supports it
/// *and* an event handler is attached, which makes progress reporting a silent
/// switch between two code paths. They were not equivalent: the batch upscale went
/// through `upscale(_:)`, which adapts the input to the provider's rate, while the
/// streaming branch handed over `context.currentAudio` untouched. Once the provider
/// started validating its rate, the same pipeline succeeded without a handler and
/// threw `sampleRateMismatch` with one.
@Suite("Streaming stages honour the provider's rate")
struct StreamingStageSampleRateTests {

    private func tone(sampleRate: Int, seconds: Double = 1.0) -> AudioBuffer {
        let count = Int(Double(sampleRate) * seconds)
        let samples = (0..<count).map { i in
            sin(2 * Float.pi * 440 * Float(i) / Float(sampleRate)) * 0.5
        }
        return AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
    }

    @Test("Upscale with a progress handler converts input to the upscaler's rate")
    func streamingUpscaleResamples() async throws {
        let engine = AudioEngine()
        let upscaler = MockStreamingUpscaler()
        await engine.register(upscaler: upscaler)

        let result = try await engine.pipeline()
            .upscale()
            .onEvent { _ in }
            .process(audio: tone(sampleRate: 44100))

        #expect(upscaler.receivedSampleRates == [16000],
                "streaming upscaler should be handed 16 kHz, got \(upscaler.receivedSampleRates)")
        #expect(result.audio?.sampleRate == 48000)
    }

    @Test("Upscale reaches the same rates with and without a progress handler")
    func streamingAndBatchAgree() async throws {
        let batchEngine = AudioEngine()
        let batchUpscaler = MockStreamingUpscaler()
        await batchEngine.register(upscaler: batchUpscaler)
        let batchResult = try await batchEngine.pipeline()
            .upscale()
            .process(audio: tone(sampleRate: 48000))

        let streamEngine = AudioEngine()
        let streamUpscaler = MockStreamingUpscaler()
        await streamEngine.register(upscaler: streamUpscaler)
        let streamResult = try await streamEngine.pipeline()
            .upscale()
            .onEvent { _ in }
            .process(audio: tone(sampleRate: 48000))

        #expect(batchUpscaler.receivedSampleRates == streamUpscaler.receivedSampleRates,
                "the progress handler changed which rate the provider saw: \(batchUpscaler.receivedSampleRates) vs \(streamUpscaler.receivedSampleRates)")
        #expect(batchResult.audio?.sampleRate == streamResult.audio?.sampleRate)
        #expect(batchResult.audio?.samples.count == streamResult.audio?.samples.count)
    }

    @Test("Enhancement with a progress handler uses the provider's resampler")
    func streamingEnhanceUsesProviderPreference() async throws {
        let engine = AudioEngine()
        let enhancer = MockEnhancer()
        await engine.register(enhancer: enhancer, for: .mossformerSE16k)

        _ = try await engine.pipeline()
            .enhance(.mossformerSE16k)
            .onEvent { _ in }
            .process(audio: tone(sampleRate: 44100))

        #expect(enhancer.receivedSampleRates == [16000],
                "streaming enhancer should be handed 16 kHz exactly once, got \(enhancer.receivedSampleRates)")
    }
}

// MARK: - What the resampler preference is worth

/// The seam existed but was empty: only USS declared a preference, so every other
/// model got cubic interpolation at the facade's edge. Cubic has no anti-aliasing
/// stage, and no reference pipeline in `Models/` resamples that way - scipy's
/// `signal.resample`, `librosa.resample`, `torchaudio.transforms.Resample` and
/// AVAudioConverter's Mastering algorithm are all band-limited.
///
/// Measured on a 48 -> 16 kHz downsample of a signal with content to 22 kHz: cubic
/// sits 131% RMS away from Mastering/max - the folded-back content is larger than
/// the signal. On a signal already under the new Nyquist the same comparison is
/// 0.56%. Aliasing is the whole difference, which is why this matters only when the
/// caller's audio is wider than the model's band.
@Suite("Anti-aliasing on downsample")
struct AntiAliasingTests {

    /// Energy at `frequency` in `samples`, by direct correlation - enough to tell
    /// whether a tone survived, without pulling in an FFT.
    private func energy(at frequency: Float, in samples: [Float], sampleRate: Int) -> Float {
        var real: Float = 0
        var imaginary: Float = 0
        for (i, sample) in samples.enumerated() {
            let phase = 2 * Float.pi * frequency * Float(i) / Float(sampleRate)
            real += sample * cos(phase)
            imaginary += sample * sin(phase)
        }
        return sqrt(real * real + imaginary * imaginary) / Float(samples.count)
    }

    /// A 15 kHz tone cannot exist at 16 kHz - it is above the 8 kHz Nyquist. An
    /// anti-aliased resampler removes it. Cubic interpolation folds it back to
    /// 16000 - 15000 = 1 kHz, inventing a tone that was never in the source.
    @Test("High quality rejects content above the target Nyquist; cubic folds it back")
    func aliasedToneIsSuppressed() throws {
        let rate = 48000
        let toneHz: Float = 15000
        let samples = (0..<rate).map { i in
            sin(2 * Float.pi * toneHz * Float(i) / Float(rate)) * 0.5
        }
        let input = AudioBuffer(samples: samples, sampleRate: rate, channels: 1)

        let high = try input.resampled(to: 16000, quality: .high)
        let cubic = try input.resampled(to: 16000, quality: .balanced)

        // 15 kHz folds to |16000 - 15000| = 1 kHz at the new rate.
        let aliasHz: Float = 1000
        let highAlias = energy(at: aliasHz, in: high.samples, sampleRate: 16000)
        let cubicAlias = energy(at: aliasHz, in: cubic.samples, sampleRate: 16000)

        #expect(cubicAlias > 0.05,
                "cubic should fold the 15 kHz tone down to 1 kHz, got \(cubicAlias)")
        #expect(highAlias < cubicAlias / 10,
                "high quality should suppress the alias: \(highAlias) vs cubic \(cubicAlias)")
    }

    /// A resampler that shifts audio in time would silently misalign it with anything
    /// derived from the original timeline - VAD segment boundaries, diarization
    /// turns, word timings - and the facade converts at the edge of every stage, so
    /// the shift would accumulate. Measured at zero lag, correlation 0.99999.
    @Test("High quality introduces no group delay relative to cubic")
    func noGroupDelay() throws {
        let rate = 48000
        let samples = (0..<rate).map { i -> Float in
            let t = Float(i) / Float(rate)
            var sum: Float = 0
            for frequency in stride(from: Float(200), to: Float(7000), by: 700) {
                sum += sin(2 * Float.pi * frequency * t)
            }
            return sum / 10
        }
        let input = AudioBuffer(samples: samples, sampleRate: rate, channels: 1)

        let high = try input.resampled(to: 16000, quality: .high)
        let cubic = try input.resampled(to: 16000, quality: .balanced)

        func correlation(lag: Int) -> Float {
            let maxLag = 64
            let count = min(high.samples.count, cubic.samples.count) - 2 * maxLag - 2
            var dot: Float = 0, normA: Float = 0, normB: Float = 0
            for i in (maxLag + 1)..<(maxLag + 1 + count) {
                let a = high.samples[i], b = cubic.samples[i + lag]
                dot += a * b; normA += a * a; normB += b * b
            }
            return dot / (sqrt(normA) * sqrt(normB) + 1e-12)
        }

        let atZero = correlation(lag: 0)
        let best = (-64...64).max { correlation(lag: $0) < correlation(lag: $1) } ?? 0

        #expect(best == 0, "high-quality resampling shifted the audio by \(best) samples")
        #expect(atZero > 0.999, "in-band correlation at zero lag was \(atZero)")
    }

    /// Below the new Nyquist there is nothing to remove, so the two agree closely.
    /// Without this the test above could pass simply because `.high` attenuates
    /// everything.
    @Test("High quality preserves content below the target Nyquist")
    func inBandToneSurvives() throws {
        let rate = 48000
        let toneHz: Float = 1000
        let samples = (0..<rate).map { i in
            sin(2 * Float.pi * toneHz * Float(i) / Float(rate)) * 0.5
        }
        let input = AudioBuffer(samples: samples, sampleRate: rate, channels: 1)

        let high = try input.resampled(to: 16000, quality: .high)
        let cubic = try input.resampled(to: 16000, quality: .balanced)

        let highEnergy = energy(at: toneHz, in: high.samples, sampleRate: 16000)
        let cubicEnergy = energy(at: toneHz, in: cubic.samples, sampleRate: 16000)

        #expect(highEnergy > 0.2, "in-band tone should survive high-quality resampling, got \(highEnergy)")
        #expect(abs(highEnergy - cubicEnergy) < 0.05,
                "in band the two should agree: \(highEnergy) vs \(cubicEnergy)")
    }
}
