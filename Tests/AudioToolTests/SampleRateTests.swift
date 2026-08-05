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
