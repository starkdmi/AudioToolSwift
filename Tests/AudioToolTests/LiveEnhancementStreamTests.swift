//
//  LiveEnhancementStreamTests.swift
//  AudioToolTests
//
//  AudioEngine.enhanceStream keeps its enhancer resident for the stream's life
//

import Foundation
import Testing
@testable import AudioTool
@testable import AudioToolCore

// MARK: - Mock

/// A streaming enhancer under residency management that behaves like the causal
/// Voice Isolator where it matters here: an unload ends every open stream with
/// `modelNotLoaded` at its next buffer or at the end of its input.
actor LiveStreamEnhancer: SpeechEnhancer, ManagedModel {
    nonisolated let sampleRate: Int = 48000
    nonisolated let inputChannels: Int = 1
    nonisolated let outputChannels: Int = 1
    nonisolated let minChunkSize: Int = 480
    nonisolated let recommendedChunkSize: Int = 4800

    nonisolated let modelId: String
    nonisolated let estimatedMemoryBytes: Int

    /// A buffer starting with this sample makes the stream fail.
    static let failureSentinel: Float = 99

    struct InjectedFailure: Error {}

    private(set) var isLoaded = false
    private(set) var unloadCount = 0
    private(set) var receivedBuffers: [AudioBuffer] = []

    init(modelId: String = "live_stream_enhancer", memoryBytes: Int = 80) {
        self.modelId = modelId
        self.estimatedMemoryBytes = memoryBytes
    }

    func load() async throws { isLoaded = true }
    func unload() async {
        isLoaded = false
        unloadCount += 1
    }
    func checkIfLoaded() async -> Bool { isLoaded }
    func reset() async {}

    func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        try push(input)
    }

    private func push(_ input: AudioBuffer) throws -> AudioBuffer {
        guard isLoaded else { throw AudioToolError.modelNotLoaded(modelId) }
        try validateSampleRate(input)
        receivedBuffers.append(input)
        if input.samples.first == Self.failureSentinel { throw InjectedFailure() }
        return AudioBuffer(
            samples: input.samples.map { $0 * 0.5 },
            sampleRate: sampleRate,
            channels: 1
        )
    }

    private func requireLoaded() throws {
        guard isLoaded else { throw AudioToolError.modelNotLoaded(modelId) }
    }

    nonisolated func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    for await chunk in input {
                        try Task.checkCancellation()
                        let output = try await self.push(chunk)
                        if case .terminated = continuation.yield(output) { return }
                    }
                    try Task.checkCancellation()
                    // Like closing a session: a stream whose model went away must
                    // not end as if it had delivered the whole signal.
                    try await self.requireLoaded()
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
        }
    }
}

// MARK: - Tests

@Suite("Live enhancement streams hold residency")
struct LiveEnhancementStreamTests {

    private static let model = EnhancementModel.voiceIsolatorRealtime

    /// A 100-byte budget with two 80-byte models: admitting either evicts the
    /// other, unless the other is leased.
    private struct Fixture {
        let engine: AudioEngine
        let residency: ModelResidency
        let enhancer: LiveStreamEnhancer
        let other: MockManagedModel

        var leases: Int {
            get async { await residency.activeUseCount(for: enhancer.modelId) }
        }

        /// Admit the other model, the thing that evicts an unleased enhancer.
        func admitOther() async throws {
            try await engine.preload(other)
        }
    }

    private func makeFixture() async throws -> Fixture {
        let residency = ModelResidency(memoryLimitBytes: 100)
        let engine = AudioEngine(configuration: .default, modelManager: residency)
        let enhancer = LiveStreamEnhancer()
        try await engine.preload(enhancer)
        await engine.register(enhancer: enhancer, for: Self.model)
        return Fixture(
            engine: engine,
            residency: residency,
            enhancer: enhancer,
            other: MockManagedModel(modelId: "other", memoryBytes: 80)
        )
    }

    private func buffer(
        _ value: Float,
        sampleRate: Int = 48000,
        channels: Int = 1
    ) -> AudioBuffer {
        AudioBuffer(
            samples: [Float](repeating: value, count: 480 * channels),
            sampleRate: sampleRate,
            channels: channels
        )
    }

    private func waitUntil(
        _ what: String,
        _ condition: () async -> Bool
    ) async throws {
        for _ in 0..<2_000 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(1))
        }
        Issue.record("\(what) did not happen within two seconds")
    }

    /// The defect `enhanceStream` exists for, so the tests below are known to be
    /// testing something.
    @Test("A provider streamed directly is evicted by another admission")
    func directStreamIsEvicted() async throws {
        let fixture = try await makeFixture()
        let (input, feed) = AsyncStream<AudioBuffer>.makeStream()
        var outputs = fixture.enhancer.stream(input).makeAsyncIterator()

        feed.yield(buffer(1))
        _ = try await outputs.next()

        try await fixture.admitOther()
        #expect(await fixture.enhancer.isLoaded == false)

        feed.yield(buffer(2))
        await #expect(throws: AudioToolError.self) {
            while try await outputs.next() != nil {}
        }
    }

    @Test("An open stream survives another model's admission, and is evictable once it ends")
    func openStreamSurvivesAdmission() async throws {
        let fixture = try await makeFixture()
        let (input, feed) = AsyncStream<AudioBuffer>.makeStream()
        var outputs = await fixture.engine
            .enhanceStream(input, model: Self.model)
            .makeAsyncIterator()

        feed.yield(buffer(1))
        let first = try await outputs.next()
        #expect(first?.samples.first == 0.5)
        #expect(await fixture.leases == 1)

        try await fixture.admitOther()
        #expect(await fixture.enhancer.isLoaded)
        #expect(await fixture.enhancer.unloadCount == 0)

        feed.yield(buffer(2))
        let second = try await outputs.next()
        #expect(second?.samples.first == 1.0)

        feed.finish()
        #expect(try await outputs.next() == nil)

        // Released before the stream reported its end, so no waiting here.
        #expect(await fixture.leases == 0)
        try await fixture.admitOther()
        #expect(await fixture.enhancer.isLoaded == false)
        #expect(await fixture.enhancer.unloadCount == 1)
    }

    @Test("A provider error ends the stream and releases the lease")
    func providerErrorReleasesLease() async throws {
        let fixture = try await makeFixture()
        let (input, feed) = AsyncStream<AudioBuffer>.makeStream()
        let output = await fixture.engine.enhanceStream(input, model: Self.model)

        feed.yield(buffer(1))
        feed.yield(buffer(LiveStreamEnhancer.failureSentinel))
        await #expect(throws: LiveStreamEnhancer.InjectedFailure.self) {
            for try await _ in output {}
        }

        #expect(await fixture.leases == 0)
        try await fixture.admitOther()
        #expect(await fixture.enhancer.isLoaded == false)
    }

    @Test("A buffer at another rate ends the stream before reaching the provider")
    func wrongRateIsRejected() async throws {
        let fixture = try await makeFixture()
        let (input, feed) = AsyncStream<AudioBuffer>.makeStream()
        let output = await fixture.engine.enhanceStream(input, model: Self.model)

        feed.yield(buffer(1))
        feed.yield(buffer(2, sampleRate: 16000))
        feed.yield(buffer(3))
        let error = await #expect(throws: AudioToolError.self) {
            for try await _ in output {}
        }
        guard case .sampleRateMismatch(expected: 48000, found: 16000)? = error else {
            Issue.record("expected a 48000/16000 sample-rate mismatch, got \(String(describing: error))")
            return
        }

        // Whether the good buffer ahead of it got through is a race with the
        // cancellation; nothing from the rejected one onward may.
        let received = await fixture.enhancer.receivedBuffers
        #expect(received.allSatisfy { $0.sampleRate == 48000 && $0.samples.first == 1 })
        #expect(await fixture.leases == 0)
        try await fixture.admitOther()
        #expect(await fixture.enhancer.isLoaded == false)
    }

    @Test("Cancelling the consumer releases the lease")
    func consumerCancellationReleasesLease() async throws {
        let fixture = try await makeFixture()
        let (input, feed) = AsyncStream<AudioBuffer>.makeStream()
        let output = await fixture.engine.enhanceStream(input, model: Self.model)
        let (started, signalStarted) = AsyncStream<Void>.makeStream()

        let consumer = Task {
            var outputs = output.makeAsyncIterator()
            _ = try await outputs.next()
            signalStarted.yield()
            while try await outputs.next() != nil {}
        }

        feed.yield(buffer(1))
        for await _ in started { break }
        try await fixture.admitOther()
        #expect(await fixture.enhancer.isLoaded)

        consumer.cancel()
        try await waitUntil("the lease release") { await fixture.leases == 0 }
        try await fixture.admitOther()
        #expect(await fixture.enhancer.isLoaded == false)
    }

    @Test("Dropping the stream releases the lease")
    func droppingStreamReleasesLease() async throws {
        let fixture = try await makeFixture()
        let (input, feed) = AsyncStream<AudioBuffer>.makeStream()

        do {
            var outputs = await fixture.engine
                .enhanceStream(input, model: Self.model)
                .makeAsyncIterator()
            feed.yield(buffer(1))
            _ = try await outputs.next()
            try await fixture.admitOther()
            #expect(await fixture.enhancer.isLoaded)
        }

        try await waitUntil("the lease release") { await fixture.leases == 0 }
        try await fixture.admitOther()
        #expect(await fixture.enhancer.isLoaded == false)
    }

    @Test("Channels are adapted per buffer")
    func channelsAreAdapted() async throws {
        let fixture = try await makeFixture()
        let (input, feed) = AsyncStream<AudioBuffer>.makeStream()
        let output = await fixture.engine.enhanceStream(input, model: Self.model)

        feed.yield(buffer(1, channels: 2))
        feed.finish()
        var outputs: [AudioBuffer] = []
        for try await chunk in output { outputs.append(chunk) }

        let received = await fixture.enhancer.receivedBuffers
        #expect(received.map(\.channels) == [1])
        #expect(received.first?.frameCount == 480)
        #expect(outputs.count == 1)
    }

    @Test("An unregistered model fails at once")
    func unregisteredModelFails() async throws {
        let engine = AudioEngine()
        let (input, _) = AsyncStream<AudioBuffer>.makeStream()
        let output = await engine.enhanceStream(input, model: .voiceIsolatorRealtime)
        await #expect(throws: AudioToolError.self) {
            for try await _ in output {}
        }
    }

    @Test("An enhancer outside residency management streams without a lease")
    func unmanagedEnhancerStreams() async throws {
        let enhancer = MockEnhancer()
        enhancer.processDelay = .zero
        enhancer.scaleFactor = 0.5
        let engine = AudioEngine()
        await engine.register(enhancer: enhancer, for: .frcrn)

        let (input, feed) = AsyncStream<AudioBuffer>.makeStream()
        let output = await engine.enhanceStream(input, model: .frcrn)
        feed.yield(buffer(1, sampleRate: 16000))
        feed.yield(buffer(2, sampleRate: 16000))
        feed.finish()

        var firstSamples: [Float] = []
        for try await chunk in output { firstSamples.append(chunk.samples[0]) }
        #expect(firstSamples == [0.5, 1.0])
    }
}
