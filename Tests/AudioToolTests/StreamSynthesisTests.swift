//
//  StreamSynthesisTests.swift
//  AudioToolTests
//
//  AudioEngine.streamSynthesis holds residency and runs off the engine actor
//

import Foundation
import Testing
@testable import AudioTool
@testable import AudioToolCore

// MARK: - Mocks

/// A streaming synthesizer under residency management. Its one stream yields
/// whatever the test feeds it, and fails if it has been unloaded in between.
actor FedStreamingSynthesizer: SpeechSynthesizer, ManagedModel {
    nonisolated let sampleRate: Int = 24000
    nonisolated let availableVoices: [String] = ["test"]
    nonisolated let modelId: String = "fed_streaming_synthesizer"
    nonisolated let estimatedMemoryBytes: Int = 80

    private nonisolated let chunks: AsyncStream<AudioBuffer>
    private(set) var isLoaded = false

    init(chunks: AsyncStream<AudioBuffer>) {
        self.chunks = chunks
    }

    func load() async throws { isLoaded = true }
    func unload() async { isLoaded = false }
    func checkIfLoaded() async -> Bool { isLoaded }

    func synthesize(_ text: String, voice: String) async throws -> AudioBuffer {
        AudioBuffer(samples: [], sampleRate: sampleRate, channels: 1)
    }

    nonisolated func streamSynthesis(_ text: String, voice: String) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                for await chunk in chunks {
                    guard await self.isLoaded else {
                        continuation.finish(throwing: AudioToolError.modelNotLoaded(modelId))
                        return
                    }
                    if case .terminated = continuation.yield(chunk) { return }
                }
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
        }
    }
}

/// An enhancer whose `inputChannels` getter blocks the thread reading it, once,
/// until ``unblock()``.
///
/// The engine reads that property synchronously on its actor, while adapting a
/// batch call's input, so a batch `enhance` through this provider holds the
/// engine actor for exactly as long as a test wants - the deterministic stand-in
/// for a long resample.
final class ActorBlockingEnhancer: SpeechEnhancer, @unchecked Sendable {
    let sampleRate: Int = 16000
    let outputChannels: Int = 1
    let minChunkSize: Int = 1
    let recommendedChunkSize: Int = 1600

    private let lock = NSLock()
    private var armed = true
    private var blocked = false
    private let release = DispatchSemaphore(value: 0)

    var inputChannels: Int {
        lock.lock()
        let shouldBlock = armed
        armed = false
        if shouldBlock { blocked = true }
        lock.unlock()
        if shouldBlock { release.wait() }
        return 1
    }

    /// Whether a reader is, or was, held in ``inputChannels``.
    var hasBlocked: Bool {
        lock.lock()
        defer { lock.unlock() }
        return blocked
    }

    /// Release the held reader. Safe to call more than once.
    func unblock() { release.signal() }

    func process(_ input: AudioBuffer) async throws -> AudioBuffer { input }
    func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { $0.finish() }
    }
    func reset() async {}
}

/// Collects a stream's output where a test can poll it from outside the engine.
actor ChunkRecorder {
    private(set) var chunks: [AudioBuffer] = []
    private(set) var ended = false
    func append(_ chunk: AudioBuffer) { chunks.append(chunk) }
    func end() { ended = true }
    var count: Int { chunks.count }

    /// Drain `stream` into this recorder, then mark it ended.
    nonisolated func record(
        _ stream: AsyncThrowingStream<AudioBuffer, Error>
    ) -> Task<Void, Error> {
        Task {
            for try await chunk in stream { await self.append(chunk) }
            await self.end()
        }
    }
}

/// Polls rather than sleeping a fixed time: the condition depends on tasks this
/// test does not drive.
func waitForCondition(
    _ what: String,
    _ condition: () async -> Bool
) async throws {
    for _ in 0..<2_000 {
        if await condition() { return }
        try await Task.sleep(for: .milliseconds(1))
    }
    Issue.record("\(what) did not happen within two seconds")
}

/// Hold the engine actor in a batch `enhance` through `blocker`, run `body`, and
/// release it however `body` exits.
func whileEngineActorIsBlocked(
    _ engine: AudioEngine,
    by blocker: ActorBlockingEnhancer,
    _ body: () async throws -> Void
) async throws {
    await engine.register(enhancer: blocker, for: .frcrn)
    let busy = Task {
        try await engine.enhance(
            AudioBuffer(samples: [Float](repeating: 0, count: 160), sampleRate: 16000),
            model: .frcrn
        )
    }
    do {
        try await waitForCondition("the engine actor blocking") { blocker.hasBlocked }
        try await body()
    } catch {
        blocker.unblock()
        _ = try? await busy.value
        throw error
    }
    blocker.unblock()
    _ = try await busy.value
}

// MARK: - Tests

@Suite("Engine synthesis streams")
struct StreamSynthesisTests {

    private static let model = SynthesisModel.appleTTS(language: "en-US")

    private struct Fixture {
        let engine: AudioEngine
        let residency: ModelResidency
        let synthesizer: FedStreamingSynthesizer
        let feed: AsyncStream<AudioBuffer>.Continuation
        let other = MockManagedModel(modelId: "other", memoryBytes: 80)

        var leases: Int {
            get async { await residency.activeUseCount(for: synthesizer.modelId) }
        }

        func admitOther() async throws {
            try await engine.preload(other)
        }

        func openStream() async -> AsyncThrowingStream<AudioBuffer, Error> {
            await engine.streamSynthesis("hello", voice: "test", model: StreamSynthesisTests.model)
        }
    }

    /// The same budget as the enhancement tests: admitting either 80-byte model
    /// evicts the other unless it is leased.
    private func makeFixture() async throws -> Fixture {
        let residency = ModelResidency(memoryLimitBytes: 100)
        let engine = AudioEngine(configuration: .default, modelManager: residency)
        let (chunks, feed) = AsyncStream<AudioBuffer>.makeStream()
        let synthesizer = FedStreamingSynthesizer(chunks: chunks)
        try await engine.preload(synthesizer)
        await engine.register(synthesizer: synthesizer, for: Self.model)
        return Fixture(engine: engine, residency: residency, synthesizer: synthesizer, feed: feed)
    }

    private func chunk(_ value: Float) -> AudioBuffer {
        AudioBuffer(samples: [Float](repeating: value, count: 240), sampleRate: 24000)
    }

    @Test("A synthesis stream keeps its synthesizer resident until it ends")
    func testStreamHoldsResidency() async throws {
        let fixture = try await makeFixture()
        var outputs = await fixture.openStream().makeAsyncIterator()

        fixture.feed.yield(chunk(1))
        #expect(try await outputs.next()?.samples.first == 1)
        #expect(await fixture.leases == 1)

        try await fixture.admitOther()
        #expect(await fixture.synthesizer.isLoaded)

        fixture.feed.yield(chunk(2))
        #expect(try await outputs.next()?.samples.first == 2)
        fixture.feed.finish()
        #expect(try await outputs.next() == nil)

        #expect(await fixture.leases == 0)
        try await fixture.admitOther()
        #expect(await fixture.synthesizer.isLoaded == false)
    }

    @Test("Cancelling the consumer releases the lease")
    func testConsumerCancellationReleasesLease() async throws {
        let fixture = try await makeFixture()
        let recorder = ChunkRecorder()
        let consumer = recorder.record(await fixture.openStream())

        fixture.feed.yield(chunk(1))
        try await waitForCondition("the first chunk") { await recorder.count == 1 }
        #expect(await fixture.leases == 1)

        consumer.cancel()
        try await waitForCondition("the lease release") { await fixture.leases == 0 }
        try await fixture.admitOther()
        #expect(await fixture.synthesizer.isLoaded == false)
    }

    /// The stream used to release its lease from the engine actor before
    /// finishing, so a batch call holding the actor held up the stream's end.
    @Test("Chunks and the stream's end arrive while the engine actor is busy")
    func testStreamCompletesWhileEngineActorIsBusy() async throws {
        let fixture = try await makeFixture()
        let recorder = ChunkRecorder()
        let consumer = recorder.record(await fixture.openStream())

        fixture.feed.yield(chunk(1))
        try await waitForCondition("the first chunk") { await recorder.count == 1 }

        try await whileEngineActorIsBlocked(fixture.engine, by: ActorBlockingEnhancer()) {
            fixture.feed.yield(chunk(2))
            try await waitForCondition("a chunk while the engine actor is held") {
                await recorder.count == 2
            }
            fixture.feed.finish()
            try await waitForCondition("the stream's end while the engine actor is held") {
                await recorder.ended
            }
        }

        try await consumer.value
        #expect(await fixture.leases == 0)
    }
}
