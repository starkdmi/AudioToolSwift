//
//  TTSStateAndCancellationTests.swift
//  AudioToolTests
//
//  Unit coverage for load-state ordering and batch cancellation
//

import Testing
@testable import AudioToolCore
@testable import AudioToolTTS

@Suite("TTS State and Cancellation Tests")
struct TTSStateAndCancellationTests {
    @Test("Download progress is accepted only for the active downloading generation")
    func testProgressGenerationGate() {
        #expect(ModelLoadStateGate.acceptsProgress(
            0.75,
            generation: 2,
            currentGeneration: 2,
            state: .downloading(progress: 0.5)
        ))
        #expect(!ModelLoadStateGate.acceptsProgress(
            0.75,
            generation: 1,
            currentGeneration: 2,
            state: .downloading(progress: 0.5)
        ))
        #expect(!ModelLoadStateGate.acceptsProgress(
            0.75,
            generation: 2,
            currentGeneration: 2,
            state: .ready
        ))
        #expect(!ModelLoadStateGate.acceptsProgress(
            0.75,
            generation: 2,
            currentGeneration: 2,
            state: .failed("load failed")
        ))
        #expect(!ModelLoadStateGate.acceptsProgress(
            0.25,
            generation: 2,
            currentGeneration: 2,
            state: .downloading(progress: 0.5)
        ))
        #expect(!ModelLoadStateGate.acceptsProgress(
            .nan,
            generation: 2,
            currentGeneration: 2,
            state: .downloading(progress: 0.5)
        ))
    }

    @Test("Batch TTS cancellation resumes a waiter exactly once")
    func testBatchCancellation() async {
        let operation = AppleTTSBatchOperation()
        let waiter = Task {
            try await withCheckedThrowingContinuation { continuation in
                operation.install(continuation)
            }
        }

        operation.cancel()
        operation.cancel()

        await #expect(throws: CancellationError.self) {
            _ = try await waiter.value
        }
    }

    @Test("A completed batch result cannot be replaced by cancellation")
    func testBatchCompletionWinsOnce() async throws {
        let operation = AppleTTSBatchOperation()
        let expected = AudioBuffer(samples: [0.25], sampleRate: 22_050)
        let waiter = Task {
            try await withCheckedThrowingContinuation { continuation in
                operation.install(continuation)
            }
        }

        operation.complete(with: .success(expected))
        operation.cancel()

        #expect(try await waiter.value == expected)
    }
}
