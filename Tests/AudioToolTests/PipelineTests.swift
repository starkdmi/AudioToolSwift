//
//  PipelineTests.swift
//  AudioTool
//
//  Tests for pipeline execution
//

import Testing
@testable import AudioTool
@testable import AudioToolCore

@Suite("Pipeline Tests")
struct PipelineTests {
    
    // MARK: - Basic Pipeline
    
    @Test("Pipeline detects audio with mock VAD")
    func testVADPipeline() async throws {
        let mockVAD = MockVAD()
        let voice = AudioEngine(
            configuration: .default,
            vad: mockVAD
        )
        
        let audio = AudioBuffer.silence(duration: 5.0, sampleRate: 16000)
        let segments = try await voice.detect(audio)
        
        #expect(segments.count == 3)
        #expect(segments[0].isSpeech == true)
        #expect(segments[1].isSpeech == false)
        #expect(segments[2].isSpeech == true)
        #expect(mockVAD.detectCallCount == 1)
    }
    
    @Test("Pipeline enhances audio with mock enhancer")
    func testEnhancementPipeline() async throws {
        let mockEnhancer = MockEnhancer()
        let voice = AudioEngine(
            configuration: .default,
            enhancer: (.mossformerSE16k, mockEnhancer)
        )
        
        let input = AudioBuffer.sine(frequency: 440, duration: 1.0, sampleRate: 16000)
        let output = try await voice.enhance(input, model: .mossformerSE16k)
        
        // Verify processing happened (mock multiplies by 0.99)
        // Check sample at index 100 since sine starts at 0
        let testIndex = 100
        #expect(abs(output.samples[testIndex] - input.samples[testIndex] * 0.99) < 0.0001)
        #expect(mockEnhancer.processCallCount == 1)
    }
    
    @Test("Pipeline transcribes with mock transcriber")
    func testTranscriptionPipeline() async throws {
        let mockTranscriber = MockTranscriber()
        let voice = AudioEngine(
            configuration: .default,
            transcriber: (.parakeet, mockTranscriber)
        )
        
        let audio = AudioBuffer.silence(duration: 3.0, sampleRate: 16000)
        let result = try await voice.transcribe(audio, model: .parakeet)
        
        #expect(result.text == "Hello world. This is a test.")
        #expect(result.segments.count == 2)
        #expect(result.language == "en")
        #expect(mockTranscriber.transcribeCallCount == 1)
    }
    
    @Test("Pipeline separates speakers with mock separator")
    func testSeparationPipeline() async throws {
        let mockSeparator = MockSeparator()
        let voice = AudioEngine(
            configuration: .default,
            separator: (.mossformer2spk, mockSeparator)
        )
        
        let audio = AudioBuffer.sine(frequency: 440, duration: 1.0, sampleRate: 16000)
        let tracks = try await voice.separate(audio, model: .mossformer2spk)
        
        #expect(tracks.count == 2)
        #expect(mockSeparator.separateCallCount == 1)
    }
    
    // MARK: - Pipeline Builder
    
    @Test("Pipeline builder chains stages correctly")
    func testPipelineBuilder() async throws {
        let mockVAD = MockVAD()
        let mockEnhancer = MockEnhancer()
        
        let voice = AudioEngine(
            configuration: .default,
            vad: mockVAD,
            enhancer: (.mossformerSE16k, mockEnhancer)
        )
        
        let result = try await voice.pipeline()
            .detect(.silero)
            .enhance(.mossformerSE16k)
            .process(audio: .silence(duration: 5.0, sampleRate: 16000))
        
        #expect(result.analysis != nil)
        #expect(result.analysis?.segments.count == 3)
        #expect(mockVAD.detectCallCount == 1)
        #expect(mockEnhancer.processCallCount >= 1)
    }

    @Test("Identified overlap tracks survive later stages and final output conversion")
    func testIdentifiedTracksPropagation() async throws {
        let diarizer = MockDiarization()
        diarizer.mockTimeline = SpeakerTimeline(segments: [
            DiarizedSegment(
                timeRange: TimeRange(start: 0, end: 1.25),
                speakerID: SpeakerID(0),
                confidence: 0.95
            ),
            DiarizedSegment(
                timeRange: TimeRange(start: 0.5, end: 1.75),
                speakerID: SpeakerID(1),
                confidence: 0.9
            ),
        ])
        let separator = MockSeparator(outputCount: 2)
        let enhancer = MockEnhancer()
        let engine = AudioEngine(
            configuration: .default,
            diarization: diarizer,
            enhancer: (.mossformerSE16k, enhancer),
            separator: (.mossformerWhamr, separator)
        )

        let pipeline = engine.pipeline()
            .diarize()
            .separateOverlap(.separate)
            .enhance(.mossformerSE16k)
        let result = try await engine.executePipeline(
            pipeline,
            audio: .sine(frequency: 220, duration: 2, sampleRate: 16_000),
            outputSampleRate: 48_000
        )

        #expect(result.identifiedTracks?.count == 2)
        #expect(result.identifiedTracks?.allSatisfy {
            $0.sourceTimeRange == TimeRange(start: 0.5, end: 1.25)
        } == true)
        #expect(result.identifiedTracks?.allSatisfy {
            $0.audio.sampleRate == 48_000
        } == true)
        #expect(result.audio?.sampleRate == 48_000)
    }
    
    @Test("Analyze runs VAD and diarization in parallel")
    func testAnalyzePipeline() async throws {
        let mockVAD = MockVAD()
        let mockDiarization = MockDiarization()
        
        let voice = AudioEngine(
            configuration: .default,
            vad: mockVAD,
            diarization: mockDiarization
        )
        
        let result = try await voice.analyze(.silence(duration: 5.0, sampleRate: 16000))
        
        #expect(result.segments.count == 3)
        #expect(result.speakers.speakerCount == 2)
        #expect(mockVAD.detectCallCount == 1)
        #expect(mockDiarization.diarizeCallCount == 1)
    }
    
    // MARK: - Conditional Execution
    
    @Test("Conditional executes then branch when true")
    func testConditionalThen() async throws {
        let mockVAD = MockVAD()
        let mockSeparator = MockSeparator()
        
        // Configure VAD to return overlapping speakers scenario
        let mockDiarization = MockDiarization()
        mockDiarization.mockTimeline = SpeakerTimeline(segments: [
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 3), speakerID: SpeakerID(0), confidence: 0.9),
            DiarizedSegment(timeRange: TimeRange(start: 2, end: 5), speakerID: SpeakerID(1), confidence: 0.85),
        ])
        
        let voice = AudioEngine(
            configuration: .default,
            vad: mockVAD,
            diarization: mockDiarization,
            separator: (.mossformerWhamr, mockSeparator)
        )
        
        let result = try await voice.pipeline()
            .analyze()
            .conditionally({ $0.analysis!.speakers.maxOverlappingSpeakers >= 2 }) {
                $0.separate(speakers: 2)
            }
            .process(audio: .silence(duration: 5.0, sampleRate: 16000))
        
        // Separation should have been called
        #expect(mockSeparator.separateCallCount == 1)
        #expect(result.separatedTracks?.count == 2)
    }
    
    @Test("Conditional skips when false")
    func testConditionalSkip() async throws {
        let mockVAD = MockVAD()
        let mockSeparator = MockSeparator()
        
        // Configure no overlapping speakers
        let mockDiarization = MockDiarization()
        mockDiarization.mockTimeline = SpeakerTimeline(segments: [
            DiarizedSegment(timeRange: TimeRange(start: 0, end: 2), speakerID: SpeakerID(0), confidence: 0.9),
            DiarizedSegment(timeRange: TimeRange(start: 3, end: 5), speakerID: SpeakerID(1), confidence: 0.85),
        ])
        
        let voice = AudioEngine(
            configuration: .default,
            vad: mockVAD,
            diarization: mockDiarization,
            separator: (.mossformerWhamr, mockSeparator)
        )
        
        let result = try await voice.pipeline()
            .analyze()
            .conditionally({ $0.analysis!.speakers.maxOverlappingSpeakers >= 2 }) {
                $0.separate(speakers: 2)
            }
            .process(audio: .silence(duration: 5.0, sampleRate: 16000))
        
        // Separation should NOT have been called
        #expect(mockSeparator.separateCallCount == 0)
        #expect(result.separatedTracks == nil)
    }
    
    // MARK: - Error Handling
    
    @Test("Pipeline throws when model not loaded")
    func testModelNotLoaded() async throws {
        let voice = AudioEngine()
        
        let audio = AudioBuffer.silence(duration: 1.0, sampleRate: 16000)
        
        do {
            _ = try await voice.detect(audio)
            Issue.record("Should have thrown modelNotLoaded")
        } catch let error as AudioToolError {
            if case .modelNotLoaded = error {
                // Expected
            } else {
                Issue.record("Wrong error type: \(error)")
            }
        }
    }
    
    // MARK: - Metrics
    
    @Test("Pipeline records execution metrics")
    func testPipelineMetrics() async throws {
        let mockVAD = MockVAD()
        mockVAD.processDelay = .milliseconds(50)
        
        let voice = AudioEngine(
            configuration: .default,
            vad: mockVAD
        )
        
        let result = try await voice.pipeline()
            .detect(.silero)
            .process(audio: .silence(duration: 1.0, sampleRate: 16000))
        
        #expect(result.metrics.totalDuration.components.seconds >= 0)
        #expect(result.metrics.stageDurations["vad"] != nil)
    }

    @Test("Parallel branches merge in declaration order and retain metrics")
    func testParallelMergeIsDeterministic() async throws {
        let enhancer = MockEnhancer()
        enhancer.scaleFactor = 0.25
        enhancer.processDelay = .milliseconds(75)
        let upscaler = MockUpscaler()
        upscaler.processDelay = .milliseconds(1)
        let engine = AudioEngine(
            configuration: .default,
            enhancer: (.mossformerSE16k, enhancer),
            upscaler: upscaler
        )
        let input = AudioBuffer(samples: [1, 2, 3, 4], sampleRate: 16_000)
        let pipeline = engine.pipeline().parallel {
            [
                PipelineBuilder().enhance(.mossformerSE16k),
                PipelineBuilder().upscale(),
            ]
        }

        let result = try await engine.executePipeline(
            pipeline,
            audio: input,
            outputSampleRate: 48_000
        )

        #expect(result.audio?.samples == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        #expect(result.metrics.stageDurations["parallel[0].enhancement"] != nil)
        #expect(result.metrics.stageDurations["parallel[1].upscaling"] != nil)
    }

    @Test("Stopping pipeline event consumption cancels in-flight inference")
    func testPipelineStreamCancellation() async throws {
        let vad = CancellationAwareVAD()
        let engine = AudioEngine(configuration: .default, vad: vad)

        for try await _ in engine.pipeline()
            .detect(.silero)
            .stream(source: .buffer(.silence(duration: 1, sampleRate: 16_000))) {
            break
        }

        // The first progress event is emitted before VAD begins. Give the producer
        // enough time to observe iterator termination; it must not finish the
        // deliberately long inference after the consumer has gone away.
        try await Task.sleep(for: .milliseconds(100))
        let counts = await vad.counts
        #expect(counts.completed == 0)
        #expect(counts.started == 0 || counts.cancelled == counts.started)
    }
}

private actor CancellationAwareVAD: VADProvider {
    nonisolated let sampleRate = 16_000
    nonisolated let inputChannels = 1
    nonisolated let outputChannels = 1
    nonisolated let minChunkSize = 512
    nonisolated let recommendedChunkSize = 16_000

    private var started = 0
    private var completed = 0
    private var cancelled = 0

    var counts: (started: Int, completed: Int, cancelled: Int) {
        (started, completed, cancelled)
    }

    func detect(_ audio: AudioBuffer) async throws -> [VADSegment] {
        try validateInputFormat(audio)
        started += 1
        do {
            try await Task.sleep(for: .seconds(10))
            completed += 1
            return []
        } catch is CancellationError {
            cancelled += 1
            throw CancellationError()
        }
    }

    nonisolated func streamDetection(
        _ audio: AsyncStream<AudioBuffer>
    ) -> AsyncThrowingStream<VADSegment, Error> {
        AsyncThrowingStream { continuation in continuation.finish() }
    }

    func reset() async {}
}

@Suite("Concurrency Tests")
struct ConcurrencyTests {
    
    @Test("BoundedChannel send and receive")
    func testBoundedChannel() async throws {
        let channel = BoundedChannel<Int>(capacity: 3)
        
        await channel.send(1)
        await channel.send(2)
        await channel.send(3)
        
        let a = await channel.receive()
        let b = await channel.receive()
        let c = await channel.receive()
        
        #expect(a == 1)
        #expect(b == 2)
        #expect(c == 3)
    }
    
    @Test("BoundedChannel close")
    func testBoundedChannelClose() async throws {
        let channel = BoundedChannel<Int>(capacity: 2)
        
        await channel.send(1)
        await channel.close()
        
        let a = await channel.receive()
        let b = await channel.receive()
        
        #expect(a == 1)
        #expect(b == nil)  // Closed, so returns nil
    }
    
    @Test("BoundedChannel async sequence")
    func testBoundedChannelSequence() async throws {
        let channel = BoundedChannel<Int>(capacity: 5)
        
        // Send values in background
        Task {
            for i in 1...5 {
                await channel.send(i)
            }
            await channel.close()
        }
        
        var received: [Int] = []
        for await value in await channel.values {
            received.append(value)
        }
        
        #expect(received == [1, 2, 3, 4, 5])
    }

    @Test("BoundedChannel timeout removes the blocked value")
    func testBoundedChannelTimeout() async throws {
        let channel = BoundedChannel<Int>(capacity: 1)
        await channel.send(1)
        let started = ContinuousClock.now

        do {
            try await channel.send(2, timeout: .milliseconds(25))
            Issue.record("expected backpressureTimeout")
        } catch let error as AudioToolError {
            guard case .backpressureTimeout = error else {
                Issue.record("unexpected error: \(error)")
                return
            }
        }

        #expect(ContinuousClock.now - started < .seconds(1),
                "timeout remained suspended behind the cancelled sender")
        #expect(await channel.receive() == 1)
        try await Task.sleep(for: .milliseconds(25))
        #expect(await channel.count == 0,
                "the timed-out value was enqueued after the call returned")
    }

    @Test("Cancelling a blocked sender removes its waiter")
    func testBoundedChannelSenderCancellation() async throws {
        let channel = BoundedChannel<Int>(capacity: 1)
        await channel.send(1)

        let sender = Task { await channel.send(2) }
        try await Task.sleep(for: .milliseconds(10))
        sender.cancel()
        _ = await sender.result

        #expect(await channel.receive() == 1)
        try await Task.sleep(for: .milliseconds(10))
        #expect(await channel.count == 0)
    }

    @Test("Cancelling an awakened sender transfers its slot")
    func testBoundedChannelAwakenedSenderCancellation() async throws {
        let channel = BoundedChannel<Int>(capacity: 1)
        await channel.send(0)

        let firstSender = Task(priority: .background) {
            await channel.send(1)
        }
        try await Task.sleep(for: .milliseconds(10))

        let secondSender = Task(priority: .background) { () -> Bool in
            do {
                try await channel.send(2, timeout: .seconds(1))
                return true
            } catch {
                return false
            }
        }
        try await Task.sleep(for: .milliseconds(10))

        // The high-priority consumer resumes before the background sender can
        // use the slot it was granted, making the cancellation-after-wake race
        // deterministic rather than timing dependent.
        let initialValue = await Task(priority: .userInitiated) {
            let value = await channel.receive()
            firstSender.cancel()
            return value
        }.value

        _ = await firstSender.result
        #expect(initialValue == 0)
        #expect(await secondSender.value,
                "the following sender remained suspended after the grant was abandoned")
        #expect(await channel.receive() == 2)
    }
    
    @Test("SegmentPool acquire and release")
    func testSegmentPool() async throws {
        let pool = SegmentPool(capacity: 4, segmentSize: 16000, sampleRate: 16000)
        
        // Acquire some buffers
        var buffer1 = await pool.acquire()
        let buffer2 = await pool.acquire()
        
        #expect(buffer1.frameCount == 16000)
        #expect(buffer2.frameCount == 16000)
        
        let stats = await pool.stats
        #expect(stats.currentlyInUse == 2)

        buffer1.samples[0] = 42
        
        // Release back
        await pool.release(buffer1)
        await pool.release(buffer2)
        
        let statsAfter = await pool.stats
        #expect(statsAfter.currentlyInUse == 0)
        #expect(statsAfter.available == 2)

        let reused = await pool.acquire()
        #expect(reused.samples[0] == 0, "reused scratch storage must be cleared")
        let finalStats = await pool.stats
        #expect(finalStats.totalAllocated == 2)
    }
}
