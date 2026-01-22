//
//  PipelineTests.swift
//  ClearVoice
//
//  Tests for pipeline execution
//

import Testing
@testable import ClearVoice
@testable import ClearVoiceCore

@Suite("Pipeline Tests")
struct PipelineTests {
    
    // MARK: - Basic Pipeline
    
    @Test("Pipeline detects audio with mock VAD")
    func testVADPipeline() async throws {
        let mockVAD = MockVAD()
        let voice = ClearVoice(
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
        let voice = ClearVoice(
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
        let voice = ClearVoice(
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
        let voice = ClearVoice(
            configuration: .default,
            separator: (.mossformer2spk, mockSeparator)
        )
        
        let audio = AudioBuffer.sine(frequency: 440, duration: 1.0, sampleRate: 16000)
        let tracks = try await voice.separate(audio, speakers: 2, model: .mossformer2spk)
        
        #expect(tracks.count == 2)
        #expect(mockSeparator.separateCallCount == 1)
    }
    
    // MARK: - Pipeline Builder
    
    @Test("Pipeline builder chains stages correctly")
    func testPipelineBuilder() async throws {
        let mockVAD = MockVAD()
        let mockEnhancer = MockEnhancer()
        
        let voice = ClearVoice(
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
    
    @Test("Analyze runs VAD and diarization in parallel")
    func testAnalyzePipeline() async throws {
        let mockVAD = MockVAD()
        let mockDiarization = MockDiarization()
        
        let voice = ClearVoice(
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
        
        let voice = ClearVoice(
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
        
        let voice = ClearVoice(
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
        let voice = ClearVoice()
        
        let audio = AudioBuffer.silence(duration: 1.0, sampleRate: 16000)
        
        do {
            _ = try await voice.detect(audio)
            Issue.record("Should have thrown modelNotLoaded")
        } catch let error as ClearVoiceError {
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
        
        let voice = ClearVoice(
            configuration: .default,
            vad: mockVAD
        )
        
        let result = try await voice.pipeline()
            .detect(.silero)
            .process(audio: .silence(duration: 1.0, sampleRate: 16000))
        
        #expect(result.metrics.totalDuration.components.seconds >= 0)
        #expect(result.metrics.stageDurations["vad"] != nil)
    }
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
    
    @Test("SegmentPool acquire and release")
    func testSegmentPool() async throws {
        let pool = SegmentPool(capacity: 4, segmentSize: 16000, sampleRate: 16000)
        
        // Acquire some buffers
        let buffer1 = await pool.acquire()
        let buffer2 = await pool.acquire()
        
        #expect(buffer1.frameCount == 16000)
        #expect(buffer2.frameCount == 16000)
        
        let stats = await pool.stats
        #expect(stats.currentlyInUse == 2)
        
        // Release back
        await pool.release(buffer1)
        await pool.release(buffer2)
        
        let statsAfter = await pool.stats
        #expect(statsAfter.currentlyInUse == 0)
        #expect(statsAfter.available == 2)
    }
}
