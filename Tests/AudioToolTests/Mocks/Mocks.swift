//
//  Mocks.swift
//  ClearVoice
//
//  Mock implementations for testing
//

import Foundation
import ClearVoiceCore

// MARK: - Mock VAD

/// Mock VAD that returns predefined segments
public final class MockVAD: VADProvider, @unchecked Sendable {
    
    public let sampleRate: Int = 16000
    public let inputChannels: Int = 1
    public let outputChannels: Int = 1
    public let minChunkSize: Int = 512
    public let recommendedChunkSize: Int = 16000
    
    public var mockSegments: [VADSegment] = [
        VADSegment(timeRange: TimeRange(start: 0.0, end: 2.5), isSpeech: true, probability: 0.95),
        VADSegment(timeRange: TimeRange(start: 2.5, end: 3.0), isSpeech: false, probability: 0.1),
        VADSegment(timeRange: TimeRange(start: 3.0, end: 5.5), isSpeech: true, probability: 0.92),
    ]
    
    public var processDelay: Duration = .milliseconds(10)
    public var detectCallCount = 0
    
    public init() {}
    
    public func detect(_ audio: AudioBuffer) async throws -> [VADSegment] {
        detectCallCount += 1
        try await Task.sleep(for: processDelay)
        return mockSegments
    }
    
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        input
    }
    
    public func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            Task {
                for await chunk in input {
                    continuation.yield(chunk)
                }
                continuation.finish()
            }
        }
    }
    
    public func streamDetection(_ audio: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<VADSegment, Error> {
        AsyncThrowingStream { continuation in
            Task {
                for segment in mockSegments {
                    try await Task.sleep(for: .milliseconds(5))
                    continuation.yield(segment)
                }
                continuation.finish()
            }
        }
    }
    
    public func reset() async {}
}

// MARK: - Mock Diarization

/// Mock diarization that returns predefined speaker timeline
public final class MockDiarization: DiarizationProvider, @unchecked Sendable {
    
    public let sampleRate: Int = 16000
    public let inputChannels: Int = 1
    public let outputChannels: Int = 1
    
    public var mockTimeline: SpeakerTimeline = SpeakerTimeline(segments: [
        DiarizedSegment(timeRange: TimeRange(start: 0.0, end: 2.5), speakerID: SpeakerID(0), confidence: 0.9),
        DiarizedSegment(timeRange: TimeRange(start: 3.0, end: 5.5), speakerID: SpeakerID(1), confidence: 0.85),
    ])
    
    public var processDelay: Duration = .milliseconds(10)
    public var diarizeCallCount = 0
    
    public init() {}
    
    public func diarize(_ audio: AudioBuffer) async throws -> SpeakerTimeline {
        diarizeCallCount += 1
        try await Task.sleep(for: processDelay)
        return mockTimeline
    }
    
    public func diarize(_ audio: AudioBuffer, vadHint: [VADSegment]) async throws -> SpeakerTimeline {
        diarizeCallCount += 1
        try await Task.sleep(for: processDelay)
        return mockTimeline
    }
    
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        input
    }
}

// MARK: - Mock Enhancer

/// Mock enhancer that applies a small modification to verify processing
public final class MockEnhancer: SpeechEnhancer, @unchecked Sendable {
    
    public let sampleRate: Int = 16000
    public let inputChannels: Int = 1
    public let outputChannels: Int = 1
    public let minChunkSize: Int = 512
    public let recommendedChunkSize: Int = 16000
    
    public var processDelay: Duration = .milliseconds(10)
    public var scaleFactor: Float = 0.99
    public var processCallCount = 0
    
    public init() {}
    
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        processCallCount += 1
        try await Task.sleep(for: processDelay)
        return AudioBuffer(
            samples: input.samples.map { $0 * scaleFactor },
            sampleRate: input.sampleRate,
            channels: input.channels
        )
    }
    
    public func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            Task {
                for await chunk in input {
                    let processed = try await process(chunk)
                    continuation.yield(processed)
                }
                continuation.finish()
            }
        }
    }
    
    public func reset() async {}
}

// MARK: - Mock Separator

/// Mock separator that splits audio into N identical copies
public final class MockSeparator: SpeechSeparator, @unchecked Sendable {
    
    public let sampleRate: Int = 16000
    public let inputChannels: Int = 1
    public let outputChannels: Int = 1
    public let supportedSpeakerCounts: [Int] = [2, 3]
    
    public var processDelay: Duration = .milliseconds(10)
    public var separateCallCount = 0
    
    public init() {}
    
    public func separate(_ audio: AudioBuffer, speakers: Int) async throws -> [AudioBuffer] {
        separateCallCount += 1
        try await Task.sleep(for: processDelay)
        
        // Return scaled copies for each "speaker"
        return (0..<speakers).map { index in
            let scale = 1.0 - Float(index) * 0.1  // Each speaker slightly different
            return AudioBuffer(
                samples: audio.samples.map { $0 * scale },
                sampleRate: audio.sampleRate,
                channels: audio.channels
            )
        }
    }
    
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        input
    }
}

// MARK: - Mock Transcriber

/// Mock transcriber with predictable output
public final class MockTranscriber: Transcriber, @unchecked Sendable {
    
    public let sampleRate: Int = 16000
    public let inputChannels: Int = 1
    public let outputChannels: Int = 1
    
    public var mockTranscription = Transcription(
        text: "Hello world. This is a test.",
        segments: [
            TranscriptionSegment(text: "Hello world.", timeRange: TimeRange(start: 0, end: 1.5), confidence: 0.95),
            TranscriptionSegment(text: "This is a test.", timeRange: TimeRange(start: 1.5, end: 3.0), confidence: 0.92),
        ],
        language: "en"
    )
    
    public var processDelay: Duration = .milliseconds(10)
    public var transcribeCallCount = 0
    
    public init() {}
    
    public func transcribe(_ audio: AudioBuffer) async throws -> Transcription {
        transcribeCallCount += 1
        try await Task.sleep(for: processDelay)
        return mockTranscription
    }
    
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        input
    }
    
    public func streamTranscription(_ audio: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<TranscriptionSegment, Error> {
        AsyncThrowingStream { continuation in
            Task {
                for segment in mockTranscription.segments {
                    try await Task.sleep(for: .milliseconds(10))
                    continuation.yield(segment)
                }
                continuation.finish()
            }
        }
    }
}

// MARK: - Mock Upscaler

/// Mock upscaler that doubles sample rate
public final class MockUpscaler: AudioUpscaler, @unchecked Sendable {
    
    public let sampleRate: Int = 16000
    public let inputChannels: Int = 1
    public let outputChannels: Int = 1
    public let inputSampleRate: Int = 16000
    public let outputSampleRate: Int = 48000
    
    public var processDelay: Duration = .milliseconds(10)
    public var processCallCount = 0
    
    public init() {}
    
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        processCallCount += 1
        try await Task.sleep(for: processDelay)
        
        // Simple upsampling by tripling samples
        var upsampled: [Float] = []
        for sample in input.samples {
            upsampled.append(sample)
            upsampled.append(sample)
            upsampled.append(sample)
        }
        
        return AudioBuffer(
            samples: upsampled,
            sampleRate: outputSampleRate,
            channels: input.channels
        )
    }
}

// MARK: - Mock Classifier

/// Mock classifier that returns predefined classifications
public final class MockClassifier: SoundClassifier, @unchecked Sendable {
    
    public let sampleRate: Int = 16000
    public let inputChannels: Int = 1
    public let outputChannels: Int = 1
    
    public var mockClassifications: [SoundClassification] = [
        SoundClassification(label: "speech", confidence: 0.9, timeRange: TimeRange(start: 0, end: 2)),
        SoundClassification(label: "music", confidence: 0.7, timeRange: TimeRange(start: 2, end: 4)),
    ]
    
    public var processDelay: Duration = .milliseconds(10)
    public var classifyCallCount = 0
    
    public init() {}
    
    public func classify(_ audio: AudioBuffer) async throws -> [SoundClassification] {
        classifyCallCount += 1
        try await Task.sleep(for: processDelay)
        return mockClassifications
    }
    
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        input
    }
}
