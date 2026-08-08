//
//  AppleSpeechTranscriber.swift
//  AudioToolSpeech
//
//  Apple SpeechAnalyzer provider for on-device speech-to-text (iOS 26+)
//
//  Note: This file requires Xcode 26+ SDK to compile. On earlier SDKs,
//  a placeholder implementation is provided.
//

import Foundation
import os

/// Transcription lifecycle. Never log recognised text: it is the user's speech, and a
/// library has no business putting it on the host app's console.
private let speechLogger = Logger(subsystem: "AudioToolSwift", category: "AppleSpeech")
import AVFoundation
import AudioToolCore

// Type alias to disambiguate from CoreAudioTypes.AudioBuffer
public typealias SpeechAudioBuffer = AudioToolCore.AudioBuffer

// Check if we're building with the iOS/macOS 26+ SDK
// SpeechAnalyzer is only available in Speech framework on iOS 26+/macOS 26+
#if canImport(Speech) && swift(>=6.0)
import Speech

// Check for SpeechAnalyzer availability using compiler version
// This type only exists in Xcode 26+ SDK
#if compiler(>=6.2)

// MARK: - Apple Speech Transcriber (Full Implementation)

private struct AppleSpeechCollection: Sendable {
    var text = ""
    var segments: [TranscriptionSegment] = []
}

/// On-device speech transcription using Apple's SpeechAnalyzer API (iOS 26+)
@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
public actor AppleSpeechTranscriber: Transcriber {
    
    // MARK: - AudioProcessor Conformance
    
    public nonisolated let sampleRate: Int = 16000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1
    
    // MARK: - Properties
    
    private let locale: Locale
    private var isModelAvailable: Bool = false
    
    // MARK: - Supported Locales
    
    public static func supportedLocales() async -> [Locale] {
        await SpeechTranscriber.supportedLocales
    }
    
    public static func isLocaleSupported(_ locale: Locale) async -> Bool {
        let locales = await SpeechTranscriber.supportedLocales
        return locales.contains(locale)
    }
    
    // MARK: - Initialization
    
    public init(locale: String = "en-US") {
        self.locale = Locale(identifier: locale)
    }
    
    public init(locale: Locale) {
        self.locale = locale
    }
    
    // MARK: - Model Loading
    
    public func load() async throws {
        // Check locale support
        let isSupported = await Self.isLocaleSupported(locale)
        guard isSupported else {
            throw AudioToolError.resourceUnavailable(
                "Locale '\(locale.identifier)' not supported by SpeechTranscriber"
            )
        }
        
        // Create a transcriber to check asset status
        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: []
        )
        
        // Check asset status for the transcriber module
        let assetStatus = await AssetInventory.status(forModules: [transcriber])
        speechLogger.debug("Asset status for \(self.locale.identifier, privacy: .public): \(String(describing: assetStatus), privacy: .public)")
        
        // System will auto-download when analysis starts if not installed
        isModelAvailable = true
    }
    
    // MARK: - Transcriber Conformance
    
    public func transcribe(_ audio: AudioToolCore.AudioBuffer) async throws -> Transcription {
        try await performTranscription(audio, onProgress: nil)
    }
    
    /// Stream transcription segments from audio stream.
    ///
    /// - Important: Apple Speech API requires complete audio for analysis.
    ///   This implementation buffers the entire input stream before processing,
    ///   so it does not provide true real-time streaming. Segments are emitted
    ///   after the input stream completes.
    ///
    /// - Parameter audio: Async stream of audio chunks
    /// - Returns: Stream of transcription segments
    public nonisolated func streamTranscription(_ audio: AsyncStream<AudioToolCore.AudioBuffer>) -> AsyncThrowingStream<TranscriptionSegment, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                guard await self.isModelAvailable else {
                    continuation.finish(throwing: AudioToolError.modelNotLoaded("Apple Speech"))
                    return
                }
                
                do {
                    var allSamples: [Float] = []
                    
                    for await chunk in audio {
                        try Task.checkCancellation()
                        guard chunk.sampleRate == self.sampleRate else {
                            throw AudioToolError.sampleRateMismatch(
                                expected: self.sampleRate,
                                found: chunk.sampleRate
                            )
                        }
                        guard chunk.channels == self.inputChannels else {
                            throw AudioToolError.channelCountMismatch(
                                expected: self.inputChannels,
                                found: chunk.channels
                            )
                        }
                        allSamples.append(contentsOf: chunk.samples)
                    }
                    guard !allSamples.isEmpty else { throw AudioToolError.emptyAudioBuffer }
                    
                    let combinedBuffer = AudioToolCore.AudioBuffer(
                        samples: allSamples,
                        sampleRate: self.sampleRate,
                        channels: 1
                    )
                    
                    let tempURL = FileManager.default.temporaryDirectory
                        .appendingPathComponent(UUID().uuidString)
                        .appendingPathExtension("wav")
                    
                    defer {
                        try? FileManager.default.removeItem(at: tempURL)
                    }
                    
                    let audioFile = try combinedBuffer.writeToTemporaryFile(at: tempURL)
                    
                    let transcriber = SpeechTranscriber(
                        locale: self.locale,
                        transcriptionOptions: [],
                        reportingOptions: [],
                        attributeOptions: []
                    )
                    
                    let analyzer = SpeechAnalyzer(modules: [transcriber])
                    let subscriptionReady = AsyncStream<Void>.makeStream()
                    let resultsTask = Task {
                        // Construct the iterator before analysis can start. Unlike
                        // Task.yield(), this handshake establishes that the result
                        // sequence has an active subscriber under scheduler pressure.
                        var resultsIterator = transcriber.results.makeAsyncIterator()
                        _ = subscriptionReady.continuation.yield(())
                        subscriptionReady.continuation.finish()
                        while let result = try await resultsIterator.next() {
                            try Task.checkCancellation()
                            guard result.isFinal,
                                  let segment = Self.segment(from: result) else { continue }
                            if case .terminated = continuation.yield(segment) {
                                throw CancellationError()
                            }
                        }
                    }

                    do {
                        var readyIterator = subscriptionReady.stream.makeAsyncIterator()
                        guard await readyIterator.next() != nil else {
                            throw CancellationError()
                        }
                        try Task.checkCancellation()
                        _ = try await analyzer.analyzeSequence(from: audioFile)
                        // Without finalization the result sequence may never close
                        // and the final hypothesis may never be emitted.
                        try await analyzer.finalizeAndFinishThroughEndOfInput()
                        try await resultsTask.value
                    } catch {
                        resultsTask.cancel()
                        _ = try? await resultsTask.value
                        throw error
                    }

                    continuation.finish()
                    
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
        }
    }
    
    // MARK: - AudioProcessor Conformance
    
    
    // MARK: - Progress-Aware Transcription
    
    /// Transcribe with progress reporting
    /// Reports progress as segments are recognized (real-time from SpeechAnalyzer)
    public func transcribe(
        _ audio: AudioToolCore.AudioBuffer,
        onProgress: ProgressCallback?
    ) async throws -> Transcription {
        try await performTranscription(audio, onProgress: onProgress)
    }

    private func performTranscription(
        _ audio: AudioToolCore.AudioBuffer,
        onProgress: ProgressCallback?
    ) async throws -> Transcription {
        guard isModelAvailable else {
            throw AudioToolError.modelNotLoaded("Apple Speech - call load() first")
        }
        try validateInputFormat(audio)
        guard !audio.samples.isEmpty else { throw AudioToolError.emptyAudioBuffer }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let audioFile = try audio.writeToTemporaryFile(at: tempURL)
        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: []
        )
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let duration = max(audio.duration, 0.001)

        await onProgress?(5)
        let subscriptionReady = AsyncStream<Void>.makeStream()
        let resultsTask = Task { () throws -> AppleSpeechCollection in
            var collection = AppleSpeechCollection()
            var reportedProgress = 5.0
            // Signal only after the result iterator exists; `Task.yield()` does
            // not guarantee that this task has run before analysis begins.
            var resultsIterator = transcriber.results.makeAsyncIterator()
            _ = subscriptionReady.continuation.yield(())
            subscriptionReady.continuation.finish()
            while let result = try await resultsIterator.next() {
                try Task.checkCancellation()
                speechLogger.debug("Result received (\(result.isFinal ? "final" : "volatile", privacy: .public)); text withheld")
                guard result.isFinal, let segment = Self.segment(from: result) else { continue }
                collection.text += String(result.text.characters)
                collection.segments.append(segment)
                reportedProgress = max(
                    reportedProgress,
                    max(5, min(segment.timeRange.end / duration, 0.95) * 100)
                )
                await onProgress?(reportedProgress)
            }
            return collection
        }

        let collection: AppleSpeechCollection
        do {
            var readyIterator = subscriptionReady.stream.makeAsyncIterator()
            guard await readyIterator.next() != nil else {
                throw CancellationError()
            }
            try Task.checkCancellation()
            _ = try await analyzer.analyzeSequence(from: audioFile)
            try await analyzer.finalizeAndFinishThroughEndOfInput()
            collection = try await resultsTask.value
        } catch {
            resultsTask.cancel()
            _ = try? await resultsTask.value
            throw error
        }

        await onProgress?(100)
        return Transcription(
            text: collection.text.trimmingCharacters(in: .whitespacesAndNewlines),
            segments: collection.segments,
            language: locale.language.languageCode?.identifier
        )
    }

    private nonisolated static func segment(
        from result: SpeechTranscriber.Result
    ) -> TranscriptionSegment? {
        let text = String(result.text.characters)
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        let start = result.range.start.seconds
        let duration = result.range.duration.seconds
        guard start.isFinite, duration.isFinite, start >= 0, duration >= 0 else { return nil }
        return TranscriptionSegment(
            text: text,
            timeRange: TimeRange(start: start, end: start + duration),
            speakerID: nil,
            confidence: 1
        )
    }
}

// MARK: - AudioBuffer Extension

extension AudioToolCore.AudioBuffer {
    @available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
    func writeToTemporaryFile(at url: URL) throws -> AVAudioFile {
        guard channels > 0, samples.count.isMultiple(of: channels) else {
            throw AudioToolError.invalidAudioFormat(
                expected: "interleaved samples divisible by channel count",
                found: "\(samples.count) samples / \(channels) channels"
            )
        }
        let frameCount = samples.count / channels
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: AVAudioChannelCount(channels),
            interleaved: false
        ) else {
            throw AudioToolError.invalidAudioFormat(
                expected: "pcmFormatFloat32 @ \(sampleRate)Hz",
                found: "unsupported format"
            )
        }
        
        let audioFile = try AVAudioFile(forWriting: url, settings: format.settings)
        
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(frameCount)
        ) else {
            throw AudioToolError.resourceUnavailable("Failed to create AVAudioPCMBuffer")
        }
        
        buffer.frameLength = AVAudioFrameCount(frameCount)
        
        if let channelData = buffer.floatChannelData {
            for frame in 0..<frameCount {
                for channel in 0..<channels {
                    channelData[channel][frame] = samples[frame * channels + channel]
                }
            }
        }
        
        try audioFile.write(from: buffer)
        
        return try AVAudioFile(forReading: url)
    }
}

#else

// MARK: - Placeholder Implementation (Pre-Xcode 26 SDK)

/// Placeholder for AppleSpeechTranscriber when building with Xcode < 26
/// 
/// This placeholder is compiled when the SpeechAnalyzer API is not available.
/// To use actual Apple Speech transcription, build with Xcode 26+ on macOS 26+.
@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
public actor AppleSpeechTranscriber: Transcriber {
    
    public nonisolated let sampleRate: Int = 16000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1
    
    public init(locale: String = "en-US") {}
    public init(locale: Locale) {}
    
    public static func supportedLocales() async -> [Locale] { [] }
    public static func isLocaleSupported(_ locale: Locale) async -> Bool { false }
    
    public func load() async throws {
        throw AudioToolError.resourceUnavailable(
            "Apple SpeechAnalyzer requires macOS 26+ and Xcode 26+ SDK. " +
            "Current SDK does not include SpeechAnalyzer API."
        )
    }
    
    public func transcribe(_ audio: AudioToolCore.AudioBuffer) async throws -> Transcription {
        throw AudioToolError.resourceUnavailable("SpeechAnalyzer not available in this SDK")
    }
    
    public nonisolated func streamTranscription(_ audio: AsyncStream<AudioToolCore.AudioBuffer>) -> AsyncThrowingStream<TranscriptionSegment, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: AudioToolError.resourceUnavailable("SpeechAnalyzer not available"))
        }
    }
    
}

#endif // compiler(>=6.2)
#endif // canImport(Speech)
