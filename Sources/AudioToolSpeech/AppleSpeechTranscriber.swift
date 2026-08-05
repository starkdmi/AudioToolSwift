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
        guard isModelAvailable else {
            throw AudioToolError.modelNotLoaded("Apple Speech - call load() first")
        }
        
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        
        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }
        
        let audioFile = try audio.writeToTemporaryFile(at: tempURL)
        
        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: []
        )
        
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        
        actor ResultCollector {
            var fullText = ""
            var segments: [TranscriptionSegment] = []
            var currentTime: Double = 0
            
            func appendResult(_ result: SpeechTranscriber.Result) {
                if result.isFinal {
                    let text = String(result.text.characters)
                    fullText += text
                    
                    let duration = Double(text.count) * 0.1
                    let segment = TranscriptionSegment(
                        text: text,
                        timeRange: TimeRange(start: currentTime, end: currentTime + duration),
                        speakerID: nil,
                        confidence: 1.0
                    )
                    segments.append(segment)
                    currentTime += duration
                }
            }
            
            func getFullText() -> String { fullText }
            func getSegments() -> [TranscriptionSegment] { segments }
        }
        
        let collector = ResultCollector()
        
        speechLogger.debug("Starting analysis...")
        
        // Start result collection FIRST (as a detached task to ensure it's listening)
        // Then run the analyzer
        // The results subscription MUST be active before analysis starts
        
        async let resultsTask: Void = Task {
            speechLogger.debug("Results task started")
            for try await result in transcriber.results {
                speechLogger.debug("Result received (\(result.isFinal ? "final" : "volatile", privacy: .public)); text withheld")
                await collector.appendResult(result)
            }
            speechLogger.debug("Results task completed")
        }.value
        
        // Give the results subscription a moment to set up
        try await Task.sleep(for: .milliseconds(100))
        
        // Now run the analysis
        speechLogger.debug("Analyzer task started")
        _ = try await analyzer.analyzeSequence(from: audioFile)
        speechLogger.debug("Analyzer task completed, finalizing...")
        
        // Signal that all audio has been processed
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        speechLogger.debug("Finalize completed")
        
        // Wait for results to finish
        try await resultsTask
        
        speechLogger.debug("All tasks completed")
        
        return Transcription(
            text: await collector.getFullText().trimmingCharacters(in: .whitespacesAndNewlines),
            segments: await collector.getSegments(),
            language: locale.language.languageCode?.identifier
        )
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
            Task {
                guard await self.isModelAvailable else {
                    continuation.finish(throwing: AudioToolError.modelNotLoaded("Apple Speech"))
                    return
                }
                
                do {
                    var allSamples: [Float] = []
                    var finalSampleRate = 16000
                    
                    for await chunk in audio {
                        allSamples.append(contentsOf: chunk.samples)
                        finalSampleRate = chunk.sampleRate
                    }
                    
                    let combinedBuffer = AudioToolCore.AudioBuffer(
                        samples: allSamples,
                        sampleRate: finalSampleRate,
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
                    
                    try await withThrowingTaskGroup(of: Void.self) { group in
                        group.addTask {
                            _ = try await analyzer.analyzeSequence(from: audioFile)
                        }
                        
                        group.addTask {
                            var currentTime: Double = 0
                            for try await result in transcriber.results {
                                let text = String(result.text.characters)
                                let duration = Double(text.count) * 0.1
                                
                                let segment = TranscriptionSegment(
                                    text: text,
                                    timeRange: TimeRange(start: currentTime, end: currentTime + duration),
                                    speakerID: nil,
                                    confidence: 1.0
                                )
                                continuation.yield(segment)
                                currentTime += duration
                            }
                        }
                        
                        try await group.waitForAll()
                    }
                    
                    continuation.finish()
                    
                } catch {
                    continuation.finish(throwing: error)
                }
            }
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
        guard isModelAvailable else {
            throw AudioToolError.modelNotLoaded("Apple Speech - call load() first")
        }
        
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        
        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }
        
        let audioFile = try audio.writeToTemporaryFile(at: tempURL)
        let audioDuration = audio.duration
        
        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: []
        )
        
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        
        actor ProgressAwareCollector {
            var fullText = ""
            var segments: [TranscriptionSegment] = []
            var currentTime: Double = 0
            let totalDuration: Double
            let onProgress: ProgressCallback?
            
            init(totalDuration: Double, onProgress: ProgressCallback?) {
                self.totalDuration = totalDuration
                self.onProgress = onProgress
            }
            
            func appendResult(_ result: SpeechTranscriber.Result) async {
                if result.isFinal {
                    let text = String(result.text.characters)
                    fullText += text
                    
                    let duration = Double(text.count) * 0.1
                    let segment = TranscriptionSegment(
                        text: text,
                        timeRange: TimeRange(start: currentTime, end: currentTime + duration),
                        speakerID: nil,
                        confidence: 1.0
                    )
                    segments.append(segment)
                    currentTime += duration
                    
                    // Report progress based on estimated time processed
                    let progress = min(currentTime / max(totalDuration, 1.0), 0.95) * 100
                    await onProgress?(progress)
                }
            }
            
            func getFullText() -> String { fullText }
            func getSegments() -> [TranscriptionSegment] { segments }
        }
        
        let collector = ProgressAwareCollector(totalDuration: audioDuration, onProgress: onProgress)
        
        await onProgress?(5.0)  // Starting
        
        async let resultsTask: Void = Task {
            for try await result in transcriber.results {
                await collector.appendResult(result)
            }
        }.value
        
        try await Task.sleep(for: .milliseconds(100))
        
        _ = try await analyzer.analyzeSequence(from: audioFile)
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        
        try await resultsTask
        
        await onProgress?(100.0)  // Complete
        
        return Transcription(
            text: await collector.getFullText().trimmingCharacters(in: .whitespacesAndNewlines),
            segments: await collector.getSegments(),
            language: locale.language.languageCode?.identifier
        )
    }
}

// MARK: - AudioBuffer Extension

extension AudioToolCore.AudioBuffer {
    @available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
    func writeToTemporaryFile(at url: URL) throws -> AVAudioFile {
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
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw AudioToolError.resourceUnavailable("Failed to create AVAudioPCMBuffer")
        }
        
        buffer.frameLength = AVAudioFrameCount(samples.count)
        
        if let channelData = buffer.floatChannelData {
            for (index, sample) in samples.enumerated() {
                channelData[0][index] = sample
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
