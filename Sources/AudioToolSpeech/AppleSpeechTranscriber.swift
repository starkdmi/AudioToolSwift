//
//  AppleSpeechTranscriber.swift
//  ClearVoiceSpeech
//
//  Apple SpeechAnalyzer provider for on-device speech-to-text (iOS 26+)
//

import Foundation
import Speech
import AVFoundation
import ClearVoiceCore

// Type alias to disambiguate from CoreAudioTypes.AudioBuffer
public typealias SpeechAudioBuffer = ClearVoiceCore.AudioBuffer

// MARK: - Apple Speech Transcriber

/// On-device speech transcription using Apple's SpeechAnalyzer API (iOS 26+)
@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
public final class AppleSpeechTranscriber: Transcriber, @unchecked Sendable {
    
    // MARK: - AudioProcessor Conformance
    
    public let sampleRate: Int = 16000
    public let inputChannels: Int = 1
    public let outputChannels: Int = 1
    
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
            throw ClearVoiceError.resourceUnavailable(
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
        print("Asset status for \(locale.identifier): \(assetStatus)")
        
        // System will auto-download when analysis starts if not installed
        isModelAvailable = true
    }
    
    // MARK: - Transcriber Conformance
    
    public func transcribe(_ audio: ClearVoiceCore.AudioBuffer) async throws -> Transcription {
        guard isModelAvailable else {
            throw ClearVoiceError.modelNotLoaded("Apple Speech - call load() first")
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
        
        print("Starting analysis...")
        
        // Start result collection FIRST (as a detached task to ensure it's listening)
        // Then run the analyzer
        // The results subscription MUST be active before analysis starts
        
        async let resultsTask: Void = Task {
            print("Results task started")
            for try await result in transcriber.results {
                print("Got result: \(result.isFinal ? "FINAL" : "volatile") - '\(String(result.text.characters))'")
                await collector.appendResult(result)
            }
            print("Results task completed")
        }.value
        
        // Give the results subscription a moment to set up
        try await Task.sleep(for: .milliseconds(100))
        
        // Now run the analysis
        print("Analyzer task started")
        try await analyzer.analyzeSequence(from: audioFile)
        print("Analyzer task completed, finalizing...")
        
        // Signal that all audio has been processed
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        print("Finalize completed")
        
        // Wait for results to finish
        try await resultsTask
        
        print("All tasks completed")
        
        return Transcription(
            text: await collector.getFullText().trimmingCharacters(in: .whitespacesAndNewlines),
            segments: await collector.getSegments(),
            language: locale.language.languageCode?.identifier
        )
    }
    
    public func streamTranscription(_ audio: AsyncStream<ClearVoiceCore.AudioBuffer>) -> AsyncThrowingStream<TranscriptionSegment, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard self.isModelAvailable else {
                    continuation.finish(throwing: ClearVoiceError.modelNotLoaded("Apple Speech"))
                    return
                }
                
                do {
                    var allSamples: [Float] = []
                    var finalSampleRate = 16000
                    
                    for await chunk in audio {
                        allSamples.append(contentsOf: chunk.samples)
                        finalSampleRate = chunk.sampleRate
                    }
                    
                    let combinedBuffer = ClearVoiceCore.AudioBuffer(
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
                            try await analyzer.analyzeSequence(from: audioFile)
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
    
    public func process(_ input: ClearVoiceCore.AudioBuffer) async throws -> ClearVoiceCore.AudioBuffer {
        return input
    }
}

// MARK: - AudioBuffer Extension

extension ClearVoiceCore.AudioBuffer {
    @available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
    func writeToTemporaryFile(at url: URL) throws -> AVAudioFile {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: AVAudioChannelCount(channels),
            interleaved: false
        ) else {
            throw ClearVoiceError.invalidAudioFormat(
                expected: "pcmFormatFloat32 @ \(sampleRate)Hz",
                found: "unsupported format"
            )
        }
        
        let audioFile = try AVAudioFile(forWriting: url, settings: format.settings)
        
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw ClearVoiceError.resourceUnavailable("Failed to create AVAudioPCMBuffer")
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
