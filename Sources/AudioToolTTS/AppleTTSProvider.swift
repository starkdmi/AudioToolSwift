//
//  AppleTTSProvider.swift
//  AudioToolTTS
//
//  Apple AVSpeechSynthesizer TTS provider for multilingual speech synthesis
//  Works in headless/CLI environments - no UI required
//

import Foundation
import AVFoundation
import AudioToolCore

final class AppleTTSBatchOperation: @unchecked Sendable {
    typealias Output = AudioToolCore.AudioBuffer

    private let lock = NSLock()
    private var result: Result<Output, Error>?
    private var continuation: CheckedContinuation<Output, Error>?
    private var synthesizer: AVSpeechSynthesizer?

    var isFinished: Bool {
        lock.lock()
        defer { lock.unlock() }
        return result != nil
    }

    func install(_ continuation: CheckedContinuation<Output, Error>) {
        lock.lock()
        if let result {
            lock.unlock()
            continuation.resume(with: result)
        } else {
            self.continuation = continuation
            lock.unlock()
        }
    }

    /// Retain the synthesizer for cancellation. Returns false if cancellation
    /// already won the race and the caller should stop starting work.
    func install(_ synthesizer: AVSpeechSynthesizer) -> Bool {
        lock.lock()
        guard result == nil else {
            lock.unlock()
            synthesizer.stopSpeaking(at: .immediate)
            return false
        }
        self.synthesizer = synthesizer
        lock.unlock()
        return true
    }

    func complete(with result: Result<Output, Error>) {
        lock.lock()
        guard self.result == nil else {
            lock.unlock()
            return
        }
        self.result = result
        let continuation = continuation
        self.continuation = nil
        synthesizer = nil
        lock.unlock()
        continuation?.resume(with: result)
    }

    func cancel() {
        lock.lock()
        guard result == nil else {
            lock.unlock()
            return
        }
        let cancellation = Result<Output, Error>.failure(CancellationError())
        result = cancellation
        let continuation = continuation
        self.continuation = nil
        let synthesizer = synthesizer
        self.synthesizer = nil
        lock.unlock()

        synthesizer?.stopSpeaking(at: .immediate)
        continuation?.resume(with: cancellation)
    }
}

private final class AppleTTSAudioCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Float] = []
    private var sampleRate: Double = 22_050
    private var complete = false

    var isComplete: Bool {
        lock.lock()
        defer { lock.unlock() }
        return complete
    }

    func append(_ newSamples: [Float], sampleRate: Double) {
        lock.lock()
        self.sampleRate = sampleRate
        samples.append(contentsOf: newSamples)
        lock.unlock()
    }

    func finish() {
        lock.lock()
        complete = true
        lock.unlock()
    }

    func snapshot() -> (samples: [Float], sampleRate: Double, isComplete: Bool) {
        lock.lock()
        defer { lock.unlock() }
        return (samples, sampleRate, complete)
    }
}

private final class AppleTTSStreamCompletion: @unchecked Sendable {
    private let lock = NSLock()
    private var complete = false
    private var cancelled = false

    var isComplete: Bool {
        lock.lock()
        defer { lock.unlock() }
        return complete
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func finish() {
        lock.lock()
        complete = true
        lock.unlock()
    }

    func cancel() {
        lock.lock()
        if !complete {
            cancelled = true
        }
        lock.unlock()
    }
}

/// Apple TTS provider using AVSpeechSynthesizer
///
/// Supports 60+ languages with no external dependencies.
/// Works offline with system voices. Fully programmatic - no UI popups.
///
/// Usage:
/// ```swift
/// let tts = TTSProviders.apple(language: "fr-FR")
/// let audio = try await tts.synthesize("Bonjour le monde", voice: "Thomas")
/// ```
public actor AppleTTSProvider: SpeechSynthesizer {
    
    // MARK: - SpeechSynthesizer Conformance
    
    public nonisolated let sampleRate: Int = 22050  // AVSpeechSynthesizer default
    
    public nonisolated var availableVoices: [String] {
        AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.hasPrefix(languageCode) || $0.language == language }
            .map { $0.identifier }
    }
    
    // MARK: - Properties
    
    private nonisolated let language: String
    private nonisolated let languageCode: String
    private nonisolated let synthesisTimeout: TimeInterval
    
    // MARK: - Initialization
    
    /// Initialize Apple TTS provider
    /// - Parameters:
    ///   - language: BCP-47 language code (e.g., "en-US", "fr-FR", "de-DE", "ru-RU")
    ///   - timeout: Maximum time to wait for synthesis completion (default: 60 seconds)
    public init(language: String = "en-US", timeout: TimeInterval = 60.0) {
        self.language = language
        self.languageCode = String(language.prefix(2))
        self.synthesisTimeout = timeout
    }
    
    // MARK: - SpeechSynthesizer Protocol
    
    /// Synthesize text to speech
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice identifier or name (e.g., "Thomas") - empty for default
    /// - Returns: Audio buffer with synthesized speech
    public func synthesize(_ text: String, voice: String) async throws -> AudioToolCore.AudioBuffer {
        let operation = AppleTTSBatchOperation()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                operation.install(continuation)
                guard !operation.isFinished else { return }

                // Run on a background thread with its own run loop.
                DispatchQueue.global(qos: .userInitiated).async {
                    let synthesizer = AVSpeechSynthesizer()
                    guard operation.install(synthesizer) else { return }

                    let utterance = AVSpeechUtterance(string: text)

                    // Set voice
                    if let selectedVoice = self.findVoice(voice) {
                        utterance.voice = selectedVoice
                    } else {
                        utterance.voice = AVSpeechSynthesisVoice(language: self.language)
                    }

                    let collector = AppleTTSAudioCollector()

                    synthesizer.write(utterance) { buffer in
                        if let pcmBuffer = buffer as? AVAudioPCMBuffer,
                           pcmBuffer.frameLength > 0 {
                            let samples = self.extractSamples(from: pcmBuffer)
                            collector.append(
                                samples,
                                sampleRate: pcmBuffer.format.sampleRate
                            )
                        } else {
                            // Completion signal
                            collector.finish()
                        }
                    }

                    // Pump the run loop until synthesis completes (required for
                    // headless environments).
                    let runLoop = RunLoop.current
                    let timeout = Date(timeIntervalSinceNow: self.synthesisTimeout)
                    while !collector.isComplete && !operation.isFinished && Date() < timeout {
                        runLoop.run(
                            mode: .default,
                            before: Date(timeIntervalSinceNow: 0.05)
                        )
                    }

                    guard !operation.isFinished else { return }
                    let snapshot = collector.snapshot()
                    if !snapshot.isComplete {
                        synthesizer.stopSpeaking(at: .immediate)
                        operation.complete(with: .failure(
                            AudioToolError.resourceUnavailable(
                                "Synthesis timed out after \(Int(self.synthesisTimeout)) seconds"
                            )
                        ))
                    } else if snapshot.samples.isEmpty {
                        operation.complete(with: .failure(
                            AudioToolError.resourceUnavailable("No audio generated")
                        ))
                    } else {
                        let result = AudioToolCore.AudioBuffer(
                            samples: snapshot.samples,
                            sampleRate: Int(snapshot.sampleRate),
                            channels: 1
                        )
                        operation.complete(with: .success(result))
                    }
                }
            }
        } onCancel: {
            operation.cancel()
        }
    }
    
    /// Stream synthesized audio chunks
    public nonisolated func streamSynthesis(_ text: String, voice: String) -> AsyncThrowingStream<AudioToolCore.AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            let completion = AppleTTSStreamCompletion()
            let producer = DispatchWorkItem {
                let synthesizer = AVSpeechSynthesizer()
                
                let utterance = AVSpeechUtterance(string: text)
                
                if let selectedVoice = self.findVoice(voice) {
                    utterance.voice = selectedVoice
                } else {
                    utterance.voice = AVSpeechSynthesisVoice(language: self.language)
                }
                
                synthesizer.write(utterance) { buffer in
                    if let pcmBuffer = buffer as? AVAudioPCMBuffer, pcmBuffer.frameLength > 0 {
                        let samples = self.extractSamples(from: pcmBuffer)
                        let chunk = AudioToolCore.AudioBuffer(
                            samples: samples,
                            sampleRate: Int(pcmBuffer.format.sampleRate),
                            channels: 1
                        )
                        if case .terminated = continuation.yield(chunk) {
                            completion.cancel()
                        }
                    } else {
                        completion.finish()
                    }
                }
                
                // Pump run loop
                let runLoop = RunLoop.current
                let timeout = Date(timeIntervalSinceNow: self.synthesisTimeout)
                while !completion.isComplete && !completion.isCancelled && Date() < timeout {
                    runLoop.run(mode: .default, before: Date(timeIntervalSinceNow: 0.05))
                }

                if completion.isCancelled {
                    synthesizer.stopSpeaking(at: .immediate)
                    continuation.finish(throwing: CancellationError())
                } else if !completion.isComplete {
                    synthesizer.stopSpeaking(at: .immediate)
                    continuation.finish(throwing: AudioToolError.resourceUnavailable(
                        "Synthesis timed out after \(Int(self.synthesisTimeout)) seconds"
                    ))
                } else {
                    continuation.finish()
                }
            }
            DispatchQueue.global(qos: .userInitiated).async(execute: producer)
            continuation.onTermination = { @Sendable _ in
                completion.cancel()
            }
        }
    }
    
    // MARK: - Voice Discovery
    
    /// Get all available voices for the configured language
    public func getVoicesForLanguage() -> [(identifier: String, name: String, quality: String)] {
        AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.hasPrefix(languageCode) || $0.language == language }
            .map { voice in
                let quality: String
                switch voice.quality {
                case .enhanced: quality = "Enhanced"
                case .premium: quality = "Premium"
                default: quality = "Default"
                }
                return (voice.identifier, voice.name, quality)
            }
    }
    
    /// List all supported languages
    public static var supportedLanguages: [String] {
        Set(AVSpeechSynthesisVoice.speechVoices().map { $0.language }).sorted()
    }
    
    // MARK: - Private Methods
    
    private nonisolated func findVoice(_ voiceSpec: String) -> AVSpeechSynthesisVoice? {
        guard !voiceSpec.isEmpty else { return nil }
        
        // Try exact identifier match
        if let voice = AVSpeechSynthesisVoice(identifier: voiceSpec) {
            return voice
        }
        
        // Try matching by name within language
        let languageVoices = AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.hasPrefix(languageCode) || $0.language == language }
        
        if let voice = languageVoices.first(where: { $0.name.lowercased() == voiceSpec.lowercased() }) {
            return voice
        }
        
        // Try partial name match
        if let voice = languageVoices.first(where: { $0.name.lowercased().contains(voiceSpec.lowercased()) }) {
            return voice
        }
        
        return nil
    }
    
    private nonisolated func extractSamples(from buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else { return [] }
        
        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        
        if channelCount == 1 {
            return Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        } else {
            // Mix stereo to mono
            var samples = [Float](repeating: 0, count: frameCount)
            for i in 0..<frameCount {
                var sum: Float = 0
                for ch in 0..<channelCount {
                    sum += channelData[ch][i]
                }
                samples[i] = sum / Float(channelCount)
            }
            return samples
        }
    }
}
