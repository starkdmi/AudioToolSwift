//
//  AppleTTSProvider.swift
//  ClearVoiceTTS
//
//  Apple AVSpeechSynthesizer TTS provider for multilingual speech synthesis
//  Works in headless/CLI environments - no UI required
//

import Foundation
import AVFoundation
import ClearVoiceCore

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
public final class AppleTTSProvider: SpeechSynthesizer, @unchecked Sendable {
    
    // MARK: - SpeechSynthesizer Conformance
    
    public let sampleRate: Int = 22050  // AVSpeechSynthesizer default
    
    public var availableVoices: [String] {
        AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.hasPrefix(languageCode) || $0.language == language }
            .map { $0.identifier }
    }
    
    // MARK: - Properties
    
    private let language: String
    private let languageCode: String
    
    // MARK: - Initialization
    
    /// Initialize Apple TTS provider
    /// - Parameter language: BCP-47 language code (e.g., "en-US", "fr-FR", "de-DE", "ru-RU")
    public init(language: String = "en-US") {
        self.language = language
        self.languageCode = String(language.prefix(2))
    }
    
    // MARK: - SpeechSynthesizer Protocol
    
    /// Synthesize text to speech
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice identifier or name (e.g., "Thomas") - empty for default
    /// - Returns: Audio buffer with synthesized speech
    public func synthesize(_ text: String, voice: String) async throws -> ClearVoiceCore.AudioBuffer {
        try await withCheckedThrowingContinuation { continuation in
            // Run on a background thread with its own run loop
            DispatchQueue.global(qos: .userInitiated).async {
                let synthesizer = AVSpeechSynthesizer()
                
                let utterance = AVSpeechUtterance(string: text)
                
                // Set voice
                if let selectedVoice = self.findVoice(voice) {
                    utterance.voice = selectedVoice
                } else {
                    utterance.voice = AVSpeechSynthesisVoice(language: self.language)
                }
                
                var allSamples: [Float] = []
                var outputSampleRate: Double = 22050
                var outputChannels: UInt32 = 1
                var isComplete = false
                var synthesisError: Error?
                
                synthesizer.write(utterance) { buffer in
                    if let pcmBuffer = buffer as? AVAudioPCMBuffer, pcmBuffer.frameLength > 0 {
                        outputSampleRate = pcmBuffer.format.sampleRate
                        outputChannels = pcmBuffer.format.channelCount
                        let samples = self.extractSamples(from: pcmBuffer)
                        allSamples.append(contentsOf: samples)
                    } else {
                        // Completion signal
                        isComplete = true
                    }
                }
                
                // Pump run loop until synthesis completes (required for headless environments)
                let runLoop = RunLoop.current
                let timeout = Date(timeIntervalSinceNow: 30.0) // 30 second timeout
                while !isComplete && Date() < timeout {
                    runLoop.run(mode: .default, before: Date(timeIntervalSinceNow: 0.05))
                }
                
                if !isComplete {
                    synthesisError = ClearVoiceError.resourceUnavailable("Synthesis timed out")
                }
                
                // Return result on main continuation
                if let error = synthesisError {
                    continuation.resume(throwing: error)
                } else if allSamples.isEmpty {
                    continuation.resume(throwing: ClearVoiceError.resourceUnavailable("No audio generated"))
                } else {
                    let result = ClearVoiceCore.AudioBuffer(
                        samples: allSamples,
                        sampleRate: Int(outputSampleRate),
                        channels: Int(outputChannels)
                    )
                    continuation.resume(returning: result)
                }
            }
        }
    }
    
    /// Stream synthesized audio chunks
    public func streamSynthesis(_ text: String, voice: String) -> AsyncThrowingStream<ClearVoiceCore.AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let synthesizer = AVSpeechSynthesizer()
                
                let utterance = AVSpeechUtterance(string: text)
                
                if let selectedVoice = self.findVoice(voice) {
                    utterance.voice = selectedVoice
                } else {
                    utterance.voice = AVSpeechSynthesisVoice(language: self.language)
                }
                
                var isComplete = false
                
                synthesizer.write(utterance) { buffer in
                    if let pcmBuffer = buffer as? AVAudioPCMBuffer, pcmBuffer.frameLength > 0 {
                        let samples = self.extractSamples(from: pcmBuffer)
                        let chunk = ClearVoiceCore.AudioBuffer(
                            samples: samples,
                            sampleRate: Int(pcmBuffer.format.sampleRate),
                            channels: Int(pcmBuffer.format.channelCount)
                        )
                        continuation.yield(chunk)
                    } else {
                        isComplete = true
                    }
                }
                
                // Pump run loop
                let runLoop = RunLoop.current
                let timeout = Date(timeIntervalSinceNow: 30.0)
                while !isComplete && Date() < timeout {
                    runLoop.run(mode: .default, before: Date(timeIntervalSinceNow: 0.05))
                }
                
                continuation.finish()
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
    
    private func findVoice(_ voiceSpec: String) -> AVSpeechSynthesisVoice? {
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
    
    private func extractSamples(from buffer: AVAudioPCMBuffer) -> [Float] {
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
