//
//  FluidAudioVADProvider.swift
//  ClearVoiceFluidAudio
//
//  Silero VAD provider using FluidAudio
//

import Foundation
import FluidAudio
import ClearVoiceCore

// MARK: - FluidAudio VAD Provider

/// Silero VAD provider using FluidAudio's VadManager
/// Implements VADProvider protocol for integration with ClearVoice pipeline
public final class FluidAudioVADProvider: VADProvider, @unchecked Sendable {
    
    // MARK: - AudioProcessor Conformance
    
    public let sampleRate: Int = 16000
    public let inputChannels: Int = 1
    public let outputChannels: Int = 1
    
    // MARK: - StreamableProcessor Conformance
    
    public var minChunkSize: Int { 512 }  // 32ms at 16kHz
    public var recommendedChunkSize: Int { 4096 }  // 256ms at 16kHz (FluidAudio default)
    
    // MARK: - Private Properties
    
    private var manager: VadManager?
    private let threshold: Float
    private let minSpeechDuration: Double
    private let minSilenceDuration: Double
    private let config: VadConfig
    
    // MARK: - Initialization
    
    /// Initialize FluidAudio VAD provider
    /// - Parameters:
    ///   - threshold: Speech probability threshold (0.0-1.0, default 0.5, lower = more sensitive)
    ///   - minSpeechDuration: Minimum speech segment duration in seconds (default 0.25)
    ///   - minSilenceDuration: Minimum silence to split segments in seconds (default 0.4)
    public init(
        threshold: Float = 0.5,
        minSpeechDuration: Double = 0.25,
        minSilenceDuration: Double = 0.4
    ) {
        self.threshold = threshold
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.config = VadConfig(defaultThreshold: threshold)
    }
    
    /// Load the VAD model
    public func load() async throws {
        manager = try await VadManager(config: config)
    }
    
    // MARK: - VADProvider Conformance
    
    /// Detect speech segments in audio
    /// - Parameter audio: Input audio buffer (16kHz mono expected)
    /// - Returns: Array of VAD segments with speech/silence labels
    public func detect(_ audio: AudioBuffer) async throws -> [VADSegment] {
        guard let manager = manager else {
            throw ClearVoiceError.modelNotLoaded("FluidAudio VAD")
        }
        
        // Convert AudioBuffer samples to FluidAudio format
        let samples = audio.samples
        
        // Configure segmentation with instance settings
        var segmentConfig = VadSegmentationConfig.default
        segmentConfig.minSpeechDuration = minSpeechDuration
        segmentConfig.minSilenceDuration = minSilenceDuration
        
        // Run speech segmentation
        let segments = try await manager.segmentSpeech(samples, config: segmentConfig)
        
        // Convert FluidAudio segments to ClearVoice VADSegment
        return segments.map { segment in
            VADSegment(
                timeRange: TimeRange(start: segment.startTime, end: segment.endTime),
                isSpeech: true,  // segmentSpeech only returns speech segments
                probability: 1.0  // FluidAudio segments are already filtered by threshold
            )
        }
    }
    
    /// Stream VAD segments as detected
    public func streamDetection(_ audio: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<VADSegment, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard let manager = self.manager else {
                    continuation.finish(throwing: ClearVoiceError.modelNotLoaded("FluidAudio VAD"))
                    return
                }
                
                var state = await manager.makeStreamState()
                
                for await chunk in audio {
                    do {
                        let result = try await manager.processStreamingChunk(
                            chunk.samples,
                            state: state,
                            config: .default,
                            returnSeconds: true,
                            timeResolution: 2
                        )
                        state = result.state
                        
                        // Emit event if speech start/end detected
                        if let event = result.event {
                            let isSpeech = event.kind == .speechStart
                            let segment = VADSegment(
                                timeRange: TimeRange(
                                    start: event.time ?? 0,
                                    end: event.time ?? 0
                                ),
                                isSpeech: isSpeech,
                                probability: Float(result.probability)
                            )
                            continuation.yield(segment)
                        }
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
                
                continuation.finish()
            }
        }
    }
    
    // MARK: - StreamableProcessor Conformance
    
    /// Stream processing (passthrough with VAD overlay)
    public func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
        // For VAD, we just pass through the audio (VAD is metadata, not audio modification)
        AsyncThrowingStream { continuation in
            Task {
                for await chunk in input {
                    continuation.yield(chunk)
                }
                continuation.finish()
            }
        }
    }
    
    /// Reset streaming state
    public func reset() async {
        // State is managed per-stream, no global reset needed
    }
    
    // MARK: - AudioProcessor Conformance
    
    /// Process audio (passthrough - VAD doesn't modify audio)
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        // VAD is detection only, return input unchanged
        return input
    }
}
