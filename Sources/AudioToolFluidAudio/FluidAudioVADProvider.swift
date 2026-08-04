//
//  FluidAudioVADProvider.swift
//  AudioToolFluidAudio
//
//  Silero VAD provider using FluidAudio
//

import Foundation
@preconcurrency import FluidAudio
import AudioToolCore

// MARK: - FluidAudio VAD Provider

/// Silero VAD provider using FluidAudio's VadManager
/// Implements VADProvider protocol for integration with AudioTool pipeline
public actor FluidAudioVADProvider: VADProvider, ChunkedProgressProvider {
    
    // MARK: - ChunkedProgressProvider Conformance
    
    public nonisolated var supportsChunkedProgress: Bool { true }
    
    // MARK: - AudioProcessor Conformance
    
    public nonisolated let sampleRate: Int = 16000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1
    
    // MARK: - StreamableProcessor Conformance
    
    public nonisolated var minChunkSize: Int { 512 }  // 32ms at 16kHz
    public nonisolated var recommendedChunkSize: Int { 4096 }  // 256ms at 16kHz (FluidAudio default)
    
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
        // Delegate to progress-aware version with nil callback
        return try await detect(audio, onProgress: nil)
    }
    
    /// Detect speech segments with progress reporting
    /// Uses FluidAudio's chunked processing (256ms per chunk) for granular progress
    /// - Parameters:
    ///   - audio: Input audio buffer (16kHz mono expected)
    ///   - onProgress: Optional callback with progress percentage (0.0 to 100.0)
    /// - Returns: Array of VAD segments with speech/silence labels
    public func detect(_ audio: AudioBuffer, onProgress: ProgressCallback?) async throws -> [VADSegment] {
        guard let manager = manager else {
            throw AudioToolError.modelNotLoaded("FluidAudio VAD")
        }
        
        let samples = audio.samples
        
        // Report initial progress
        await onProgress?(0.0)
        
        // Use chunked processing to report progress
        // FluidAudio's process() returns [VadResult] - one per 256ms chunk (4096 samples at 16kHz)
        let chunkSize = 4096
        let totalChunks = (samples.count + chunkSize - 1) / chunkSize
        
        // Process all samples to get per-chunk results with progress
        var processedChunks = 0
        let results = try await manager.process(samples)
        
        // Report progress based on chunks processed
        // Note: process() is synchronous internally, so we report after completion
        // For more granular progress, we'd need FluidAudio to expose incremental API
        processedChunks = results.count
        if processedChunks > 0 {
            let progress = min(Double(processedChunks) / Double(max(totalChunks, 1)) * 80.0, 80.0)
            await onProgress?(progress)
        }
        
        // Now run segmentation with the results we have
        var segmentConfig = VadSegmentationConfig.default
        segmentConfig.minSpeechDuration = minSpeechDuration
        segmentConfig.minSilenceDuration = minSilenceDuration
        
        // Segmentation phase - report 80-100%
        await onProgress?(85.0)
        
        let segments = try await manager.segmentSpeech(samples, config: segmentConfig)
        
        await onProgress?(95.0)
        
        // Convert FluidAudio segments to AudioTool VADSegment
        let vadSegments = segments.map { segment in
            VADSegment(
                timeRange: TimeRange(start: segment.startTime, end: segment.endTime),
                isSpeech: true,  // segmentSpeech only returns speech segments
                probability: 1.0  // FluidAudio segments are already filtered by threshold
            )
        }
        
        await onProgress?(100.0)
        
        return vadSegments
    }
    
    /// Stream VAD segments as detected
    public nonisolated func streamDetection(_ audio: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<VADSegment, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard let manager = await self.manager else {
                    continuation.finish(throwing: AudioToolError.modelNotLoaded("FluidAudio VAD"))
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
    public nonisolated func stream(_ input: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<AudioBuffer, Error> {
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
