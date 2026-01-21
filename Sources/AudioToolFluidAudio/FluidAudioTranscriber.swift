//
//  FluidAudioTranscriber.swift
//  ClearVoiceFluidAudio
//
//  Parakeet v3 transcription provider using FluidAudio
//

import Foundation
@preconcurrency import FluidAudio
import ClearVoiceCore

// MARK: - Model Version Enum

/// Parakeet model version selection
public enum ParakeetVersion: Sendable {
    case v2  // English-only, faster
    case v3  // Multilingual (English + others)
    
    var asrVersion: AsrModelVersion {
        switch self {
        case .v2: return .v2
        case .v3: return .v3
        }
    }
}

// MARK: - FluidAudio Transcriber

/// Parakeet v3 transcription provider using FluidAudio's ASR
/// Implements Transcriber protocol for integration with ClearVoice pipeline
public actor FluidAudioTranscriber: Transcriber {
    
    // MARK: - AudioProcessor Conformance
    
    public nonisolated let sampleRate: Int = 16000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1
    
    // MARK: - Private Properties
    
    private var asrManager: AsrManager?
    private var models: AsrModels?
    private let version: ParakeetVersion
    
    // MARK: - Initialization
    
    /// Initialize Parakeet transcriber
    /// - Parameter version: Model version (.v3 for multilingual, .v2 for English-only)
    public init(version: ParakeetVersion = .v3) {
        self.version = version
    }
    
    /// Load the ASR model (downloads if needed)
    public func load() async throws {
        models = try await AsrModels.downloadAndLoad(version: version.asrVersion)
        asrManager = AsrManager(config: .default)
        try await asrManager?.initialize(models: models!)
    }
    
    // MARK: - Transcriber Conformance
    
    /// Transcribe audio to text
    /// - Parameter audio: Input audio buffer (16kHz mono expected)
    /// - Returns: Transcription with text
    public func transcribe(_ audio: AudioBuffer) async throws -> Transcription {
        guard let manager = asrManager else {
            throw ClearVoiceError.modelNotLoaded("FluidAudio ASR")
        }
        
        let result = try await manager.transcribe(audio.samples)
        
        // FluidAudio ASRResult has .text property
        // Segments with timestamps may require additional configuration
        return Transcription(
            text: result.text,
            segments: [],  // Basic implementation - segments require additional API
            language: nil  // Language detection not exposed in basic API
        )
    }
    
    /// Stream transcription segments from audio stream.
    ///
    /// - Important: FluidAudio ASR is batch-oriented and requires complete audio.
    ///   This implementation buffers the entire input stream before processing,
    ///   so it does not provide true real-time streaming. A single segment is
    ///   emitted after the input stream completes.
    ///
    /// - Parameter audio: Async stream of audio chunks
    /// - Returns: Stream of transcription segments
    public nonisolated func streamTranscription(_ audio: AsyncStream<AudioBuffer>) -> AsyncThrowingStream<TranscriptionSegment, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard let manager = await self.asrManager else {
                    continuation.finish(throwing: ClearVoiceError.modelNotLoaded("FluidAudio ASR"))
                    return
                }
                
                // Collect audio chunks for batch processing
                var allSamples = [Float]()
                
                for await chunk in audio {
                    allSamples.append(contentsOf: chunk.samples)
                }
                
                // Transcribe collected audio
                do {
                    let result = try await manager.transcribe(allSamples)
                    
                    // Emit as single segment for basic implementation
                    let duration = Double(allSamples.count) / 16000.0
                    let segment = TranscriptionSegment(
                        text: result.text,
                        timeRange: TimeRange(start: 0, end: duration),
                        speakerID: nil,
                        confidence: 1.0
                    )
                    continuation.yield(segment)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - AudioProcessor Conformance
    
    /// Process audio (passthrough - transcription doesn't modify audio)
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        return input
    }
    
    // MARK: - Progress-Aware Transcription
    
    /// Transcribe with progress reporting
    /// Since FluidAudio ASR is batch-oriented, this reports:
    /// - 0% at start, 50% when processing starts, 100% when complete
    /// For real per-segment progress, use streaming API or pre-segmented audio
    public func transcribe(_ audio: AudioBuffer, onProgress: ProgressCallback?) async throws -> Transcription {
        guard let manager = asrManager else {
            throw ClearVoiceError.modelNotLoaded("FluidAudio ASR")
        }
        
        // Report initial progress
        await onProgress?(0.0)
        
        // Start transcription - this is batch, so we can't get intermediate progress
        let result = try await manager.transcribe(audio.samples)
        
        // Report near-complete
        await onProgress?(90.0)
        
        // Build transcription result
        let transcription = Transcription(
            text: result.text,
            segments: [],  // Basic implementation - segments require additional API
            language: nil  // Language detection not exposed in basic API
        )
        
        // Report complete
        await onProgress?(100.0)
        
        return transcription
    }
}
