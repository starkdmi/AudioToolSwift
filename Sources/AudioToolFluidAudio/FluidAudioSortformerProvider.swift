//
//  FluidAudioSortformerProvider.swift
//  ClearVoiceFluidAudio
//
//  Sortformer speaker diarization provider using FluidAudio
//  End-to-end neural model from NVIDIA, optimized for ≤4 speakers
//

import Foundation
@preconcurrency import FluidAudio
import ClearVoiceCore

// MARK: - FluidAudio Sortformer Provider

/// Sortformer speaker diarization provider using FluidAudio's SortformerDiarizer
/// Uses NVIDIA's end-to-end neural model for real-time streaming diarization
///
/// Key characteristics:
/// - Single neural network (no pipeline stages)
/// - Real-time streaming capable (0.32s - 30s latency options)
/// - Native overlapping speech handling (scores all 4 speakers per frame)
/// - Optimized for ≤4 speakers (performance degrades beyond)
/// - ~120x RTF on Apple Silicon
///
/// ## Configuration Presets
///
/// | Config | Latency | Use Case |
/// |--------|---------|----------|
/// | `.default` | ~1.04s | Real-time streaming (recommended) |
/// | `.nvidiaLowLatency` | ~1.04s | NVIDIA benchmark config |
/// | `.nvidiaHighLatency` | ~30.4s | Batch, best quality |
///
/// - Warning: CoreML models have static input shapes baked in at conversion time.
///   The `.default` config matches the model shipped with FluidAudio v0.10.0+.
///   Other configurations may require separately converted models.
///
/// ## Recommended Factory Methods
///
/// ```swift
/// // Low latency (recommended for streaming)
/// let provider = FluidAudioProviders.sortformerLowLatency()
///
/// // High latency (best quality for batch processing)
/// let provider = FluidAudioProviders.sortformerHighLatency()
/// ```
///
/// Use `FluidAudioDiarizationProvider` (Pyannote) for:
/// - Scenarios with >4 speakers
/// - Non-English audio
public actor FluidAudioSortformerProvider: DiarizationProvider {
    
    // MARK: - AudioProcessor Conformance
    
    public nonisolated let sampleRate: Int = 16000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1
    
    // MARK: - Private Properties
    
    private var diarizer: SortformerDiarizer?
    private let config: SortformerConfig
    
    // MARK: - Public Properties
    
    /// The Sortformer configuration used by this provider
    public nonisolated var configuration: SortformerConfig {
        config
    }
    
    /// Estimated latency in seconds based on configuration
    ///
    /// Calculated as: `(chunkLen + rightContext) × subsamplingFactor × melStride / sampleRate`
    public nonisolated var estimatedLatency: Double {
        let chunkLen = Double(config.chunkLen)
        let rightContext = Double(config.chunkRightContext)
        let subsamplingFactor = Double(config.subsamplingFactor)
        let melStride = Double(config.melStride)
        let sampleRate = Double(config.sampleRate)
        return (chunkLen + rightContext) * subsamplingFactor * melStride / sampleRate
    }
    
    // MARK: - Initialization
    
    /// Initialize Sortformer diarization provider
    /// - Parameter config: Sortformer configuration (default for low latency)
    public init(config: SortformerConfig = .default) {
        self.config = config
    }
    
    /// Load the Sortformer models (downloads from HuggingFace if needed)
    ///
    /// - Note: The CoreML model shipped with FluidAudio has static input shapes that must match
    ///   the configuration. The `.default` config is guaranteed to work. Other configurations
    ///   may require specially converted models.
    public func load() async throws {
        // Validate config compatibility with shipped model
        validateConfigCompatibility()
        
        // Create diarizer with config
        diarizer = SortformerDiarizer(config: config)
        
        // Load models from HuggingFace (auto-downloads and caches)
        let models = try await SortformerModelInference.loadFromHuggingFace(config: config)
        
        // Initialize diarizer with loaded models
        diarizer?.initialize(models: models)
    }
    
    /// Validates that the configuration is compatible with available models
    /// Logs a warning if using a configuration that may not have a matching CoreML model
    private func validateConfigCompatibility() {
        let defaultConfig = SortformerConfig.default
        
        // Check if config matches the default (shipped) model
        if config.isCompatible(with: defaultConfig) {
            return // Config matches shipped model
        }
        
        // Check if using NVIDIA low latency (similar to default)
        let nvidiaLowConfig = SortformerConfig.nvidiaLowLatency
        if config.isCompatible(with: nvidiaLowConfig) {
            print("[Sortformer] Using NVIDIA low latency config - may require matching CoreML model")
            return
        }
        
        // Check if using NVIDIA high latency
        let nvidiaHighConfig = SortformerConfig.nvidiaHighLatency
        if config.isCompatible(with: nvidiaHighConfig) {
            print("[Sortformer] Warning: High latency config requires separately converted CoreML model")
            print("[Sortformer] If you experience runtime errors, use sortformerLowLatency() instead")
            return
        }
        
        // Custom configuration
        print("[Sortformer] Warning: Custom config may not match available CoreML model")
        print("[Sortformer] Config: chunkMelFrames=\(config.chunkMelFrames), fifoLen=\(config.fifoLen), spkcacheLen=\(config.spkcacheLen)")
    }
    
    // MARK: - DiarizationProvider Conformance
    
    /// Identify speakers in audio (from URL - recommended)
    /// This loads audio from file and processes in batch mode
    public func diarize(url: URL) async throws -> SpeakerTimeline {
        guard let diarizer = diarizer, diarizer.isAvailable else {
            throw ClearVoiceError.modelNotLoaded("FluidAudio Sortformer")
        }
        
        // Load audio at 16kHz mono using FluidAudio's converter
        let converter = AudioConverter()
        let samples = try converter.resampleAudioFile(path: url.path)
        
        return try diarizeInternal(samples: samples)
    }
    
    /// Identify speakers in audio (from samples)
    /// - Parameter audio: Input audio buffer (16kHz mono expected)
    /// - Returns: Speaker timeline with labeled segments
    public func diarize(_ audio: AudioBuffer) async throws -> SpeakerTimeline {
        guard let diarizer = diarizer, diarizer.isAvailable else {
            throw ClearVoiceError.modelNotLoaded("FluidAudio Sortformer")
        }
        
        return try diarizeInternal(samples: audio.samples)
    }
    
    /// Diarize with VAD hint for efficiency
    /// Note: Sortformer performs its own speaker detection, VAD hint is informational only
    public func diarize(_ audio: AudioBuffer, vadHint: [VADSegment]) async throws -> SpeakerTimeline {
        // Sortformer handles speaker activity internally
        return try await diarize(audio)
    }
    
    // MARK: - Internal Processing
    
    private func diarizeInternal(samples: [Float]) throws -> SpeakerTimeline {
        guard let diarizer = diarizer else {
            throw ClearVoiceError.modelNotLoaded("FluidAudio Sortformer")
        }
        
        // Use batch processing for complete audio
        let timeline = try diarizer.processComplete(samples)
        
        // Convert Sortformer timeline to ClearVoice DiarizedSegment format
        // Sortformer timeline.segments is [[SortformerSegment]] - array per speaker
        var segments: [DiarizedSegment] = []
        
        for (speakerIndex, speakerSegments) in timeline.segments.enumerated() {
            for segment in speakerSegments {
                segments.append(DiarizedSegment(
                    timeRange: TimeRange(start: Double(segment.startTime), end: Double(segment.endTime)),
                    speakerID: SpeakerID(speakerIndex),
                    confidence: 1.0
                ))
            }
        }
        
        // Sort segments chronologically by start time
        segments.sort { $0.timeRange.start < $1.timeRange.start }
        
        return SpeakerTimeline(segments: segments)
    }
    
    // MARK: - AudioProcessor Conformance
    
    /// Process audio (passthrough - diarization doesn't modify audio)
    public func process(_ input: AudioBuffer) async throws -> AudioBuffer {
        return input
    }
    
    // MARK: - Streaming Support
    
    /// Process audio samples for streaming diarization
    /// Returns chunk result when enough audio has been processed
    /// - Parameter samples: Audio samples to add (16kHz mono)
    /// - Returns: Optional result with speaker predictions for this chunk
    public func processChunk(_ samples: [Float]) throws -> SortformerChunkResult? {
        guard let diarizer = diarizer else {
            throw ClearVoiceError.modelNotLoaded("FluidAudio Sortformer")
        }
        
        return try diarizer.processSamples(samples)
    }
    
    /// Get current accumulated timeline (for streaming mode)
    public var currentTimeline: SpeakerTimeline? {
        guard let diarizer = diarizer else { return nil }
        
        let timeline = diarizer.timeline
        var segments: [DiarizedSegment] = []
        
        for (speakerIndex, speakerSegments) in timeline.segments.enumerated() {
            for segment in speakerSegments {
                segments.append(DiarizedSegment(
                    timeRange: TimeRange(start: Double(segment.startTime), end: Double(segment.endTime)),
                    speakerID: SpeakerID(speakerIndex),
                    confidence: 1.0
                ))
            }
        }
        
        segments.sort { $0.timeRange.start < $1.timeRange.start }
        return SpeakerTimeline(segments: segments)
    }
    
    /// Reset streaming state for new audio session
    public func resetStreamingState() {
        diarizer?.reset()
    }
}
