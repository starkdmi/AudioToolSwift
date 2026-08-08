//
//  FluidAudioSortformerProvider.swift
//  AudioToolFluidAudio
//
//  Sortformer speaker diarization provider using FluidAudio
//  End-to-end neural model from NVIDIA, optimized for ≤4 speakers
//

import Foundation
import os
@preconcurrency import FluidAudio
import AudioToolCore

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
/// | `.balancedV2_1` | ~1.04s | NVIDIA benchmark config |
/// | `.highContextV2_1` | ~30.4s | Batch, best quality |
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
public actor FluidAudioSortformerProvider: DiarizationProvider, SpeakerIdentifier, ChunkedProgressProvider {

    /// Configuration warnings. These matter - a mismatched config produces a runtime
    /// CoreML shape failure - but they belong in the log, not on the host app's stdout.
    private static let logger = Logger(subsystem: "AudioToolSwift", category: "Sortformer")

    
    // MARK: - ChunkedProgressProvider Conformance
    
    /// Sortformer uses batch processing for production quality, so chunked progress is not available.
    /// Progress reporting is limited to 0% and 100%.
    public nonisolated var supportsChunkedProgress: Bool { false }
    
    // MARK: - AudioProcessor Conformance
    
    public nonisolated let sampleRate: Int = 16000
    public nonisolated let inputChannels: Int = 1
    public nonisolated let outputChannels: Int = 1
    
    // MARK: - Private Properties
    
    private var diarizer: SortformerDiarizer?
    private let config: SortformerConfig
    private let preprocessNormalization: PreprocessNormalization
    
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
    /// - Parameters:
    ///   - config: Sortformer configuration (default for low latency)
    ///   - preprocessNormalization: Audio normalization before processing (default: .none)
    public init(
        config: SortformerConfig = .default,
        preprocessNormalization: PreprocessNormalization = .none
    ) {
        self.config = config
        self.preprocessNormalization = preprocessNormalization
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
        let models = try await SortformerModels.loadFromHuggingFace(config: config)
        
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
        let nvidiaLowConfig = SortformerConfig.balancedV2_1
        if config.isCompatible(with: nvidiaLowConfig) {
            Self.logger.notice("Using the balanced v2.1 config; it may require a matching CoreML model")
            return
        }
        
        // Check if using NVIDIA high latency
        let nvidiaHighConfig = SortformerConfig.highContextV2_1
        if config.isCompatible(with: nvidiaHighConfig) {
            Self.logger.warning(
                "High-context config requires a separately converted CoreML model; use sortformerLowLatency() if loading fails")
            return
        }
        
        // Custom configuration
        Self.logger.warning("""
            Custom Sortformer config may not match any available CoreML model \
            (chunkMelFrames=\(self.config.chunkMelFrames, privacy: .public), \
            fifoLen=\(self.config.fifoLen, privacy: .public), \
            spkcacheLen=\(self.config.spkcacheLen, privacy: .public))
            """)
    }
    
    // MARK: - DiarizationProvider Conformance
    
    /// Identify speakers in audio (from URL - recommended)
    /// This loads audio from file and processes in batch mode
    public func diarize(url: URL) async throws -> SpeakerTimeline {
        guard let diarizer = diarizer, diarizer.isAvailable else {
            throw AudioToolError.modelNotLoaded("FluidAudio Sortformer")
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
            throw AudioToolError.modelNotLoaded("FluidAudio Sortformer")
        }
        try validateInputFormat(audio)
        
        return try diarizeInternal(samples: audio.samples)
    }
    
    /// Diarize with VAD hint for efficiency
    /// Note: Sortformer performs its own speaker detection, VAD hint is informational only
    public func diarize(_ audio: AudioBuffer, vadHint: [VADSegment]) async throws -> SpeakerTimeline {
        // Sortformer handles speaker activity internally
        return try await diarize(audio)
    }
    
    /// Diarize with progress reporting
    /// 
    /// - Note: Uses batch processing (`processComplete`) for production quality.
    ///   Progress reporting is limited to 0% and 100% to preserve quality.
    ///   Streaming mode (`processSamples`) produces slightly different results
    ///   due to limited lookahead context at chunk boundaries.
    ///
    /// - Parameters:
    ///   - audio: Input audio buffer (16kHz mono expected)
    ///   - onProgress: Callback with progress percentage (0.0 to 100.0)
    /// - Returns: Speaker timeline with labeled segments
    public func diarize(_ audio: AudioBuffer, onProgress: ProgressCallback?) async throws -> SpeakerTimeline {
        // Report initial progress
        await onProgress?(0.0)
        
        // Use batch processing for production quality (identical to diarize(_:))
        let result = try await diarize(audio)
        
        // Report completion
        await onProgress?(100.0)
        
        return result
    }
    
    // MARK: - Internal Processing
    
    private func diarizeInternal(samples: [Float]) throws -> SpeakerTimeline {
        guard let diarizer = diarizer else {
            throw AudioToolError.modelNotLoaded("FluidAudio Sortformer")
        }
        
        // Apply preprocessing normalization if configured
        let processedSamples = applyPreprocessNormalization(samples)
        
        // Use batch processing for complete audio
        let timeline = try diarizer.processComplete(processedSamples)
        
        return Self.speakerTimeline(from: timeline)
    }

    /// Convert a FluidAudio `DiarizerTimeline` into our `SpeakerTimeline`.
    ///
    /// The timeline no longer exposes a flat `segments` array indexed by speaker; it
    /// keys speakers by slot and each one carries its own segments. Finalized and
    /// tentative segments are both included: a caller asking for a timeline wants
    /// everything heard so far, and for completed audio the tentative set is empty.
    private static func speakerTimeline(from timeline: DiarizerTimeline) -> SpeakerTimeline {
        var segments: [DiarizedSegment] = []

        for (speakerIndex, speaker) in timeline.speakers {
            for segment in speaker.finalizedSegments + speaker.tentativeSegments {
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

    /// Apply preprocessing normalization to audio samples
    private func applyPreprocessNormalization(_ samples: [Float]) -> [Float] {
        switch preprocessNormalization {
        case .none:
            return samples
            
        case .peak(let targetDB):
            // Peak normalization: scale so max sample reaches targetDB
            let maxSample = samples.map { abs($0) }.max() ?? 0.0
            guard maxSample > 0 else { return samples }
            
            // Convert targetDB to linear scale (targetDB is relative to 1.0)
            let targetLinear = pow(10.0, targetDB / 20.0)
            let scale = targetLinear / maxSample
            return samples.map { $0 * scale }
            
        case .rms(let targetDB):
            // RMS normalization: scale based on root-mean-square energy
            let sumSquares = samples.reduce(0.0) { $0 + $1 * $1 }
            let rms = sqrt(sumSquares / Float(samples.count))
            guard rms > 0 else { return samples }
            
            // Convert targetDB to linear scale
            let targetLinear = pow(10.0, targetDB / 20.0)
            let scale = targetLinear / rms
            
            // Apply scaling but prevent clipping
            var normalized = samples.map { $0 * scale }
            let maxAfter = normalized.map { abs($0) }.max() ?? 0.0
            if maxAfter > 1.0 {
                // Reduce gain to prevent clipping
                let clipScale = 0.99 / maxAfter
                normalized = normalized.map { $0 * clipScale }
            }
            return normalized
        }
    }
    
    // MARK: - Streaming Support
    
    /// Process audio samples for streaming diarization
    /// Returns chunk result when enough audio has been processed
    /// - Parameter samples: Audio samples to add (16kHz mono)
    /// - Returns: Optional result with speaker predictions for this chunk
    public func processChunk(_ samples: [Float]) throws -> DiarizerChunkResult? {
        guard let diarizer = diarizer else {
            throw AudioToolError.modelNotLoaded("FluidAudio Sortformer")
        }

        return try diarizer.process(samples: samples)?.chunkResult
    }

    /// Get current accumulated timeline (for streaming mode)
    public var currentTimeline: SpeakerTimeline? {
        guard let diarizer = diarizer else { return nil }
        return Self.speakerTimeline(from: diarizer.timeline)
    }
    
    /// Reset streaming state for new audio session
    public func resetStreamingState() {
        diarizer?.reset()
    }
    
    // MARK: - Speaker Identification
    
    /// Identify which speaker slot a separated audio track belongs to.
    ///
    /// This method uses the preserved spkcache from previous diarization to identify
    /// which speaker slot a new audio segment belongs to. The spkcache contains learned
    /// speaker characteristics from the initial diarization pass.
    ///
    /// **Important:** This method must be called on the same Sortformer instance that
    /// performed the initial diarization, as it relies on the preserved internal state.
    ///
    /// ## Workflow
    /// 1. Run `diarize()` on full audio → populates spkcache with speaker characteristics
    /// 2. Separate overlapping regions using MossFormer2
    /// 3. Call `identifySpeaker()` on each separated track → returns slot assignment
    ///
    /// ## Resampling
    /// The input audio is automatically resampled to 16kHz if needed.
    /// For separated tracks from WHAMR (8kHz), they will be upsampled.
    ///
    /// - Parameter audio: Audio segment to identify (will be resampled to 16kHz)
    /// - Returns: Speaker identification with slot, confidence, and probabilities
    /// - Throws: `AudioToolError.modelNotLoaded` if Sortformer not initialized
    ///
    /// Example:
    /// ```swift
    /// // After diarization and separation...
    /// let identification = try await sortformer.identifySpeaker(separatedTrack)
    /// print("Track belongs to speaker slot \(identification.speakerSlot)")
    /// print("Confidence: \(identification.confidence)")
    /// ```
    public func identifySpeaker(_ audio: AudioBuffer) async throws -> SpeakerIdentification {
        guard let diarizer = diarizer, diarizer.isAvailable else {
            throw AudioToolError.modelNotLoaded("FluidAudio Sortformer")
        }
        try validateInputChannels(audio)
        
        // Resample to 16kHz if needed (e.g., from 8kHz WHAMR output)
        let inputSamples: [Float]
        if audio.sampleRate != sampleRate {
            inputSamples = resampleAudio(audio.samples, fromRate: audio.sampleRate, toRate: sampleRate)
        } else {
            inputSamples = audio.samples
        }
        
        // Process the audio through Sortformer.
        // Note: this streaming path preserves spkcache state, so the speaker
        // characteristics learned during the previous diarization still apply.
        let update = try diarizer.process(samples: inputSamples)

        guard let chunkResult = update?.chunkResult else {
            // Audio too short for full chunk - use batch processing with state preservation
            return try identifySpeakerBatch(inputSamples)
        }

        // Aggregate probabilities across all frames
        let numSpeakers = 4
        let frameCount = chunkResult.finalizedFrameCount

        // Calculate average probabilities across all frames
        var avgProbs = [Float](repeating: 0, count: numSpeakers)
        for frame in 0..<frameCount {
            for speaker in 0..<numSpeakers {
                let prob = chunkResult.probability(speaker: speaker, frame: frame, numSpeakers: numSpeakers)
                avgProbs[speaker] += prob
            }
        }
        
        // Normalize by frame count
        if frameCount > 0 {
            for i in 0..<numSpeakers {
                avgProbs[i] /= Float(frameCount)
            }
        }
        
        // Find dominant speaker
        let maxProb = avgProbs.max() ?? 0
        let speakerSlot = avgProbs.firstIndex(of: maxProb) ?? 0
        
        return SpeakerIdentification(
            speakerSlot: speakerSlot,
            confidence: maxProb,
            allProbabilities: avgProbs,
            averageProbabilities: avgProbs,
            frameCount: frameCount
        )
    }
    
    /// Batch identification for short audio segments
    ///
    /// When audio is too short for streaming chunks, use batch processing
    /// while preserving the spkcache state.
    private func identifySpeakerBatch(_ samples: [Float]) throws -> SpeakerIdentification {
        guard let diarizer = diarizer else {
            throw AudioToolError.modelNotLoaded("FluidAudio Sortformer")
        }
        
        // For very short audio, processComplete would reset state
        // Instead, we pad the audio to minimum length and use streaming mode
        let numSpeakers = 4
        let minDuration = 0.5 // Minimum 0.5 seconds for reliable identification
        let minSamples = Int(minDuration * Double(sampleRate))
        
        // Pad short audio to minimum length
        let processingSamples: [Float]
        if samples.count < minSamples {
            var paddedSamples = samples
            paddedSamples.append(contentsOf: [Float](repeating: 0, count: minSamples - samples.count))
            processingSamples = paddedSamples
        } else {
            processingSamples = samples
        }
        
        // Process the audio (padded or original)
        if let result = try diarizer.process(samples: processingSamples)?.chunkResult {
            let frameCount = result.finalizedFrameCount
            var avgProbs = [Float](repeating: 0, count: numSpeakers)

            for frame in 0..<frameCount {
                for speaker in 0..<numSpeakers {
                    avgProbs[speaker] += result.probability(speaker: speaker, frame: frame, numSpeakers: numSpeakers)
                }
            }
            
            if frameCount > 0 {
                for i in 0..<numSpeakers {
                    avgProbs[i] /= Float(frameCount)
                }
            }
            
            let maxProb = avgProbs.max() ?? 0
            let speakerSlot = avgProbs.firstIndex(of: maxProb) ?? 0
            
            return SpeakerIdentification(
                speakerSlot: speakerSlot,
                confidence: maxProb,
                allProbabilities: avgProbs,
                averageProbabilities: avgProbs,
                frameCount: frameCount
            )
        }
        
        // Fallback: Return low-confidence identification
        return SpeakerIdentification(
            speakerSlot: 0,
            confidence: 0,
            allProbabilities: [Float](repeating: 0.25, count: numSpeakers),
            averageProbabilities: nil,
            frameCount: 0
        )
    }
    
    /// Identify multiple separated tracks and return sorted by speaker slot
    ///
    /// Convenience method for identifying all tracks from a separation operation.
    ///
    /// - Parameter tracks: Array of separated audio tracks
    /// - Returns: Array of (track, identification) pairs sorted by speaker slot
    /// - Throws: `AudioToolError.modelNotLoaded` if Sortformer not initialized
    public func identifySpeakers(_ tracks: [AudioBuffer]) async throws -> [(audio: AudioBuffer, identification: SpeakerIdentification)] {
        var results: [(audio: AudioBuffer, identification: SpeakerIdentification)] = []
        
        for track in tracks {
            let identification = try await identifySpeaker(track)
            results.append((track, identification))
        }
        
        // Sort by speaker slot for consistent ordering
        return results.sorted { $0.identification.speakerSlot < $1.identification.speakerSlot }
    }
    
    // MARK: - Private Helpers
    
    /// Resample audio samples using linear interpolation
    ///
    /// Simple but effective resampling for speaker identification purposes.
    /// Uses linear interpolation which is sufficient for re-identification
    /// where we're comparing spectral characteristics, not preserving fidelity.
    ///
    /// - Parameters:
    ///   - samples: Input audio samples
    ///   - fromRate: Source sample rate
    ///   - toRate: Target sample rate
    /// - Returns: Resampled audio samples
    private nonisolated func resampleAudio(_ samples: [Float], fromRate: Int, toRate: Int) -> [Float] {
        guard fromRate != toRate else { return samples }
        guard !samples.isEmpty else { return [] }
        
        let ratio = Double(toRate) / Double(fromRate)
        let newLength = Int(Double(samples.count) * ratio)
        
        guard newLength > 0 else { return [] }
        
        var resampled = [Float](repeating: 0, count: newLength)
        
        for i in 0..<newLength {
            let sourceIndex = Double(i) / ratio
            let index = Int(sourceIndex)
            let fraction = Float(sourceIndex - Double(index))
            
            if index < samples.count - 1 {
                resampled[i] = samples[index] * (1 - fraction) + samples[index + 1] * fraction
            } else if index < samples.count {
                resampled[i] = samples[index]
            }
        }
        
        return resampled
    }
}
