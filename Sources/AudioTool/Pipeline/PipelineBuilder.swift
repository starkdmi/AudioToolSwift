//
//  PipelineBuilder.swift
//  AudioTool
//
//  Declarative pipeline builder for chaining audio processing stages
//

import Foundation
import AudioToolCore

// MARK: - Pipeline Stage

/// Internal representation of a pipeline stage
/// A single step in a pipeline.
///
/// Internal deliberately. It was public but unconstructable - both its initialiser
/// and `PipelineBuilder.stages` were internal - so nothing outside could make or read
/// one, while a public enum meant every new stage type would have been a
/// source-breaking change for consumers who could not use it in the first place.
/// Build pipelines through ``PipelineBuilder``.
internal struct PipelineStage: Sendable {
    internal enum StageType: Sendable {
        case detect(VADModel)
        case diarize
        case analyze
        case enhance(EnhancementModel)
        case separate(speakers: Int, useOriginal: Bool)
        case separateOverlap(handling: OverlapHandling, useOriginal: Bool)
        case separateUSS(targets: [SoundEmbedding])
        case upscale
        case transcribe(TranscriptionModel)
        case mergeTranscriptionWithDiarization
        case classify
        case conditional(@Sendable (PipelineContext) -> Bool, then: [PipelineStage], else: [PipelineStage])
        case parallel([[PipelineStage]])
        case forEach(@Sendable (PipelineBuilder) -> PipelineBuilder)
    }
    
    let type: StageType
    let name: String
    
    init(type: StageType, name: String) {
        self.type = type
        self.name = name
    }
}

// MARK: - Pipeline Builder

/// Declarative pipeline builder
public struct PipelineBuilder: Sendable {
    
    internal var stages: [PipelineStage] = []
    /// The engine that will run this pipeline.
    ///
    /// Weak on purpose: a builder is a value type that callers hold, and a strong
    /// reference would keep the engine - and every loaded model with it - alive for as
    /// long as the builder existed. The cost is that dropping the engine while holding
    /// a builder turns into a "Pipeline not attached" error at `process` time rather
    /// than a compile-time or construction-time failure. Keep the engine alive for as
    /// long as you intend to run the pipeline.
    internal weak var voice: AudioEngine?
    
    internal var onStageCompleteHandler: (@Sendable (String, Duration) async -> Void)?
    internal var onSegmentHandler: (@Sendable (PipelineEvent) async -> Void)?
    
    // Internal initializer
    init(voice: AudioEngine? = nil) {
        self.voice = voice
    }
    
    // MARK: - Analysis Stages
    
    /// Add VAD detection
    public func detect(_ model: VADModel = .silero) -> PipelineBuilder {
        var builder = self
        builder.stages.append(PipelineStage(type: .detect(model), name: "vad"))
        return builder
    }
    
    /// Add speaker diarization
    public func diarize() -> PipelineBuilder {
        var builder = self
        builder.stages.append(PipelineStage(type: .diarize, name: "diarization"))
        return builder
    }
    
    /// Add combined analysis (VAD + Diarization in parallel)
    public func analyze() -> PipelineBuilder {
        var builder = self
        builder.stages.append(PipelineStage(type: .analyze, name: "analysis"))
        return builder
    }
    
    // MARK: - Processing Stages
    
    /// Add enhancement
    public func enhance(_ model: EnhancementModel = .mossformerSE16k) -> PipelineBuilder {
        var builder = self
        builder.stages.append(PipelineStage(type: .enhance(model), name: "enhancement"))
        return builder
    }
    
    /// Add speaker separation
    public func separate(speakers: Int, useOriginal: Bool = false) -> PipelineBuilder {
        var builder = self
        builder.stages.append(PipelineStage(
            type: .separate(speakers: speakers, useOriginal: useOriginal),
            name: "separation"
        ))
        return builder
    }
    
    /// Add speaker separation with model selection based on overlap count
    ///
    /// Automatically selects the appropriate model:
    /// - 2 overlapping speakers: WHAMR model (best for noisy environments)
    /// - 3 overlapping speakers: 3spk model
    /// - 4+ speakers: No separation (too complex)
    ///
    /// Example usage with diarization:
    /// ```swift
    /// pipeline
    ///     .diarize()
    ///     .conditionally({ $0.analysis!.speakers.maxOverlappingSpeakers >= 2 }) {
    ///         $0.separateOverlappingSpeakers(useOriginal: true)
    ///     }
    /// ```
    ///
    /// - Parameters:
    ///   - useOriginal: Whether to use original audio (true) or enhanced audio (false)
    /// - Returns: Updated pipeline builder
    /// - Note: This adds a conditional stage that auto-selects the model based on diarization results
    public func separateOverlappingSpeakers(useOriginal: Bool = true) -> PipelineBuilder {
        // Add conditional separation based on maxOverlappingSpeakers
        conditionally({ context in
            guard let speakers = context.analysis?.speakers else { return false }
            return speakers.maxOverlappingSpeakers == 2
        }, then: { builder in
            builder.separate(speakers: 2, useOriginal: useOriginal)
        }, else: { builder in
            // Check for 3 speakers
            builder.conditionally({ context in
                guard let speakers = context.analysis?.speakers else { return false }
                return speakers.maxOverlappingSpeakers == 3
            }, then: { innerBuilder in
                innerBuilder.separate(speakers: 3, useOriginal: useOriginal)
            })
        })
    }
    
    /// Separate overlapping speech regions with speaker re-identification.
    ///
    /// This stage detects overlap regions from diarization, separates the speakers
    /// using MossFormer2, and re-identifies each track using Sortformer's preserved
    /// spkcache state.
    ///
    /// ## How It Works
    /// 1. Finds overlapping time ranges from diarization
    /// 2. For each overlap, selects appropriate model (WHAMR for 2, 3spk for 3)
    /// 3. Separates the mixed audio into individual tracks
    /// 4. Re-identifies each track using the diarizer's preserved speaker state
    /// 5. Emits events for each detected overlap and identified track
    ///
    /// ## Requirements
    /// - Must run after `.diarize()` stage
    /// - Requires Sortformer diarization provider (preserves spkcache)
    /// - Requires separation model (WHAMR and/or 3spk)
    ///
    /// ## Events Emitted
    /// - `.overlapDetected(timeRange:speakerCount:)` for each overlap region
    /// - `.trackIdentified(track:)` for each separated track
    ///
    /// - Note: Speaker re-identification after separation is not yet implemented.
    ///   Tracks are returned without speaker IDs. Use `.separate` mode (default).
    ///
    /// Example:
    /// ```swift
    /// let result = try await voice.pipeline()
    ///     .detect(.silero)
    ///     .diarize()
    ///     .separateOverlap(.separate)
    ///     .transcribe(.parakeet)
    ///     .onEvent { event in
    ///         switch event {
    ///         case .overlapDetected(let range, let count):
    ///             print("Overlap at \(range): \(count) speakers")
    ///         case .trackIdentified(let track):
    ///             print("Separated track \(track.trackIndex)")
    ///         default: break
    ///         }
    ///     }
    ///     .process(audio: audio)
    ///
    /// // Access separated tracks (without speaker IDs)
    /// for track in result.identifiedTracks ?? [] {
    ///     print("Track \(track.trackIndex): \(track.audio.duration)s")
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - handling: How to handle overlaps (default: `.separate`)
    ///   - useOriginal: Use original audio (true) or enhanced audio (false)
    /// - Returns: Updated pipeline builder
    public func separateOverlap(
        _ handling: OverlapHandling = .separate,
        useOriginal: Bool = true
    ) -> PipelineBuilder {
        var builder = self
        builder.stages.append(PipelineStage(
            type: .separateOverlap(handling: handling, useOriginal: useOriginal),
            name: "overlapSeparation"
        ))
        return builder
    }
    
    /// Add Universal Sound Separation (USS)
    ///
    /// Separates specific sound types from audio using USS (ResUNet30 + FiLM conditioning).
    /// Progress is reported per-embedding when multiple types are requested.
    ///
    /// Example:
    /// ```swift
    /// pipeline
    ///     .separateUSS([.music, .animal, .nature])  // Extract music, animals, nature
    /// ```
    ///
    /// - Parameter targets: Sound targets to extract (e.g., `.music`, `.animal`, `.speech`,
    ///   or a custom `SoundEmbedding` built from AudioSet class indices)
    /// - Returns: Updated pipeline builder
    public func separateUSS(_ targets: [SoundEmbedding]) -> PipelineBuilder {
        var builder = self
        builder.stages.append(PipelineStage(
            type: .separateUSS(targets: targets),
            name: "uss"
        ))
        return builder
    }
    
    /// Add Universal Sound Separation for a single target
    ///
    /// - Parameter target: Sound target to extract
    /// - Returns: Updated pipeline builder
    public func separateUSS(_ target: SoundEmbedding) -> PipelineBuilder {
        separateUSS([target])
    }
    
    /// Add upscaling
    public func upscale() -> PipelineBuilder {
        var builder = self
        builder.stages.append(PipelineStage(type: .upscale, name: "upscaling"))
        return builder
    }
    
    /// Add transcription
    public func transcribe(_ model: TranscriptionModel = .parakeet) -> PipelineBuilder {
        var builder = self
        builder.stages.append(PipelineStage(type: .transcribe(model), name: "transcription"))
        return builder
    }
    
    /// Merge transcription with diarization to create speaker-attributed transcription
    ///
    /// This stage combines a `Transcription` result with a `SpeakerTimeline` to produce
    /// a `DiarizedTranscription` where each segment is assigned to the speaker with the
    /// most overlap at that time range.
    ///
    /// ## Requirements
    /// - Must run after both `.transcribe()` and `.diarize()` stages
    /// - Can be used after `.parallel()` that runs transcription and diarization concurrently
    ///
    /// ## Example: Parallel Transcription + Diarization
    /// ```swift
    /// let result = try await voice.pipeline()
    ///     .parallel {[
    ///         PipelineBuilder().transcribe(.whisperLarge),
    ///         PipelineBuilder().diarize()
    ///     ]}
    ///     .mergeTranscriptionWithDiarization()
    ///     .process(audio: audio)
    ///
    /// // Access diarized transcription
    /// for segment in result.diarizedTranscription?.segments ?? [] {
    ///     let speaker = segment.speakerID?.id ?? "Unknown"
    ///     let overlap = segment.isOverlapRegion ? " [OVERLAP]" : ""
    ///     print("\(speaker): \(segment.text)\(overlap)")
    /// }
    /// ```
    ///
    /// ## Overlap Handling
    /// When a transcription segment spans a region where multiple speakers are active:
    /// - `speakerID` is set to the speaker with the most overlap duration
    /// - `activeSpeakers` contains all speakers active during the segment
    /// - `isOverlapRegion` returns `true`
    /// - `attributionConfidence` is lower to indicate uncertainty
    ///
    /// - Returns: Updated pipeline builder
    public func mergeTranscriptionWithDiarization() -> PipelineBuilder {
        var builder = self
        builder.stages.append(PipelineStage(
            type: .mergeTranscriptionWithDiarization,
            name: "mergeTranscriptionDiarization"
        ))
        return builder
    }
    
    /// Add sound classification
    public func classify() -> PipelineBuilder {
        var builder = self
        builder.stages.append(PipelineStage(type: .classify, name: "classification"))
        return builder
    }
    
    // MARK: - Conditional Execution
    
    /// Conditionally add stages
    public func conditionally(
        _ condition: @escaping @Sendable (PipelineContext) -> Bool,
        then thenBuilder: @escaping @Sendable (PipelineBuilder) -> PipelineBuilder
    ) -> PipelineBuilder {
        var builder = self
        let thenStages = thenBuilder(PipelineBuilder()).stages
        builder.stages.append(PipelineStage(
            type: .conditional(condition, then: thenStages, else: []),
            name: "conditional"
        ))
        return builder
    }
    
    /// Conditionally add stages with else branch
    public func conditionally(
        _ condition: @escaping @Sendable (PipelineContext) -> Bool,
        then thenBuilder: @escaping @Sendable (PipelineBuilder) -> PipelineBuilder,
        else elseBuilder: @escaping @Sendable (PipelineBuilder) -> PipelineBuilder
    ) -> PipelineBuilder {
        var builder = self
        let thenStages = thenBuilder(PipelineBuilder()).stages
        let elseStages = elseBuilder(PipelineBuilder()).stages
        builder.stages.append(PipelineStage(
            type: .conditional(condition, then: thenStages, else: elseStages),
            name: "conditional"
        ))
        return builder
    }
    
    // MARK: - Parallel Execution
    
    /// Execute multiple branches in parallel
    public func parallel(
        _ branches: @escaping @Sendable () -> [PipelineBuilder]
    ) -> PipelineBuilder {
        var builder = self
        let branchStages = branches().map(\.stages)
        builder.stages.append(PipelineStage(
            type: .parallel(branchStages),
            name: "parallel"
        ))
        return builder
    }
    
    // MARK: - Iteration
    
    /// Apply stages to each item (e.g., each separated speaker)
    public func forEach(
        _ transform: @escaping @Sendable (PipelineBuilder) -> PipelineBuilder
    ) -> PipelineBuilder {
        var builder = self
        builder.stages.append(PipelineStage(
            type: .forEach(transform),
            name: "forEach"
        ))
        return builder
    }
    
    // MARK: - Callbacks
    
    /// Callback when stage completes
    public func onStageComplete(
        _ handler: @escaping @Sendable (String, Duration) async -> Void
    ) -> PipelineBuilder {
        var builder = self
        builder.onStageCompleteHandler = handler
        return builder
    }
    
    /// Callback for each pipeline event
    public func onEvent(
        _ handler: @escaping @Sendable (PipelineEvent) async -> Void
    ) -> PipelineBuilder {
        var builder = self
        builder.onSegmentHandler = handler
        return builder
    }
    
    // MARK: - Execution
    
    /// Execute pipeline on audio buffer (batch mode)
    public func process(audio: AudioBuffer) async throws -> PipelineResult {
        guard let voice = voice else {
            throw AudioToolError.pipelineConfigurationInvalid("Pipeline not attached to AudioEngine instance")
        }
        return try await voice.executePipeline(self, audio: audio, eventHandler: onSegmentHandler)
    }
    
    /// Execute pipeline on audio source (batch mode)
    public func process(source: AudioSource) async throws -> PipelineResult {
        guard let voice = voice else {
            throw AudioToolError.pipelineConfigurationInvalid("Pipeline not attached to AudioEngine instance")
        }
        
        let audio: AudioBuffer
        switch source {
        case .file(let url):
            audio = try await voice.loadAudio(from: url)
        case .buffer(let buffer):
            audio = buffer
        }
        
        return try await voice.executePipeline(self, audio: audio, eventHandler: onSegmentHandler)
    }
    
    /// Execute pipeline with streaming output
    public func stream(source: AudioSource) -> AsyncThrowingStream<PipelineEvent, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    guard let voice = voice else {
                        throw AudioToolError.pipelineConfigurationInvalid("Pipeline not attached to AudioEngine instance")
                    }
                    
                    let audio: AudioBuffer
                    switch source {
                    case .file(let url):
                        audio = try await voice.loadAudio(from: url)
                    case .buffer(let buffer):
                        audio = buffer
                    }
                    
                    // Execute with event streaming
                    _ = try await voice.executePipeline(self, audio: audio) { event in
                        if case .terminated = continuation.yield(event) {
                            // This callback runs in the producer task. Mark it
                            // cancelled immediately so the pipeline's next stage or
                            // chunk check stops even before onTermination is delivered.
                            withUnsafeCurrentTask { task in task?.cancel() }
                        }
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                producer.cancel()
            }
        }
    }
}
