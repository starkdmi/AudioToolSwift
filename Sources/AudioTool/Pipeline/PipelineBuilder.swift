//
//  PipelineBuilder.swift
//  ClearVoice
//
//  Declarative pipeline builder for chaining audio processing stages
//

import Foundation
import ClearVoiceCore

// MARK: - Pipeline Stage

/// Internal representation of a pipeline stage
public struct PipelineStage: Sendable {
    public enum StageType: Sendable {
        case detect(VADModel)
        case diarize
        case analyze
        case enhance(EnhancementModel)
        case separate(speakers: Int, useOriginal: Bool)
        case upscale
        case transcribe(TranscriptionModel)
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
    internal weak var voice: ClearVoice?
    
    internal var onStageCompleteHandler: (@Sendable (String, Duration) async -> Void)?
    internal var onSegmentHandler: (@Sendable (PipelineEvent) async -> Void)?
    
    // Internal initializer
    init(voice: ClearVoice? = nil) {
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
            throw ClearVoiceError.pipelineConfigurationInvalid("Pipeline not attached to ClearVoice instance")
        }
        return try await voice.executePipeline(self, audio: audio)
    }
    
    /// Execute pipeline on audio source (batch mode)
    public func process(source: AudioSource) async throws -> PipelineResult {
        guard let voice = voice else {
            throw ClearVoiceError.pipelineConfigurationInvalid("Pipeline not attached to ClearVoice instance")
        }
        
        let audio: AudioBuffer
        switch source {
        case .file(let url):
            audio = try await voice.loadAudio(from: url)
        case .buffer(let buffer):
            audio = buffer
        }
        
        return try await voice.executePipeline(self, audio: audio)
    }
    
    /// Execute pipeline with streaming output
    public func stream(source: AudioSource) -> AsyncThrowingStream<PipelineEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    guard let voice = voice else {
                        throw ClearVoiceError.pipelineConfigurationInvalid("Pipeline not attached to ClearVoice instance")
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
                        continuation.yield(event)
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
