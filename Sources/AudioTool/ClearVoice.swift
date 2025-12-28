//
//  ClearVoice.swift
//  ClearVoice
//
//  Main unified API actor for audio ML processing
//

import Foundation
import ClearVoiceCore

/// Main entry point - thread-safe model coordinator
public actor ClearVoice {
    
    // MARK: - Configuration
    
    private let configuration: ClearVoiceConfiguration
    
    // MARK: - Model Providers (injectable for testing)
    
    internal var vadProvider: (any VADProvider)?
    internal var diarizationProvider: (any DiarizationProvider)?
    internal var enhancerProviders: [EnhancementModel: any SpeechEnhancer] = [:]
    internal var separatorProviders: [SeparationModel: any SpeechSeparator] = [:]
    internal var transcriberProviders: [TranscriptionModel: any Transcriber] = [:]
    internal var upscalerProvider: (any AudioUpscaler)?
    internal var classifierProvider: (any SoundClassifier)?
    
    // MARK: - Initialization
    
    /// Production initializer
    public init(configuration: ClearVoiceConfiguration = .default) {
        self.configuration = configuration
    }
    
    /// Testing initializer with injectable providers
    internal init(
        configuration: ClearVoiceConfiguration = .default,
        vad: (any VADProvider)? = nil,
        diarization: (any DiarizationProvider)? = nil,
        enhancer: (EnhancementModel, any SpeechEnhancer)? = nil,
        separator: (SeparationModel, any SpeechSeparator)? = nil,
        transcriber: (TranscriptionModel, any Transcriber)? = nil,
        upscaler: (any AudioUpscaler)? = nil,
        classifier: (any SoundClassifier)? = nil
    ) {
        self.configuration = configuration
        self.vadProvider = vad
        self.diarizationProvider = diarization
        if let (model, provider) = enhancer {
            self.enhancerProviders[model] = provider
        }
        if let (model, provider) = separator {
            self.separatorProviders[model] = provider
        }
        if let (model, provider) = transcriber {
            self.transcriberProviders[model] = provider
        }
        self.upscalerProvider = upscaler
        self.classifierProvider = classifier
    }
    
    // MARK: - Provider Registration (for external libraries like ClearVoiceMLX, ClearVoiceCoreML)
    
    /// Register an enhancement provider
    public func register(enhancer: any SpeechEnhancer, for model: EnhancementModel) {
        self.enhancerProviders[model] = enhancer
    }
    
    /// Register a separator provider
    public func register(separator: any SpeechSeparator, for model: SeparationModel) {
        self.separatorProviders[model] = separator
    }
    
    /// Register an upscaler provider
    public func register(upscaler: any AudioUpscaler) {
        self.upscalerProvider = upscaler
    }
    
    // MARK: - Audio I/O
    
    /// Load audio from file
    public func loadAudio(from url: URL) async throws -> AudioBuffer {
        // TODO: Integrate SwiftAudio for actual loading
        // For now, throw not implemented
        throw ClearVoiceError.audioFileNotFound(url)
    }
    
    /// Save audio to file
    public func saveAudio(_ buffer: AudioBuffer, to url: URL, format: AudioFormat = .wav) async throws {
        // TODO: Integrate SwiftAudio for actual saving
        throw ClearVoiceError.resourceUnavailable("Audio export not implemented")
    }
    
    // MARK: - Analysis
    
    /// Voice activity detection
    public func detect(
        _ audio: AudioBuffer,
        model: VADModel = .silero
    ) async throws -> [VADSegment] {
        guard let vad = vadProvider else {
            throw ClearVoiceError.modelNotLoaded("VAD")
        }
        return try await vad.detect(audio)
    }
    
    /// Speaker diarization
    public func diarize(
        _ audio: AudioBuffer,
        vadHint: [VADSegment]? = nil
    ) async throws -> SpeakerTimeline {
        guard let diarizer = diarizationProvider else {
            throw ClearVoiceError.modelNotLoaded("Diarization")
        }
        
        if let hint = vadHint {
            return try await diarizer.diarize(audio, vadHint: hint)
        } else {
            return try await diarizer.diarize(audio)
        }
    }
    
    /// Combined VAD + Diarization (parallel execution)
    public func analyze(_ audio: AudioBuffer) async throws -> AnalysisResult {
        async let vadResult = detect(audio)
        async let diarizeResult = diarize(audio)
        
        return AnalysisResult(
            segments: try await vadResult,
            speakers: try await diarizeResult
        )
    }
    
    // MARK: - Enhancement
    
    /// Enhance full audio
    public func enhance(
        _ audio: AudioBuffer,
        model: EnhancementModel = .mossformerSE16k
    ) async throws -> AudioBuffer {
        guard let enhancer = enhancerProviders[model] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        return try await enhancer.process(audio)
    }
    
    /// Enhance only speech segments (VAD-gated)
    public func enhance(
        _ audio: AudioBuffer,
        segments: [VADSegment],
        model: EnhancementModel = .mossformerSE16k
    ) async throws -> AudioBuffer {
        guard let enhancer = enhancerProviders[model] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        
        var result = audio
        let speechSegments = segments.filter(\.isSpeech)
        
        for segment in speechSegments {
            let chunk = audio.slice(segment.timeRange.start..<segment.timeRange.end)
            let enhanced = try await enhancer.process(chunk)
            result = result.replacing(segment.timeRange.start..<segment.timeRange.end, with: enhanced)
        }
        
        return result
    }
    
    // MARK: - Separation
    
    /// Separate speakers
    public func separate(
        _ audio: AudioBuffer,
        speakers: Int,
        model: SeparationModel = .mossformer2spk
    ) async throws -> [AudioBuffer] {
        guard let separator = separatorProviders[model] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        return try await separator.separate(audio, speakers: speakers)
    }
    
    // MARK: - Upscaling
    
    /// Super-resolution upscaling
    public func upscale(_ audio: AudioBuffer) async throws -> AudioBuffer {
        guard let upscaler = upscalerProvider else {
            throw ClearVoiceError.modelNotLoaded("Upscaler")
        }
        return try await upscaler.process(audio)
    }
    
    // MARK: - Transcription
    
    /// Transcribe audio
    public func transcribe(
        _ audio: AudioBuffer,
        model: TranscriptionModel = .parakeet
    ) async throws -> Transcription {
        guard let transcriber = transcriberProviders[model] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        return try await transcriber.transcribe(audio)
    }
    
    // MARK: - Classification
    
    /// Classify sounds
    public func classify(_ audio: AudioBuffer) async throws -> [SoundClassification] {
        guard let classifier = classifierProvider else {
            throw ClearVoiceError.modelNotLoaded("Classifier")
        }
        return try await classifier.classify(audio)
    }
    
    // MARK: - Pipeline Builder
    
    /// Create declarative pipeline
    public nonisolated func pipeline() -> PipelineBuilder {
        PipelineBuilder(voice: self)
    }
    
    // MARK: - Pipeline Execution
    
    /// Execute pipeline (internal)
    internal func executePipeline(
        _ pipeline: PipelineBuilder,
        audio: AudioBuffer,
        eventHandler: (@Sendable (PipelineEvent) async -> Void)? = nil
    ) async throws -> PipelineResult {
        let startTime = ContinuousClock.now
        var metrics = PipelineMetrics()
        
        var context = PipelineContext(
            analysis: nil,
            currentAudio: audio,
            originalAudio: audio
        )
        
        var result = PipelineResult(metrics: metrics)
        
        for stage in pipeline.stages {
            try Task.checkCancellation()
            
            let stageStart = ContinuousClock.now
            
            switch stage.type {
            case .detect(let model):
                let segments = try await detect(context.currentAudio, model: model)
                let timeline = SpeakerTimeline(segments: [])
                context = PipelineContext(
                    analysis: AnalysisResult(segments: segments, speakers: timeline),
                    currentAudio: context.currentAudio,
                    originalAudio: context.originalAudio
                )
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: result.separatedTracks,
                    transcription: result.transcription,
                    classifications: result.classifications,
                    analysis: context.analysis,
                    metrics: result.metrics
                )
                
            case .diarize:
                let timeline = try await diarize(context.currentAudio)
                let segments = context.analysis?.segments ?? []
                context = PipelineContext(
                    analysis: AnalysisResult(segments: segments, speakers: timeline),
                    currentAudio: context.currentAudio,
                    originalAudio: context.originalAudio
                )
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: result.separatedTracks,
                    transcription: result.transcription,
                    classifications: result.classifications,
                    analysis: context.analysis,
                    metrics: result.metrics
                )
                
            case .analyze:
                let analysis = try await analyze(context.currentAudio)
                context = PipelineContext(
                    analysis: analysis,
                    currentAudio: context.currentAudio,
                    originalAudio: context.originalAudio
                )
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: result.separatedTracks,
                    transcription: result.transcription,
                    classifications: result.classifications,
                    analysis: analysis,
                    metrics: result.metrics
                )
                await eventHandler?(.analysisComplete(analysis))
                
            case .enhance(let model):
                let enhanced: AudioBuffer
                if let segments = context.analysis?.speechSegments, !segments.isEmpty {
                    enhanced = try await enhance(context.currentAudio, segments: segments, model: model)
                } else {
                    enhanced = try await enhance(context.currentAudio, model: model)
                }
                context = PipelineContext(
                    analysis: context.analysis,
                    currentAudio: enhanced,
                    originalAudio: context.originalAudio
                )
                result = PipelineResult(
                    audio: enhanced,
                    separatedTracks: result.separatedTracks,
                    transcription: result.transcription,
                    classifications: result.classifications,
                    analysis: result.analysis,
                    metrics: result.metrics
                )
                
            case .separate(let speakers, let useOriginal):
                let inputAudio = useOriginal ? context.originalAudio : context.currentAudio
                let tracks = try await separate(inputAudio, speakers: speakers)
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: tracks,
                    transcription: result.transcription,
                    classifications: result.classifications,
                    analysis: result.analysis,
                    metrics: result.metrics
                )
                
            case .upscale:
                let upscaled = try await upscale(context.currentAudio)
                context = PipelineContext(
                    analysis: context.analysis,
                    currentAudio: upscaled,
                    originalAudio: context.originalAudio
                )
                result = PipelineResult(
                    audio: upscaled,
                    separatedTracks: result.separatedTracks,
                    transcription: result.transcription,
                    classifications: result.classifications,
                    analysis: result.analysis,
                    metrics: result.metrics
                )
                
            case .transcribe(let model):
                let transcription = try await transcribe(context.currentAudio, model: model)
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: result.separatedTracks,
                    transcription: transcription,
                    classifications: result.classifications,
                    analysis: result.analysis,
                    metrics: result.metrics
                )
                for segment in transcription.segments {
                    await eventHandler?(.transcriptionSegment(segment))
                }
                
            case .classify:
                let classifications = try await classify(context.currentAudio)
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: result.separatedTracks,
                    transcription: result.transcription,
                    classifications: classifications,
                    analysis: result.analysis,
                    metrics: result.metrics
                )
                
            case .conditional(let condition, let thenStages, let elseStages):
                let stagesToRun = condition(context) ? thenStages : elseStages
                if !stagesToRun.isEmpty {
                    var subBuilder = PipelineBuilder(voice: self)
                    subBuilder.stages = stagesToRun
                    result = try await executePipeline(subBuilder, audio: context.currentAudio, eventHandler: eventHandler)
                    if let audio = result.audio {
                        context = PipelineContext(
                            analysis: result.analysis ?? context.analysis,
                            currentAudio: audio,
                            originalAudio: context.originalAudio
                        )
                    }
                }
                
            case .parallel(let branches):
                // Execute all branches in parallel
                // Capture values before entering task group to satisfy Swift 6 strict concurrency
                let currentAudio = context.currentAudio
                try await withThrowingTaskGroup(of: PipelineResult.self) { group in
                    for branchStages in branches {
                        group.addTask {
                            var subBuilder = PipelineBuilder(voice: self)
                            subBuilder.stages = branchStages
                            return try await self.executePipeline(subBuilder, audio: currentAudio, eventHandler: eventHandler)
                        }
                    }
                    
                    // Collect results (merge logic could be more sophisticated)
                    for try await branchResult in group {
                        if let transcription = branchResult.transcription {
                            result = PipelineResult(
                                audio: result.audio,
                                separatedTracks: result.separatedTracks,
                                transcription: transcription,
                                classifications: result.classifications ?? branchResult.classifications,
                                analysis: result.analysis,
                                metrics: result.metrics
                            )
                        }
                        if let classifications = branchResult.classifications {
                            result = PipelineResult(
                                audio: result.audio,
                                separatedTracks: result.separatedTracks,
                                transcription: result.transcription,
                                classifications: classifications,
                                analysis: result.analysis,
                                metrics: result.metrics
                            )
                        }
                    }
                }
                
            case .forEach(let transform):
                // Apply transform to each separated track
                if let tracks = result.separatedTracks {
                    var processedTracks: [AudioBuffer] = []
                    for track in tracks {
                        let subBuilder = transform(PipelineBuilder(voice: self))
                        let trackResult = try await executePipeline(subBuilder, audio: track, eventHandler: eventHandler)
                        if let processedAudio = trackResult.audio {
                            processedTracks.append(processedAudio)
                        }
                    }
                    result = PipelineResult(
                        audio: result.audio,
                        separatedTracks: processedTracks,
                        transcription: result.transcription,
                        classifications: result.classifications,
                        analysis: result.analysis,
                        metrics: result.metrics
                    )
                }
            }
            
            let stageDuration = ContinuousClock.now - stageStart
            metrics.stageDurations[stage.name] = stageDuration
            await pipeline.onStageCompleteHandler?(stage.name, stageDuration)
            await eventHandler?(.stageComplete(stage: stage.name, duration: stageDuration))
        }
        
        metrics.totalDuration = ContinuousClock.now - startTime
        
        return PipelineResult(
            audio: result.audio,
            separatedTracks: result.separatedTracks,
            transcription: result.transcription,
            classifications: result.classifications,
            analysis: result.analysis,
            metrics: metrics
        )
    }
    
    // MARK: - Model Management
    
    /// Preload models into memory
    public func preload(_ models: [any ModelIdentifier]) async throws {
        // TODO: Implement model loading
        for model in models {
            print("Preloading: \(model.modelName)")
        }
    }
    
    /// Current memory usage
    public var memoryUsage: Int {
        // TODO: Track actual memory
        0
    }
}
