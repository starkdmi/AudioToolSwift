//
//  ClearVoice.swift
//  ClearVoice
//
//  Main unified API actor for audio ML processing
//

import Foundation
import AVFoundation
import ClearVoiceCore
@preconcurrency import AudioUtils


/// Main entry point - thread-safe model coordinator
public actor ClearVoice {
    
    // MARK: - Configuration
    
    private let configuration: ClearVoiceConfiguration
    
    // MARK: - Model Lifecycle Management
    
    /// Model lifecycle manager for memory tracking and LRU eviction
    public let modelManager: ModelLifecycleManager
    
    // MARK: - Model Providers (injectable for testing)
    
    internal var vadProvider: (any VADProvider)?
    internal var diarizationProvider: (any DiarizationProvider)?
    internal var enhancerProviders: [EnhancementModel: any SpeechEnhancer] = [:]
    internal var separatorProviders: [SeparationModel: any SpeechSeparator] = [:]
    internal var transcriberProviders: [TranscriptionModel: any Transcriber] = [:]
    internal var upscalerProvider: (any AudioUpscaler)?
    internal var classifierProvider: (any SoundClassifier)?
    internal var ussProvider: (any UniversalSoundSeparator)?
    internal var synthesizerProviders: [String: any SpeechSynthesizer] = [:]
    internal var translatorProviders: [TranslationModel: any TextTranslator] = [:]
    internal var textPreprocessorProviders: [String: any TextPreprocessor] = [:]
    
    // MARK: - Initialization
    
    /// Production initializer
    public init(
        configuration: ClearVoiceConfiguration = .default,
        modelManager: ModelLifecycleManager? = nil
    ) {
        self.configuration = configuration
        self.modelManager = modelManager ?? ModelLifecycleManager(
            memoryLimitBytes: configuration.modelMemoryLimit
        )
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
        self.modelManager = ModelLifecycleManager(
            memoryLimitBytes: configuration.modelMemoryLimit
        )
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
    
    /// Register a synthesizer provider
    public func register(synthesizer: any SpeechSynthesizer, for model: SynthesisModel) {
        self.synthesizerProviders[model.modelName] = synthesizer
    }
    
    /// Register a translation provider
    public func register(translator: any TextTranslator, for model: TranslationModel) {
        self.translatorProviders[model] = translator
    }
    
    /// Register a transcriber provider
    public func register(transcriber: any Transcriber, for model: TranscriptionModel) {
        self.transcriberProviders[model] = transcriber
    }
    
    /// Register a USS (Universal Sound Separation) provider
    public func register(uss: any UniversalSoundSeparator) {
        self.ussProvider = uss
    }
    
    /// Register a text preprocessor provider (e.g., RUAccent for Russian stress marking)
    public func register(preprocessor: any TextPreprocessor, for model: TextPreprocessorModel) {
        self.textPreprocessorProviders[model.modelName] = preprocessor
    }
    
    // MARK: - Audio I/O
    
    /// Load audio from file
    ///
    /// Uses SwiftAudio (AudioUtils) to load and resample audio files.
    /// Supports WAV, CAF, MP3, M4A, and other formats supported by AVFoundation.
    ///
    /// - Parameters:
    ///   - url: Path to audio file
    ///   - targetSampleRate: Optional target sample rate for resampling. If nil, preserves source sample rate.
    /// - Returns: Loaded audio buffer
    public func loadAudio(from url: URL, targetSampleRate: Int? = nil) async throws -> ClearVoiceCore.AudioBuffer {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ClearVoiceError.audioFileNotFound(url)
        }
        
        // Configure AudioLoader - preserve source rate when targetSampleRate is nil
        let config: AudioLoader.Configuration
        if let targetRate = targetSampleRate {
            config = AudioLoader.Configuration(
                targetSampleRate: Double(targetRate),
                maxDuration: 3600.0,  // 1 hour max
                normalizationMode: .none,
                resamplingMethod: .auto
            )
        } else {
            // Preserve source sample rate - use a placeholder value since resamplingMethod is .none
            config = AudioLoader.Configuration(
                targetSampleRate: 48000,  // Ignored when resamplingMethod is .none
                maxDuration: 3600.0,
                normalizationMode: .none,
                resamplingMethod: .none
            )
        }
        
        let loader = AudioLoader(config: config)
        
        do {
            let audioData = try loader.loadMonoBuffer(from: url)
            return ClearVoiceCore.AudioBuffer(
                samples: audioData.samples,
                sampleRate: audioData.sampleRate,
                channels: 1
            )
        } catch let error as AudioLoaderError {
            // Map AudioLoader errors to ClearVoice errors
            switch error {
            case .fileNotFound(let path):
                throw ClearVoiceError.audioFileNotFound(URL(fileURLWithPath: path))
            case .unsupportedFormat(let ext):
                throw ClearVoiceError.invalidAudioFormat(expected: "wav/mp3/m4a/flac", found: ext)
            case .fileTooLarge, .audioTooLong:
                throw ClearVoiceError.resourceUnavailable(error.localizedDescription)
            default:
                throw ClearVoiceError.resourceUnavailable(error.localizedDescription)
            }
        }
    }
    
    /// Save audio to file
    ///
    /// Exports audio buffer to a file in the specified format using AudioUtils.
    ///
    /// - Parameters:
    ///   - buffer: Audio buffer to save
    ///   - url: Destination file path
    ///   - format: Output format (default: .wav)
    public func saveAudio(_ buffer: ClearVoiceCore.AudioBuffer, to url: URL, format: AudioFormat = .wav) async throws {
        // Map ClearVoice format to AudioSaver format
        let saverConfig: AudioSaver.Configuration
        switch format {
        case .wav:
            saverConfig = AudioSaver.Configuration(
                sampleRate: Double(buffer.sampleRate),
                bitDepth: .float32,
                fileFormat: .wav
            )
        case .m4a:
            saverConfig = AudioSaver.Configuration(
                sampleRate: Double(buffer.sampleRate),
                bitDepth: .float32,
                fileFormat: .m4a(bitRate: 128000)
            )
        case .mp3:
            // MP3 not directly supported, fall back to WAV
            saverConfig = AudioSaver.Configuration(
                sampleRate: Double(buffer.sampleRate),
                bitDepth: .int16,
                fileFormat: .wav
            )
        case .flac:
            // FLAC not in AudioSaver, use WAV with high bit depth
            saverConfig = AudioSaver.Configuration(
                sampleRate: Double(buffer.sampleRate),
                bitDepth: .int24,
                fileFormat: .wav
            )
        }
        
        let saver = AudioSaver(config: saverConfig)
        
        // Convert AudioBuffer to AudioData
        let audioData = AudioData(
            samples: buffer.samples,
            sampleRate: buffer.sampleRate
        )
        
        do {
            try saver.saveBuffer(audioData, to: url)
        } catch let error as AudioSaverError {
            throw ClearVoiceError.resourceUnavailable(error.localizedDescription)
        }
    }
    
    // MARK: - Analysis
    
    /// Voice activity detection
    public func detect(
        _ audio: ClearVoiceCore.AudioBuffer,
        model: VADModel = .silero
    ) async throws -> [VADSegment] {
        guard let vad = vadProvider else {
            throw ClearVoiceError.modelNotLoaded("VAD")
        }
        
        // Resample to model's expected sample rate if needed
        let input = try audio.resampled(to: model.sampleRate)
        return try await vad.detect(input)
    }
    
    /// Speaker diarization
    public func diarize(
        _ audio: ClearVoiceCore.AudioBuffer,
        vadHint: [VADSegment]? = nil
    ) async throws -> SpeakerTimeline {
        guard let diarizer = diarizationProvider else {
            throw ClearVoiceError.modelNotLoaded("Diarization")
        }
        
        // Diarization typically expects 16kHz
        let input = try audio.resampled(to: 16000)
        
        if let hint = vadHint {
            return try await diarizer.diarize(input, vadHint: hint)
        } else {
            return try await diarizer.diarize(input)
        }
    }
    
    /// Combined VAD + Diarization (parallel execution)
    public func analyze(_ audio: ClearVoiceCore.AudioBuffer) async throws -> AnalysisResult {
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
        _ audio: ClearVoiceCore.AudioBuffer,
        model: EnhancementModel = .mossformerSE16k
    ) async throws -> ClearVoiceCore.AudioBuffer {
        guard let enhancer = enhancerProviders[model] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        
        // Resample to model's expected sample rate if needed
        let input = try audio.resampled(to: model.sampleRate)
        let output = try await enhancer.process(input)
        
        // Resample back to original rate if different
        if audio.sampleRate != model.sampleRate {
            return try output.resampled(to: audio.sampleRate)
        }
        return output
    }
    
    /// Enhance only speech segments (VAD-gated)
    public func enhance(
        _ audio: ClearVoiceCore.AudioBuffer,
        segments: [VADSegment],
        model: EnhancementModel = .mossformerSE16k
    ) async throws -> ClearVoiceCore.AudioBuffer {
        guard let enhancer = enhancerProviders[model] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        
        // Resample full audio to model's expected sample rate
        let resampledAudio = try audio.resampled(to: model.sampleRate)
        
        var result = resampledAudio
        let speechSegments = segments.filter(\.isSpeech)
        
        for segment in speechSegments {
            // Scale time ranges to resampled audio
            let scaledStart = segment.timeRange.start
            let scaledEnd = segment.timeRange.end
            let chunk = resampledAudio.slice(scaledStart..<scaledEnd)
            let enhanced = try await enhancer.process(chunk)
            result = result.replacing(scaledStart..<scaledEnd, with: enhanced)
        }
        
        // Resample back to original rate if different
        if audio.sampleRate != model.sampleRate {
            return try result.resampled(to: audio.sampleRate)
        }
        return result
    }
    
    // MARK: - Separation
    
    /// Separate speakers
    public func separate(
        _ audio: ClearVoiceCore.AudioBuffer,
        speakers: Int,
        model: SeparationModel = .mossformer2spk
    ) async throws -> [ClearVoiceCore.AudioBuffer] {
        guard let separator = separatorProviders[model] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        
        // Resample to model's expected sample rate if needed
        let input = try audio.resampled(to: model.sampleRate)
        let outputs = try await separator.separate(input, speakers: speakers)
        
        // Resample all outputs back to original rate if different
        if audio.sampleRate != model.sampleRate {
            return try outputs.map { try $0.resampled(to: audio.sampleRate) }
        }
        return outputs
    }
    
    // MARK: - Upscaling
    
    /// Super-resolution upscaling
    public func upscale(_ audio: ClearVoiceCore.AudioBuffer) async throws -> ClearVoiceCore.AudioBuffer {
        guard let upscaler = upscalerProvider else {
            throw ClearVoiceError.modelNotLoaded("Upscaler")
        }
        
        // Upscaler expects 16kHz input, outputs 48kHz
        let input = try audio.resampled(to: 16000)
        return try await upscaler.process(input)
    }
    
    // MARK: - Transcription
    
    /// Transcribe audio
    public func transcribe(
        _ audio: ClearVoiceCore.AudioBuffer,
        model: TranscriptionModel = .parakeet
    ) async throws -> Transcription {
        guard let transcriber = transcriberProviders[model] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        
        // Resample to model's expected sample rate if needed
        let input = try audio.resampled(to: model.sampleRate)
        return try await transcriber.transcribe(input)
    }
    
    /// Transcribe audio with progress reporting
    /// - Parameters:
    ///   - audio: Audio buffer to transcribe
    ///   - model: Transcription model to use
    ///   - onProgress: Progress callback (0.0 to 100.0) as segments are recognized
    /// - Returns: Complete transcription result
    public func transcribe(
        _ audio: ClearVoiceCore.AudioBuffer,
        model: TranscriptionModel = .parakeet,
        onProgress: ProgressCallback?
    ) async throws -> Transcription {
        guard let transcriber = transcriberProviders[model] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        
        // Resample to model's expected sample rate if needed
        let input = try audio.resampled(to: model.sampleRate)
        return try await transcriber.transcribe(input, onProgress: onProgress)
    }
    
    // MARK: - Classification
    
    /// Classify sounds
    public func classify(_ audio: ClearVoiceCore.AudioBuffer) async throws -> [SoundClassification] {
        guard let classifier = classifierProvider else {
            throw ClearVoiceError.modelNotLoaded("Classifier")
        }
        return try await classifier.classify(audio)
    }
    
    // MARK: - Speech Synthesis
    
    /// Synthesize text to speech
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice identifier (must be loaded by the provider)
    ///   - model: Synthesis model to use
    /// - Returns: Audio buffer with synthesized speech
    public func synthesize(
        _ text: String,
        voice: String,
        model: SynthesisModel = .kokoro(language: .americanEnglish, voice: "af_heart")
    ) async throws -> ClearVoiceCore.AudioBuffer {
        guard let synthesizer = synthesizerProviders[model.modelName] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        return try await synthesizer.synthesize(text, voice: voice)
    }
    
    /// Stream synthesized audio chunks
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice identifier
    ///   - model: Synthesis model to use
    /// - Returns: Async stream of audio chunks
    public func streamSynthesis(
        _ text: String,
        voice: String,
        model: SynthesisModel = .kokoro(language: .americanEnglish, voice: "af_heart")
    ) -> AsyncThrowingStream<ClearVoiceCore.AudioBuffer, Error> {
        guard let synthesizer = synthesizerProviders[model.modelName] else {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: ClearVoiceError.modelNotLoaded(model.modelName))
            }
        }
        return synthesizer.streamSynthesis(text, voice: voice)
    }
    
    // MARK: - Translation
    
    /// Translate text
    /// - Parameters:
    ///   - text: Source text to translate
    ///   - source: Source language code (BCP-47) or nil for auto-detect
    ///   - target: Target language code (BCP-47)
    ///   - model: Translation model to use
    /// - Returns: Translation result
    public func translate(
        _ text: String,
        from source: String? = nil,
        to target: String,
        model: TranslationModel = .appleTranslation
    ) async throws -> TranslationResult {
        guard let translator = translatorProviders[model] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        return try await translator.translate(text, from: source, to: target)
    }
    
    /// Translate batch of texts
    /// - Parameters:
    ///   - texts: Array of source texts
    ///   - source: Source language code (BCP-47) or nil for auto-detect
    ///   - target: Target language code (BCP-47)
    ///   - model: Translation model to use
    /// - Returns: Batch translation result
    public func translateBatch(
        _ texts: [String],
        from source: String? = nil,
        to target: String,
        model: TranslationModel = .appleTranslation
    ) async throws -> BatchTranslationResult {
        guard let translator = translatorProviders[model] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        return try await translator.translateBatch(texts, from: source, to: target)
    }
    
    // MARK: - Text Preprocessing
    
    /// Preprocess text (e.g., add stress marks for Russian TTS)
    ///
    /// Text preprocessing transforms input text before TTS synthesis.
    /// For example, RUAccent adds stress marks to Russian text.
    ///
    /// - Parameters:
    ///   - text: Input text to preprocess
    ///   - model: Preprocessing model to use
    /// - Returns: Preprocessed text
    ///
    /// Example:
    /// ```swift
    /// let stressed = try await voice.preprocess("привет мир", model: .ruaccent(profile: .balanced))
    /// // stressed = "прив+ет м+ир"
    /// ```
    public func preprocess(
        _ text: String,
        model: TextPreprocessorModel
    ) throws -> String {
        guard let preprocessor = textPreprocessorProviders[model.modelName] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        return try preprocessor.process(text)
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
        audio: ClearVoiceCore.AudioBuffer,
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
            let stageName = stage.name
            
            switch stage.type {
            case .detect(let model):
                await eventHandler?(.progress(stage: stageName, percent: 0))
                
                // Use progress-aware detection if available
                let progressCallback: ProgressCallback? = if let handler = eventHandler {
                    { @Sendable percent in
                        await handler(.progress(stage: stageName, percent: percent))
                    }
                } else {
                    nil
                }
                
                let segments: [VADSegment]
                if let vad = vadProvider {
                    // Resample to model's expected sample rate if needed
                    let input = try context.currentAudio.resampled(to: model.sampleRate)
                    segments = try await vad.detect(input, onProgress: progressCallback)
                } else {
                    throw ClearVoiceError.modelNotLoaded("VAD")
                }
                
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
                await eventHandler?(.progress(stage: stageName, percent: 100))
                
            case .diarize:
                await eventHandler?(.progress(stage: stageName, percent: 0))
                
                // Use progress-aware diarization if available
                let progressCallback: ProgressCallback? = if let handler = eventHandler {
                    { @Sendable percent in
                        await handler(.progress(stage: stageName, percent: percent))
                    }
                } else {
                    nil
                }
                
                let timeline: SpeakerTimeline
                if let diarizer = diarizationProvider {
                    // Diarization typically expects 16kHz
                    let input = try context.currentAudio.resampled(to: 16000)
                    timeline = try await diarizer.diarize(input, onProgress: progressCallback)
                } else {
                    throw ClearVoiceError.modelNotLoaded("Diarization")
                }
                
                let segments = context.analysis?.segments ?? []
                context = PipelineContext(
                    analysis: AnalysisResult(segments: segments, speakers: timeline),
                    currentAudio: context.currentAudio,
                    originalAudio: context.originalAudio
                )
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: result.separatedTracks,
                    ussSeparated: result.ussSeparated,
                    transcription: result.transcription,
                    classifications: result.classifications,
                    analysis: context.analysis,
                    metrics: result.metrics
                )
                await eventHandler?(.progress(stage: stageName, percent: 100))
                
            case .analyze:
                await eventHandler?(.progress(stage: stageName, percent: 0))
                let analysis = try await analyze(context.currentAudio)
                context = PipelineContext(
                    analysis: analysis,
                    currentAudio: context.currentAudio,
                    originalAudio: context.originalAudio
                )
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: result.separatedTracks,
                    ussSeparated: result.ussSeparated,
                    transcription: result.transcription,
                    classifications: result.classifications,
                    analysis: analysis,
                    metrics: result.metrics
                )
                await eventHandler?(.analysisComplete(analysis))
                await eventHandler?(.progress(stage: stageName, percent: 100))
                
            case .enhance(let model):
                await eventHandler?(.progress(stage: stageName, percent: 0))
                guard let enhancer = enhancerProviders[model] else {
                    throw ClearVoiceError.modelNotLoaded(model.modelName)
                }
                let speechSegments = context.analysis?.speechSegments ?? []
                let enhanced: ClearVoiceCore.AudioBuffer
                if let eventHandler, let streamable = enhancer as? StreamableOutput {
                    let resampledAudio = try context.currentAudio.resampled(to: model.sampleRate)
                    if !speechSegments.isEmpty {
                        let totalSpeechSamples = speechSegments.reduce(0) { partial, segment in
                            partial + Int(segment.timeRange.duration * Double(model.sampleRate))
                        }
                        var processedSamples = 0
                        var resultBuffer = resampledAudio
                        for segment in speechSegments {
                            let chunk = resampledAudio.slice(segment.timeRange.start..<segment.timeRange.end)
                            var segmentSamples: [Float] = []
                            segmentSamples.reserveCapacity(chunk.samples.count)
                            for try await processedChunk in streamable.processStream(chunk) {
                                segmentSamples.append(contentsOf: processedChunk.samples)
                                processedSamples += processedChunk.samples.count
                                let percent = min(100.0, Double(processedSamples) / Double(max(totalSpeechSamples, 1)) * 100)
                                await eventHandler(.progress(stage: stageName, percent: percent))
                            }
                            let processedSegment = AudioBuffer(
                                samples: segmentSamples,
                                sampleRate: resampledAudio.sampleRate,
                                channels: resampledAudio.channels
                            )
                            resultBuffer = resultBuffer.replacing(segment.timeRange.start..<segment.timeRange.end, with: processedSegment)
                        }
                        if context.currentAudio.sampleRate != model.sampleRate {
                            enhanced = try resultBuffer.resampled(to: context.currentAudio.sampleRate)
                        } else {
                            enhanced = resultBuffer
                        }
                    } else {
                        let totalSamples = resampledAudio.samples.count
                        var processedSamples = 0
                        var streamedSamples: [Float] = []
                        streamedSamples.reserveCapacity(totalSamples)
                        for try await processedChunk in streamable.processStream(resampledAudio) {
                            streamedSamples.append(contentsOf: processedChunk.samples)
                            processedSamples += processedChunk.samples.count
                            let percent = min(100.0, Double(processedSamples) / Double(max(totalSamples, 1)) * 100)
                            await eventHandler(.progress(stage: stageName, percent: percent))
                        }
                        let streamed = AudioBuffer(
                            samples: streamedSamples,
                            sampleRate: resampledAudio.sampleRate,
                            channels: resampledAudio.channels
                        )
                        if context.currentAudio.sampleRate != model.sampleRate {
                            enhanced = try streamed.resampled(to: context.currentAudio.sampleRate)
                        } else {
                            enhanced = streamed
                        }
                    }
                } else {
                    if !speechSegments.isEmpty {
                        enhanced = try await enhance(context.currentAudio, segments: speechSegments, model: model)
                    } else {
                        enhanced = try await enhance(context.currentAudio, model: model)
                    }
                }
                await eventHandler?(.progress(stage: stageName, percent: 100))
                context = PipelineContext(
                    analysis: context.analysis,
                    currentAudio: enhanced,
                    originalAudio: context.originalAudio
                )
                result = PipelineResult(
                    audio: enhanced,
                    separatedTracks: result.separatedTracks,
                    ussSeparated: result.ussSeparated,
                    transcription: result.transcription,
                    classifications: result.classifications,
                    analysis: result.analysis,
                    metrics: result.metrics
                )
                
            case .separate(let speakers, let useOriginal):
                await eventHandler?(.progress(stage: stageName, percent: 0))
                let inputAudio = useOriginal ? context.originalAudio : context.currentAudio
                let tracks = try await separate(inputAudio, speakers: speakers)
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: tracks,
                    ussSeparated: result.ussSeparated,
                    transcription: result.transcription,
                    classifications: result.classifications,
                    analysis: result.analysis,
                    metrics: result.metrics
                )
                await eventHandler?(.progress(stage: stageName, percent: 100))
                
            case .separateUSS(let types):
                await eventHandler?(.progress(stage: stageName, percent: 0))
                guard let uss = ussProvider else {
                    throw ClearVoiceError.modelNotLoaded("USS")
                }
                
                // Resample to USS sample rate (32kHz)
                let ussInput = try context.currentAudio.resampled(to: 32000)
                
                // Use progress-aware multi-type separation
                let progressCallback: ProgressCallback? = if let handler = eventHandler {
                    { @Sendable percent in
                        await handler(.progress(stage: stageName, percent: percent))
                    }
                } else {
                    nil
                }
                
                // Process each type and emit per-embedding progress
                var ussSeparatedResults: [USSSoundType: ClearVoiceCore.AudioBuffer] = [:]
                for (idx, type) in types.enumerated() {
                    let separated = try await uss.separateSound(ussInput, type: type)
                    ussSeparatedResults[type] = separated
                    
                    // Emit per-embedding event
                    await eventHandler?(.ussSeparated(type: type, audio: separated))
                    
                    // Emit progress per embedding
                    let percent = Double(idx + 1) / Double(types.count) * 100.0
                    await progressCallback?(percent)
                }
                
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: result.separatedTracks,
                    ussSeparated: ussSeparatedResults,
                    transcription: result.transcription,
                    classifications: result.classifications,
                    analysis: result.analysis,
                    metrics: result.metrics
                )
                // Note: 100% progress already emitted in the loop above
                
            case .upscale:
                await eventHandler?(.progress(stage: stageName, percent: 0))
                guard let upscaler = upscalerProvider else {
                    throw ClearVoiceError.modelNotLoaded("Upscaler")
                }
                
                let upscaled: ClearVoiceCore.AudioBuffer
                
                // Use streaming for progress reporting if supported
                if let eventHandler, let streamable = upscaler as? StreamableOutput {
                    let inputAudio = context.currentAudio
                    let totalSamples = inputAudio.samples.count
                    var processedSamples = 0
                    var streamedSamples: [Float] = []
                    streamedSamples.reserveCapacity(totalSamples * 3) // Upscaling typically 3x samples
                    
                    for try await processedChunk in streamable.processStream(inputAudio) {
                        streamedSamples.append(contentsOf: processedChunk.samples)
                        processedSamples += processedChunk.samples.count
                        // Estimate progress based on input samples (3x output expected)
                        let estimatedTotalOutput = totalSamples * 3
                        let percent = min(100.0, Double(processedSamples) / Double(max(estimatedTotalOutput, 1)) * 100)
                        await eventHandler(.progress(stage: stageName, percent: percent))
                    }
                    
                    upscaled = AudioBuffer(
                        samples: streamedSamples,
                        sampleRate: upscaler.outputSampleRate,
                        channels: inputAudio.channels
                    )
                } else {
                    // Fallback to batch processing
                    upscaled = try await upscale(context.currentAudio)
                }
                
                context = PipelineContext(
                    analysis: context.analysis,
                    currentAudio: upscaled,
                    originalAudio: context.originalAudio
                )
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: result.separatedTracks,
                    ussSeparated: result.ussSeparated,
                    transcription: result.transcription,
                    classifications: result.classifications,
                    analysis: context.analysis,
                    metrics: result.metrics
                )
                await eventHandler?(.progress(stage: stageName, percent: 100))
                
            case .transcribe(let model):
                await eventHandler?(.progress(stage: stageName, percent: 0))
                
                // Use progress-aware transcription
                // For batch transcribers this reports 100% at end; streaming transcribers can report per-segment
                let transcription = try await transcribe(context.currentAudio, model: model) { percent in
                    await eventHandler?(.progress(stage: stageName, percent: percent))
                }
                
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: result.separatedTracks,
                    ussSeparated: result.ussSeparated,
                    transcription: transcription,
                    classifications: result.classifications,
                    analysis: result.analysis,
                    metrics: result.metrics
                )
                
                // Emit per-segment events for consumers
                for segment in transcription.segments {
                    await eventHandler?(.transcriptionSegment(segment))
                }
                await eventHandler?(.progress(stage: stageName, percent: 100))
                
            case .classify:
                await eventHandler?(.progress(stage: stageName, percent: 0))
                let classifications = try await classify(context.currentAudio)
                result = PipelineResult(
                    audio: result.audio,
                    separatedTracks: result.separatedTracks,
                    ussSeparated: result.ussSeparated,
                    transcription: result.transcription,
                    classifications: classifications,
                    analysis: result.analysis,
                    metrics: result.metrics
                )
                await eventHandler?(.progress(stage: stageName, percent: 100))
                
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
                                ussSeparated: result.ussSeparated ?? branchResult.ussSeparated,
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
                                ussSeparated: result.ussSeparated ?? branchResult.ussSeparated,
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
                    var processedTracks: [ClearVoiceCore.AudioBuffer] = []
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
                        ussSeparated: result.ussSeparated,
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
            ussSeparated: result.ussSeparated,
            transcription: result.transcription,
            classifications: result.classifications,
            analysis: result.analysis,
            metrics: metrics
        )
    }
    
    // MARK: - Model Management
    
    /// Preload models into memory
    ///
    /// Note: Model loading is handled by individual providers via their `load()` methods.
    /// This method exists for batch preloading hints but actual loading should be done
    /// through provider registration and explicit load calls.
    ///
    /// Example:
    /// ```swift
    /// let provider = MossFormer2SE48KProvider()
    /// try await provider.load()
    /// clearVoice.register(enhancer: provider, for: .mossformer2SE48K)
    /// ```
    public func preload(_ models: [any ModelIdentifier]) async throws {
        // Model preloading is handled by providers - this is a no-op hint
        // Providers should be loaded via their load() methods before registration
    }
    
    // MARK: - Model Lifecycle Convenience
    
    /// Unload a specific model by ID
    ///
    /// Calls the model's `unload()` method and removes it from tracking.
    /// Use this to free GPU/memory when a model is no longer needed.
    ///
    /// - Parameter modelId: The model identifier (e.g., "mossformer2_se_48k")
    public func unloadModel(_ modelId: String) async {
        await modelManager.unload(modelId: modelId)
    }
    
    /// Unload all models
    ///
    /// Calls `unload()` on all tracked models and clears the registry.
    /// Useful for memory cleanup before app termination or switching contexts.
    public func unloadAllModels() async {
        await modelManager.unloadAll()
    }
    
    /// Get model lifecycle statistics
    ///
    /// Returns current memory usage, loaded model count, and eviction stats.
    public var modelStats: ModelLifecycleManager.Stats {
        get async { await modelManager.stats }
    }
    
    /// Current memory usage in bytes
    ///
    /// Uses mach task_info to get the resident memory size of the current process.
    public nonisolated var memoryUsage: Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        return result == KERN_SUCCESS ? Int(info.resident_size) : 0
    }
}
