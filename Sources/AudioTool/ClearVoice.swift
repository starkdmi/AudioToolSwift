//
//  ClearVoice.swift
//  ClearVoice
//
//  Main unified API actor for audio ML processing
//

import Foundation
import AVFoundation
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
    internal var synthesizerProviders: [String: any SpeechSynthesizer] = [:]
    internal var translatorProviders: [TranslationModel: any TextTranslator] = [:]
    internal var textPreprocessorProviders: [String: any TextPreprocessor] = [:]
    
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
    ///   - targetSampleRate: Optional target sample rate for resampling
    /// - Returns: Loaded audio buffer
    public func loadAudio(from url: URL, targetSampleRate: Int? = nil) async throws -> ClearVoiceCore.AudioBuffer {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ClearVoiceError.audioFileNotFound(url)
        }
        
        // Load audio using AVFoundation
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)
        
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw ClearVoiceError.resourceUnavailable("Failed to create audio buffer")
        }
        
        try audioFile.read(into: buffer)
        
        guard let channelData = buffer.floatChannelData else {
            throw ClearVoiceError.resourceUnavailable("No audio data in buffer")
        }
        
        // Convert to mono if stereo
        var samples: [Float]
        if format.channelCount > 1 {
            // Mix down to mono
            samples = [Float](repeating: 0, count: Int(frameCount))
            for ch in 0..<Int(format.channelCount) {
                for i in 0..<Int(frameCount) {
                    samples[i] += channelData[ch][i]
                }
            }
            for i in 0..<samples.count {
                samples[i] /= Float(format.channelCount)
            }
        } else {
            samples = Array(UnsafeBufferPointer(start: channelData[0], count: Int(frameCount)))
        }
        
        let sourceSampleRate = Int(format.sampleRate)
        let targetRate = targetSampleRate ?? sourceSampleRate
        
        // Resample if needed
        if sourceSampleRate != targetRate {
            samples = resampleAudio(samples, from: sourceSampleRate, to: targetRate)
        }
        
        return AudioBuffer(samples: samples, sampleRate: targetRate, channels: 1)
    }
    
    /// Save audio to file
    ///
    /// Exports audio buffer to a file in the specified format.
    ///
    /// - Parameters:
    ///   - buffer: Audio buffer to save
    ///   - url: Destination file path
    ///   - format: Output format (default: .wav)
    public func saveAudio(_ buffer: ClearVoiceCore.AudioBuffer, to url: URL, format: AudioFormat = .wav) async throws {
        guard let avFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(buffer.sampleRate),
            channels: AVAudioChannelCount(buffer.channels),
            interleaved: false
        ) else {
            throw ClearVoiceError.invalidAudioFormat(
                expected: "pcmFormatFloat32",
                found: "unsupported"
            )
        }
        
        // Determine file settings based on format
        let settings: [String: Any]
        switch format {
        case .wav:
            settings = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: Double(buffer.sampleRate),
                AVNumberOfChannelsKey: buffer.channels,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsNonInterleaved: false
            ]
        case .m4a:
            settings = [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: Double(buffer.sampleRate),
                AVNumberOfChannelsKey: buffer.channels,
                AVEncoderBitRateKey: 128000
            ]
        case .mp3:
            // Note: MP3 encoding not directly supported, use WAV format
            settings = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: Double(buffer.sampleRate),
                AVNumberOfChannelsKey: buffer.channels,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false
            ]
        case .flac:
            settings = [
                AVFormatIDKey: kAudioFormatFLAC,
                AVSampleRateKey: Double(buffer.sampleRate),
                AVNumberOfChannelsKey: buffer.channels
            ]
        }
        
        let audioFile = try AVAudioFile(forWriting: url, settings: settings)
        
        guard let pcmBuffer = AVAudioPCMBuffer(
            pcmFormat: avFormat,
            frameCapacity: AVAudioFrameCount(buffer.samples.count)
        ) else {
            throw ClearVoiceError.resourceUnavailable("Failed to create PCM buffer")
        }
        
        pcmBuffer.frameLength = AVAudioFrameCount(buffer.samples.count)
        
        if let channelData = pcmBuffer.floatChannelData {
            for (index, sample) in buffer.samples.enumerated() {
                channelData[0][index] = sample
            }
        }
        
        try audioFile.write(from: pcmBuffer)
    }
    
    /// Simple linear interpolation resampling
    private func resampleAudio(_ samples: [Float], from sourceSampleRate: Int, to targetSampleRate: Int) -> [Float] {
        guard sourceSampleRate != targetSampleRate else { return samples }
        
        let ratio = Double(targetSampleRate) / Double(sourceSampleRate)
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
    
    // MARK: - Analysis
    
    /// Voice activity detection
    public func detect(
        _ audio: ClearVoiceCore.AudioBuffer,
        model: VADModel = .silero
    ) async throws -> [VADSegment] {
        guard let vad = vadProvider else {
            throw ClearVoiceError.modelNotLoaded("VAD")
        }
        return try await vad.detect(audio)
    }
    
    /// Speaker diarization
    public func diarize(
        _ audio: ClearVoiceCore.AudioBuffer,
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
        return try await enhancer.process(audio)
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
        _ audio: ClearVoiceCore.AudioBuffer,
        speakers: Int,
        model: SeparationModel = .mossformer2spk
    ) async throws -> [ClearVoiceCore.AudioBuffer] {
        guard let separator = separatorProviders[model] else {
            throw ClearVoiceError.modelNotLoaded(model.modelName)
        }
        return try await separator.separate(audio, speakers: speakers)
    }
    
    // MARK: - Upscaling
    
    /// Super-resolution upscaling
    public func upscale(_ audio: ClearVoiceCore.AudioBuffer) async throws -> ClearVoiceCore.AudioBuffer {
        guard let upscaler = upscalerProvider else {
            throw ClearVoiceError.modelNotLoaded("Upscaler")
        }
        return try await upscaler.process(audio)
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
        return try await transcriber.transcribe(audio)
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
                let enhanced: ClearVoiceCore.AudioBuffer
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
