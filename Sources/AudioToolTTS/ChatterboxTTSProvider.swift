//
//  ChatterboxTTSProvider.swift
//  ClearVoiceTTS
//
//  ChatterBox Multilingual TTS provider with MLX backend
//

import Foundation
import ClearVoice
import ClearVoiceCore
import ClearVoiceFluidAudio
import ChatterboxMLXSwift
import MLX
import MLXNN
import MLXRandom
import AudioUtils

/// ChatterBox TTS provider supporting 23 languages with voice cloning
///
/// ChatterBox is Resemble AI's production-grade open source TTS model with MIT license.
///
/// ## Supported Languages (23)
/// Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi,
/// Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian,
/// Swedish, Swahili, Turkish, Chinese.
///
/// ### Language Support Levels
/// - **Full (18)**: Works by default
/// - **Preprocessed (1)**: `ru` - uses RUAccent for stress marks (automatic)
/// - **Experimental (4)**: `zh`, `ja`, `ko`, `he` - missing text preprocessing
///
/// ## Model Repositories (HuggingFace)
/// - `starkdmi/chatterbox` (fp32)
/// - `starkdmi/chatterbox-fp16`
/// - `starkdmi/chatterbox-8bit`
/// - `starkdmi/chatterbox-6bit`
/// - `starkdmi/chatterbox-4bit`
///
/// ## Parameter Tips
///
/// **General Use (TTS and Voice Agents):**
/// - Default settings (`exaggeration=0.5`, `cfgWeight=0.5`) work well for most prompts
/// - If reference speaker has fast speaking style, lower `cfgWeight` to ~0.3
///
/// **Expressive or Dramatic Speech:**
/// - Try lower `cfgWeight` (~0.3) and increase `exaggeration` to ~0.7 or higher
/// - Higher `exaggeration` tends to speed up speech; reducing `cfgWeight` compensates
///
/// **Cross-language:**
/// - Ensure reference clip matches the specified language tag
/// - To mitigate accent transfer, set `cfgWeight` to 0
///
/// ## Usage
/// ```swift
/// // Basic usage with default voice (no reference needed)
/// let tts = TTSProviders.chatterbox(precision: .fp16, language: .english)
/// try await tts.load()
/// let audio = try await tts.synthesize("Hello world!", voice: "")
///
/// // Voice cloning with reference audio
/// let tts = TTSProviders.chatterbox(precision: .fp16, language: .english)
/// try await tts.load()
/// try await tts.setReferenceAudio(from: referenceURL)
/// let audio = try await tts.synthesize("Hello world!", voice: "")
///
/// // Expressive speech
/// await tts.configure(exaggeration: 0.7, cfgWeight: 0.3)
/// let expressiveAudio = try await tts.synthesize("Wow, that's amazing!", voice: "")
///
/// // Russian with automatic RUAccent
/// let tts = TTSProviders.chatterbox(language: .russian)
/// try await tts.load()
/// // RUAccent is applied automatically if configured
/// let audio = try await tts.synthesize("Привет мир", voice: "")
/// ```
public actor ChatterboxTTSProvider: SpeechSynthesizer {
    
    // MARK: - Public Properties
    
    /// Base HuggingFace repository (without precision suffix)
    public static let baseRepo = "starkdmi/chatterbox"
    
    /// Default precision
    public static let defaultPrecision: ModelPrecision = .fp32
    
    /// Current loading state for UI binding
    public private(set) var state: ModelState = .notLoaded
    
    /// Observable state stream for UI
    public nonisolated var stateStream: AsyncStream<ModelState> {
        AsyncStream { continuation in
            Task { await addStateContinuation(continuation) }
        }
    }
    
    private func addStateContinuation(_ continuation: AsyncStream<ModelState>.Continuation) {
        stateContinuations.append(continuation)
    }
    
    // MARK: - SpeechSynthesizer Conformance
    
    public nonisolated let sampleRate: Int = 24000
    
    /// ChatterBox uses voice cloning - no preset voices
    public nonisolated var availableVoices: [String] { [] }
    
    // MARK: - Private Properties
    
    private var voiceEncoder: VoiceEncoder?
    private var t3: T3?
    private var s3Tokenizer: S3TokenizerV2?
    private var s3Gen: S3Token2Wav?
    private var textTokenizer: MTLTokenizer?
    
    private var modelPath: URL?
    private nonisolated let language: ChatterboxLanguage
    private nonisolated let repo: String
    private nonisolated let precision: ModelPrecision
    
    /// Optional Russian text preprocessor
    private var ruAccentProvider: RUAccentProvider?
    private nonisolated let useRuAccent: Bool
    private nonisolated let convertToStressMarks: Bool
    
    /// Reference audio for voice cloning (optional - uses default voice if not set)
    private var referenceWav: MLXArray?
    private var referenceSampleRate: Int?
    private var speakerEmbedding: MLXArray?
    private var refDict: [String: MLXArray]?
    
    /// Default voice conditioning from conds.safetensors (loaded during load())
    private var defaultT3Cond: T3Cond?
    private var defaultRefDict: [String: MLXArray]?
    private var hasDefaultVoice: Bool = false
    
    /// VAD trimmer for removing start/end artifacts (breathing/noise)
    private var vadProvider: FluidAudioVADProvider?
    private var vadTrimEnabled: Bool = true  // Enabled by default - ChatterBox produces start/end artifacts
    
    /// State stream continuations
    private var stateContinuations: [AsyncStream<ModelState>.Continuation] = []
    
    /// Synthesis parameters
    private var exaggeration: Float = 0.5
    private var t3Temperature: Float = 0.8
    private var t3CfgWeight: Float = 0.5
    private var t3RepetitionPenalty: Float = 2.0
    private var t3MinP: Float = 0.05
    private var t3TopP: Float = 1.0
    private var seed: UInt64 = 0
    
    // MARK: - Initialization
    
    /// Initialize ChatterBox TTS provider with precision-based repo selection
    /// - Parameters:
    ///   - precision: Model precision (fp32, fp16, 8bit, 6bit, 4bit)
    ///   - language: Target language for synthesis
    ///   - useRuAccent: Enable automatic RUAccent for Russian (default: true)
    ///   - convertToStressMarks: Convert + to Unicode stress marks for Russian (default: true)
    public init(
        precision: ModelPrecision = ChatterboxTTSProvider.defaultPrecision,
        language: ChatterboxLanguage = .english,
        useRuAccent: Bool = true,
        convertToStressMarks: Bool = true
    ) {
        self.precision = precision
        // ChatterBox repos use different suffix pattern
        self.repo = switch precision {
        case .fp32: Self.baseRepo
        case .fp16: "\(Self.baseRepo)-fp16"
        case .bit8: "\(Self.baseRepo)-8bit"
        case .bit6: "\(Self.baseRepo)-6bit"
        case .bit4: "\(Self.baseRepo)-4bit"
        default: precision.repo(base: Self.baseRepo)
        }
        self.language = language
        self.useRuAccent = useRuAccent
        self.convertToStressMarks = convertToStressMarks
    }
    
    /// Initialize with explicit HuggingFace repo (for custom repos)
    /// - Parameters:
    ///   - repo: Full HuggingFace repository ID
    ///   - language: Target language for synthesis
    ///   - useRuAccent: Enable automatic RUAccent for Russian (default: true)
    ///   - convertToStressMarks: Convert + to Unicode stress marks (default: true)
    public init(
        repo: String,
        language: ChatterboxLanguage = .english,
        useRuAccent: Bool = true,
        convertToStressMarks: Bool = true
    ) {
        self.precision = .fp32
        self.repo = repo
        self.language = language
        self.useRuAccent = useRuAccent
        self.convertToStressMarks = convertToStressMarks
    }
    
    /// Initialize with explicit local path (skips download)
    /// - Parameters:
    ///   - modelPath: Path to ChatterBox model weights directory
    ///   - language: Target language for synthesis
    ///   - useRuAccent: Enable automatic RUAccent for Russian (default: true)
    ///   - convertToStressMarks: Convert + to Unicode stress marks (default: true)
    public init(
        modelPath: URL,
        language: ChatterboxLanguage = .english,
        useRuAccent: Bool = true,
        convertToStressMarks: Bool = true
    ) {
        self.modelPath = modelPath
        self.precision = .fp32
        self.repo = ChatterboxTTSProvider.baseRepo
        self.language = language
        self.useRuAccent = useRuAccent
        self.convertToStressMarks = convertToStressMarks
    }
    
    // MARK: - Model Loading
    
    /// Load model, downloading if necessary
    public func load() async throws {
        // Check for explicit path first
        if let path = modelPath {
            try await loadFromPath(path)
            return
        }
        
        // Check if already downloaded
        if let cached = ModelDownloader.shared.localPath(for: repo) {
            try await loadFromPath(cached)
            return
        }
        
        // Download from HuggingFace
        updateState(.downloading(progress: 0))
        
        do {
            let path = try await ModelDownloader.shared.downloadAndGetPath(
                repo: repo,
                matching: ["*.safetensors", "*.json"]
            ) { [weak self] progress in
                Task { @MainActor in
                    await self?.updateState(.downloading(progress: progress.fractionCompleted))
                }
            }
            
            try await loadFromPath(path)
        } catch {
            updateState(.failed(error.localizedDescription))
            throw error
        }
    }
    
    /// Load from a specific path
    private func loadFromPath(_ path: URL) async throws {
        updateState(.loading)
        
        do {
            // Find the model weights
            var weightsPath = path
            
            // If path is a directory, find the safetensors file
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: path.path, isDirectory: &isDir), isDir.boolValue {
                let files = try FileManager.default.contentsOfDirectory(at: path, includingPropertiesForKeys: nil)
                if let modelFile = files.first(where: { $0.lastPathComponent == "model.safetensors" }) {
                    weightsPath = modelFile
                } else if let anyFile = files.first(where: { $0.pathExtension == "safetensors" }) {
                    weightsPath = anyFile
                } else {
                    throw ClearVoiceError.modelNotFound("No .safetensors file found in \(path.lastPathComponent)")
                }
            }
            
            // Load all weights
            let weightsURL = URL(fileURLWithPath: weightsPath.path)
            let weightsData = try Data(contentsOf: weightsURL)
            let allWeights = try MLX.loadArrays(data: weightsData)
            
            // Load quantization config if present
            let quantizationConfig = loadQuantizationConfig(weightsPath: weightsPath.path)
            
            // Initialize models
            voiceEncoder = VoiceEncoder()
            t3 = T3(.multilingual())
            s3Tokenizer = S3TokenizerV2(name: "speech_tokenizer_v2_25hz")
            s3Gen = S3Token2Wav()
            
            // Separate weights by module prefix
            var veWeights: [String: MLXArray] = [:]
            var t3Weights: [String: MLXArray] = [:]
            var s3genWeights: [String: MLXArray] = [:]
            
            for (key, value) in allWeights {
                if key.hasPrefix("ve.") {
                    veWeights[String(key.dropFirst("ve.".count))] = value
                } else if key.hasPrefix("t3.") {
                    t3Weights[String(key.dropFirst("t3.".count))] = value
                } else if key.hasPrefix("s3gen.") {
                    s3genWeights[String(key.dropFirst("s3gen.".count))] = value
                } else if key.hasPrefix("flow.") ||
                          key.hasPrefix("mel2wav.") ||
                          key.hasPrefix("speaker_encoder.") ||
                          key.hasPrefix("f0_predictor.") {
                    s3genWeights[key] = value
                }
            }
            
            // Sanitize and load weights
            let sanitizedVE = voiceEncoder!.sanitize(weights: veWeights)
            let sanitizedT3 = t3!.sanitize(weights: t3Weights)
            let sanitizedS3Gen = s3Gen!.sanitize(s3genWeights)
            
            try updateModule(voiceEncoder!, name: "ve", weights: sanitizedVE, quantization: quantizationConfig)
            try updateModule(t3!, name: "t3", weights: sanitizedT3, quantization: quantizationConfig)
            
            let expectedMissing: Set<String> = [
                "flowEncoder.flow.encoder.embed.pos_enc.pe",
                "flowEncoder.flow.encoder.up_embed.pos_enc.pe",
                "mel2wav.stft_window",
                "trim_fade",
            ]
            try updateModule(s3Gen!, name: "s3gen", weights: sanitizedS3Gen, quantization: quantizationConfig, expectedMissing: expectedMissing)
            
            // Load S3Tokenizer weights - check bundled first, then HF cache
            let s3TokenizerWeights = try resolveS3TokenizerWeights(allWeights: allWeights, baseDir: weightsPath.deletingLastPathComponent())
            let sanitizedS3Tok = s3Tokenizer!.sanitize(weights: s3TokenizerWeights)
            let expectedTokMissing: Set<String> = ["encoder.freqs.0", "encoder.freqs.1"]
            try updateModule(s3Tokenizer!, name: "s3tokenizer", weights: sanitizedS3Tok, quantization: nil, expectedMissing: expectedTokMissing)
            
            // Load text tokenizer
            let baseDir = weightsPath.deletingLastPathComponent()
            let tokenizerPath = resolveTokenizerJsonPath(baseDir: baseDir)
            if let tokenizerPath = tokenizerPath {
                textTokenizer = try MTLTokenizer(tokenizerJSONPath: tokenizerPath)
            } else {
                throw ClearVoiceError.modelNotFound("Missing tokenizer JSON (grapheme_mtl_merged_expanded_v1.json)")
            }
            
            // Load default voice conditioning from conds.safetensors (optional)
            let condsPath = baseDir.appendingPathComponent("conds.safetensors")
            if FileManager.default.fileExists(atPath: condsPath.path) {
                try loadDefaultConds(from: condsPath)
            }
            
            // Load VAD provider for trimming start/end artifacts (enabled by default)
            // Parameters tuned for TTS output based on Silero VAD best practices:
            // Optimized for ChatterBox TTS artifact trimming (tested 2024-01):
            // - threshold 0.9: Required to detect low-level breathing/whisper artifacts
            // - minSpeechDuration 0.1: Catch brief intonations at speech boundaries
            // - minSilenceDuration 0.1: Standard silence detection
            if vadTrimEnabled {
                let vad = FluidAudioVADProvider(
                    threshold: 0.9,
                    minSpeechDuration: 0.1,
                    minSilenceDuration: 0.1
                )
                try await vad.load()
                self.vadProvider = vad
            }
            
            modelPath = path
            updateState(.ready)
        } catch {
            updateState(.failed(error.localizedDescription))
            throw error
        }
    }
    
    // MARK: - State Updates
    
    private func updateState(_ newState: ModelState) {
        state = newState
        for continuation in stateContinuations {
            continuation.yield(newState)
        }
    }
    
    // MARK: - Reference Audio
    
    /// Set reference audio for voice cloning from URL
    /// - Parameter url: Path to reference audio file (WAV recommended)
    public func setReferenceAudio(from url: URL) async throws {
        guard let voiceEncoder = voiceEncoder else {
            throw ClearVoiceError.modelNotLoaded("ChatterBox")
        }
        
        let info = try AudioLoader().getAudioInfo(from: url.path)
        let config = AudioLoader.Configuration(
            targetSampleRate: info.sampleRate,
            normalizationMode: .none,
            resamplingMethod: .none
        )
        let loader = AudioLoader(config: config)
        let audio = try loader.load(from: url.path)
        
        // Convert to mono if needed
        let wav = audio.ndim == 1 ? audio : MLX.mean(audio, axis: 0)
        
        referenceWav = wav
        referenceSampleRate = Int(info.sampleRate)
        
        // Compute speaker embedding - MUST resample to 16kHz first (VoiceEncoder expects 16kHz)
        let refWav16kForVE = resampleAudioPolyphase(wav, origSR: Int(info.sampleRate), targetSR: S3_SR)
        speakerEmbedding = voiceEncoder.embedsFromWavs([refWav16kForVE], sampleRate: S3_SR)
        
        // Precompute reference dictionary if s3Gen is loaded
        if let s3Gen = s3Gen, let s3Tokenizer = s3Tokenizer {
            let refWav24k = resampleAudioPolyphase(wav, origSR: Int(info.sampleRate), targetSR: S3GEN_SR)
            let decCondLen = 10 * S3GEN_SR
            let refWav24kTrim = refWav24k[0..<min(decCondLen, refWav24k.shape[0])]
            let refWav16kFrom24 = resampleAudioPolyphase(refWav24kTrim, origSR: S3GEN_SR, targetSR: S3_SR)
            
            let s3genMel = logMelSpectrogramCompat(refWav16kFrom24, nMels: 128)
            let s3genMelBatch = s3genMel.expandedDimensions(axis: 0)
            let s3genMelLen = MLXArray([Int32(s3genMelBatch.shape[2])])
            let (s3genTokens, s3genTokenLens) = s3Tokenizer(s3genMelBatch, s3genMelLen)
            
            refDict = s3Gen.embed_ref(
                ref_wav: refWav24kTrim.expandedDimensions(axis: 0),
                ref_sr: S3GEN_SR,
                ref_speech_tokens: s3genTokens,
                ref_speech_token_lens: s3genTokenLens
            )
        }
    }
    
    /// Set reference audio from an AudioBuffer
    /// - Parameter audio: Audio buffer with reference speech
    public func setReferenceAudio(_ audio: ClearVoiceCore.AudioBuffer) async throws {
        guard let voiceEncoder = voiceEncoder else {
            throw ClearVoiceError.modelNotLoaded("ChatterBox")
        }
        
        let wav = MLXArray(audio.samples)
        let sr = audio.sampleRate
        
        // Convert to mono if needed
        let monoWav = wav.ndim == 1 ? wav : MLX.mean(wav, axis: 0)
        
        referenceWav = monoWav
        referenceSampleRate = sr
        
        // Compute speaker embedding
        speakerEmbedding = voiceEncoder.embedsFromWavs([monoWav], sampleRate: sr)
        
        // Precompute reference dictionary
        if let s3Gen = s3Gen, let s3Tokenizer = s3Tokenizer {
            let refWav24k = resampleAudioPolyphase(monoWav, origSR: sr, targetSR: S3GEN_SR)
            let decCondLen = 10 * S3GEN_SR
            let refWav24kTrim = refWav24k[0..<min(decCondLen, refWav24k.shape[0])]
            let refWav16kFrom24 = resampleAudioPolyphase(refWav24kTrim, origSR: S3GEN_SR, targetSR: S3_SR)
            
            let s3genMel = logMelSpectrogramCompat(refWav16kFrom24, nMels: 128)
            let s3genMelBatch = s3genMel.expandedDimensions(axis: 0)
            let s3genMelLen = MLXArray([Int32(s3genMelBatch.shape[2])])
            let (s3genTokens, s3genTokenLens) = s3Tokenizer(s3genMelBatch, s3genMelLen)
            
            refDict = s3Gen.embed_ref(
                ref_wav: refWav24kTrim.expandedDimensions(axis: 0),
                ref_sr: S3GEN_SR,
                ref_speech_tokens: s3genTokens,
                ref_speech_token_lens: s3genTokenLens
            )
        }
    }
    
    // MARK: - Default Voice Loading
    
    /// Load default voice conditioning from conds.safetensors
    /// - Parameter path: Path to conds.safetensors file
    private func loadDefaultConds(from path: URL) throws {
        let data = try Data(contentsOf: path)
        let tensors = try MLX.loadArrays(data: data)
        
        // Load T3 conditioning
        guard let speakerEmb = tensors["t3.speaker_emb"] else {
            throw ClearVoiceError.resourceUnavailable("Missing t3.speaker_emb in conds.safetensors")
        }
        
        let condTokens = tensors["t3.cond_prompt_speech_tokens"]?.asType(.int32)
        let condEmb = tensors["t3.cond_prompt_speech_emb"]
        let emotionAdv = tensors["t3.emotion_adv"]
        
        defaultT3Cond = T3Cond(
            speaker_emb: speakerEmb,
            cond_prompt_speech_tokens: condTokens,
            cond_prompt_speech_emb: condEmb,
            emotion_adv: emotionAdv ?? MLX.ones([1, 1, 1]) * 0.5
        )
        
        // Load S3Gen reference dictionary
        guard let promptToken = tensors["gen.prompt_token"],
              let promptTokenLen = tensors["gen.prompt_token_len"],
              let promptFeat = tensors["gen.prompt_feat"],
              let embedding = tensors["gen.embedding"] else {
            throw ClearVoiceError.resourceUnavailable("Missing gen.* keys in conds.safetensors")
        }
        
        var promptFeatLen = tensors["gen.prompt_feat_len"]?.asType(.int32)
        if promptFeatLen == nil {
            let length: Int
            if promptFeat.ndim >= 2 {
                length = promptFeat.dim(1)
            } else {
                length = promptFeat.dim(0)
            }
            promptFeatLen = MLXArray([Int32(length)])
        }
        
        defaultRefDict = [
            "prompt_token": promptToken.asType(.int32),
            "prompt_token_len": promptTokenLen.asType(.int32),
            "prompt_feat": promptFeat,
            "prompt_feat_len": promptFeatLen!,
            "embedding": embedding,
        ]
        
        hasDefaultVoice = true
    }
    
    // MARK: - RUAccent Configuration
    
    /// Configure RUAccent provider for Russian text preprocessing
    /// - Parameter provider: Configured RUAccentProvider instance
    public func configureRuAccent(_ provider: RUAccentProvider) {
        self.ruAccentProvider = provider
    }
    
    // MARK: - VAD Trimmer Configuration
    
    /// Configure VAD-based audio trimming to remove start/end artifacts
    ///
    /// ChatterBox may produce breathing sounds or noise at the start/end of generated audio.
    /// VAD trimming detects speech boundaries and trims non-speech segments.
    ///
    /// - Parameters:
    ///   - enabled: Enable VAD trimming (default: true)
    ///   - provider: Optional custom VAD provider (default: auto-created SileroVAD)
    public func configureVadTrimmer(enabled: Bool = true, provider: FluidAudioVADProvider? = nil) async throws {
        self.vadTrimEnabled = enabled
        
        if enabled && provider == nil && self.vadProvider == nil {
            // Create and load default VAD provider
            let vad = FluidAudioVADProvider(
                threshold: 0.5,  // Standard threshold for clear speech/silence boundary
                minSpeechDuration: 0.1,  // Shorter min duration for TTS output
                minSilenceDuration: 0.1  // Detect short silences
            )
            try await vad.load()
            self.vadProvider = vad
        } else if let provider = provider {
            self.vadProvider = provider
        }
    }
    
    /// Disable VAD trimming
    public func disableVadTrimmer() {
        self.vadTrimEnabled = false
    }
    
    // MARK: - Synthesis Parameters
    
    /// Configure synthesis parameters for customizing voice output
    ///
    /// ## Parameter Guide
    ///
    /// **General Use (TTS and Voice Agents):**
    /// - Default settings (`exaggeration=0.5`, `cfgWeight=0.5`) work well for most prompts
    /// - If reference speaker has fast speaking style, lower `cfgWeight` to ~0.3
    ///
    /// **Expressive or Dramatic Speech:**
    /// - Try lower `cfgWeight` (~0.3) and increase `exaggeration` to ~0.7 or higher
    /// - Higher `exaggeration` tends to speed up speech; reducing `cfgWeight` compensates
    ///
    /// **Cross-language:**
    /// - Ensure reference clip matches the language being synthesized
    /// - To mitigate accent transfer from reference, set `cfgWeight` to 0
    ///
    /// - Parameters:
    ///   - exaggeration: Emotion/intensity control (0.0-1.0+). Higher = more expressive, faster speech. Default: 0.5
    ///   - temperature: Sampling temperature for speech tokens. Higher = more variation. Default: 0.8
    ///   - cfgWeight: Classifier-free guidance weight. Lower = slower/more deliberate. Default: 0.5
    ///   - repetitionPenalty: Penalty for repeating tokens. Default: 2.0
    ///   - minP: Minimum probability threshold for sampling. Default: 0.05
    ///   - topP: Nucleus sampling threshold. Default: 1.0
    ///   - seed: Random seed for reproducible results. Default: 0 (random)
    public func configure(
        exaggeration: Float = 0.5,
        temperature: Float = 0.8,
        cfgWeight: Float = 0.5,
        repetitionPenalty: Float = 2.0,
        minP: Float = 0.05,
        topP: Float = 1.0,
        seed: UInt64 = 0
    ) {
        self.exaggeration = exaggeration
        self.t3Temperature = temperature
        self.t3CfgWeight = cfgWeight
        self.t3RepetitionPenalty = repetitionPenalty
        self.t3MinP = minP
        self.t3TopP = topP
        self.seed = seed
    }
    
    // MARK: - SpeechSynthesizer Protocol
    
    /// Synthesize text to speech
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice identifier (ignored - uses reference audio or default voice)
    /// - Returns: Audio buffer with synthesized speech (24kHz mono)
    public func synthesize(_ text: String, voice: String) async throws -> ClearVoiceCore.AudioBuffer {
        guard let t3 = t3, let s3Gen = s3Gen, let textTokenizer = textTokenizer else {
            throw ClearVoiceError.modelNotLoaded("ChatterBox")
        }
        
        // Use reference audio if set, otherwise fall back to default voice
        let useRefDict: [String: MLXArray]
        let useSpeakerEmbedding: MLXArray
        let useT3Cond: T3Cond?
        
        if let refDict = refDict, let speakerEmbedding = speakerEmbedding {
            // User provided reference audio
            useRefDict = refDict
            useSpeakerEmbedding = speakerEmbedding
            useT3Cond = nil  // Will build T3Cond from speaker embedding
        } else if hasDefaultVoice, let defaultRefDict = defaultRefDict, let defaultT3Cond = defaultT3Cond {
            // Use default voice from conds.safetensors
            useRefDict = defaultRefDict
            useSpeakerEmbedding = defaultT3Cond.speaker_emb
            useT3Cond = defaultT3Cond
        } else {
            throw ClearVoiceError.resourceUnavailable("No reference audio set and no default voice available. Either call setReferenceAudio() or use a model repo with conds.safetensors.")
        }
        
        // Preprocess text
        let processedText = await preprocessText(text)
        
        // Initialize random seed
        MLXRandom.seed(seed)
        
        // Tokenize text
        let textTokenIds = textTokenizer.textToTokens(processedText, languageId: language.rawValue)
        var textTokens = MLXArray(textTokenIds, [1, textTokenIds.count]).asType(.int32)
        
        // Apply CFG if needed
        if t3CfgWeight > 0.0 {
            textTokens = concatenated([textTokens, textTokens], axis: 0)
        }
        
        // Add start/end tokens
        let batch = textTokens.shape[0]
        let sot = MLXArray(
            Array(repeating: Int32(t3.hp.start_text_token), count: batch),
            [batch, 1]
        )
        let eot = MLXArray(
            Array(repeating: Int32(t3.hp.stop_text_token), count: batch),
            [batch, 1]
        )
        textTokens = concatenated([sot, textTokens, eot], axis: 1)
        
        // Compute or use T3 conditioning
        var t3Cond: T3Cond
        
        if let precomputedT3Cond = useT3Cond {
            // Use precomputed T3Cond from conds.safetensors
            t3Cond = precomputedT3Cond
            // Update exaggeration
            t3Cond.emotion_adv = MLX.ones([1, 1, 1]) * exaggeration
        } else {
            // Compute T3 conditioning from reference audio
            guard let refWav = referenceWav, let refSr = referenceSampleRate else {
                throw ClearVoiceError.resourceUnavailable("Reference audio not set")
            }
            
            let encCondLen = 6 * S3_SR
            let refWav16kFull = resampleAudioPolyphase(refWav, origSR: refSr, targetSR: S3_SR)
            let refWav16k = refWav16kFull[0..<min(encCondLen, refWav16kFull.shape[0])]
            
            var t3Mel = logMelSpectrogramCompat(refWav16k, nMels: 128)
            let maxMelFrames = t3.hp.speech_cond_prompt_len * 4
            if t3Mel.dim(1) > maxMelFrames {
                t3Mel = t3Mel[0..., 0..<maxMelFrames]
            }
            
            let t3MelBatch = t3Mel.expandedDimensions(axis: 0)
            let t3MelLen = MLXArray([Int32(t3MelBatch.shape[2])])
            guard let s3Tokenizer = s3Tokenizer else {
                throw ClearVoiceError.modelNotLoaded("S3Tokenizer")
            }
            let (t3Tokens, _) = s3Tokenizer(t3MelBatch, t3MelLen)
            let plen = t3.hp.speech_cond_prompt_len
            let t3PromptTokens = t3Tokens[0..., 0..<min(plen, t3Tokens.shape[1])]
            
            let veMean = MLX.mean(useSpeakerEmbedding, axis: 0, keepDims: true)
            
            t3Cond = T3Cond(
                speaker_emb: veMean,
                cond_prompt_speech_tokens: t3PromptTokens,
                emotion_adv: MLX.ones([1, 1, 1]) * exaggeration
            )
        }
        
        // Run T3 inference
        MLXRandom.seed(seed)
        let rawSpeechTokens = t3.inference(
            t3_cond: &t3Cond,
            text_tokens: textTokens,
            initial_speech_tokens: nil,
            max_new_tokens: nil,
            temperature: t3Temperature,
            top_p: t3TopP,
            min_p: t3MinP,
            repetition_penalty: t3RepetitionPenalty,
            cfg_weight: t3CfgWeight,
            language_id: language.rawValue
        )
        
        // Filter speech tokens
        let dropped = dropInvalidTokens(rawSpeechTokens)
        let speechTokens = filterValidTokens(dropped)
        
        // Generate audio
        let fullWav = s3Gen(speechTokens: speechTokens.asType(.int32), refDict: useRefDict, finalize: true)
        let wavMono = fullWav.ndim == 1 ? fullWav : fullWav[0]
        
        // Convert to samples
        let samples = wavMono.asArray(Float.self)
        var audioBuffer = ClearVoiceCore.AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
        
        // Apply VAD trimming if enabled (removes start/end artifacts like breathing/noise)
        if vadTrimEnabled, let vad = vadProvider {
            audioBuffer = try await trimAudioWithVAD(audioBuffer, using: vad)
        }
        
        return audioBuffer
    }
    
    /// Trim audio using VAD to remove non-speech segments at start and end
    private func trimAudioWithVAD(_ audio: ClearVoiceCore.AudioBuffer, using vad: FluidAudioVADProvider) async throws -> ClearVoiceCore.AudioBuffer {
        // Resample to 16kHz for VAD using simple linear interpolation
        // (polyphase resampling preserves high frequencies which causes VAD to over-detect speech)
        let vadSampleRate = 16000
        let resampledAudio: ClearVoiceCore.AudioBuffer
        
        if audio.sampleRate != vadSampleRate {
            // Use simple linear interpolation resampling
            let ratio = Double(vadSampleRate) / Double(audio.sampleRate)
            let newCount = Int(Double(audio.samples.count) * ratio)
            var resampled = [Float](repeating: 0, count: newCount)
            
            for i in 0..<newCount {
                let srcPos = Double(i) / ratio
                let srcIdx = Int(srcPos)
                let frac = Float(srcPos - Double(srcIdx))
                
                if srcIdx + 1 < audio.samples.count {
                    resampled[i] = audio.samples[srcIdx] * (1 - frac) + audio.samples[srcIdx + 1] * frac
                } else if srcIdx < audio.samples.count {
                    resampled[i] = audio.samples[srcIdx]
                }
            }
            resampledAudio = ClearVoiceCore.AudioBuffer(samples: resampled, sampleRate: vadSampleRate, channels: 1)
        } else {
            resampledAudio = audio
        }
        
        // Detect speech segments
        let segments = try await vad.detect(resampledAudio)
        
        #if DEBUG
        print("[VAD] Detected \(segments.count) speech segments")
        #endif
        
        // If no speech detected, return original (avoid trimming everything)
        guard !segments.isEmpty else {
            #if DEBUG
            print("[VAD] No speech detected, returning original audio")
            #endif
            return audio
        }
        
        // Find first and last speech boundaries
        let firstSpeechStart = segments.first!.timeRange.start
        let lastSpeechEnd = segments.last!.timeRange.end
        let audioDuration = audio.duration
        
        #if DEBUG
        print("[VAD] Trimming: \(String(format: "%.3f", audioDuration))s -> \(String(format: "%.3f", lastSpeechEnd + 0.01))s")
        #endif
        
        // Convert times to sample indices in original audio
        let startSample = max(0, Int(firstSpeechStart * Double(audio.sampleRate)))
        let endSample = min(audio.samples.count, Int(lastSpeechEnd * Double(audio.sampleRate)))
        
        // Add 10ms padding after last speech segment (tighter than Python's 30ms for cleaner TTS output)
        let paddingSamples = Int(0.01 * Double(audio.sampleRate))
        let paddedStart = max(0, startSample)  // No padding at start
        let paddedEnd = min(audio.samples.count, endSample + paddingSamples)
        
        // Calculate what we're trimming
        let trimmedFromStart = paddedStart
        let trimmedFromEnd = audio.samples.count - paddedEnd
        
        #if DEBUG
        print("[VAD] Trimming: \(trimmedFromStart) samples from start, \(trimmedFromEnd) samples from end")
        #endif
        
        // Only trim if there's something to trim
        if paddedStart == 0 && paddedEnd == audio.samples.count {
            #if DEBUG
            print("[VAD] Nothing to trim, speech spans entire audio")
            #endif
            return audio
        }
        
        // Extract trimmed audio
        let trimmedSamples = Array(audio.samples[paddedStart..<paddedEnd])
        
        #if DEBUG
        let trimmedDuration = Double(trimmedSamples.count) / Double(audio.sampleRate)
        print("[VAD] Trimmed: \(String(format: "%.3f", audioDuration - trimmedDuration))s removed")
        #endif
        
        return ClearVoiceCore.AudioBuffer(samples: trimmedSamples, sampleRate: audio.sampleRate, channels: 1)
    }
    
    
    /// Stream synthesized audio chunks
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice identifier (ignored)
    /// - Returns: Async stream of audio chunks
    public nonisolated func streamSynthesis(_ text: String, voice: String) -> AsyncThrowingStream<ClearVoiceCore.AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // For now, generate full audio and yield as single chunk
                    // TODO: Implement sentence-level streaming for lower latency
                    let audio = try await synthesize(text, voice: voice)
                    continuation.yield(audio)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Text Preprocessing
    
    private func preprocessText(_ text: String) async -> String {
        var processedText = text
        
        // Apply RUAccent for Russian if enabled
        if language == .russian && useRuAccent {
            if let ruAccent = ruAccentProvider {
                do {
                    processedText = try ruAccent.process(text)
                    if convertToStressMarks {
                        processedText = RUAccentProvider.convertToStressMarks(processedText)
                    }
                } catch {
                    // Continue with original text if RUAccent fails
                    // This follows the "graceful fallback" requirement
                }
            }
        }
        
        return processedText
    }
    
    // MARK: - AudioProcessor Conformance
    
    public var inputChannels: Int { 0 }  // Text input
    public var outputChannels: Int { 1 }
    
    public func process(_ input: ClearVoiceCore.AudioBuffer) async throws -> ClearVoiceCore.AudioBuffer {
        throw ClearVoiceError.pipelineConfigurationInvalid("ChatterBox is a synthesizer, use synthesize() instead of process()")
    }
    
    // MARK: - Private Helpers
    
    private func dropInvalidTokens(_ tokens: MLXArray) -> MLXArray {
        let flat = tokens.flattened().asArray(Int32.self)
        var start = 0
        var end = flat.count
        if let sosIdx = flat.firstIndex(of: 6561) {
            start = sosIdx + 1
        }
        if let eosIdx = flat.firstIndex(of: 6562) {
            end = eosIdx
        }
        if end <= start {
            return MLXArray.zeros([0], dtype: .int32)
        }
        let slice = Array(flat[start..<end])
        return MLXArray(slice)
    }
    
    private func filterValidTokens(_ tokens: MLXArray) -> MLXArray {
        let flat = tokens.flattened().asArray(Int32.self)
        let filtered = flat.filter { $0 < 6561 }
        return MLXArray(filtered, [1, filtered.count])
    }
    
    private struct QuantizationConfig {
        let groupSize: Int
        let bits: Int
    }
    
    private func loadQuantizationConfig(weightsPath: String) -> QuantizationConfig? {
        let weightsURL = URL(fileURLWithPath: weightsPath)
        let baseURL = weightsURL.hasDirectoryPath ? weightsURL : weightsURL.deletingLastPathComponent()
        let configURL = baseURL.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL) else {
            return nil
        }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let quant = json["quantization"] as? [String: Any],
              let bits = intValue(quant["bits"]),
              let groupSize = intValue(quant["group_size"]) else {
            return nil
        }
        return QuantizationConfig(groupSize: groupSize, bits: bits)
    }
    
    private func intValue(_ value: Any?) -> Int? {
        if let n = value as? Int { return n }
        if let n = value as? Double { return Int(n) }
        return nil
    }
    
    private func hasQuantizedWeights(_ weights: [String: MLXArray]) -> Bool {
        weights.keys.contains { $0.hasSuffix(".scales") }
    }
    
    private func updateModule(
        _ module: Module,
        name: String,
        weights: [String: MLXArray],
        quantization: QuantizationConfig?,
        expectedMissing: Set<String> = []
    ) throws {
        // Apply quantization if needed
        if hasQuantizedWeights(weights), let quantization = quantization {
            let keySet = Set(weights.keys)
            quantize(
                model: module,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                filter: { path, _ in keySet.contains("\(path).scales") }
            )
        }
        
        let currWeights = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        let filtered = weights.filter { currWeights.keys.contains($0.key) && !expectedMissing.contains($0.key) }
        let nested = NestedDictionary<String, MLXArray>.unflattened(filtered)
        try module.update(parameters: nested, verify: [.shapeMismatch])
        module.train(false)
    }
    
    private func stripWeightsPrefix(_ weights: [String: MLXArray], prefixes: [String]) -> [String: MLXArray] {
        for prefix in prefixes {
            let filtered = weights.filter { $0.key.hasPrefix(prefix) }
            if !filtered.isEmpty {
                var stripped: [String: MLXArray] = [:]
                stripped.reserveCapacity(filtered.count)
                for (key, value) in filtered {
                    stripped[String(key.dropFirst(prefix.count))] = value
                }
                return stripped
            }
        }
        return weights
    }
    
    private func resolveTokenizerJsonPath(baseDir: URL) -> String? {
        let candidates = [
            "grapheme_mtl_merged_expanded_v1.json",
            "tokenizer.json",
        ]
        for name in candidates {
            let path = baseDir.appendingPathComponent(name).path
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        return nil
    }
    
    /// Resolve S3Tokenizer weights - bundled in model.safetensors or from HuggingFace cache
    private func resolveS3TokenizerWeights(allWeights: [String: MLXArray], baseDir: URL) throws -> [String: MLXArray] {
        let prefixes = ["s3_tokenizer.", "s3tokenizer.", "s3Tokenizer."]
        
        // Check if bundled in model.safetensors
        let hasBundled = allWeights.keys.contains { key in
            prefixes.contains { key.hasPrefix($0) }
        }
        if hasBundled {
            return stripWeightsPrefix(allWeights, prefixes: prefixes)
        }
        
        // Check for separate file in model directory
        let candidates = [
            "s3tokenizer.safetensors",
            "s3_tokenizer.safetensors",
            "s3tokenizer/model.safetensors",
            "s3_tokenizer/model.safetensors",
        ]
        for name in candidates {
            let path = baseDir.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: path.path) {
                let data = try Data(contentsOf: path)
                return try MLX.loadArrays(data: data)
            }
        }
        
        // Fall back to HuggingFace cache (mlx-community/S3TokenizerV2)
        if let hfPath = resolveHFS3TokenizerPath() {
            let data = try Data(contentsOf: URL(fileURLWithPath: hfPath))
            return try MLX.loadArrays(data: data)
        }
        
        throw ClearVoiceError.modelNotFound("S3Tokenizer weights not found in model bundle or HuggingFace cache")
    }
    
    /// Find S3TokenizerV2 in HuggingFace cache
    private func resolveHFS3TokenizerPath() -> String? {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser
        let hubBase = home.appendingPathComponent(".cache/huggingface/hub")
        let modelDir = hubBase.appendingPathComponent("models--mlx-community--S3TokenizerV2")
        let snapshotsDir = modelDir.appendingPathComponent("snapshots")
        
        guard let entries = try? fm.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }
        
        // Sort by modification date, use most recent
        let sorted = entries.sorted { lhs, rhs in
            let lhsDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let rhsDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return lhsDate < rhsDate
        }
        
        for entry in sorted.reversed() {
            let candidate = entry.appendingPathComponent("model.safetensors").path
            if fm.fileExists(atPath: candidate) {
                return candidate
            }
        }
        
        return nil
    }
}
