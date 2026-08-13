//
//  ChatterboxTTSProvider.swift
//  AudioToolTTS
//
//  ChatterBox Multilingual TTS provider with MLX backend
//

import Foundation
import os

/// VAD trimming diagnostics for synthesised speech.
private let chatterboxLogger = Logger(subsystem: "AudioToolSwift", category: "ChatterboxTTS")
import AudioTool
import AudioToolCore
import AudioToolFluidAudio
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
    public static let baseRepo = ModelRepository.chatterboxFP32

    /// Which repository holds a given precision.
    ///
    /// ChatterBox names its quantized repos `-8bit`/`-6bit`/`-4bit` rather than the
    /// `ModelPrecision.repoSuffix` spelling, so the mapping cannot be derived and
    /// has to be written down. Written down *once*: the benchmark catalog needs the
    /// same answer to report where a case's weights come from, and a second copy
    /// would be free to disagree - which is the failure `ModelRepository` exists to
    /// prevent.
    public static func repository(for precision: ModelPrecision) -> String {
        switch precision {
        case .fp32: baseRepo
        case .fp16: "\(baseRepo)-fp16"
        case .bit8: "\(baseRepo)-8bit"
        case .bit6: "\(baseRepo)-6bit"
        case .bit4: "\(baseRepo)-4bit"
        default: precision.repo(base: baseRepo)
        }
    }

    /// Peak footprint at a given precision, for the benchmark's headroom gate.
    ///
    /// These were `550_000_000` at 4-bit and scaled by precision, which was wrong in
    /// the one direction that matters: the 4-bit checkpoint is 650 MB on its own, so
    /// the estimate sat below the weights alone. `Headroom.check` uses this to decide
    /// whether a case is safe to start, and an underestimate green-lights a run that
    /// then exhausts memory - the failure the gate exists to prevent.
    ///
    /// Measured peak footprint on an M1 Pro / 16 GB, three runs each:
    ///
    /// | precision | checkpoint | peak | overhead |
    /// | --- | ---: | ---: | ---: |
    /// | fp32 | 2580 MiB | 4017 MiB | 1437 |
    /// | fp16 | 1293 MiB | 3058 MiB | 1765 |
    /// | 8bit |  917 MiB | 2287 MiB | 1370 |
    /// | 6bit |  769 MiB | 2250 MiB | 1481 |
    /// | 4bit |  620 MiB | 2205 MiB | 1585 |
    ///
    /// Peak barely moves across the integer widths, because the weights are mmap'd
    /// lazily - `weightsMaterializedAtLoad` is false and the load delta is 48 MiB -
    /// and what dominates is run-time allocation: activations, T3's KV cache and the
    /// S3Gen flow decoder, all computed in float whatever the weights are stored as.
    /// Quantizing this model moves peak memory far less than the file sizes suggest.
    ///
    /// So: checkpoint size plus a flat 1.8 GiB, which is above every measured
    /// overhead including fp16's 1765 MiB. 1.6 GiB was tried first and put fp16's
    /// estimate 168 MiB *below* its measurement - the one direction a headroom gate
    /// must never err in.
    public static func estimatedMemoryBytes(for precision: ModelPrecision) -> Int {
        let runtimeOverhead = 1_800_000_000
        let checkpoint: Int = switch precision {
        case .fp32: 2_580_000_000
        case .fp16, .bf16: 1_290_000_000
        case .int8, .bit8: 900_000_000
        case .int6, .bit6: 750_000_000
        case .int4, .bit4: 650_000_000
        }
        return checkpoint + runtimeOverhead
    }
    
    /// Default precision
    public static let defaultPrecision: ModelPrecision = .fp32
    
    /// Current loading state for UI binding
    public private(set) var state: ModelState = .notLoaded
    
    /// Observable state stream for UI
    public nonisolated var stateStream: AsyncStream<ModelState> {
        stateBroadcaster.makeStream()
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
    private var loadGeneration: UInt64 = 0

    /// One load at a time; see ``ModelLoadGate``.
    private let loadGate = ModelLoadGate()
    private nonisolated let language: ChatterboxLanguage
    private nonisolated let repo: String
    private nonisolated let precision: ModelPrecision
    private nonisolated let modelIdentity: String
    
    /// Optional Russian text preprocessor
    private var ruAccentProvider: RUAccentProvider?
    private nonisolated let useRuAccent: Bool
    private nonisolated let convertToStressMarks: Bool
    
    /// Reference audio for voice cloning (optional - uses default voice if not set)
    private var referenceWav: MLXArray?
    private var referenceSampleRate: Int?
    private var speakerEmbedding: MLXArray?
    private var refDict: [String: MLXArray]?

    /// User-provided reference audio is durable configuration. Keep a mono CPU
    /// copy across residency eviction and rebuild all MLX conditioning tensors
    /// from it after the model is loaded again.
    private var referenceAudioConfiguration: AudioToolCore.AudioBuffer?
    
    /// Default voice conditioning from conds.safetensors (loaded during load())
    private var defaultT3Cond: T3Cond?
    private var defaultRefDict: [String: MLXArray]?
    private var hasDefaultVoice: Bool = false
    
    /// VAD trimmer for removing start/end artifacts (breathing/noise)
    private var vadProvider: FluidAudioVADProvider?
    private var vadTrimEnabled: Bool = true  // Enabled by default - ChatterBox produces start/end artifacts
    
    /// Kept outside actor isolation so subscription and cancellation registration
    /// are synchronous and cannot overtake each other.
    private nonisolated let stateBroadcaster = AsyncStateBroadcaster(
        initialState: ModelState.notLoaded
    )
    
    /// Synthesis parameters
    private var exaggeration: Float = 0.5
    private var t3Temperature: Float = 0.8
    private var t3CfgWeight: Float = 0.5
    /// Upstream `mtl_tts.generate`'s default. It was 2.0, inherited from the vendored
    /// Python port rather than from the model authors, and 2.0 is what made generation
    /// run away: with T3's stopping heuristics disabled it loops to the token limit -
    /// 40 s and 32 s of audio for two sentences that take ~3 s to say - while 1.2
    /// terminates on its own EOS every time, and does so before any of those heuristics
    /// bite. They were containing a problem this value created.
    private var t3RepetitionPenalty: Float = 1.2
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
        let repo = Self.repository(for: precision)
        self.precision = precision
        self.repo = repo
        self.modelIdentity = repo
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
        self.modelIdentity = repo
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
        self.modelIdentity = modelPath.standardizedFileURL.path
        self.language = language
        self.useRuAccent = useRuAccent
        self.convertToStressMarks = convertToStressMarks
    }
    
    // MARK: - Model Loading
    
    /// Load model, downloading if necessary
    public func load() async throws {
        try await loadGate.run { [self] in try await performLoad() }
    }

    /// The load itself.
    ///
    /// The generation counter below already stops a load that began before an
    /// `unload()` from publishing after it. What it does not stop is two concurrent
    /// `load()` calls both allocating: the second bumps the generation, so the first
    /// does the work and then throws `CancellationError` at its caller, who asked
    /// for nothing more than a loaded model. ``ModelLoadGate`` gives both callers the
    /// one load they wanted.
    private func performLoad() async throws {
        loadGeneration &+= 1
        let generation = loadGeneration

        // Check for explicit path first
        if let path = modelPath {
            try await loadFromPath(path, generation: generation)
            return
        }

        let requiredFiles = ModelFiles.chatterboxRequired(for: precision)
        let downloadFiles = ModelFiles.chatterboxDownload(for: precision)
        
        // Check if already downloaded
        if let cached = ModelDownloader.shared.localPath(
            for: repo,
            matching: requiredFiles
        ) {
            try await loadFromPath(cached, generation: generation)
            return
        }
        
        // Download from HuggingFace
        updateState(.downloading(progress: 0))
        
        do {
            let path = try await ModelDownloader.shared.downloadAndGetPath(
                repo: repo,
                matching: downloadFiles
            ) { [weak self] progress in
                Task {
                    await self?.updateDownloadProgress(
                        progress.fractionCompleted,
                        generation: generation
                    )
                }
            }

            guard ModelDownloader.hasRequiredFiles(
                at: path,
                patterns: requiredFiles
            ) else {
                throw AudioToolError.modelNotFound(
                    "Downloaded Chatterbox snapshot is missing model or tokenizer files"
                )
            }
            try await loadFromPath(path, generation: generation)
        } catch {
            if generation == loadGeneration {
                clearLoadedState()
                GPU.clearCache()
                updateState(.failed(error.localizedDescription))
            }
            throw error
        }
    }
    
    /// Load from a specific path
    private func loadFromPath(_ path: URL, generation: UInt64) async throws {
        guard generation == loadGeneration else { throw CancellationError() }
        updateState(.loading)

        do {
            // Prepare the only asynchronous dependency before publishing any model
            // components. Actor reentrancy during this download must not expose a
            // half-built Chatterbox graph to synthesis calls.
            let candidateVAD: FluidAudioVADProvider?
            if vadTrimEnabled {
                let vad = FluidAudioVADProvider(
                    threshold: 0.9,
                    minSpeechDuration: 0.1,
                    minSilenceDuration: 0.1
                )
                try await vad.load()
                candidateVAD = vad
            } else {
                candidateVAD = nil
            }
            guard generation == loadGeneration else { throw CancellationError() }

            // Never carry conditioning tensors from an older model instance into a
            // reload. The CPU configuration below remains available for rebuilding.
            referenceWav = nil
            referenceSampleRate = nil
            speakerEmbedding = nil
            refDict = nil
            defaultT3Cond = nil
            defaultRefDict = nil
            hasDefaultVoice = false

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
                    throw AudioToolError.modelNotFound("No .safetensors file found in \(path.lastPathComponent)")
                }
            }
            
            // Load all weights
            let allWeights = try MLX.loadArrays(url: weightsPath)
            
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
                throw AudioToolError.modelNotFound("Missing tokenizer JSON (grapheme_mtl_merged_expanded_v1.json)")
            }
            
            // Load default voice conditioning from conds.safetensors (optional)
            let condsPath = baseDir.appendingPathComponent("conds.safetensors")
            if FileManager.default.fileExists(atPath: condsPath.path) {
                try loadDefaultConds(from: condsPath)
            }

            if let referenceAudioConfiguration {
                try rebuildReferenceConditioning(from: referenceAudioConfiguration)
            }
            
            vadProvider = candidateVAD
            modelPath = path
            guard generation == loadGeneration else { throw CancellationError() }
            updateState(.ready)
        } catch {
            if generation == loadGeneration {
                clearLoadedState()
                GPU.clearCache()
                updateState(.failed(error.localizedDescription))
            }
            throw error
        }
    }
    
    // MARK: - State Updates
    
    private func updateState(_ newState: ModelState) {
        state = newState
        stateBroadcaster.send(newState)
    }

    private func updateDownloadProgress(_ progress: Double, generation: UInt64) {
        guard progress.isFinite else { return }
        let clampedProgress = min(1, max(0, progress))
        guard ModelLoadStateGate.acceptsProgress(
            clampedProgress,
            generation: generation,
            currentGeneration: loadGeneration,
            state: state
        ) else { return }
        updateState(.downloading(progress: clampedProgress))
    }

    private func clearLoadedState() {
        voiceEncoder = nil
        t3 = nil
        s3Tokenizer = nil
        s3Gen = nil
        textTokenizer = nil
        referenceWav = nil
        referenceSampleRate = nil
        speakerEmbedding = nil
        refDict = nil
        defaultT3Cond = nil
        defaultRefDict = nil
        hasDefaultVoice = false
        vadProvider = nil
    }
    
    // MARK: - Reference Audio
    
    /// Set reference audio for voice cloning from URL
    /// - Parameter url: Path to reference audio file (WAV recommended)
    public func setReferenceAudio(from url: URL) async throws {
        guard voiceEncoder != nil, s3Gen != nil, s3Tokenizer != nil else {
            throw AudioToolError.modelNotLoaded("ChatterBox")
        }
        
        let info = try AudioLoader().getAudioInfo(from: url.path)
        let config = AudioLoader.Configuration(
            targetSampleRate: info.sampleRate,
            normalizationMode: .none,
            resamplingMethod: .none
        )
        let loader = AudioLoader(config: config)
        let audio = try loader.load(from: url.path)

        let reference = AudioToolCore.AudioBuffer(
            samples: audio.asArray(Float.self),
            sampleRate: Int(info.sampleRate),
            channels: 1
        )
        try configureReferenceAudio(reference)
    }
    
    /// Set reference audio from an AudioBuffer
    /// - Parameter audio: Audio buffer with reference speech
    public func setReferenceAudio(_ audio: AudioToolCore.AudioBuffer) async throws {
        guard voiceEncoder != nil, s3Gen != nil, s3Tokenizer != nil else {
            throw AudioToolError.modelNotLoaded("ChatterBox")
        }

        try configureReferenceAudio(audio)
    }

    private func configureReferenceAudio(
        _ audio: AudioToolCore.AudioBuffer
    ) throws {
        let mono = try audio.converted(toChannels: 1)
        guard !mono.isEmpty else { throw AudioToolError.emptyAudioBuffer }
        guard !mono.samples.contains(where: { !$0.isFinite }) else {
            throw AudioToolError.invalidAudioFormat(
                expected: "finite reference samples",
                found: "NaN or infinity"
            )
        }

        // Build first and commit the durable configuration only after all
        // conditioning is available, so a bad replacement does not erase a
        // previously working voice.
        try rebuildReferenceConditioning(from: mono)
        referenceAudioConfiguration = mono
    }

    private func rebuildReferenceConditioning(
        from audio: AudioToolCore.AudioBuffer
    ) throws {
        guard let voiceEncoder, let s3Gen, let s3Tokenizer else {
            throw AudioToolError.modelNotLoaded("ChatterBox")
        }

        let wav = MLXArray(audio.samples)
        let sourceSampleRate = audio.sampleRate

        // Band-limited sinc, not `resampleAudioPolyphase`, for every conditioning
        // resample here. scipy's default filter puts its cutoff at Nyquist and folds
        // content from above 8 kHz back into the band - measured -20.9 dB alias
        // rejection against -55.8 dB for this one - and the S3 tokenizer resolves the
        // difference into different speech tokens. `S3Gen.embed_ref` is the one place
        // that must keep the scipy filter, because the reference uses it there too.
        //
        // VoiceEncoder expects 16 kHz. Resample explicitly so both file and
        // AudioBuffer configuration follow the same conditioning path.
        let wav16k = resampleAudioKaiserBest(
            wav,
            origSR: sourceSampleRate,
            targetSR: S3_SR
        )
        let newSpeakerEmbedding = voiceEncoder.embedsFromWavs(
            [wav16k],
            sampleRate: S3_SR
        )

        let wav24k = resampleAudioKaiserBest(
            wav,
            origSR: sourceSampleRate,
            targetSR: S3GEN_SR
        )
        let decoderConditioningLength = 10 * S3GEN_SR
        let wav24kTrimmed = wav24k[0..<min(decoderConditioningLength, wav24k.shape[0])]
        let wav16kFrom24k = resampleAudioKaiserBest(
            wav24kTrimmed,
            origSR: S3GEN_SR,
            targetSR: S3_SR
        )
        let mel = logMelSpectrogramCompat(wav16kFrom24k, nMels: 128)
        let melBatch = mel.expandedDimensions(axis: 0)
        let melLength = MLXArray([Int32(melBatch.shape[2])])
        let (tokens, tokenLengths) = s3Tokenizer(melBatch, melLength)
        let newRefDict = s3Gen.embed_ref(
            ref_wav: wav24kTrimmed.expandedDimensions(axis: 0),
            ref_sr: S3GEN_SR,
            ref_speech_tokens: tokens,
            ref_speech_token_lens: tokenLengths
        )

        eval([newSpeakerEmbedding] + Array(newRefDict.values))
        referenceWav = wav
        referenceSampleRate = sourceSampleRate
        speakerEmbedding = newSpeakerEmbedding
        refDict = newRefDict
    }

    /// T3's encoder-conditioning speech tokens, from the reference audio.
    ///
    /// Its own method because it has two callers with nothing else in common:
    /// `generate` needs it to build `T3Cond`, and the parity suite needs it without
    /// generating, since everything past this point samples. It is not folded into
    /// `rebuildReferenceConditioning` because it does not have to be - it depends on
    /// nothing that `setReferenceAudio` computes, only on the stored reference audio,
    /// and computing it eagerly would charge every caller for a tokenizer pass they
    /// may never use.
    ///
    /// The 6 s window is T3's encoder-conditioning length, separate from the 10 s
    /// decoder window `rebuildReferenceConditioning` applies; on a reference shorter
    /// than 6 s neither bites and the two token sets coincide.
    private func t3PromptSpeechTokens() throws -> MLXArray {
        guard let t3, let s3Tokenizer else {
            throw AudioToolError.modelNotLoaded("ChatterBox")
        }
        guard let refWav = referenceWav, let refSr = referenceSampleRate else {
            throw AudioToolError.resourceUnavailable("Reference audio not set")
        }

        let encCondLen = 6 * S3_SR
        // Same band-limited sinc as `rebuildReferenceConditioning`; these tokens go
        // through the same tokenizer and are sensitive the same way.
        let refWav16kFull = resampleAudioKaiserBest(refWav, origSR: refSr, targetSR: S3_SR)
        let refWav16k = refWav16kFull[0..<min(encCondLen, refWav16kFull.shape[0])]

        var t3Mel = logMelSpectrogramCompat(refWav16k, nMels: 128)
        // Trim the mel before tokenizing, not the tokens after: 4 mel frames per
        // token, and tokenizing the untrimmed mel yields plen+1.
        let maxMelFrames = t3.hp.speech_cond_prompt_len * 4
        if t3Mel.dim(1) > maxMelFrames {
            t3Mel = t3Mel[0..., 0..<maxMelFrames]
        }

        let t3MelBatch = t3Mel.expandedDimensions(axis: 0)
        let t3MelLen = MLXArray([Int32(t3MelBatch.shape[2])])
        let (t3Tokens, _) = s3Tokenizer(t3MelBatch, t3MelLen)
        let plen = t3.hp.speech_cond_prompt_len
        return t3Tokens[0..., 0..<min(plen, t3Tokens.shape[1])]
    }

    /// A materialized conditioning tensor. `MLXArray` is not `Sendable`, so the
    /// values are read out here, inside the actor, rather than handed across.
    internal struct ConditioningSnapshot: Sendable {
        let shape: [Int]
        let values: [Float]
    }

    /// The conditioning this provider is holding, keyed as `conds.safetensors` keys it.
    ///
    /// Exists for the parity suite. Conditioning is the deterministic half of
    /// Chatterbox - everything past it samples - so it is the part that can be held
    /// to a reference at all, and it is private state otherwise. Reading it through
    /// the provider rather than rebuilding the sequence in the test is deliberate:
    /// a test that re-derives `prepareConditioning` measures the test's copy of it.
    internal func conditioningTensors() -> [String: ConditioningSnapshot] {
        var arrays: [String: MLXArray] = [:]
        if let speakerEmbedding {
            arrays["t3.speaker_emb"] = speakerEmbedding
        }
        // Derived on demand rather than stored: `generate` computes it the same way
        // from the same input, so this is the shipped path and not a parallel one.
        if referenceWav != nil, let tokens = try? t3PromptSpeechTokens() {
            arrays["t3.cond_prompt_speech_tokens"] = tokens
        }
        for (key, value) in refDict ?? [:] {
            arrays["gen.\(key)"] = value
        }
        guard !arrays.isEmpty else { return [:] }

        eval(Array(arrays.values))
        return arrays.mapValues {
            ConditioningSnapshot(
                shape: $0.shape,
                values: $0.asType(.float32).flattened().asArray(Float.self)
            )
        }
    }

    // MARK: - Default Voice Loading
    
    /// Load default voice conditioning from conds.safetensors
    /// - Parameter path: Path to conds.safetensors file
    private func loadDefaultConds(from path: URL) throws {
        let tensors = try MLX.loadArrays(url: path)
        
        // Load T3 conditioning
        guard let speakerEmb = tensors["t3.speaker_emb"] else {
            throw AudioToolError.resourceUnavailable("Missing t3.speaker_emb in conds.safetensors")
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
            throw AudioToolError.resourceUnavailable("Missing gen.* keys in conds.safetensors")
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
    ///   - temperature: Sampling temperature for speech tokens, greater than 0. Higher = more variation. Default: 0.8
    ///   - cfgWeight: Classifier-free guidance weight, 0 or greater. Lower = slower/more deliberate. Default: 0.5
    ///   - repetitionPenalty: Penalty for repeating tokens, greater than 0. Default: 1.2
    ///     (upstream's value). Raising it towards 2.0 makes the model loop instead of
    ///     emitting EOS; see ``t3RepetitionPenalty``.
    ///   - minP: Minimum probability threshold for sampling, 0...1. Default: 0.05
    ///   - topP: Nucleus sampling threshold, 0...1. Default: 1.0
    ///   - seed: Random seed for reproducible results. Default: 0, which draws a
    ///     fresh seed for each synthesis.
    /// - Throws: ``AudioToolError/pipelineConfigurationInvalid(_:)`` for a value
    ///   outside its range. `temperature` and `repetitionPenalty` are *divisors*
    ///   inside T3's sampling loop, so zero or a non-finite value did not produce
    ///   unusual speech - it produced infinities, then NaN logits, then whatever a
    ///   sampler does with a NaN distribution.
    /// The seed a synthesis should run under.
    ///
    /// `seed == 0` is documented as "random" and now behaves that way: a fresh draw
    /// per call. Any other value is used as given, which is what makes a configured
    /// seed reproducible.
    private func resolvedSeed() -> UInt64 {
        seed == 0 ? UInt64.random(in: 1...UInt64.max) : seed
    }

    public func configure(
        exaggeration: Float = 0.5,
        temperature: Float = 0.8,
        cfgWeight: Float = 0.5,
        repetitionPenalty: Float = 1.2,
        minP: Float = 0.05,
        topP: Float = 1.0,
        seed: UInt64 = 0
    ) throws {
        func validate(_ value: Float, _ name: String, _ isValid: Bool, _ expected: String) throws {
            guard value.isFinite, isValid else {
                throw AudioToolError.pipelineConfigurationInvalid(
                    "Chatterbox \(name) must be \(expected); got \(value)."
                )
            }
        }

        try validate(exaggeration, "exaggeration", exaggeration >= 0, "0 or greater")
        try validate(temperature, "temperature", temperature > 0, "greater than 0")
        try validate(cfgWeight, "cfgWeight", cfgWeight >= 0, "0 or greater")
        try validate(
            repetitionPenalty, "repetitionPenalty", repetitionPenalty > 0, "greater than 0"
        )
        try validate(minP, "minP", (0...1).contains(minP), "between 0 and 1")
        try validate(topP, "topP", (0...1).contains(topP), "between 0 and 1")

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
    public func synthesize(_ text: String, voice: String) async throws -> AudioToolCore.AudioBuffer {
        try Task.checkCancellation()
        guard let t3 = t3, let s3Gen = s3Gen, let textTokenizer = textTokenizer else {
            throw AudioToolError.modelNotLoaded("ChatterBox")
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
            throw AudioToolError.resourceUnavailable("No reference audio set and no default voice available. Either call setReferenceAudio() or use a model repo with conds.safetensors.")
        }
        
        // Preprocess text
        let processedText = await preprocessText(text)
        
        // One seeding per synthesis, resolved from `seed`.
        //
        // `MLXRandom.seed` is process-global state, not per-actor: two providers
        // synthesizing concurrently reseed each other's sampler mid-generation, and
        // nothing here can prevent that short of threading an explicit key through
        // every sampling call in the vendored T3 implementation. What is fixed is
        // the documented contract - seed 0 meant "random" and was in fact the fixed
        // seed zero - and the double reseed: this used to be applied here and again
        // immediately before T3 inference, so the first had no effect at all.
        let synthesisSeed = resolvedSeed()
        MLXRandom.seed(synthesisSeed)
        
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
            let veMean = MLX.mean(useSpeakerEmbedding, axis: 0, keepDims: true)

            t3Cond = T3Cond(
                speaker_emb: veMean,
                cond_prompt_speech_tokens: try t3PromptSpeechTokens(),
                emotion_adv: MLX.ones([1, 1, 1]) * exaggeration
            )
        }
        
        // Run T3 inference
        let rawSpeechTokens = try t3.inference(
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
        try Task.checkCancellation()
        
        // Filter speech tokens
        let dropped = dropInvalidTokens(rawSpeechTokens)
        let speechTokens = filterValidTokens(dropped)
        
        // Generate audio
        let fullWav = s3Gen(speechTokens: speechTokens.asType(.int32), refDict: useRefDict, finalize: true)
        try Task.checkCancellation()
        let wavMono = fullWav.ndim == 1 ? fullWav : fullWav[0]

        // Convert to samples
        var samples = wavMono.asArray(Float.self)

        // Drop the final speech token's audio.
        //
        // Upstream `mtl_tts.generate` truncates the waveform to `n_tokens - 1` tokens'
        // worth of samples: the last token is emitted just before EOS with degraded
        // attention and decodes to ~40 ms of noise. This is the multilingual model's own
        // fix for the end-of-utterance artifact - `tts.py` has no equivalent - and it is
        // deterministic, unlike the VAD trim below, which infers the boundary from the
        // very noise this removes.
        let speechTokenCount = speechTokens.dim(speechTokens.ndim - 1)
        if speechTokenCount > 1 {
            let keep = (speechTokenCount - 1) * (sampleRate / S3_TOKEN_RATE)
            if keep < samples.count {
                samples.removeLast(samples.count - keep)
            }
        }

        var audioBuffer = AudioToolCore.AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
        
        // Apply VAD trimming if enabled (removes start/end artifacts like breathing/noise)
        if vadTrimEnabled, let vad = vadProvider {
            audioBuffer = try await trimAudioWithVAD(audioBuffer, using: vad)
        }
        
        return audioBuffer
    }
    
    /// Trim audio using VAD to remove non-speech segments at start and end
    private func trimAudioWithVAD(_ audio: AudioToolCore.AudioBuffer, using vad: FluidAudioVADProvider) async throws -> AudioToolCore.AudioBuffer {
        // Resample to 16kHz for VAD using simple linear interpolation
        // (polyphase resampling preserves high frequencies which causes VAD to over-detect speech)
        let vadSampleRate = 16000
        let resampledAudio: AudioToolCore.AudioBuffer
        
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
            resampledAudio = AudioToolCore.AudioBuffer(samples: resampled, sampleRate: vadSampleRate, channels: 1)
        } else {
            resampledAudio = audio
        }
        
        // Detect speech segments
        let segments = try await vad.detect(resampledAudio)
        
        #if DEBUG
        chatterboxLogger.debug("Detected \(segments.count) speech segments")
        #endif
        
        // If no speech detected, return original (avoid trimming everything)
        guard !segments.isEmpty else {
            #if DEBUG
            chatterboxLogger.debug("No speech detected, returning original audio")
            #endif
            return audio
        }
        
        // Find first and last speech boundaries
        let firstSpeechStart = segments.first!.timeRange.start
        let lastSpeechEnd = segments.last!.timeRange.end
        let audioDuration = audio.duration
        
        #if DEBUG
        chatterboxLogger.debug("Trimming: \(String(format: "%.3f", audioDuration))s -> \(String(format: "%.3f", lastSpeechEnd + 0.01))s")
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
        chatterboxLogger.debug("Trimming: \(trimmedFromStart) samples from start, \(trimmedFromEnd) samples from end")
        #endif
        
        // Only trim if there's something to trim
        if paddedStart == 0 && paddedEnd == audio.samples.count {
            #if DEBUG
            chatterboxLogger.debug("Nothing to trim, speech spans entire audio")
            #endif
            return audio
        }
        
        // Extract trimmed audio
        let trimmedSamples = Array(audio.samples[paddedStart..<paddedEnd])
        
        #if DEBUG
        let trimmedDuration = Double(trimmedSamples.count) / Double(audio.sampleRate)
        chatterboxLogger.debug("Trimmed: \(String(format: "%.3f", audioDuration - trimmedDuration))s removed")
        #endif
        
        return AudioToolCore.AudioBuffer(samples: trimmedSamples, sampleRate: audio.sampleRate, channels: 1)
    }
    
    
    /// Stream synthesized audio chunks
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice identifier (ignored)
    /// - Returns: Async stream of audio chunks
    public nonisolated func streamSynthesis(_ text: String, voice: String) -> AsyncThrowingStream<AudioToolCore.AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    // For now, generate full audio and yield as single chunk
                    // TODO: Implement sentence-level streaming for lower latency
                    let audio = try await synthesize(text, voice: voice)
                    try Task.checkCancellation()
                    if case .terminated = continuation.yield(audio) {
                        throw CancellationError()
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

        // Upstream applies `punc_norm` immediately before tokenization. RUAccent runs
        // first so it sees natural text; `puncNorm` only rewrites punctuation and
        // whitespace, so it leaves the stress marks RUAccent inserts alone.
        return puncNorm(processedText)
    }
    
    // MARK: - AudioProcessor Conformance
    
    public var inputChannels: Int { 0 }  // Text input
    public var outputChannels: Int { 1 }
    
    public func process(_ input: AudioToolCore.AudioBuffer) async throws -> AudioToolCore.AudioBuffer {
        throw AudioToolError.pipelineConfigurationInvalid("ChatterBox is a synthesizer, use synthesize() instead of process()")
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
                return try MLX.loadArrays(url: path)
            }
        }
        
        // Fall back to HuggingFace cache (mlx-community/S3TokenizerV2)
        if let hfPath = resolveHFS3TokenizerPath() {
            return try MLX.loadArrays(url: URL(fileURLWithPath: hfPath))
        }
        
        throw AudioToolError.modelNotFound("S3Tokenizer weights not found in model bundle or HuggingFace cache")
    }
    
    /// Find the pinned S3TokenizerV2 snapshot in the model cache.
    ///
    /// Through ``ModelDownloader``, so the pin applies. This used to walk
    /// `~/.cache/huggingface/hub` itself and take the snapshot with the newest
    /// modification date: whichever revision happened to be fetched last won,
    /// including one fetched from `main` by some other tool, and the registry entry
    /// for this repository governed nothing.
    private func resolveHFS3TokenizerPath() -> String? {
        ModelDownloader.shared.localPath(
            for: ModelRepository.s3Tokenizer,
            matching: ["model.safetensors"]
        )?.appendingPathComponent("model.safetensors").path
    }
}

// MARK: - ManagedModel

extension ChatterboxTTSProvider: ManagedModel {
    public nonisolated var modelId: String {
        "chatterbox:\(modelIdentity):\(precision.rawValue):\(language.rawValue)"
    }

    public nonisolated var estimatedMemoryBytes: Int {
        Self.estimatedMemoryBytes(for: precision)
    }

    public func checkIfLoaded() async -> Bool {
        voiceEncoder != nil && t3 != nil && s3Tokenizer != nil && s3Gen != nil && textTokenizer != nil
    }

    public func unload() async {
        // Bracketed: the gate stays shut until this method's state reset is
        // done, so a concurrent load cannot publish a model into the gap and
        // have it wiped by the lines below.
        let teardown = await loadGate.beginTeardown()
        defer { loadGate.endTeardown(teardown) }
        loadGeneration &+= 1
        clearLoadedState()
        // `referenceAudioConfiguration` is CPU-backed user configuration and is
        // intentionally retained for rebuildReferenceConditioning(from:).
        updateState(.notLoaded)
        GPU.clearCache()
    }
}
