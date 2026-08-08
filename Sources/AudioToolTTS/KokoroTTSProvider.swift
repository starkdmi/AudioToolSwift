//
//  KokoroTTSProvider.swift
//  AudioToolTTS
//
//  Kokoro TTS provider with MisakiSwift G2P (MIT license)
//

import Foundation
import AudioTool
import AudioToolCore
import KokoroSwift
import MLX
import MLXNN
import CoreMedia

/// Typealias to avoid conflict with CoreAudio.AudioBuffer
public typealias TTSAudioBuffer = AudioToolCore.AudioBuffer

/// Kokoro TTS provider using MisakiSwift for G2P (MIT license, no ESpeakNG)
///
/// ## Performance
/// ~5.5-11.7x RTF on Apple Silicon
///
/// ## Supported Languages
/// All languages use misaki[en] except Japanese (misaki[ja]) and Chinese (misaki[zh]):
///
/// | Language | Code | Voices | Example |
/// |----------|------|--------|---------|
/// | 🇺🇸 American English | `a` | 20 (11F, 9M) | `af_heart`, `am_adam` |
/// | 🇬🇧 British English | `b` | 8 (4F, 4M) | `bf_emma`, `bm_george` |
/// | 🇯🇵 Japanese | `j` | 5 (4F, 1M) | `jf_alpha`, `jm_kumo` |
/// | 🇨🇳 Mandarin Chinese | `z` | 8 (4F, 4M) | `zf_xiaoxiao`, `zm_yunxi` |
/// | 🇪🇸 Spanish | `e` | 3 (1F, 2M) | `ef_dora`, `em_alex` |
/// | 🇫🇷 French | `f` | 1 (1F) | `ff_siwis` |
/// | 🇮🇳 Hindi | `h` | 4 (2F, 2M) | `hf_alpha`, `hm_omega` |
/// | 🇮🇹 Italian | `i` | 2 (1F, 1M) | `if_sara`, `im_nicola` |
/// | 🇧🇷 Brazilian Portuguese | `p` | 3 (1F, 2M) | `pf_dora`, `pm_alex` |
///
/// ## Voice Naming Convention
/// - First letter: Language code
/// - Second letter: Gender (f=female, m=male)
/// - Rest: Voice name
///
/// ## Usage
/// ```swift
/// // Basic usage with default voice
/// let tts = TTSProviders.kokoro()
/// try await tts.load()
/// let audio = try await tts.synthesize("Hello world!", voice: "af_heart")
///
/// // Using KokoroVoice enum
/// let voice = KokoroVoice.af_bella
/// let audio = try await tts.synthesize("Hello!", voice: voice.rawValue)
///
/// // Get available voices for a language
/// let italianVoices = KokoroLanguage.italian.availableVoices  // [.if_sara, .im_nicola]
/// let defaultVoice = KokoroLanguage.french.defaultVoice  // .ff_siwis
///
/// // Multilingual synthesis
/// let tts = KokoroTTSProvider(language: .italian)
/// try await tts.load()
/// let audio = try await tts.synthesize("Ciao mondo!", voice: "if_sara")
/// ```
///
/// ## Quality Grades
/// Voices are graded A-D based on training data quality and quantity.
/// Best quality voices: `af_heart` (A), `af_bella` (A-), `bf_emma` (B-)
///
/// For full voice quality info: https://huggingface.co/mlx-community/Kokoro-82M-bf16/blob/main/VOICES.md
public actor KokoroTTSProvider: SpeechSynthesizer {
    
    // MARK: - Public Properties
    
    /// Base HuggingFace repository (without precision suffix)
    public static let baseRepo = ModelRepository.kokoroBase
    
    /// Default precision
    public static let defaultPrecision: ModelPrecision = .bf16
    
    /// Current loading state for UI binding
    public private(set) var state: ModelState = .notLoaded
    
    /// Observable state stream for UI
    public nonisolated var stateStream: AsyncStream<ModelState> {
        stateBroadcaster.makeStream()
    }
    
    // MARK: - SpeechSynthesizer Conformance
    
    public nonisolated let sampleRate: Int = 24000
    
    /// Available voices for this language (uses default list since embeddings are loaded lazily)
    public nonisolated var availableVoices: [String] {
        language.availableVoices.map { $0.rawValue }
    }
    
    // MARK: - Private Properties
    
    private var tts: KokoroTTS?
    private var modelPath: URL?
    private var loadGeneration: UInt64 = 0
    private nonisolated let language: KokoroLanguage
    private nonisolated let repo: String
    private nonisolated let precision: ModelPrecision
    private nonisolated let modelIdentity: String
    
    /// Voice embeddings cache (voice name -> MLXArray)
    private var voiceEmbeddings: [String: MLXArray] = [:]

    /// External voice files are configuration, not resident model state. Keep
    /// their paths across eviction so their MLX embeddings can be reconstructed
    /// after the model is loaded again.
    private var customVoiceSources: [String: URL] = [:]
    
    /// Kept outside actor isolation so subscription and cancellation registration
    /// are synchronous and cannot overtake each other.
    private nonisolated let stateBroadcaster = AsyncStateBroadcaster(
        initialState: ModelState.notLoaded
    )
    
    // MARK: - Initialization
    
    /// Initialize Kokoro TTS provider with precision-based repo selection
    /// - Parameters:
    ///   - precision: Model precision (determines repo: Kokoro-82M-bf16, Kokoro-82M-4bit, etc.)
    ///   - language: Target language for synthesis
    public init(
        precision: ModelPrecision = KokoroTTSProvider.defaultPrecision,
        language: KokoroLanguage = .americanEnglish
    ) {
        let repo = precision.repo(base: KokoroTTSProvider.baseRepo)
        self.precision = precision
        self.repo = repo
        self.modelIdentity = repo
        self.language = language
    }
    
    /// Initialize with explicit HuggingFace repo (for custom repos)
    /// - Parameters:
    ///   - repo: Full HuggingFace repository ID
    ///   - language: Target language for synthesis
    public init(
        repo: String,
        language: KokoroLanguage = .americanEnglish
    ) {
        self.precision = .bf16
        self.repo = repo
        self.modelIdentity = repo
        self.language = language
    }
    
    /// Initialize with explicit local path (skips download)
    /// - Parameters:
    ///   - modelPath: Path to Kokoro model weights directory
    ///   - language: Target language for synthesis
    public init(modelPath: URL, language: KokoroLanguage = .americanEnglish) {
        self.modelPath = modelPath
        self.precision = .bf16
        self.repo = KokoroTTSProvider.baseRepo
        self.modelIdentity = modelPath.standardizedFileURL.path
        self.language = language
    }
    
    // MARK: - Model Loading
    
    /// Load model, downloading if necessary
    public func load() async throws {
        loadGeneration &+= 1
        let generation = loadGeneration

        // Check for explicit path first
        if let path = modelPath {
            try await loadFromPath(path, generation: generation)
            return
        }

        let requiredFiles = ModelFiles.kokoro
        
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
                matching: requiredFiles
            ) { [weak self] progress in
                Task {
                    await self?.updateDownloadProgress(
                        progress.fractionCompleted,
                        generation: generation
                    )
                }
            }
            
            try await loadFromPath(path, generation: generation)
        } catch {
            if generation == loadGeneration {
                tts = nil
                voiceEmbeddings.removeAll()
                GPU.clearCache()
                updateState(.failed(error.localizedDescription))
            }
            throw error
        }
    }
    
    /// Load from a specific path (can be file or directory)
    private func loadFromPath(_ path: URL, generation: UInt64) async throws {
        guard generation == loadGeneration else { throw CancellationError() }
        updateState(.loading)
        
        do {
            // Find the safetensors model file
            // KokoroTTS expects the actual .safetensors FILE path, not directory
            var weightsPath = path
            
            // If path is a directory, find the safetensors file
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: path.path, isDirectory: &isDir), isDir.boolValue {
                // Look for kokoro*.safetensors or *.safetensors
                let files = try FileManager.default.contentsOfDirectory(at: path, includingPropertiesForKeys: nil)
                if let kokoroFile = files.first(where: { $0.lastPathComponent.hasPrefix("kokoro") && $0.pathExtension == "safetensors" }) {
                    weightsPath = kokoroFile
                } else if let anyFile = files.first(where: { $0.pathExtension == "safetensors" }) {
                    weightsPath = anyFile
                } else {
                    throw AudioToolError.modelNotFound("No .safetensors file found in \(path.lastPathComponent)")
                }
            }
            
            // KokoroSwift uses .misaki G2P by default (MIT license, no ESpeakNG)
            let candidate = try KokoroTTS(modelPath: weightsPath, g2p: .misaki)
            guard generation == loadGeneration else { throw CancellationError() }
            tts = candidate
            modelPath = path
            voiceEmbeddings.removeAll(keepingCapacity: true)
            
            // Try to auto-load voices from voices subdirectory (use original path, not weightsPath)
            let voicesDir = path.appendingPathComponent("voices")
            if FileManager.default.fileExists(atPath: voicesDir.path) {
                try? loadVoices(from: voicesDir, retainForReload: false)
            }

            // Custom voices override bundled voices with the same name. A missing
            // or invalid configured file makes reload fail explicitly instead of
            // leaving a provider that reports ready but cannot synthesize that voice.
            for (_, source) in customVoiceSources.sorted(by: { $0.key < $1.key }) {
                _ = try loadVoice(from: source, retainForReload: false)
            }
            
            guard generation == loadGeneration else { throw CancellationError() }
            updateState(.ready)
        } catch {
            if generation == loadGeneration {
                tts = nil
                voiceEmbeddings.removeAll()
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
    
    // MARK: - Voice Loading
    
    /// Load a voice embedding from .npy or .safetensors file
    /// - Parameter voicePath: Path to voice style file
    /// - Returns: Voice name (filename without extension)
    @discardableResult
    public func loadVoice(from voicePath: URL) throws -> String {
        try loadVoice(from: voicePath, retainForReload: true)
    }

    @discardableResult
    private func loadVoice(
        from voicePath: URL,
        retainForReload: Bool
    ) throws -> String {
        let voiceName = voicePath.deletingPathExtension().lastPathComponent
        
        // Load voice embedding based on file extension
        let embedding: MLXArray
        if voicePath.pathExtension == "safetensors" {
            let arrays = try MLX.loadArrays(url: voicePath)
            guard let voiceArray = arrays["voice"] ?? arrays.values.first else {
                throw AudioToolError.resourceUnavailable("No voice tensor found in \(voicePath.lastPathComponent)")
            }
            embedding = voiceArray
        } else {
            // .npy format - load directly
            embedding = try MLX.loadArray(url: voicePath)
        }
        
        voiceEmbeddings[voiceName] = embedding
        if retainForReload {
            customVoiceSources[voiceName] = voicePath.standardizedFileURL
        }
        return voiceName
    }
    
    /// Load all voice embeddings from a directory
    /// - Parameter directory: Directory containing .npy or .safetensors voice files
    public func loadVoices(from directory: URL) throws {
        try loadVoices(from: directory, retainForReload: true)
    }

    private func loadVoices(
        from directory: URL,
        retainForReload: Bool
    ) throws {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "npy" || file.pathExtension == "safetensors" {
            _ = try loadVoice(from: file, retainForReload: retainForReload)
        }
    }
    
    // MARK: - Voice Mixing
    
    /// Mix multiple voice embeddings with weighted interpolation
    ///
    /// Creates a blended voice by interpolating between multiple voice embeddings.
    /// This allows creating custom voices with characteristics from multiple speakers.
    ///
    /// Example:
    /// ```swift
    /// // 70% Bella, 30% Sarah
    /// let mixed = try tts.mixVoices([("af_bella", 0.7), ("af_sarah", 0.3)])
    /// let audio = try await tts.synthesize("Hello!", voiceEmbedding: mixed)
    /// ```
    ///
    /// - Parameter voices: Array of (voiceName, weight) tuples. Weights are normalized automatically.
    /// - Returns: Blended MLXArray voice embedding
    /// - Throws: `AudioToolError.resourceUnavailable` if any voice is not loaded
    public func mixVoices(_ voices: [(name: String, weight: Float)]) throws -> MLXArray {
        guard !voices.isEmpty else {
            throw AudioToolError.resourceUnavailable("No voices provided for mixing")
        }
        
        // Normalize weights to sum to 1.0
        let totalWeight = voices.map(\.weight).reduce(0, +)
        guard totalWeight > 0 else {
            throw AudioToolError.resourceUnavailable("Total weight must be greater than 0")
        }
        
        var mixed: MLXArray? = nil
        for (name, weight) in voices {
            guard let embedding = voiceEmbeddings[name] else {
                throw AudioToolError.resourceUnavailable("Voice '\(name)' not loaded. Call loadVoice(from:) first.")
            }
            let normalizedWeight = weight / totalWeight
            let weighted = embedding * MLXArray(normalizedWeight)
            mixed = mixed.map { $0 + weighted } ?? weighted
        }
        
        return mixed!
    }
    
    /// Synthesize text using a raw voice embedding (for mixed voices)
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voiceEmbedding: Raw MLXArray voice embedding (from mixVoices or custom)
    /// - Returns: Audio buffer with synthesized speech (24kHz mono)
    public func synthesize(_ text: String, voiceEmbedding: MLXArray) async throws -> AudioToolCore.AudioBuffer {
        guard let tts = tts else {
            throw AudioToolError.modelNotLoaded("KokoroTTS")
        }
        
        // Map KokoroLanguage to KokoroSwift.Language
        let kokoroLang: KokoroSwift.Language = switch language {
        case .americanEnglish: .enUS
        case .britishEnglish: .enGB
        case .japanese: .japanese
        case .chinese: .chinese
        case .spanish: .spanish
        case .french: .french
        case .hindi: .hindi
        case .italian: .italian
        case .portuguese: .portuguese
        }
        
        // Generate audio using the raw embedding
        let (samples, _) = try tts.generateAudio(
            voice: voiceEmbedding,
            language: kokoroLang,
            text: text
        )
        
        return AudioToolCore.AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
    }
    
    // MARK: - SpeechSynthesizer Protocol
    
    /// Synthesize text to speech
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice name (must be loaded via `loadVoice`)
    /// - Returns: Audio buffer with synthesized speech (24kHz mono)
    public func synthesize(_ text: String, voice: String) async throws -> AudioToolCore.AudioBuffer {
        guard let tts = tts else {
            throw AudioToolError.modelNotLoaded("KokoroTTS")
        }
        
        guard let voiceEmbedding = voiceEmbeddings[voice] else {
            throw AudioToolError.resourceUnavailable("Voice '\(voice)' not loaded. Call loadVoice(from:) first.")
        }
        
        // Map KokoroLanguage to KokoroSwift.Language
        // All use misaki[en] except Japanese (misaki[ja]) and Chinese (misaki[zh])
        let kokoroLang: KokoroSwift.Language = switch language {
        case .americanEnglish: .enUS
        case .britishEnglish: .enGB
        case .japanese: .japanese
        case .chinese: .chinese
        case .spanish: .spanish
        case .french: .french
        case .hindi: .hindi
        case .italian: .italian
        case .portuguese: .portuguese
        }
        
        // Generate audio using Kokoro with MisakiSwift G2P
        // Returns ([Float], [MToken]?) tuple
        let (samples, _) = try tts.generateAudio(
            voice: voiceEmbedding,
            language: kokoroLang,
            text: text
        )
        
        return AudioToolCore.AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
    }
    
    /// Stream synthesized audio chunks
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice name
    /// - Returns: Async stream of audio chunks
    public nonisolated func streamSynthesis(_ text: String, voice: String) -> AsyncThrowingStream<AudioToolCore.AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    // TODO: Implement sentence-level streaming for lower latency
                    // For now, generate full audio and yield as single chunk
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
    
    // MARK: - AudioProcessor Conformance (inherited via SpeechSynthesizer)
    
    public var inputChannels: Int { 0 }  // Text input, no audio channels
    public var outputChannels: Int { 1 }
    
    public func process(_ input: AudioToolCore.AudioBuffer) async throws -> AudioToolCore.AudioBuffer {
        throw AudioToolError.pipelineConfigurationInvalid("KokoroTTS is a synthesizer, use synthesize() instead of process()")
    }
}

// MARK: - ManagedModel

extension KokoroTTSProvider: ManagedModel {
    public nonisolated var modelId: String {
        "kokoro:\(modelIdentity):\(precision.rawValue):\(language.rawValue)"
    }

    public nonisolated var estimatedMemoryBytes: Int {
        switch precision {
        case .fp32: 360_000_000
        case .fp16, .bf16: 220_000_000
        case .int8, .bit8: 150_000_000
        case .bit6: 130_000_000
        case .int4, .bit4: 110_000_000
        }
    }

    public func checkIfLoaded() async -> Bool { tts != nil }

    public func unload() async {
        loadGeneration &+= 1
        tts = nil
        voiceEmbeddings.removeAll()
        // `customVoiceSources` is durable configuration and is intentionally
        // retained so loadFromPath(_:) can reconstruct these embeddings.
        updateState(.notLoaded)
        GPU.clearCache()
    }
}

// MARK: - Voice Matching Extension

extension KokoroTTSProvider {
    
    /// Create a blended voice from voice match results
    ///
    /// Convenience method to convert `VoiceMatchResult` weights directly
    /// to a synthesizable voice embedding.
    ///
    /// Example:
    /// ```swift
    /// let matcher = KokoroVoiceMatcher()
    /// let result = try await matcher.matchVoice(...)
    /// let blended = try tts.blendedVoice(from: result)
    /// let audio = try await tts.synthesize("Hello!", voiceEmbedding: blended)
    /// ```
    ///
    /// - Parameter matchResult: Voice match result from `KokoroVoiceMatcher`
    /// - Returns: Blended MLXArray voice embedding
    /// - Throws: `AudioToolError.resourceUnavailable` if any voice is not loaded
    public func blendedVoice(from matchResult: VoiceMatchResult) throws -> MLXArray {
        try mixVoices(matchResult.weights)
    }
    
    /// Get voice embedding for a single voice
    ///
    /// - Parameter voiceId: Voice identifier (e.g., "af_bella")
    /// - Returns: Voice embedding if loaded, nil otherwise
    public func voiceEmbedding(for voiceId: String) -> MLXArray? {
        voiceEmbeddings[voiceId]
    }
    
    /// Get all loaded voice embeddings as arrays for matching
    ///
    /// Converts MLXArray embeddings to Float arrays for use with
    /// `KokoroVoiceMatcher`.
    ///
    /// - Note: Kokoro voice embeddings are different from speaker embeddings.
    ///   Voice matching uses speaker embeddings from synthesized audio.
    public var voiceEmbeddingsAsArrays: [String: [Float]] {
        var result: [String: [Float]] = [:]
        for (name, embedding) in voiceEmbeddings {
            result[name] = embedding.asArray(Float.self)
        }
        return result
    }
}
