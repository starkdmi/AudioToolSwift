//
//  KokoroTTSProvider.swift
//  ClearVoiceTTS
//
//  Kokoro TTS provider with MisakiSwift G2P (MIT license)
//

import Foundation
import ClearVoice
import ClearVoiceCore
import KokoroSwift
import MLX
import MLXNN
import CoreMedia

/// Typealias to avoid conflict with CoreAudio.AudioBuffer
public typealias TTSAudioBuffer = ClearVoiceCore.AudioBuffer

/// Kokoro TTS provider using MisakiSwift for G2P (MIT license, no ESpeakNG)
///
/// Performance: ~5.5-11.7x RTF on Apple Silicon
/// Languages: en-US, en-GB
///
/// Usage:
/// ```swift
/// let tts = TTSProviders.kokoro()  // Auto-downloads from HuggingFace
/// try await tts.load()  // Downloads if needed, shows progress
/// let audio = try await tts.synthesize("Hello world!", voice: "af_heart")
/// ```
public final class KokoroTTSProvider: SpeechSynthesizer, @unchecked Sendable {
    
    // MARK: - Public Properties
    
    /// Base HuggingFace repository (without precision suffix)
    public static let baseRepo = "mlx-community/Kokoro-82M"
    
    /// Default precision
    public static let defaultPrecision: ModelPrecision = .bf16
    
    /// Current loading state for UI binding
    public private(set) var state: ModelState = .notLoaded
    
    /// Observable state stream for UI
    public var stateStream: AsyncStream<ModelState> {
        AsyncStream { continuation in
            stateContinuations.append(continuation)
        }
    }
    
    // MARK: - SpeechSynthesizer Conformance
    
    public let sampleRate: Int = 24000
    
    public var availableVoices: [String] {
        Array(voiceEmbeddings.keys)
    }
    
    // MARK: - Private Properties
    
    private var tts: KokoroTTS?
    private var modelPath: URL?
    private let language: KokoroLanguage
    private let repo: String
    private let precision: ModelPrecision
    
    /// Voice embeddings cache (voice name -> MLXArray)
    private var voiceEmbeddings: [String: MLXArray] = [:]
    
    /// State stream continuations
    private var stateContinuations: [AsyncStream<ModelState>.Continuation] = []
    
    // MARK: - Initialization
    
    /// Initialize Kokoro TTS provider with precision-based repo selection
    /// - Parameters:
    ///   - precision: Model precision (determines repo: Kokoro-82M-bf16, Kokoro-82M-4bit, etc.)
    ///   - language: Target language for synthesis
    public init(
        precision: ModelPrecision = KokoroTTSProvider.defaultPrecision,
        language: KokoroLanguage = .americanEnglish
    ) {
        self.precision = precision
        self.repo = precision.repo(base: KokoroTTSProvider.baseRepo)
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
        self.language = language
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
                matching: ["*.safetensors", "voices/*.npy", "config.json"]
            ) { [weak self] progress in
                self?.updateState(.downloading(progress: progress.fractionCompleted))
            }
            
            try await loadFromPath(path)
        } catch {
            updateState(.failed(error.localizedDescription))
            throw error
        }
    }
    
    /// Load from a specific path (can be file or directory)
    private func loadFromPath(_ path: URL) async throws {
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
                    throw ClearVoiceError.modelNotFound("No .safetensors file found in \(path.lastPathComponent)")
                }
            }
            
            // KokoroSwift uses .misaki G2P by default (MIT license, no ESpeakNG)
            tts = KokoroTTS(modelPath: weightsPath, g2p: .misaki)
            modelPath = path
            
            // Try to auto-load voices from voices subdirectory (use original path, not weightsPath)
            let voicesDir = path.appendingPathComponent("voices")
            if FileManager.default.fileExists(atPath: voicesDir.path) {
                try? loadVoices(from: voicesDir)
            }
            
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
    
    // MARK: - Voice Loading
    
    /// Load a voice embedding from .npy or .safetensors file
    /// - Parameter voicePath: Path to voice style file
    /// - Returns: Voice name (filename without extension)
    @discardableResult
    public func loadVoice(from voicePath: URL) throws -> String {
        let voiceName = voicePath.deletingPathExtension().lastPathComponent
        
        // Load voice embedding based on file extension
        let embedding: MLXArray
        if voicePath.pathExtension == "safetensors" {
            let arrays = try MLX.loadArrays(url: voicePath)
            guard let voiceArray = arrays["voice"] ?? arrays.values.first else {
                throw ClearVoiceError.resourceUnavailable("No voice tensor found in \(voicePath.lastPathComponent)")
            }
            embedding = voiceArray
        } else {
            // .npy format - load directly
            embedding = try MLX.loadArray(url: voicePath)
        }
        
        voiceEmbeddings[voiceName] = embedding
        return voiceName
    }
    
    /// Load all voice embeddings from a directory
    /// - Parameter directory: Directory containing .npy or .safetensors voice files
    public func loadVoices(from directory: URL) throws {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "npy" || file.pathExtension == "safetensors" {
            _ = try loadVoice(from: file)
        }
    }
    
    // MARK: - SpeechSynthesizer Protocol
    
    /// Synthesize text to speech
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice name (must be loaded via `loadVoice`)
    /// - Returns: Audio buffer with synthesized speech (24kHz mono)
    public func synthesize(_ text: String, voice: String) async throws -> ClearVoiceCore.AudioBuffer {
        guard let tts = tts else {
            throw ClearVoiceError.modelNotLoaded("KokoroTTS")
        }
        
        guard let voiceEmbedding = voiceEmbeddings[voice] else {
            throw ClearVoiceError.resourceUnavailable("Voice '\(voice)' not loaded. Call loadVoice(from:) first.")
        }
        
        // Map KokoroLanguage to KokoroSwift.Language
        let kokoroLang: KokoroSwift.Language = switch language {
        case .americanEnglish: .enUS
        case .britishEnglish: .enGB
        case .japanese: .enUS  // Fallback - Japanese not in current kokoro-ios
        case .chinese: .enUS   // Fallback - Chinese not in current kokoro-ios
        }
        
        // Generate audio using Kokoro with MisakiSwift G2P
        // Returns ([Float], [MToken]?) tuple
        let (samples, _) = try tts.generateAudio(
            voice: voiceEmbedding,
            language: kokoroLang,
            text: text
        )
        
        return ClearVoiceCore.AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
    }
    
    /// Stream synthesized audio chunks
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice name
    /// - Returns: Async stream of audio chunks
    public func streamSynthesis(_ text: String, voice: String) -> AsyncThrowingStream<ClearVoiceCore.AudioBuffer, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // TODO: Implement sentence-level streaming for lower latency
                    // For now, generate full audio and yield as single chunk
                    let audio = try await synthesize(text, voice: voice)
                    continuation.yield(audio)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - AudioProcessor Conformance (inherited via SpeechSynthesizer)
    
    public var inputChannels: Int { 0 }  // Text input, no audio channels
    public var outputChannels: Int { 1 }
    
    public func process(_ input: ClearVoiceCore.AudioBuffer) async throws -> ClearVoiceCore.AudioBuffer {
        throw ClearVoiceError.pipelineConfigurationInvalid("KokoroTTS is a synthesizer, use synthesize() instead of process()")
    }
}
